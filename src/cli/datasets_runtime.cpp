#include "datasets_runtime.hpp"
#include "../util/file_util.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace trailhead {

std::string dataset_lib_sh() {
    // Bash helpers shared by batch_run, sbatch_gen, and local_run. Each test
    // wrapper calls th_ds_ensure(...) before running and th_ds_finish(...)
    // after. State lives under .trailhead/datasets/<name>/ on whatever
    // filesystem the wrapper is executing on (compute node shared FS for
    // SLURM, project root for local).
    return R"BASH(#!/bin/bash
# Trailhead dataset helpers — generated; do not edit by hand.
# th_ds_ensure <name>           : fetch dataset if not present (flock + sentinel)
# th_ds_finish <name> <test>    : record consumer; rm path + caches when refcount hits expected
#
# Per-dataset state: .trailhead/datasets/<name>/{path,fetch,cache_paths,expected,finished.txt,done,cleaned}

_th_ds_for_each_dep() {
  # _th_ds_for_each_dep <name> <fn> [arg...]
  # Calls <fn> <dep> [arg...] for each dataset listed in <name>/depends_on.
  local name="$1"; local fn="$2"; shift 2
  local d=".trailhead/datasets/${name}"
  if [ ! -f "${d}/depends_on" ]; then return 0; fi
  while IFS= read -r dep; do
    [ -n "$dep" ] && "$fn" "$dep" "$@"
  done < "${d}/depends_on"
}

th_ds_ensure() {
  local name="$1"
  local d=".trailhead/datasets/${name}"
  if [ ! -d "$d" ]; then return 0; fi
  # Walk dependencies first (DAG order: leaves fetched before parents).
  _th_ds_for_each_dep "$name" th_ds_ensure
  if [ -f "${d}/done" ] && [ ! -f "${d}/cleaned" ]; then return 0; fi
  if [ -f "${d}/cleaned" ]; then
    # Cleaned mid-run because expected was undercounted; refetch.
    rm -f "${d}/done" "${d}/cleaned"
  fi
  local fetch_cmd
  fetch_cmd=$(cat "${d}/fetch" 2>/dev/null || true)
  ( exec 200>"${d}/fetch.lock"
    flock 200
    if [ ! -f "${d}/done" ]; then
      echo "[trailhead] dataset fetch: ${name}"
      if bash -c "$fetch_cmd"; then
        touch "${d}/done"
      else
        echo "TRAILHEAD:fail:dataset_${name}_fetch" >&2
        flock -u 200; exit 1
      fi
    fi
    flock -u 200
  )
}

th_ds_finish() {
  local name="$1"
  local test_name="$2"
  local d=".trailhead/datasets/${name}"
  if [ ! -d "$d" ]; then return 0; fi
  ( exec 201>"${d}/finish.lock"
    flock 201
    echo "$test_name" >> "${d}/finished.txt"
    if [ -f "${d}/cleaned" ]; then flock -u 201; return 0; fi
    local fin exp
    fin=$(wc -l < "${d}/finished.txt" 2>/dev/null || echo 0)
    exp=$(cat "${d}/expected" 2>/dev/null || echo 0)
    if [ "$fin" -ge "$exp" ] && [ "$exp" -gt 0 ]; then
      touch "${d}/cleaned"
      local p
      p=$(cat "${d}/path" 2>/dev/null || true)
      [ -n "$p" ] && rm -rf "$p" 2>/dev/null
      if [ -f "${d}/cache_paths" ]; then
        while IFS= read -r line; do
          # Globs allowed in cache_paths; intentionally unquoted expansion.
          [ -n "$line" ] && rm -rf $line 2>/dev/null
        done < "${d}/cache_paths"
      fi
      echo "[trailhead] dataset cleaned: ${name} (after ${fin}/${exp})"
    fi
    flock -u 201
  )
  # Cascade finish to dependencies — each transitive dep counts this consumer.
  _th_ds_for_each_dep "$name" th_ds_finish "$test_name"
}
)BASH";
}

bool write_dataset_lib(const std::string& th_dir) {
    fs::mkdir_p(th_dir + "/lib");
    return fs::write_file_atomic(th_dir + "/lib/datasets.sh", dataset_lib_sh());
}

// Walk `name` and everything reachable through `depends_on`, appending to
// `out` in dependency-first order. Throws on a cycle; throws on dangling dep.
static void walk_deps_dfs(const Registry& reg,
                            const std::string& name,
                            std::unordered_set<std::string>& visited,
                            std::unordered_set<std::string>& on_stack,
                            std::vector<std::string>& out)
{
    if (visited.count(name)) return;
    if (on_stack.count(name))
        throw std::runtime_error("dataset dependency cycle involving '" + name + "'");
    auto it = reg.datasets.find(name);
    if (it == reg.datasets.end())
        throw std::runtime_error("dataset '" + name + "' referenced but not defined");
    on_stack.insert(name);
    for (const auto& dep : it->second.depends_on)
        walk_deps_dfs(reg, dep, visited, on_stack, out);
    on_stack.erase(name);
    visited.insert(name);
    out.push_back(name);
}

std::unordered_map<std::string, int> dataset_expected_counts(
        const Registry& reg,
        const std::vector<std::string>& selected_test_names)
{
    std::unordered_set<std::string> sel(selected_test_names.begin(), selected_test_names.end());
    std::unordered_map<std::string, int> counts;
    for (const auto& t : reg.tests) {
        if (!sel.count(t.name)) continue;
        // Each test contributes one consumer to every dataset in the
        // transitive closure of its declared datasets.
        std::unordered_set<std::string> seen;
        std::unordered_set<std::string> on_stack;
        std::vector<std::string> closure;
        for (const auto& d : t.datasets) {
            if (!reg.datasets.count(d)) continue;
            try { walk_deps_dfs(reg, d, seen, on_stack, closure); }
            catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                throw;
            }
        }
        for (const auto& d : closure) ++counts[d];
    }
    return counts;
}

std::vector<std::string> required_build_targets(
        const Registry& reg,
        const std::vector<std::string>& selected_test_names)
{
    auto counts = dataset_expected_counts(reg, selected_test_names);
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (const auto& [name, _exp] : counts) {
        auto it = reg.datasets.find(name);
        if (it == reg.datasets.end()) continue;
        for (const auto& tgt : it->second.requires_targets) {
            if (tgt.empty()) continue;
            if (seen.insert(tgt).second) out.push_back(tgt);
        }
    }
    return out;
}

std::vector<std::string> init_dataset_state(const Registry& reg,
                                              const std::vector<std::string>& selected_test_names,
                                              const std::string& th_dir)
{
    auto counts = dataset_expected_counts(reg, selected_test_names);

    std::vector<std::string> touched;
    for (const auto& [name, expected] : counts) {
        auto it = reg.datasets.find(name);
        if (it == reg.datasets.end()) continue;
        const auto& ds = it->second;

        std::string d = th_dir + "/datasets/" + name;
        fs::mkdir_p(d);

        fs::write_file_atomic(d + "/path",      ds.path + "\n");
        // Sub-registry datasets author their fetch_cmd with paths relative to
        // the sub-registry root (where CMakeLists.txt lives); wrap in a
        // subshell that cd's into sub_dir so mkdir/curl/etc. land there.
        std::string fetch = ds.fetch_cmd;
        if (!ds.sub_dir.empty())
            fetch = "cd " + ds.sub_dir + " && (" + fetch + ")";
        fs::write_file_atomic(d + "/fetch",     fetch + "\n");
        std::string caches;
        for (const auto& cp : ds.cache_paths) caches += cp + "\n";
        fs::write_file_atomic(d + "/cache_paths", caches);
        std::string deps;
        for (const auto& dp : ds.depends_on) deps += dp + "\n";
        fs::write_file_atomic(d + "/depends_on", deps);
        fs::write_file_atomic(d + "/expected",  std::to_string(expected) + "\n");
        // Reset refcount + cleanup state for this run.
        fs::write_file_atomic(d + "/finished.txt", "");
        std::remove((d + "/cleaned").c_str());
        // `done` left intact so cached fetches survive across invocations.
        touched.push_back(name);
    }
    return touched;
}

} // namespace trailhead
