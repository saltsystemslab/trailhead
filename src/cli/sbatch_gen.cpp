#include "sbatch_gen.hpp"
#include "datasets_runtime.hpp"
#include "../util/file_util.hpp"
#include <sstream>
#include <algorithm>
#include <map>
#include <set>

namespace trailhead {

// Classify a setup command into an execution phase by its leading verb. Lower
// phases run first. Returns -1 for anything unrecognised (callers then keep
// such steps strictly sequential to be safe).
//   0 prep · 1 download · 2 extract · 3 move/install · 4 cleanup
static int setup_phase(const std::string& cmd) {
    size_t b = cmd.find_first_not_of(" \t");
    if (b == std::string::npos) return -1;
    size_t e = cmd.find_first_of(" \t", b);
    std::string tok = cmd.substr(b, e == std::string::npos ? std::string::npos : e - b);
    if (auto sl = tok.rfind('/'); sl != std::string::npos) tok = tok.substr(sl + 1);

    auto is = [&](std::initializer_list<const char*> opts) {
        for (const char* o : opts) if (tok == o) return true;
        return false;
    };
    if (is({"mkdir"}))                                                    return 0;
    if (is({"curl", "wget", "aria2c", "git"}))                            return 1;
    if (is({"unzip", "tar", "gunzip", "gzip", "bunzip2", "bzip2",
            "xz", "unxz", "zstd", "unzstd", "7z", "7za"}))                return 2;
    if (is({"mv", "cp", "install", "ln"}))                                return 3;
    if (is({"rm", "rmdir"}))                                              return 4;
    return -1;
}

std::vector<std::vector<std::pair<int, std::string>>>
plan_setup_stages(const std::vector<std::string>& setup) {
    std::vector<std::vector<std::pair<int, std::string>>> stages;

    bool has_barrier = std::find(setup.begin(), setup.end(), "---") != setup.end();

    // Index non-barrier items in original order — these indices are stable for
    // a given setup list, so per-item sentinels/locks line up across chunks.
    std::vector<std::pair<int, std::string>> items;
    { int idx = 0; for (const auto& s : setup) if (s != "---") items.push_back({idx++, s}); }
    if (items.empty()) return stages;

    if (has_barrier) {
        // Manual barriers: split on "---", dropping empty groups.
        stages.push_back({});
        int idx = 0;
        for (const auto& s : setup) {
            if (s == "---") { stages.push_back({}); continue; }
            stages.back().push_back({idx++, s});
        }
        stages.erase(std::remove_if(stages.begin(), stages.end(),
            [](const auto& v) { return v.empty(); }), stages.end());
        return stages;
    }

    // Auto mode: only group by phase if every step is recognised; otherwise
    // run sequentially so an unknown command's ordering is never broken.
    bool all_known = std::all_of(items.begin(), items.end(),
        [](const auto& it) { return setup_phase(it.second) >= 0; });
    if (!all_known) {
        for (const auto& it : items) stages.push_back({it});
        return stages;
    }

    // std::map keeps phases ordered; insertion order within a phase is preserved.
    std::map<int, std::vector<std::pair<int, std::string>>> buckets;
    for (const auto& it : items) buckets[setup_phase(it.second)].push_back(it);
    for (auto& [_, vec] : buckets) stages.push_back(std::move(vec));
    return stages;
}

// Emit #SBATCH header lines for a node profile merged with sbatch_defaults
static std::string sbatch_headers(const NodeProfile& node,
                                   const SbatchDefaults& defs,
                                   const std::string& job_name)
{
    std::ostringstream o;
    auto line = [&](const std::string& key, const std::string& val) {
        if (!val.empty()) o << "#SBATCH --" << key << "=" << val << "\n";
    };
    auto line_int = [&](const std::string& key, int val, int skip_if = 0) {
        if (val != skip_if) o << "#SBATCH --" << key << "=" << val << "\n";
    };

    o << "#SBATCH --job-name=" << job_name << "\n";
    line("partition", node.partition);

    // Hardware targeting: gpu_type (--gres=gpu:<type>) takes priority over nodelist
    if (!node.gpu_type.empty()) {
        // Request a specific GPU model across the partition (no nodelist pinning)
        o << "#SBATCH --gres=gpu:" << node.gpu_type << "\n";
    } else if (!node.nodelist.empty()) {
        // Pin to a specific node; request 1 GPU of whatever type is there
        line("nodelist", node.nodelist);
        o << "#SBATCH --gres=gpu:1\n";
    }

    line_int("nodes",         node.nodes,         0);
    line_int("ntasks",        node.ntasks,        0);
    line_int("cpus-per-task", node.cpus_per_task, 0);
    // Always emit --time so an empty profile time can't fall through to the QOS
    // DefaultTime (often only minutes); default to 1h.
    o << "#SBATCH --time=" << (node.time.empty() ? "01:00:00" : node.time) << "\n";
    if (!node.account.empty()) line("account", node.account);
    line("output",  defs.output_pattern);
    line("error",   defs.error_pattern);

    // Extra directives from the node profile
    for (const auto& [k, v] : node.extra)
        line(k, v);

    return o.str();
}

// Extract the remote path from an rsync_dest string "user@host:/path" → "/path".
// Returns empty string if not parseable.
static std::string remote_path_from_rsync_dest(const std::string& rsync_dest) {
    auto colon = rsync_dest.find(':');
    if (colon == std::string::npos || colon + 1 >= rsync_dest.size()) return "";
    return rsync_dest.substr(colon + 1);
}

// Resolve the remote working directory for a script.
// rsync_dest is always the *parent* directory; project_name (last component of
// local project_root) is appended so the path matches where do_rsync puts the files.
static std::string resolve_effective_root(const std::string& bc_rsync_dest,
                                           const std::string& node_rsync_dest,
                                           const std::string& fallback,
                                           const std::string& project_name = "")
{
    for (const auto& d : {bc_rsync_dest, node_rsync_dest}) {
        if (!d.empty()) {
            auto rp = remote_path_from_rsync_dest(d);
            if (!rp.empty())
                return project_name.empty() ? rp : rp + "/" + project_name;
        }
    }
    return fallback;
}

// Replace all occurrences of `from` with `to` in `s`
static std::string str_replace_all(std::string s,
                                    const std::string& from,
                                    const std::string& to)
{
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Build the body of a script (preamble + configure + commands)
// build_dir:      the directory cmake builds into for this node (e.g. "build_h200")
// configure_cmd:  cmake configure command from the build config (run once on the node)
// node_preamble:  per-node shell lines emitted after project preamble (module loads, exports, etc.)
// parent_project: absolute remote path of the root project (non-empty only for sub-registry tests)
// parent_setup:   root project setup commands, run before own setup when parent_project is set
static std::string sbatch_body(const std::vector<const TestEntry*>& tests,
                                const SbatchDefaults& defs,
                                const std::string& project_root,
                                const std::string& build_dir,
                                const std::string& configure_cmd,
                                const std::vector<std::string>& setup = {},
                                const std::vector<std::string>& node_preamble = {},
                                const std::string& parent_project = {},
                                const std::vector<std::string>& parent_setup = {},
                                const std::vector<std::string>& extra_targets = {},
                                int build_jobs = 1)
{
    std::ostringstream o;

    if (!node_preamble.empty()) {
        // Node preamble takes full control: skip project-level modules and preamble entirely.
        // This avoids duplication when the sub-registry preamble overlaps with the node preamble.
        for (const auto& line : node_preamble)
            o << line << "\n";
    } else {
        // No node preamble: use project-level modules + preamble
        for (const auto& mod : defs.modules)
            o << "module load " << mod << "\n";
        if (!defs.modules.empty()) o << "\n";
        for (const auto& line : defs.preamble)
            o << line << "\n";
    }
    o << "\n";

    // Set TRAILHEAD_JOB_ID so reporter.hpp labels results as sbatch-<id>
    o << "export TRAILHEAD_JOB_ID=$SLURM_JOB_ID\n";
    o << "export TRAILHEAD_ENABLED=1\n";
    if (!build_dir.empty())
        o << "export TRAILHEAD_BUILD_DIR=" << build_dir << "\n";
    if (!project_root.empty())
        o << "cd " << project_root << "\n";

    // Source dataset helpers if any test in this script lists a dataset.
    // The runtime is a no-op if .trailhead/datasets/<name>/ is absent (e.g.
    // when this script is submitted manually without running the dataset
    // init pre-step), so wrapping is always safe.
    bool any_datasets = false;
    for (const TestEntry* t : tests)
        if (!t->datasets.empty()) { any_datasets = true; break; }
    if (any_datasets)
        o << "[ -f .trailhead/lib/datasets.sh ] && source .trailhead/lib/datasets.sh\n";
    o << "\n";

    // Root project setup: run before own setup for sub-registry tests that depend on it.
    if (!parent_project.empty() && !parent_setup.empty()) {
        o << "if [ ! -f " << parent_project << "/.trailhead/setup_done ]; then\n";
        o << "  (\n";
        o << "    cd " << parent_project << "\n";
        for (const auto& s : parent_setup)
            o << "    " << s << "\n";
        o << "    touch .trailhead/setup_done\n";
        o << "  )\n";
        o << "fi\n\n";
    }

    // Project setup: submodule init, dataset downloads, etc.
    // Guarded by .trailhead/setup_done so it only runs once per remote workspace.
    // Steps are grouped into stages by plan_setup_stages(): explicit "---"
    // barriers split manually, otherwise recognised dataset-prep verbs are
    // phased (downloads in parallel, then extracts, then moves), and anything
    // unrecognised stays strictly sequential.
    if (!setup.empty()) {
        o << "if [ ! -f .trailhead/setup_done ]; then\n";
        for (const auto& stage : plan_setup_stages(setup)) {
            if (stage.size() == 1) {
                // Tolerate a non-zero exit (e.g. `mkdir datasets` when it already
                // exists) so a non-idempotent step doesn't abort the whole job
                // under `set -e`; log it instead and continue.
                o << "  " << stage[0].second
                  << " || echo \"[trailhead] setup step exited $? (continuing)\"\n";
            } else {
                for (const auto& [_, cmd] : stage)
                    o << "  " << cmd << " &\n";
                o << "  wait\n";   // bare `wait` returns 0, so a bg failure won't trip set -e
            }
        }
        o << "  touch .trailhead/setup_done\n";
        o << "fi\n\n";
    }

    // Build phase: configure + cmake --build. Multiple per-test jobs can land on
    // the same node and share this build tree, so serialise the whole phase with
    // an flock keyed by the build dir — concurrent cmake/ninja runs on one tree
    // otherwise corrupt the cache and race on object files ("file lock held" /
    // stale file errors). The lock lives in the current project's .trailhead, so
    // parent and sub-registry build trees get independent locks. The check for an
    // already-configured tree and the per-target builds all run inside the lock,
    // so only one job configures and only one builds a given target at a time.
    if (!build_dir.empty()) {
        std::string lock_key = build_dir;
        for (auto& c : lock_key) if (c == '/') c = '_';
        o << "mkdir -p .trailhead\n";
        o << "exec 200>\".trailhead/build_" << lock_key << ".lock\"\n";
        // Bounded wait: a stale lock (e.g. a prior holder killed at the SLURM
        // time limit, or flock-over-NFS not releasing) must not make us block
        // forever and hit our own time limit. After the wait we proceed anyway —
        // the configure check below is idempotent and the build is incremental.
        o << "flock -w \"${TRAILHEAD_BUILD_LOCK_WAIT:-600}\" 200 "
             "|| echo \"[trailhead] build lock wait timed out — proceeding\"\n";

        // Configure once (auto-detects GPU arch on the compute node). For the
        // cmake ".." run-from-build-dir form, cd into the build dir first.
        if (!configure_cmd.empty()) {
            std::string cfg = str_replace_all(configure_cmd, "-B build", "-B " + build_dir);
            bool from_build = cfg.size() >= 3 && cfg.substr(cfg.size() - 3) == " ..";
            o << "if [ ! -f " << build_dir << "/Makefile ] && [ ! -f " << build_dir << "/build.ninja ]; then\n";
            if (from_build)
                o << "  mkdir -p " << build_dir << " && (cd " << build_dir << " && " << cfg << ")\n";
            else
                o << "  " << cfg << "\n";
            o << "  if [ $? -ne 0 ]; then echo \"TRAILHEAD:build_fail\"; flock -u 200; exit 1; fi\n";
            o << "fi\n";
        }

        // Build dataset-required targets (converters, etc.) plus each test's
        // target, deduplicated. Done up front under the lock so the per-test
        // loop below only runs binaries — no build commands touch the tree once
        // the lock is released.
        std::vector<std::string> build_targets;
        std::set<std::string> seen_tgt;
        for (const auto& tgt : extra_targets)
            if (seen_tgt.insert(tgt).second) build_targets.push_back(tgt);
        for (const TestEntry* t : tests)
            if (!t->build_name.empty() && !t->target.empty()
                && seen_tgt.insert(t->target).second)
                build_targets.push_back(t->target);
        int j = std::max(1, build_jobs);
        for (const auto& tgt : build_targets)
            o << "cmake --build " << build_dir << " --target " << tgt
              << " -j " << j
              << " || { echo \"TRAILHEAD:build_fail\"; flock -u 200; exit 1; }\n";

        o << "flock -u 200\n\n";
    }

    for (const TestEntry* t : tests) {
        o << "# " << (t->label.empty() ? t->name : t->label) << "\n";
        // Substitute build dir in the run cmd (e.g. "build/tests/foo" → "build_h200/tests/foo")
        std::string cmd = str_replace_all(t->cmd, "build/", build_dir + "/");

        // When workdir is default and a build is linked, run from inside the build directory.
        // Strip build_dir/ references from cmd since we're already cd'd there.
        std::string effective_wd = t->workdir;
        if ((effective_wd.empty() || effective_wd == ".") && !build_dir.empty() && !t->build_name.empty()) {
            effective_wd = build_dir;
            cmd = str_replace_all(cmd, "./" + build_dir + "/", "./");
            cmd = str_replace_all(cmd, build_dir + "/", "./");
        }

        // Datasets: ensure before, finish-with-refcount-cleanup after. `|| true`
        // so set -e doesn't trip on a transient flock or missing helper.
        for (const auto& d : t->datasets)
            o << "declare -F th_ds_ensure >/dev/null && th_ds_ensure "
              << d << " || true\n";

        // Run the test, capturing exit code so finish always executes even
        // when the test (or surrounding `set -e`) would otherwise abort.
        if (!t->datasets.empty()) o << "_th_rc=0\n";
        if (!effective_wd.empty() && effective_wd != ".") {
            if (!t->datasets.empty())
                o << "( cd " << effective_wd << " && " << cmd << " ) || _th_rc=$?\n";
            else
                o << "(\n  cd " << effective_wd << "\n  " << cmd << "\n)\n";
        } else {
            if (!t->datasets.empty())
                o << "( " << cmd << " ) || _th_rc=$?\n";
            else
                o << cmd << "\n";
        }

        if (!t->datasets.empty()) {
            for (const auto& d : t->datasets)
                o << "declare -F th_ds_finish >/dev/null && th_ds_finish "
                  << d << " " << t->name << " || true\n";
            o << "[ \"$_th_rc\" -eq 0 ] || true   # don't propagate test failure to set-e\n";
        }
        o << "\n";
    }

    return o.str();
}

// For sub-registry builds where effective_root already navigates into the sub-dir,
// strip the sub-dir prefix from a test's workdir so it stays relative to the new root.
static std::string adjust_workdir(const std::string& workdir, const std::string& sub_dir_name) {
    if (sub_dir_name.empty()) return workdir;
    if (workdir == sub_dir_name) return ".";
    if (workdir.rfind(sub_dir_name + "/", 0) == 0)
        return workdir.substr(sub_dir_name.size() + 1);
    return workdir;
}

// Return the sbatch_defaults for a test: sub-registry's own if available, else parent's.
static const SbatchDefaults& effective_defs(const Registry& reg, const TestEntry& t) {
    if (!t.sub_dir.empty()) {
        auto it = reg.sub_sbatch_defaults.find(t.sub_dir);
        if (it != reg.sub_sbatch_defaults.end()) return it->second;
    }
    return reg.sbatch_defaults;
}

// Return the setup steps for a test: sub-registry's own if available, else parent's.
static const std::vector<std::string>& effective_setup(const Registry& reg, const TestEntry& t) {
    if (!t.sub_dir.empty()) {
        auto it = reg.sub_setups.find(t.sub_dir);
        if (it != reg.sub_setups.end()) return it->second;
        // Test belongs to a sub-registry that has no setup — don't fall through to parent's setup
        static const std::vector<std::string> empty;
        return empty;
    }
    return reg.setup;
}

// Append the sub-dir name to effective_root when a sub-registry build relies on the node
// rsync_dest (has no build-specific rsync_dest). Returns the sub-dir name used (or "").
static std::string maybe_append_sub_dir(std::string& effective_root,
                                         const BuildConfig& bc,
                                         const std::string& node_rsync)
{
    if (bc.sub_dir.empty() || (bc.rsync_dest.empty() && node_rsync.empty())) return "";
    std::string name = bc.sub_dir;
    auto sl = name.rfind('/');
    if (sl != std::string::npos) name = name.substr(sl + 1);
    effective_root += "/" + name;
    return name;
}

std::vector<std::pair<std::string,std::string>>
generate_sbatch(const Registry& reg, const SbatchOptions& opts)
{
    std::vector<std::pair<std::string,std::string>> out;

    // rsync_dest is always the parent dir; append the local project name to get the remote path
    std::string project_name;
    {
        auto sl = opts.project_root.rfind('/');
        project_name = (sl != std::string::npos && sl + 1 < opts.project_root.size())
            ? opts.project_root.substr(sl + 1) : opts.project_root;
    }

    if (opts.split) {
        // One script per test
        for (const auto& t : reg.tests) {
            std::ostringstream script;
            script << "#!/bin/bash\n";

            // Emit hardware headers if a node was specified (via opts.node_name)
            const NodeProfile* node_ptr = nullptr;
            const SbatchDefaults& defs = effective_defs(reg, t);
            std::string job_name = defs.job_name_prefix + "-" + t.name;
            if (!opts.node_name.empty()) {
                auto it = reg.nodes.find(opts.node_name);
                if (it != reg.nodes.end()) {
                    node_ptr = &it->second;
                    script << sbatch_headers(it->second, defs, job_name);
                }
            } else {
                script << "#SBATCH --job-name=" << job_name << "\n";
                script << "# No hardware target — use 'trailhead gen --node <profile>'\n";
            }
            script << "\n";

            // Resolve build dir, configure_cmd, and effective project root
            std::string build_dir     = "build";
            std::string configure_cmd;
            std::string effective_root = opts.project_root;
            std::string node_rsync = node_ptr ? node_ptr->rsync_dest : "";
            std::string sub_dir_name;
            std::string root_remote;
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    if (!bit->second.dir.empty()) build_dir = bit->second.dir;
                    configure_cmd = bit->second.configure_cmd;
                    effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                            opts.project_root, project_name);
                    root_remote = effective_root;
                    sub_dir_name = maybe_append_sub_dir(effective_root, bit->second, node_rsync);
                }
            }
            // Node-specific build dir: explicit setting wins; otherwise append node
            // name to the config's dir to avoid arch collisions across nodes.
            if (node_ptr) {
                if (!node_ptr->build_dir.empty())
                    build_dir = node_ptr->build_dir;
                else if (!opts.node_name.empty())
                    build_dir = build_dir + "_" + opts.node_name;
            }

            // No linked build config — fall back to any registered build config
            if (configure_cmd.empty()) {
                for (const auto& [bname, bc] : reg.builds) {
                    if (!bc.configure_cmd.empty()) {
                        configure_cmd = bc.configure_cmd;
                        if (effective_root == opts.project_root) {
                            effective_root = resolve_effective_root(bc.rsync_dest, node_rsync,
                                                                    opts.project_root, project_name);
                            root_remote = effective_root;
                            maybe_append_sub_dir(effective_root, bc, node_rsync);
                        }
                        break;
                    }
                }
            }

            if (configure_cmd.find("{{arch}}") != std::string::npos) {
                std::string arch_val = (node_ptr && !node_ptr->cuda_arch.empty())
                    ? node_ptr->cuda_arch
                    : "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')";
                configure_cmd = str_replace_all(configure_cmd, "{{arch}}", arch_val);
            }
            TestEntry t_adj = t;
            t_adj.workdir = adjust_workdir(t.workdir, sub_dir_name);

            std::string parent_project;
            std::vector<std::string> parent_setup_vec;
            if (!t.sub_dir.empty() && !reg.setup.empty() && !root_remote.empty()
                    && root_remote != effective_root) {
                parent_project = root_remote;
                parent_setup_vec = reg.setup;
            }

            // Datasets used by this single test contribute requires_targets.
            std::vector<std::string> extra_tgts =
                required_build_targets(reg, {t.name});

            script << sbatch_body({&t_adj}, defs, effective_root,
                                  build_dir, configure_cmd, effective_setup(reg, t),
                                  node_ptr ? node_ptr->preamble : std::vector<std::string>{},
                                  parent_project, parent_setup_vec,
                                  extra_tgts,
                                  node_ptr ? node_ptr->cpus_per_task : 1);

            out.push_back({t.name + ".sbatch", script.str()});
        }
    } else {
        // Single combined script
        std::ostringstream script;
        script << "#!/bin/bash\n";

        // Use opts.node_name for headers if specified
        const NodeProfile* node_ptr = nullptr;
        std::string job_name = reg.sbatch_defaults.job_name_prefix + "-all";
        if (!opts.node_name.empty()) {
            auto it = reg.nodes.find(opts.node_name);
            if (it != reg.nodes.end()) {
                node_ptr = &it->second;
                script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
            }
        } else {
            script << "#SBATCH --job-name=" << job_name << "\n";
            script << "#SBATCH --output=" << reg.sbatch_defaults.output_pattern << "\n";
            script << "#SBATCH --error="  << reg.sbatch_defaults.error_pattern  << "\n";
            script << "# No hardware target — use 'trailhead gen --node <profile>'\n";
        }
        script << "\n";

        std::string build_dir = "build";
        std::string configure_cmd;
        std::string effective_root = opts.project_root;
        std::string node_rsync = node_ptr ? node_ptr->rsync_dest : "";
        if (node_ptr) {
            if (!node_ptr->build_dir.empty())
                build_dir = node_ptr->build_dir;
            // else: auto-generate after we know the build config name
        }
        // Pick configure_cmd and remote root from whichever build config is referenced first
        std::string first_build_name;
        for (const auto& t : reg.tests) {
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    first_build_name = t.build_name;
                    configure_cmd = bit->second.configure_cmd;
                    effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                            opts.project_root, project_name);
                    break;
                }
            }
        }
        // Fall back to any registered build config if none was linked
        if (configure_cmd.empty()) {
            for (const auto& [bname, bc] : reg.builds) {
                if (!bc.configure_cmd.empty()) {
                    configure_cmd = bc.configure_cmd;
                    if (effective_root == opts.project_root)
                        effective_root = resolve_effective_root(bc.rsync_dest, node_rsync,
                                                                opts.project_root, project_name);
                    break;
                }
            }
        }

        // Auto-generate build dir: append node name to config's dir
        if (node_ptr && node_ptr->build_dir.empty() && !opts.node_name.empty()) {
            build_dir = build_dir + "_" + opts.node_name;
        }

        if (node_ptr && !node_ptr->cuda_arch.empty())
            configure_cmd = str_replace_all(configure_cmd, "{{arch}}", node_ptr->cuda_arch);
        std::vector<const TestEntry*> ptrs;
        std::vector<std::string> all_names;
        for (const auto& t : reg.tests) {
            ptrs.push_back(&t);
            all_names.push_back(t.name);
        }
        std::vector<std::string> extra_tgts = required_build_targets(reg, all_names);
        script << sbatch_body(ptrs, reg.sbatch_defaults, effective_root,
                              build_dir, configure_cmd, reg.setup,
                              node_ptr ? node_ptr->preamble : std::vector<std::string>{},
                              {}, {}, extra_tgts,
                              node_ptr ? node_ptr->cpus_per_task : 1);

        out.push_back({"run_all.sbatch", script.str()});
    }

    return out;
}

bool write_sbatch(const std::string& trailhead_dir,
                  const Registry& reg,
                  const SbatchOptions& opts)
{
    std::string sbatch_dir = trailhead_dir + "/sbatch";
    fs::mkdir_p(sbatch_dir);
    auto scripts = generate_sbatch(reg, opts);
    bool ok = true;
    for (const auto& [name, content] : scripts) {
        std::string path = sbatch_dir + "/" + name;
        auto slash = path.rfind('/');
        if (slash != std::string::npos) fs::mkdir_p(path.substr(0, slash));
        if (!fs::write_file_atomic(path, content)) ok = false;
    }
    return ok;
}

std::string generate_test_script(const TestEntry& test,
                                  const std::string& node_name,
                                  const Registry& reg,
                                  const SbatchOptions& opts)
{
    std::ostringstream script;
    script << "#!/bin/bash\n";

    const SbatchDefaults& defs = effective_defs(reg, test);
    std::string job_name = defs.job_name_prefix + "-" + test.name;
    const NodeProfile* node_ptr = nullptr;
    if (!node_name.empty()) {
        auto it = reg.nodes.find(node_name);
        if (it != reg.nodes.end()) {
            node_ptr = &it->second;
            script << sbatch_headers(it->second, defs, job_name);
        }
    }
    script << "\n";

    std::string project_name;
    {
        auto sl = opts.project_root.rfind('/');
        project_name = (sl != std::string::npos && sl + 1 < opts.project_root.size())
            ? opts.project_root.substr(sl + 1) : opts.project_root;
    }

    std::string build_dir     = "build";
    std::string configure_cmd;
    std::string effective_root = opts.project_root;
    std::string node_rsync = node_ptr ? node_ptr->rsync_dest : "";
    std::string sub_dir_name;
    std::string root_remote;  // root project remote path (before sub-dir appended)
    if (!test.build_name.empty()) {
        auto bit = reg.builds.find(test.build_name);
        if (bit != reg.builds.end()) {
            if (!bit->second.dir.empty()) build_dir = bit->second.dir;
            configure_cmd = bit->second.configure_cmd;
            effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                    opts.project_root, project_name);
            root_remote = effective_root;
            sub_dir_name = maybe_append_sub_dir(effective_root, bit->second, node_rsync);
        }
    }
    if (node_ptr) {
        if (!node_ptr->build_dir.empty())
            build_dir = node_ptr->build_dir;
        else if (!node_name.empty())
            build_dir = build_dir + "_" + node_name;
    }
    if (configure_cmd.empty()) {
        for (const auto& [bname, bc] : reg.builds) {
            if (!bc.configure_cmd.empty()) {
                configure_cmd = bc.configure_cmd;
                if (effective_root == opts.project_root) {
                    effective_root = resolve_effective_root(bc.rsync_dest, node_rsync,
                                                            opts.project_root, project_name);
                    root_remote = effective_root;
                    maybe_append_sub_dir(effective_root, bc, node_rsync);
                }
                break;
            }
        }
    }

    if (configure_cmd.find("{{arch}}") != std::string::npos) {
        std::string arch_val = (node_ptr && !node_ptr->cuda_arch.empty())
            ? node_ptr->cuda_arch
            : "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')";
        configure_cmd = str_replace_all(configure_cmd, "{{arch}}", arch_val);
    }
    TestEntry test_adj = test;
    test_adj.workdir = adjust_workdir(test.workdir, sub_dir_name);

    // For sub-registry tests: also run root setup if the root has setup commands
    std::string parent_project;
    std::vector<std::string> parent_setup_vec;
    if (!test.sub_dir.empty() && !reg.setup.empty() && !root_remote.empty()
            && root_remote != effective_root) {
        parent_project = root_remote;
        parent_setup_vec = reg.setup;
    }

    std::vector<std::string> extra_tgts = required_build_targets(reg, {test.name});

    script << sbatch_body({&test_adj}, defs, effective_root,
                          build_dir, configure_cmd, effective_setup(reg, test),
                          node_ptr ? node_ptr->preamble : std::vector<std::string>{},
                          parent_project, parent_setup_vec, extra_tgts,
                          node_ptr ? node_ptr->cpus_per_task : 1);
    return script.str();
}

} // namespace trailhead
