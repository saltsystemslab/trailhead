#pragma once
#include "json.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace trailhead {

// ── Build configuration ───────────────────────────────────────────────────
// Defines a cmake (or any) build that tests can share.
// Run order per test group:
//   1. configure_cmd  — once, only if build dir is absent
//   2. rsync          — once per group, if rsync_dest is set
//   3. per-test:  cmake --build <dir> --target <test.target>
//   4. per-test:  run test cmd (via sh -c, supports &&, pipes, multi-line)
struct BuildConfig {
    std::string name;
    std::string dir;            // build output directory (e.g. "./build")
    std::string configure_cmd;  // cmake configure step (run once if dir absent)
    std::string build_cmd;      // fallback full build (used by `build run` command)
    // rsync: sync source tree to remote before building/running
    std::string rsync_src;      // local source dir to sync (default: project root)
    std::string rsync_dest;     // remote destination (e.g. user@host:/path/to/project)
    // Set at load time for builds merged from sub-registries. Not serialized.
    // Relative path from project root to the sub-registry root (e.g. "andes_benchmarks").
    std::string sub_dir;
};

// ── Node profile ─────────────────────────────────────────────────────────
struct NodeProfile {
    std::string name;
    std::string partition;

    // Two ways to target hardware — set one or the other, not both:
    //   nodelist:  pins a specific node  (e.g. "d4067" for RTX 6000)
    //              generates: --nodelist=d4067  --gres=gpu:1
    //   gpu_type:  requests a GPU model across the partition
    //              (e.g. "h200" → --gres=gpu:h200, no --nodelist)
    std::string nodelist;
    std::string gpu_type;

    int nodes         = 1;
    int ntasks        = 1;
    int cpus_per_task = 1;
    std::string time  = "01:00:00";
    std::string account;
    // Remote rsync destination for this node (user@host:/path).
    // When set, trailhead rsyncs the project here before submitting sbatch jobs.
    // Takes precedence over any rsync_dest set on BuildConfig.
    std::string rsync_dest;
    // Per-node build directory. cmake auto-detects GPU arch when run on the node,
    // so each node type needs its own build dir to avoid clobbering each other.
    // If empty, falls back to the build config's dir ("build").
    std::string build_dir;
    // CUDA compute capability (e.g. "90" for H200, "86" for RTX 3090).
    // Substituted as {{arch}} in build config configure_cmd at script-generation and local-run time.
    std::string cuda_arch;
    std::unordered_map<std::string,std::string> extra;
};

// ── Global sbatch defaults ────────────────────────────────────────────────
struct SbatchDefaults {
    std::string output_pattern  = ".trailhead/slurm-%j.out";
    std::string error_pattern   = ".trailhead/slurm-%j.err";
    std::string job_name_prefix = "th";
    std::vector<std::string> preamble;
    std::vector<std::string> modules;
};

// ── Test entry ────────────────────────────────────────────────────────────
struct TestEntry {
    std::string name;
    std::string label;
    // cmd is run via sh -c, so it can be multi-line or contain &&, pipes, etc.
    // Example: "cd results && ./build/my_test --verbose && diff out.txt ref.txt"
    std::string cmd;
    std::string workdir       = ".";
    int timeout_sec           = 300;
    std::vector<std::string> tags;
    std::string build_name;     // references BuildConfig::name; empty = no build step
    // cmake target to build before running this test.
    // Defaults to test name when build_name is set (override with --target "").
    // Rebuilt with: cmake --build <build.dir> --target <target>
    std::string target;
    // Hardware requirement hint for display and compatibility warnings.
    // Values: "" or "any" = no constraint, "gpu" = needs GPU, "cpu" = CPU-only
    std::string requires_hw;
    // Set at load time for tests merged from sub-registries. Not serialized.
    // Relative path from project root to the sub-registry root (e.g. "andes_benchmarks").
    std::string sub_dir;
};

// ── Registry ──────────────────────────────────────────────────────────────
struct Registry {
    int version = 1;
    std::unordered_map<std::string, BuildConfig>  builds;
    std::unordered_map<std::string, NodeProfile>  nodes;
    SbatchDefaults sbatch_defaults;
    std::vector<TestEntry> tests;
    // One-time project setup commands: run before cmake configure in every
    // sbatch script, and via `trailhead setup run` for local bootstrapping.
    // Typical use: submodule init, dataset downloads, etc.
    std::vector<std::string> setup;
    // Relative paths to sub-registry roots (e.g. git submodules).
    // Tests from each are merged into the view at load time.
    std::vector<std::string> sub_registries;
    // Per-sub-registry sbatch_defaults, keyed by sub_rel path (e.g. "gunrock").
    // Populated by merge_sub_registries. Not serialized.
    std::unordered_map<std::string, SbatchDefaults> sub_sbatch_defaults;
};

// ── Serialisation ─────────────────────────────────────────────────────────

inline BuildConfig build_from_json(const std::string& name, const JsonValue& v) {
    BuildConfig b;
    b.name          = name;
    b.dir           = v.get_str("dir");
    b.configure_cmd = v.get_str("configure_cmd");
    b.build_cmd     = v.get_str("build_cmd");
    b.rsync_src     = v.get_str("rsync_src");
    b.rsync_dest    = v.get_str("rsync_dest");
    return b;
}

inline JsonValue build_to_json(const BuildConfig& b) {
    JsonObject obj;
    if (!b.dir.empty())           obj.push_back({"dir",           b.dir});
    if (!b.configure_cmd.empty()) obj.push_back({"configure_cmd", b.configure_cmd});
    if (!b.build_cmd.empty())     obj.push_back({"build_cmd",     b.build_cmd});
    if (!b.rsync_src.empty())     obj.push_back({"rsync_src",     b.rsync_src});
    if (!b.rsync_dest.empty())    obj.push_back({"rsync_dest",    b.rsync_dest});
    return JsonValue(std::move(obj));
}

inline NodeProfile node_from_json(const std::string& name, const JsonValue& v) {
    NodeProfile n;
    n.name         = name;
    n.partition    = v.get_str("partition");
    n.nodelist     = v.get_str("nodelist");
    n.gpu_type     = v.get_str("gpu_type");
    n.nodes        = (int)v.get_int("nodes", 1);
    n.ntasks       = (int)v.get_int("ntasks", 1);
    n.cpus_per_task= (int)v.get_int("cpus_per_task", 1);
    n.time         = v.get_str("time", "01:00:00");
    n.account      = v.get_str("account");
    n.rsync_dest   = v.get_str("rsync_dest");
    n.build_dir    = v.get_str("build_dir");
    n.cuda_arch    = v.get_str("cuda_arch");
    static const std::vector<std::string> known = {
        "partition","nodelist","gpu_type","nodes","ntasks","cpus_per_task","time","account","rsync_dest","build_dir","cuda_arch"
    };
    if (v.is_object()) {
        for (const auto& [k, val] : v.as_object()) {
            bool recognised = false;
            for (const auto& kk : known) if (k == kk) { recognised = true; break; }
            if (!recognised && val.is_string())
                n.extra[k] = val.as_string();
        }
    }
    return n;
}

inline JsonValue node_to_json(const NodeProfile& n) {
    JsonObject obj;
    auto add = [&](const std::string& k, const std::string& v) {
        if (!v.empty()) obj.push_back({k, v});
    };
    add("partition", n.partition);
    add("nodelist",  n.nodelist);
    add("gpu_type",  n.gpu_type);
    if (n.nodes > 1)         obj.push_back({"nodes",         JsonValue((int64_t)n.nodes)});
    if (n.ntasks > 1)        obj.push_back({"ntasks",        JsonValue((int64_t)n.ntasks)});
    if (n.cpus_per_task > 1) obj.push_back({"cpus_per_task", JsonValue((int64_t)n.cpus_per_task)});
    add("time",      n.time);
    add("account",    n.account);
    add("rsync_dest", n.rsync_dest);
    add("build_dir",  n.build_dir);
    add("cuda_arch",  n.cuda_arch);
    for (const auto& [k, v] : n.extra) add(k, v);
    return JsonValue(std::move(obj));
}

inline SbatchDefaults sbatch_defaults_from_json(const JsonValue& v) {
    SbatchDefaults d;
    d.output_pattern  = v.get_str("output_pattern", ".trailhead/slurm-%j.out");
    d.error_pattern   = v.get_str("error_pattern",  ".trailhead/slurm-%j.err");
    d.job_name_prefix = v.get_str("job_name_prefix", "th");
    d.preamble = v.get_str_array("preamble");
    d.modules  = v.get_str_array("modules");
    return d;
}

inline JsonValue sbatch_defaults_to_json(const SbatchDefaults& d) {
    JsonObject obj;
    obj.push_back({"output_pattern",  d.output_pattern});
    obj.push_back({"error_pattern",   d.error_pattern});
    obj.push_back({"job_name_prefix", d.job_name_prefix});
    JsonArray pre, mods;
    for (const auto& s : d.preamble) pre.push_back(JsonValue(s));
    for (const auto& s : d.modules)  mods.push_back(JsonValue(s));
    obj.push_back({"preamble", std::move(pre)});
    obj.push_back({"modules",  std::move(mods)});
    return JsonValue(std::move(obj));
}

inline TestEntry test_from_json(const JsonValue& v) {
    TestEntry t;
    t.name         = v.get_str("name");
    t.label        = v.get_str("label");
    t.cmd          = v.get_str("cmd");
    t.workdir      = v.get_str("workdir", ".");
    t.timeout_sec  = (int)v.get_int("timeout_sec", 300);
    t.tags         = v.get_str_array("tags");
    t.build_name   = v.get_str("build");
    t.target       = v.get_str("target");
    t.requires_hw  = v.get_str("requires");
    return t;
}

inline JsonValue test_to_json(const TestEntry& t) {
    JsonObject obj;
    obj.push_back({"name",        t.name});
    obj.push_back({"label",       t.label});
    obj.push_back({"cmd",         t.cmd});
    obj.push_back({"workdir",     t.workdir});
    obj.push_back({"timeout_sec", JsonValue((int64_t)t.timeout_sec)});
    JsonArray tags;
    for (const auto& tag : t.tags) tags.push_back(JsonValue(tag));
    obj.push_back({"tags", std::move(tags)});
    if (!t.build_name.empty())              obj.push_back({"build",    t.build_name});
    if (!t.target.empty())                  obj.push_back({"target",   t.target});
    if (!t.requires_hw.empty() &&
        t.requires_hw != "any")             obj.push_back({"requires", t.requires_hw});
    return JsonValue(std::move(obj));
}

// ── Load / save ───────────────────────────────────────────────────────────

Registry registry_from_json(const JsonValue& root);
JsonValue registry_to_json(const Registry& reg);

std::optional<Registry> load_registry(const std::string& trailhead_dir);
bool save_registry(const std::string& trailhead_dir, const Registry& reg);

// Load sub-registries declared in reg.sub_registries and merge their tests in.
// project_root is the parent directory of .trailhead/.
// Merged tests have sub_dir set (non-empty) and names prefixed with their submodule name.
void merge_sub_registries(Registry& reg, const std::string& project_root);

Registry make_default_registry();

} // namespace trailhead
