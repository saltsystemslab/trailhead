#include "registry.hpp"
#include "../util/file_util.hpp"
#include <stdexcept>

namespace trailhead {

Registry registry_from_json(const JsonValue& root) {
    Registry reg;
    reg.version = (int)root.get_int("version", 1);

    // Builds map
    const JsonValue* builds_val = root.get("builds");
    if (builds_val && builds_val->is_object()) {
        for (const auto& [name, bv] : builds_val->as_object()) {
            reg.builds[name] = build_from_json(name, bv);
        }
    }

    // Nodes map
    const JsonValue* nodes_val = root.get("nodes");
    if (nodes_val && nodes_val->is_object()) {
        for (const auto& [name, nv] : nodes_val->as_object()) {
            reg.nodes[name] = node_from_json(name, nv);
        }
    }

    // sbatch_defaults
    const JsonValue* defs = root.get("sbatch_defaults");
    if (defs && defs->is_object()) {
        reg.sbatch_defaults = sbatch_defaults_from_json(*defs);
    }

    // Tests
    const JsonValue* tests_val = root.get("tests");
    if (tests_val && tests_val->is_array()) {
        for (const auto& tv : tests_val->as_array()) {
            reg.tests.push_back(test_from_json(tv));
        }
    }

    // Setup commands
    reg.setup = root.get_str_array("setup");

    // Sub-registry paths
    reg.sub_registries = root.get_str_array("sub_registries");

    return reg;
}

JsonValue registry_to_json(const Registry& reg) {
    JsonObject root;
    root.push_back({"version", JsonValue((int64_t)reg.version)});

    // Builds — skip entries that came from sub-registries (contain '/')
    JsonObject builds_obj;
    for (const auto& [name, bc] : reg.builds) {
        if (name.find('/') == std::string::npos)
            builds_obj.push_back({name, build_to_json(bc)});
    }
    root.push_back({"builds", JsonValue(std::move(builds_obj))});

    // Nodes
    JsonObject nodes_obj;
    for (const auto& [name, np] : reg.nodes) {
        nodes_obj.push_back({name, node_to_json(np)});
    }
    root.push_back({"nodes", JsonValue(std::move(nodes_obj))});

    // sbatch_defaults
    root.push_back({"sbatch_defaults", sbatch_defaults_to_json(reg.sbatch_defaults)});

    // Tests — skip tests from sub-registries
    JsonArray tests_arr;
    for (const auto& t : reg.tests) {
        if (t.sub_dir.empty())
            tests_arr.push_back(test_to_json(t));
    }
    root.push_back({"tests", JsonValue(std::move(tests_arr))});

    // Setup commands
    JsonArray setup_arr;
    for (const auto& s : reg.setup) setup_arr.push_back(JsonValue(s));
    root.push_back({"setup", JsonValue(std::move(setup_arr))});

    // Sub-registry paths
    JsonArray sub_arr;
    for (const auto& s : reg.sub_registries) sub_arr.push_back(JsonValue(s));
    root.push_back({"sub_registries", JsonValue(std::move(sub_arr))});

    return JsonValue(std::move(root));
}

std::optional<Registry> load_registry(const std::string& trailhead_dir) {
    std::string path = trailhead_dir + "/registry.json";
    auto content = fs::read_file(path);
    if (!content) return std::nullopt;
    try {
        auto root = json_parse(*content);
        return registry_from_json(root);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse registry.json: " + std::string(e.what()));
    }
}

bool save_registry(const std::string& trailhead_dir, const Registry& reg) {
    fs::mkdir_p(trailhead_dir);
    std::string path = trailhead_dir + "/registry.json";
    std::string content = json_emit(registry_to_json(reg));
    return fs::write_file_atomic(path, content);
}

void merge_sub_registries(Registry& reg, const std::string& project_root) {
    for (const auto& sub_rel : reg.sub_registries) {
        std::string sub_abs = project_root + "/" + sub_rel;
        std::string sub_th  = sub_abs + "/.trailhead/registry.json";
        auto content = fs::read_file(sub_th);
        if (!content) continue;

        Registry sub;
        try { sub = registry_from_json(json_parse(*content)); }
        catch (...) { continue; }

        // The display name is the last path component (e.g. "andes_benchmarks")
        std::string sub_name = sub_rel;
        auto slash = sub_name.rfind('/');
        if (slash != std::string::npos) sub_name = sub_name.substr(slash + 1);

        // Merge build configs — prefix to avoid collisions; record sub_dir for local execution
        for (const auto& [bname, bc] : sub.builds) {
            std::string key = sub_name + "/" + bname;
            if (!reg.builds.count(key)) {
                BuildConfig merged = bc;
                merged.sub_dir = sub_rel;
                reg.builds[key] = merged;
            }
        }

        // Merge tests
        for (auto t : sub.tests) {
            t.name     = sub_name + "/" + t.name;
            t.sub_dir  = sub_rel;
            if (!t.build_name.empty()) t.build_name = sub_name + "/" + t.build_name;
            // Remap non-trivial workdirs to be relative to project root
            if (!t.workdir.empty() && t.workdir != ".")
                t.workdir = sub_rel + "/" + t.workdir;
            reg.tests.push_back(std::move(t));
        }

        // Merge node profiles — parent takes priority; fill in missing fields from sub
        for (const auto& [nname, np] : sub.nodes) {
            if (!reg.nodes.count(nname)) {
                reg.nodes[nname] = np;
            } else {
                auto& pn = reg.nodes[nname];
                if (pn.rsync_dest.empty() && !np.rsync_dest.empty()) pn.rsync_dest = np.rsync_dest;
                if (pn.cuda_arch.empty()  && !np.cuda_arch.empty())  pn.cuda_arch  = np.cuda_arch;
                if (pn.preamble.empty()   && !np.preamble.empty())   pn.preamble   = np.preamble;
            }
        }

        // Store sub-registry's own sbatch_defaults for per-test lookup
        reg.sub_sbatch_defaults[sub_rel] = sub.sbatch_defaults;
    }
}

Registry make_default_registry() {
    Registry reg;
    reg.sbatch_defaults.output_pattern  = ".trailhead/slurm-%j.out";
    reg.sbatch_defaults.error_pattern   = ".trailhead/slurm-%j.err";
    reg.sbatch_defaults.job_name_prefix = "th";
    reg.sbatch_defaults.preamble        = {"set -euo pipefail"};

    // Example node profiles — edit partition/nodelist/gpu_type to match your cluster
    NodeProfile rtx6000;
    rtx6000.name          = "rtx6000";
    rtx6000.partition     = "SaltSystemsLab";
    rtx6000.nodelist      = "d4067";  // pin to specific node
    rtx6000.cpus_per_task = 8;
    rtx6000.time          = "01:00:00";
    reg.nodes["rtx6000"]  = rtx6000;

    NodeProfile h200;
    h200.name          = "h200";
    h200.partition     = "gpu-interactive";  // shared partition, picked by GPU model
    h200.gpu_type      = "h200";             // → --gres=gpu:h200
    h200.cpus_per_task = 8;
    h200.time          = "02:00:00";
    reg.nodes["h200"]  = h200;

    return reg;
}

} // namespace trailhead
