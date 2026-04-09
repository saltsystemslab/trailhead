#include "sbatch_gen.hpp"
#include "../util/file_util.hpp"
#include <sstream>
#include <algorithm>

namespace trailhead {

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
    line("time",    node.time);
    if (!node.account.empty()) line("account", node.account);
    line("output",  defs.output_pattern);
    line("error",   defs.error_pattern);

    // Extra directives from the node profile
    for (const auto& [k, v] : node.extra)
        line(k, v);

    return o.str();
}

// Build the body of a script (modules + preamble + commands)
static std::string sbatch_body(const std::vector<const TestEntry*>& tests,
                                const SbatchDefaults& defs,
                                const std::string& project_root)
{
    std::ostringstream o;

    // Module loads
    for (const auto& mod : defs.modules)
        o << "module load " << mod << "\n";
    if (!defs.modules.empty()) o << "\n";

    // Preamble lines
    for (const auto& line : defs.preamble)
        o << line << "\n";
    if (!defs.preamble.empty()) o << "\n";

    // Set TRAILHEAD_JOB_ID so reporter.hpp labels results as sbatch-<id>
    o << "export TRAILHEAD_JOB_ID=$SLURM_JOB_ID\n";
    if (!project_root.empty())
        o << "cd " << project_root << "\n";
    o << "\n";

    for (const TestEntry* t : tests) {
        o << "# " << (t->label.empty() ? t->name : t->label) << "\n";
        // If test has a cmake target, rebuild it before running
        if (!t->build_name.empty() && !t->target.empty()) {
            o << "cmake --build build --target " << t->target << "\n";
        }
        if (!t->workdir.empty() && t->workdir != ".") {
            // Run cmd in subshell so workdir doesn't affect subsequent tests
            o << "(\n  cd " << t->workdir << "\n  sh -c " << "'" << t->cmd << "'\n)\n\n";
        } else {
            // cmd via sh -c to support multi-line / && / pipes
            o << "sh -c '" << t->cmd << "'\n\n";
        }
    }

    return o.str();
}

std::vector<std::pair<std::string,std::string>>
generate_sbatch(const Registry& reg, const SbatchOptions& opts)
{
    std::vector<std::pair<std::string,std::string>> out;

    if (opts.split) {
        // One script per test
        for (const auto& t : reg.tests) {
            std::ostringstream script;
            script << "#!/bin/bash\n";

            // Find node profile
            std::string job_name = reg.sbatch_defaults.job_name_prefix + "-" + t.name;
            if (!t.node_profile.empty()) {
                auto it = reg.nodes.find(t.node_profile);
                if (it != reg.nodes.end()) {
                    script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
                } else {
                    script << "# WARNING: node profile '" << t.node_profile << "' not found\n";
                }
            } else {
                script << "# No node profile assigned — add one with: trailhead node add\n";
            }
            script << "\n";
            script << sbatch_body({&t}, reg.sbatch_defaults, opts.project_root);

            out.push_back({t.name + ".sbatch", script.str()});
        }
    } else {
        // Single combined script
        // Group tests by node profile for a warning if they differ
        std::string dominant_profile;
        for (const auto& t : reg.tests)
            if (!t.node_profile.empty()) { dominant_profile = t.node_profile; break; }

        bool mixed = false;
        for (const auto& t : reg.tests)
            if (!t.node_profile.empty() && t.node_profile != dominant_profile) { mixed = true; break; }

        std::ostringstream script;
        script << "#!/bin/bash\n";
        if (mixed) {
            script << "# WARNING: tests use different node profiles. Use --split for per-test scripts.\n";
        }

        // Use dominant profile headers (or bare defaults if none)
        std::string job_name = reg.sbatch_defaults.job_name_prefix + "-all";
        if (!dominant_profile.empty()) {
            auto it = reg.nodes.find(dominant_profile);
            if (it != reg.nodes.end())
                script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
        } else {
            script << "#SBATCH --job-name=" << job_name << "\n";
            script << "#SBATCH --output=" << reg.sbatch_defaults.output_pattern << "\n";
            script << "#SBATCH --error="  << reg.sbatch_defaults.error_pattern  << "\n";
        }
        script << "\n";

        std::vector<const TestEntry*> ptrs;
        for (const auto& t : reg.tests) ptrs.push_back(&t);
        script << sbatch_body(ptrs, reg.sbatch_defaults, opts.project_root);

        out.push_back({"run_all.sbatch", script.str()});
    }

    return out;
}

bool write_sbatch(const std::string& trailhead_dir,
                  const Registry& reg,
                  const SbatchOptions& opts)
{
    auto scripts = generate_sbatch(reg, opts);
    bool ok = true;
    for (const auto& [name, content] : scripts) {
        std::string path = trailhead_dir + "/" + name;
        if (!fs::write_file_atomic(path, content)) ok = false;
    }
    return ok;
}

} // namespace trailhead
