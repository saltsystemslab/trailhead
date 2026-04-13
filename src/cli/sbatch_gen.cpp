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

// Extract the remote path from an rsync_dest string "user@host:/path" → "/path".
// Returns empty string if not parseable.
static std::string remote_path_from_rsync_dest(const std::string& rsync_dest) {
    auto colon = rsync_dest.find(':');
    if (colon == std::string::npos || colon + 1 >= rsync_dest.size()) return "";
    return rsync_dest.substr(colon + 1);
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

// Build the body of a script (modules + preamble + configure + commands)
// build_dir:     the directory cmake builds into for this node (e.g. "build_h200")
// configure_cmd: cmake configure command from the build config (run once on the node)
static std::string sbatch_body(const std::vector<const TestEntry*>& tests,
                                const SbatchDefaults& defs,
                                const std::string& project_root,
                                const std::string& build_dir,
                                const std::string& configure_cmd)
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

    // Configure step: run cmake on the compute node so it auto-detects GPU arch.
    // Only runs if the build directory doesn't already exist (once per node).
    if (!configure_cmd.empty() && !build_dir.empty()) {
        std::string cfg = str_replace_all(configure_cmd, "-B build", "-B " + build_dir);
        o << "[ -d " << build_dir << " ] || " << cfg << "\n\n";
    }

    for (const TestEntry* t : tests) {
        o << "# " << (t->label.empty() ? t->name : t->label) << "\n";
        // If test has a cmake target, rebuild it in the node's build dir
        if (!t->build_name.empty() && !t->target.empty()) {
            o << "cmake --build " << build_dir << " --target " << t->target << "\n";
        }
        // Substitute build dir in the run cmd (e.g. "build/tests/foo" → "build_h200/tests/foo")
        std::string cmd = str_replace_all(t->cmd, "build/", build_dir + "/");
        if (!t->workdir.empty() && t->workdir != ".") {
            o << "(\n  cd " << t->workdir << "\n  sh -c '" << cmd << "'\n)\n\n";
        } else {
            o << "sh -c '" << cmd << "'\n\n";
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

            // Resolve node profile
            const NodeProfile* node_ptr = nullptr;
            std::string job_name = reg.sbatch_defaults.job_name_prefix + "-" + t.name;
            if (!t.node_profile.empty()) {
                auto it = reg.nodes.find(t.node_profile);
                if (it != reg.nodes.end()) {
                    node_ptr = &it->second;
                    script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
                } else {
                    script << "# WARNING: node profile '" << t.node_profile << "' not found\n";
                }
            } else {
                script << "# No node profile assigned — add one with: trailhead node add\n";
            }
            script << "\n";

            // Resolve build dir, configure_cmd, and effective project root
            std::string build_dir     = "build";
            std::string configure_cmd;
            std::string effective_root = opts.project_root;
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    if (!bit->second.dir.empty()) build_dir = bit->second.dir;
                    configure_cmd = bit->second.configure_cmd;
                    // Use the remote path as the working directory in the script
                    if (!bit->second.rsync_dest.empty()) {
                        std::string rp = remote_path_from_rsync_dest(bit->second.rsync_dest);
                        if (!rp.empty()) effective_root = rp;
                    }
                }
            }
            // Node-specific build dir overrides the build config's dir
            if (node_ptr && !node_ptr->build_dir.empty())
                build_dir = node_ptr->build_dir;

            script << sbatch_body({&t}, reg.sbatch_defaults, effective_root,
                                  build_dir, configure_cmd);

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

        // For combined scripts, use the dominant node's build_dir (best-effort)
        std::string build_dir = "build";
        std::string configure_cmd;
        std::string effective_root = opts.project_root;
        if (!dominant_profile.empty()) {
            auto nit = reg.nodes.find(dominant_profile);
            if (nit != reg.nodes.end() && !nit->second.build_dir.empty())
                build_dir = nit->second.build_dir;
        }
        // Pick configure_cmd and remote root from whichever build config is referenced first
        for (const auto& t : reg.tests) {
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    configure_cmd = bit->second.configure_cmd;
                    if (!bit->second.rsync_dest.empty()) {
                        std::string rp = remote_path_from_rsync_dest(bit->second.rsync_dest);
                        if (!rp.empty()) effective_root = rp;
                    }
                    break;
                }
            }
        }

        std::vector<const TestEntry*> ptrs;
        for (const auto& t : reg.tests) ptrs.push_back(&t);
        script << sbatch_body(ptrs, reg.sbatch_defaults, effective_root,
                              build_dir, configure_cmd);

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
        if (!fs::write_file_atomic(path, content)) ok = false;
    }
    return ok;
}

std::string generate_test_script(const TestEntry& test,
                                  const std::string& node_name,
                                  const Registry& reg,
                                  const SbatchOptions& opts)
{
    TestEntry t = test;
    t.node_profile = node_name;

    Registry tmp;
    tmp.builds         = reg.builds;
    tmp.nodes          = reg.nodes;
    tmp.sbatch_defaults = reg.sbatch_defaults;
    tmp.tests          = {t};

    SbatchOptions split_opts  = opts;
    split_opts.split          = true;

    auto scripts = generate_sbatch(tmp, split_opts);
    return scripts.empty() ? "" : scripts[0].second;
}

} // namespace trailhead
