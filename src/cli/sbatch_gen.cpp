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

// Resolve the remote working directory for a script.
// Prefers the build config's rsync_dest; falls back to the node profile's rsync_dest.
static std::string resolve_effective_root(const std::string& bc_rsync_dest,
                                           const std::string& node_rsync_dest,
                                           const std::string& fallback)
{
    for (const auto& d : {bc_rsync_dest, node_rsync_dest}) {
        if (!d.empty()) {
            auto rp = remote_path_from_rsync_dest(d);
            if (!rp.empty()) return rp;
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

// Build the body of a script (modules + preamble + configure + commands)
// build_dir:     the directory cmake builds into for this node (e.g. "build_h200")
// configure_cmd: cmake configure command from the build config (run once on the node)
static std::string sbatch_body(const std::vector<const TestEntry*>& tests,
                                const SbatchDefaults& defs,
                                const std::string& project_root,
                                const std::string& build_dir,
                                const std::string& configure_cmd,
                                const std::vector<std::string>& setup = {})
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

    // Project setup: submodule init, dataset downloads, etc.
    // Runs before cmake configure so the source tree is complete.
    if (!setup.empty()) {
        for (const auto& s : setup)
            o << s << "\n";
        o << "\n";
    }

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
            o << "(\n  cd " << t->workdir << "\n" << cmd << "\n)\n\n";
        } else {
            o << cmd << "\n\n";
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

            // Emit hardware headers if a node was specified (via opts.node_name)
            const NodeProfile* node_ptr = nullptr;
            std::string job_name = reg.sbatch_defaults.job_name_prefix + "-" + t.name;
            if (!opts.node_name.empty()) {
                auto it = reg.nodes.find(opts.node_name);
                if (it != reg.nodes.end()) {
                    node_ptr = &it->second;
                    script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
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
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    if (!bit->second.dir.empty()) build_dir = bit->second.dir;
                    configure_cmd = bit->second.configure_cmd;
                    effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                            opts.project_root);
                }
            }
            // Node-specific build dir overrides the build config's dir
            if (node_ptr && !node_ptr->build_dir.empty())
                build_dir = node_ptr->build_dir;

            // No linked build config — fall back to any registered build config
            if (configure_cmd.empty()) {
                for (const auto& [bname, bc] : reg.builds) {
                    if (!bc.configure_cmd.empty()) {
                        configure_cmd = bc.configure_cmd;
                        if (effective_root == opts.project_root)
                            effective_root = resolve_effective_root(bc.rsync_dest, node_rsync,
                                                                    opts.project_root);
                        break;
                    }
                }
            }

            if (node_ptr && !node_ptr->cuda_arch.empty())
                configure_cmd = str_replace_all(configure_cmd, "{{arch}}", node_ptr->cuda_arch);
            script << sbatch_body({&t}, reg.sbatch_defaults, effective_root,
                                  build_dir, configure_cmd, reg.setup);

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
        if (node_ptr && !node_ptr->build_dir.empty())
            build_dir = node_ptr->build_dir;
        // Pick configure_cmd and remote root from whichever build config is referenced first
        for (const auto& t : reg.tests) {
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    configure_cmd = bit->second.configure_cmd;
                    effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                            opts.project_root);
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
                                                                opts.project_root);
                    break;
                }
            }
        }

        if (node_ptr && !node_ptr->cuda_arch.empty())
            configure_cmd = str_replace_all(configure_cmd, "{{arch}}", node_ptr->cuda_arch);
        std::vector<const TestEntry*> ptrs;
        for (const auto& t : reg.tests) ptrs.push_back(&t);
        script << sbatch_body(ptrs, reg.sbatch_defaults, effective_root,
                              build_dir, configure_cmd, reg.setup);

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
    std::ostringstream script;
    script << "#!/bin/bash\n";

    std::string job_name = reg.sbatch_defaults.job_name_prefix + "-" + test.name;
    const NodeProfile* node_ptr = nullptr;
    if (!node_name.empty()) {
        auto it = reg.nodes.find(node_name);
        if (it != reg.nodes.end()) {
            node_ptr = &it->second;
            script << sbatch_headers(it->second, reg.sbatch_defaults, job_name);
        }
    }
    script << "\n";

    std::string build_dir     = "build";
    std::string configure_cmd;
    std::string effective_root = opts.project_root;
    std::string node_rsync = node_ptr ? node_ptr->rsync_dest : "";
    if (!test.build_name.empty()) {
        auto bit = reg.builds.find(test.build_name);
        if (bit != reg.builds.end()) {
            if (!bit->second.dir.empty()) build_dir = bit->second.dir;
            configure_cmd = bit->second.configure_cmd;
            effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                    opts.project_root);
        }
    }
    if (node_ptr && !node_ptr->build_dir.empty())
        build_dir = node_ptr->build_dir;
    if (configure_cmd.empty()) {
        for (const auto& [bname, bc] : reg.builds) {
            if (!bc.configure_cmd.empty()) {
                configure_cmd = bc.configure_cmd;
                if (effective_root == opts.project_root)
                    effective_root = resolve_effective_root(bc.rsync_dest, node_rsync,
                                                            opts.project_root);
                break;
            }
        }
    }

    if (node_ptr && !node_ptr->cuda_arch.empty())
        configure_cmd = str_replace_all(configure_cmd, "{{arch}}", node_ptr->cuda_arch);
    script << sbatch_body({&test}, reg.sbatch_defaults, effective_root,
                          build_dir, configure_cmd, reg.setup);
    return script.str();
}

} // namespace trailhead
