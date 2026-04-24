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
    // For cmake ".." (run-from-build-dir) form, cd into the build dir first.
    if (!configure_cmd.empty() && !build_dir.empty()) {
        std::string cfg = str_replace_all(configure_cmd, "-B build", "-B " + build_dir);
        bool from_build = cfg.size() >= 3 && cfg.substr(cfg.size() - 3) == " ..";
        if (from_build) {
            o << "[ -f " << build_dir << "/CMakeCache.txt ] || "
              << "(mkdir -p " << build_dir << " && cd " << build_dir << " && " << cfg << ")\n\n";
        } else {
            o << "[ -d " << build_dir << " ] || " << cfg << "\n\n";
        }
    }

    for (const TestEntry* t : tests) {
        o << "# " << (t->label.empty() ? t->name : t->label) << "\n";
        // If test has a cmake target, rebuild it in the node's build dir
        if (!t->build_name.empty() && !t->target.empty()) {
            o << "cmake --build " << build_dir << " --target " << t->target << "\n";
        }
        // Substitute build dir in the run cmd (e.g. "build/tests/foo" → "build_h200/tests/foo")
        std::string cmd = str_replace_all(t->cmd, "build/", build_dir + "/");

        // When workdir is default and a build is linked, run from inside the build directory.
        // Strip any "build_dir/" prefix from cmd since we're already there.
        std::string effective_wd = t->workdir;
        if ((effective_wd.empty() || effective_wd == ".") && !build_dir.empty() && !t->build_name.empty()) {
            effective_wd = build_dir;
            const std::string prefix = build_dir + "/";
            if (cmd.rfind(prefix, 0) == 0)
                cmd = "./" + cmd.substr(prefix.size());
        }

        if (!effective_wd.empty() && effective_wd != ".") {
            o << "(\n  cd " << effective_wd << "\n  " << cmd << "\n)\n\n";
        } else {
            o << cmd << "\n\n";
        }
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
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
                    if (!bit->second.dir.empty()) build_dir = bit->second.dir;
                    configure_cmd = bit->second.configure_cmd;
                    effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                            opts.project_root, project_name);
                    sub_dir_name = maybe_append_sub_dir(effective_root, bit->second, node_rsync);
                }
            }
            // Node-specific build dir: explicit setting wins; otherwise default to build_<node>
            if (node_ptr) {
                if (!node_ptr->build_dir.empty())
                    build_dir = node_ptr->build_dir;
                else if (!opts.node_name.empty())
                    build_dir = "build_" + opts.node_name;
            }

            // No linked build config — fall back to any registered build config
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

            if (configure_cmd.find("{{arch}}") != std::string::npos) {
                std::string arch_val = (node_ptr && !node_ptr->cuda_arch.empty())
                    ? node_ptr->cuda_arch
                    : "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')";
                configure_cmd = str_replace_all(configure_cmd, "{{arch}}", arch_val);
            }
            TestEntry t_adj = t;
            t_adj.workdir = adjust_workdir(t.workdir, sub_dir_name);
            script << sbatch_body({&t_adj}, defs, effective_root,
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
        if (node_ptr) {
            if (!node_ptr->build_dir.empty())
                build_dir = node_ptr->build_dir;
            else if (!opts.node_name.empty())
                build_dir = "build_" + opts.node_name;
        }
        // Pick configure_cmd and remote root from whichever build config is referenced first
        for (const auto& t : reg.tests) {
            if (!t.build_name.empty()) {
                auto bit = reg.builds.find(t.build_name);
                if (bit != reg.builds.end()) {
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
    if (!test.build_name.empty()) {
        auto bit = reg.builds.find(test.build_name);
        if (bit != reg.builds.end()) {
            if (!bit->second.dir.empty()) build_dir = bit->second.dir;
            configure_cmd = bit->second.configure_cmd;
            effective_root = resolve_effective_root(bit->second.rsync_dest, node_rsync,
                                                    opts.project_root, project_name);
            sub_dir_name = maybe_append_sub_dir(effective_root, bit->second, node_rsync);
        }
    }
    if (node_ptr) {
        if (!node_ptr->build_dir.empty())
            build_dir = node_ptr->build_dir;
        else if (!node_name.empty())
            build_dir = "build_" + node_name;
    }
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

    if (configure_cmd.find("{{arch}}") != std::string::npos) {
        std::string arch_val = (node_ptr && !node_ptr->cuda_arch.empty())
            ? node_ptr->cuda_arch
            : "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')";
        configure_cmd = str_replace_all(configure_cmd, "{{arch}}", arch_val);
    }
    TestEntry test_adj = test;
    test_adj.workdir = adjust_workdir(test.workdir, sub_dir_name);
    script << sbatch_body({&test_adj}, defs, effective_root,
                          build_dir, configure_cmd, reg.setup);
    return script.str();
}

} // namespace trailhead
