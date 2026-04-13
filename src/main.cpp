#include "core/registry.hpp"
#include "core/result_store.hpp"
#include "cli/visualizer.hpp"
#include "cli/sbatch_gen.hpp"
#include "cli/remote_run.hpp"
#include "cli/local_run.hpp"
#include "util/file_util.hpp"
#include "util/process.hpp"
#include "util/ansi.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

// ── Argument parsing helpers ──────────────────────────────────────────────

struct Args {
    std::vector<std::string> positional;
    std::unordered_map<std::string,std::string> flags; // --key=val or --key val

    static Args parse(int argc, char** argv, int start = 1) {
        Args a;
        for (int i = start; i < argc; ++i) {
            std::string s(argv[i]);
            if (s.size() > 2 && s[0]=='-' && s[1]=='-') {
                size_t eq = s.find('=');
                if (eq != std::string::npos) {
                    a.flags[s.substr(2, eq-2)] = s.substr(eq+1);
                } else {
                    // Next arg is value if it doesn't start with -
                    std::string key = s.substr(2);
                    if (i+1 < argc && argv[i+1][0] != '-') {
                        a.flags[key] = argv[++i];
                    } else {
                        a.flags[key] = "1"; // boolean flag
                    }
                }
            } else {
                a.positional.push_back(s);
            }
        }
        return a;
    }

    bool has(const std::string& k) const { return flags.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = flags.find(k);
        return it != flags.end() ? it->second : def;
    }
    int get_int(const std::string& k, int def = 0) const {
        auto it = flags.find(k);
        if (it == flags.end()) return def;
        try { return std::stoi(it->second); } catch (...) { return def; }
    }
    bool flag(const std::string& k) const { return has(k) && get(k) != "0"; }
};

// ── Require .trailhead/ exists ────────────────────────────────────────────

static std::string require_trailhead(bool quiet = false) {
    auto d = trailhead::fs::find_trailhead_dir();
    if (!d) {
        if (!quiet)
            std::cerr << trailhead::ansi::RED << "Error:" << trailhead::ansi::RESET
                      << " No .trailhead/ found. Run: trailhead init\n";
        std::exit(1);
    }
    return *d;
}

static trailhead::Registry load_reg(const std::string& th_dir) {
    auto reg = trailhead::load_registry(th_dir);
    if (!reg) {
        std::cerr << trailhead::ansi::RED << "Error:" << trailhead::ansi::RESET
                  << " Could not load registry.json\n";
        std::exit(1);
    }
    return *reg;
}

// ── Subcommand: init ──────────────────────────────────────────────────────

static int cmd_init(const Args& args) {
    (void)args;
    std::string cwd;
    char buf[4096];
    if (getcwd(buf, sizeof(buf))) cwd = buf;

    std::string th_dir = cwd + "/.trailhead";
    if (trailhead::fs::is_dir(th_dir)) {
        std::cout << trailhead::ansi::YELLOW << "Already initialised:" << trailhead::ansi::RESET
                  << " " << th_dir << "\n";
        return 0;
    }

    trailhead::fs::mkdir_p(th_dir + "/results");
    trailhead::Registry reg = trailhead::make_default_registry();
    trailhead::save_registry(th_dir, reg);

    std::cout << trailhead::ansi::BGREEN << "Initialized" << trailhead::ansi::RESET
              << " " << th_dir << "\n\n";
    std::cout << "Next steps:\n";
    std::cout << "  trailhead node add --name <profile>  # define a node/SLURM profile\n";
    std::cout << "  trailhead add --name <test> --cmd <cmd>  # register a test\n";
    std::cout << "  trailhead run --all                   # run all tests locally\n";
    std::cout << "  trailhead gen --split                 # generate sbatch scripts\n";
    return 0;
}

// ── Subcommand: node ──────────────────────────────────────────────────────

static int cmd_node(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n"
                     "  trailhead node add  --name <n> --partition <p>\n"
                     "                      [--nodelist <node>]    # pin specific node (e.g. d4067)\n"
                     "                      [--gpu-type <model>]   # OR pick GPU model (e.g. h200)\n"
                     "                      [--cpus <n>] [--time <HH:MM:SS>]\n"
                     "                      [--nodes <n>] [--ntasks <n>] [--account <a>]\n"
                     "  trailhead node list\n"
                     "  trailhead node remove <name>\n"
                     "\n"
                     "  --nodelist and --gpu-type are mutually exclusive:\n"
                     "    --nodelist d4067          → pins node, adds --gres=gpu:1\n"
                     "    --gpu-type h200           → requests model, adds --gres=gpu:h200\n";
        return 0;
    }
    std::string action(argv[2]);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    if (action == "list") {
        if (reg.nodes.empty()) {
            std::cout << trailhead::ansi::DIM << "No node profiles defined.\n" << trailhead::ansi::RESET;
            return 0;
        }
        // Find longest name for alignment
        size_t max_name = 4;
        for (const auto& [n, _] : reg.nodes) max_name = std::max(max_name, n.size());

        using namespace trailhead::ansi;
        std::cout << BOLD << pad("NAME", (int)max_name+2)
                  << " PARTITION          TARGET              CPUS  TIME\n" << RESET;
        for (const auto& [name, np] : reg.nodes) {
            // Show either gpu_type or nodelist as the hardware target
            std::string target = !np.gpu_type.empty()
                ? ("gpu:" + np.gpu_type)
                : (!np.nodelist.empty() ? ("node:" + np.nodelist) : "-");
            std::cout << pad(name, (int)max_name+2)
                      << " " << pad(np.partition, 18)
                      << " " << pad(target, 18)
                      << " " << pad(std::to_string(np.cpus_per_task), 5)
                      << " " << np.time << "\n";
        }
        return 0;
    }

    if (action == "add") {
        Args args = Args::parse(argc, argv, 3);
        std::string name = args.get("name");
        if (name.empty()) {
            std::cerr << "Error: --name is required\n"; return 1;
        }
        trailhead::NodeProfile np;
        np.name         = name;
        np.partition    = args.get("partition");
        np.nodelist     = args.get("nodelist");   // pin specific node
        np.gpu_type     = args.get("gpu-type");   // OR request GPU model (e.g. h200, a100)
        np.nodes        = args.get_int("nodes", 1);
        np.ntasks       = args.get_int("ntasks", 1);
        np.cpus_per_task= args.get_int("cpus", 1);
        np.time         = args.get("time", "01:00:00");
        np.account      = args.get("account");

        if (!np.nodelist.empty() && !np.gpu_type.empty()) {
            std::cerr << "Error: specify --nodelist OR --gpu-type, not both.\n";
            return 1;
        }

        reg.nodes[name] = np;
        trailhead::save_registry(th_dir, reg);
        std::cout << trailhead::ansi::BGREEN << "Added node profile:" << trailhead::ansi::RESET
                  << " " << name << "\n";
        return 0;
    }

    if (action == "remove") {
        if (argc < 4) { std::cerr << "Error: specify a profile name\n"; return 1; }
        std::string name(argv[3]);
        if (!reg.nodes.count(name)) {
            std::cerr << "Error: node profile '" << name << "' not found\n"; return 1;
        }
        reg.nodes.erase(name);
        trailhead::save_registry(th_dir, reg);
        std::cout << trailhead::ansi::YELLOW << "Removed node profile:" << trailhead::ansi::RESET
                  << " " << name << "\n";
        return 0;
    }

    std::cerr << "Unknown node action: " << action << "\n";
    return 1;
}

// ── Subcommand: build ─────────────────────────────────────────────────────

static int cmd_build(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n"
                     "  trailhead build add --name <n> --dir <build-dir>\n"
                     "                      --configure <cmake-configure-cmd>\n"
                     "                      --build <cmake-build-cmd>\n"
                     "  trailhead build list\n"
                     "  trailhead build remove <name>\n"
                     "  trailhead build run <name>   # manually trigger configure+build\n"
                     "\n"
                     "Example:\n"
                     "  trailhead build add --name release --dir ./build \\\n"
                     "    --configure \"cmake -B build -DCMAKE_BUILD_TYPE=Release\" \\\n"
                     "    --build \"cmake --build build -j8\"\n";
        return 0;
    }
    std::string action(argv[2]);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    if (action == "list") {
        if (reg.builds.empty()) {
            std::cout << trailhead::ansi::DIM << "No build configs defined.\n" << trailhead::ansi::RESET;
            return 0;
        }
        size_t max_name = 4;
        for (const auto& [n, _] : reg.builds) max_name = std::max(max_name, n.size());

        using namespace trailhead::ansi;
        std::cout << BOLD << pad("NAME", (int)max_name+2) << " DIR              BUILD CMD\n" << RESET;
        for (const auto& [name, bc] : reg.builds) {
            std::cout << pad(name, (int)max_name+2)
                      << " " << pad(bc.dir, 16)
                      << " " << bc.build_cmd << "\n";
        }
        return 0;
    }

    if (action == "add") {
        Args args = Args::parse(argc, argv, 3);
        std::string name = args.get("name");
        if (name.empty()) { std::cerr << "Error: --name is required\n"; return 1; }

        trailhead::BuildConfig bc;
        bc.name          = name;
        bc.dir           = args.get("dir");
        bc.configure_cmd = args.get("configure");
        bc.build_cmd     = args.get("build");
        bc.rsync_src     = args.get("rsync-src");
        bc.rsync_dest    = args.get("rsync-dest");

        if (bc.build_cmd.empty()) {
            std::cerr << "Error: --build is required (e.g. \"cmake --build build -j8\")\n";
            return 1;
        }
        reg.builds[name] = bc;
        trailhead::save_registry(th_dir, reg);
        std::cout << trailhead::ansi::BGREEN << "Added build config:" << trailhead::ansi::RESET
                  << " " << name << "\n";
        if (!bc.configure_cmd.empty()) std::cout << "  configure:  " << bc.configure_cmd << "\n";
        std::cout                               << "  build:      " << bc.build_cmd << "\n";
        if (!bc.rsync_dest.empty())
            std::cout << "  rsync:      " << (bc.rsync_src.empty() ? "<project root>" : bc.rsync_src)
                      << " → " << bc.rsync_dest << "\n";
        return 0;
    }

    if (action == "remove") {
        if (argc < 4) { std::cerr << "Error: specify a build name\n"; return 1; }
        std::string name(argv[3]);
        if (!reg.builds.count(name)) { std::cerr << "Error: build '" << name << "' not found\n"; return 1; }
        reg.builds.erase(name);
        trailhead::save_registry(th_dir, reg);
        std::cout << trailhead::ansi::YELLOW << "Removed build config:" << trailhead::ansi::RESET
                  << " " << name << "\n";
        return 0;
    }

    if (action == "run") {
        if (argc < 4) { std::cerr << "Error: specify a build name\n"; return 1; }
        std::string name(argv[3]);
        auto it = reg.builds.find(name);
        if (it == reg.builds.end()) { std::cerr << "Error: build '" << name << "' not found\n"; return 1; }
        const auto& bc = it->second;

        // Run configure if build dir doesn't exist and configure_cmd is set
        bool need_configure = !bc.configure_cmd.empty() &&
                              !bc.dir.empty() && !trailhead::fs::is_dir(bc.dir);
        if (need_configure) {
            std::cout << trailhead::ansi::BOLD << "Configuring:" << trailhead::ansi::RESET
                      << " " << bc.configure_cmd << "\n";
            auto r = trailhead::proc::run(bc.configure_cmd, {}, {}, 300, "");
            if (r.exit_code != 0) {
                std::cout << r.stdout_str << r.stderr_str;
                std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "Configure failed") << "\n";
                return 1;
            }
        }
        std::cout << trailhead::ansi::BOLD << "Building:" << trailhead::ansi::RESET
                  << " " << bc.build_cmd << "\n";
        auto r = trailhead::proc::run(bc.build_cmd, {}, {}, 600, "", [](const std::string& line) {
            std::cout << "  " << line << "\n";
        });
        if (r.exit_code != 0) {
            if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
            std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "Build failed") << "\n";
            return 1;
        }
        std::cout << trailhead::ansi::color(trailhead::ansi::BGREEN, "Build succeeded") << "\n";
        return 0;
    }

    std::cerr << "Unknown build action: " << action << "\n";
    return 1;
}

// ── Subcommand: add ───────────────────────────────────────────────────────

static int cmd_add(int argc, char** argv) {
    Args args = Args::parse(argc, argv, 2);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    std::string name = args.get("name");
    std::string cmd  = args.get("cmd");
    if (name.empty() || cmd.empty()) {
        std::cerr << "Usage: trailhead add --name <n> --cmd <cmd> [--label <l>]\n"
                     "                     [--build <config>] [--target <cmake-target>]\n"
                     "                     [--node <profile>] [--tag <t>]\n"
                     "                     [--timeout <sec>] [--workdir <dir>]\n"
                     "\n"
                     "  --cmd supports shell syntax (&&, pipes, multi-line via semicolons):\n"
                     "    --cmd \"./build/my_test --verbose\"\n"
                     "    --cmd \"./build/my_test && diff out.txt ref.txt\"\n"
                     "\n"
                     "  --target: cmake target to rebuild before running this test.\n"
                     "    Defaults to test name when --build is set.\n"
                     "    Use --target \"\" to skip per-test rebuild.\n";
        if (!reg.builds.empty()) {
            std::cout << "\nAvailable build configs:\n";
            for (const auto& [n, bc] : reg.builds)
                std::cout << "  " << n << "  (" << bc.build_cmd << ")\n";
        }
        if (!reg.nodes.empty()) {
            std::cout << "\nAvailable node profiles:\n";
            for (const auto& [n, np] : reg.nodes)
                std::cout << "  " << n << "  (" << np.partition << ")\n";
        }
        return 1;
    }

    for (const auto& t : reg.tests) {
        if (t.name == name) {
            std::cerr << "Error: test '" << name << "' already exists. Remove it first.\n";
            return 1;
        }
    }

    std::string build = args.get("build");
    if (!build.empty() && !reg.builds.count(build)) {
        std::cerr << "Error: build config '" << build << "' not found.\n";
        if (!reg.builds.empty()) {
            std::cout << "Available configs:";
            for (const auto& [n, _] : reg.builds) std::cout << " " << n;
            std::cout << "\n";
        } else {
            std::cout << "Create one first: trailhead build add --name <n> --build <cmd>\n";
        }
        return 1;
    }

    std::string node = args.get("node");
    if (!node.empty() && !reg.nodes.count(node)) {
        std::cerr << "Error: node profile '" << node << "' not found.\n";
        if (!reg.nodes.empty()) {
            std::cout << "Available profiles:";
            for (const auto& [n, _] : reg.nodes) std::cout << " " << n;
            std::cout << "\n";
        }
        return 1;
    }

    trailhead::TestEntry t;
    t.name         = name;
    t.label        = args.get("label");
    t.cmd          = cmd;
    t.workdir      = args.get("workdir", ".");
    t.timeout_sec  = args.get_int("timeout", 300);
    t.node_profile = node;
    t.build_name   = build;

    // target: explicit --target, or defaults to test name when build is set,
    // or empty string to disable per-test rebuild
    if (args.has("target")) {
        t.target = args.get("target"); // explicit (may be empty to disable)
    } else if (!build.empty()) {
        t.target = name;               // default: rebuild this test's cmake target
    }

    std::string tag_str = args.get("tag");
    if (!tag_str.empty()) {
        std::istringstream ss(tag_str);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) t.tags.push_back(tok);
    }

    reg.tests.push_back(t);
    trailhead::save_registry(th_dir, reg);

    std::cout << trailhead::ansi::BGREEN << "Added test:" << trailhead::ansi::RESET << " " << name;
    if (!build.empty()) std::cout << "  (build: " << build << ", target: " << t.target << ")";
    if (!node.empty())  std::cout << "  (node: "  << node  << ")";
    std::cout << "\n";
    return 0;
}

// ── Subcommand: list ──────────────────────────────────────────────────────

static int cmd_list(int argc, char** argv) {
    (void)argc; (void)argv;
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);
    trailhead::print_status(th_dir, reg);
    return 0;
}

// ── Subcommand: remove ────────────────────────────────────────────────────

static int cmd_remove(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: trailhead remove <name>\n"; return 1; }
    std::string name(argv[2]);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    auto it = std::find_if(reg.tests.begin(), reg.tests.end(),
        [&](const trailhead::TestEntry& t) { return t.name == name; });
    if (it == reg.tests.end()) {
        std::cerr << "Error: test '" << name << "' not found\n"; return 1;
    }
    reg.tests.erase(it);
    trailhead::save_registry(th_dir, reg);
    std::cout << trailhead::ansi::YELLOW << "Removed test:" << trailhead::ansi::RESET
              << " " << name << "\n";
    return 0;
}

// ── Subcommand: run ───────────────────────────────────────────────────────

// Run a build config (configure if needed, then build). Returns true on success.
static bool run_build(const trailhead::BuildConfig& bc) {
    // Configure: only if build dir is absent and configure_cmd is set
    bool need_configure = !bc.configure_cmd.empty() &&
                          !bc.dir.empty() && !trailhead::fs::is_dir(bc.dir);
    if (need_configure) {
        std::cout << trailhead::ansi::BOLD << "  configure:" << trailhead::ansi::RESET
                  << " " << bc.configure_cmd << "\n";
        auto r = trailhead::proc::run(bc.configure_cmd, {}, {}, 300, "");
        if (r.exit_code != 0) {
            std::cout << r.stdout_str;
            if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
            std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "  Configure failed\n");
            return false;
        }
    }
    std::cout << trailhead::ansi::BOLD << "  build:    " << trailhead::ansi::RESET
              << " " << bc.build_cmd << "\n";
    auto r = trailhead::proc::run(bc.build_cmd, {}, {}, 600, "");
    if (r.exit_code != 0) {
        std::cout << r.stdout_str;
        if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
        std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "  Build failed\n");
        return false;
    }
    return true;
}

static int cmd_run(int argc, char** argv) {
    Args args = Args::parse(argc, argv, 2);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    bool no_build = args.flag("no-build");  // skip build step
    std::string filter_tag = args.get("tag");
    bool run_all = args.flag("all");

    std::vector<std::string> names = args.positional;
    names.erase(std::remove(names.begin(), names.end(), "run"), names.end());

    std::vector<trailhead::TestEntry*> to_run;
    for (auto& t : reg.tests) {
        if (run_all || names.empty()) {
            if (filter_tag.empty() ||
                std::find(t.tags.begin(), t.tags.end(), filter_tag) != t.tags.end())
                to_run.push_back(&t);
        } else {
            if (std::find(names.begin(), names.end(), t.name) != names.end())
                to_run.push_back(&t);
        }
    }

    if (to_run.empty()) {
        std::cerr << "No tests matched. Use --all or specify test names.\n";
        return 1;
    }

    std::string results_dir = th_dir + "/results";
    trailhead::fs::mkdir_p(results_dir);

    // ── Per-build-group: configure (once) + rsync (once) ──────────────────
    if (!no_build) {
        std::vector<std::string> builds_seen;
        for (const auto* t : to_run) {
            if (t->build_name.empty()) continue;
            if (std::find(builds_seen.begin(), builds_seen.end(), t->build_name) != builds_seen.end()) continue;
            builds_seen.push_back(t->build_name);

            auto it = reg.builds.find(t->build_name);
            if (it == reg.builds.end()) {
                std::cerr << "Warning: build config '" << t->build_name << "' not found\n";
                continue;
            }
            const auto& bc = it->second;
            std::cout << trailhead::ansi::BOLD << "Build [" << t->build_name << "]\n" << trailhead::ansi::RESET;

            // Configure: only if build dir is absent
            bool need_configure = !bc.configure_cmd.empty() &&
                                  !bc.dir.empty() && !trailhead::fs::is_dir(bc.dir);
            if (need_configure) {
                std::cout << "  configure: " << bc.configure_cmd << "\n";
                auto r = trailhead::proc::run(bc.configure_cmd, {}, {}, 300, "", nullptr, true);
                if (r.exit_code != 0) {
                    std::cout << r.stdout_str;
                    if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
                    std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "  Configure failed\n");
                    return 1;
                }
            }

            // Rsync: once per build group
            if (!bc.rsync_dest.empty()) {
                auto root = trailhead::fs::find_trailhead_root();
                std::string src = bc.rsync_src.empty()
                    ? (root ? *root + "/" : "./")
                    : bc.rsync_src;
                std::string rsync_cmd = "rsync -avz --exclude='.trailhead/' "
                    + src + " " + bc.rsync_dest;
                std::cout << "  rsync:     " << src << " → " << bc.rsync_dest << "\n";
                auto r = trailhead::proc::run(rsync_cmd, {}, {}, 120, "", nullptr, true);
                if (r.exit_code != 0) {
                    std::cout << r.stdout_str;
                    if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
                    std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "  Rsync failed\n");
                    return 1;
                }
            }
            std::cout << "\n";
        }
    }

    // ── Test step: per-test target build + run ─────────────────────────────
    int failed_count = 0;
    int idx = 0;
    for (auto* t : to_run) {
        ++idx;
        std::cout << trailhead::ansi::BOLD
                  << "[" << idx << "/" << to_run.size() << "]"
                  << trailhead::ansi::RESET << " " << t->name;
        std::cout.flush();

        // Per-test cmake target build
        if (!no_build && !t->build_name.empty() && !t->target.empty()) {
            auto it = reg.builds.find(t->build_name);
            if (it != reg.builds.end()) {
                const auto& bc = it->second;
                std::string target_cmd = "cmake --build " + bc.dir
                    + " --target " + t->target;
                std::cout << "\n  cmake --build " << bc.dir << " --target " << t->target << " ...";
                std::cout.flush();
                auto r = trailhead::proc::run(target_cmd, {}, {}, 300, "", nullptr, true);
                if (r.exit_code != 0) {
                    std::cout << "\n";
                    if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
                    std::cerr << trailhead::ansi::color(trailhead::ansi::BRED, "  Build failed\n");
                    ++failed_count;
                    continue;
                }
            }
        }
        std::cout << " ...";
        std::cout.flush();

        std::unordered_map<std::string,std::string> env;
        env["TRAILHEAD_RESULTS_DIR"] = results_dir;

        // Run cmd via sh -c so multi-line / && / pipes work
        int64_t t_start = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        auto result = trailhead::proc::run(t->cmd, {}, env, t->timeout_sec, t->workdir,
                                           nullptr, /*use_shell=*/true);

        int64_t wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() - t_start;

        // Check stdout for TRAILHEAD: markers (from trailhead.h)
        std::string remaining_stdout;
        trailhead::TestResult stdout_result;
        stdout_result.name  = t->name;
        stdout_result.exit_code = result.exit_code;
        trailhead::parse_trailhead_output(result.stdout_str, stdout_result, &remaining_stdout);
        bool has_markers = (stdout_result.passed + stdout_result.failed > 0)
                         || !stdout_result.timings.empty();

        if (result.timed_out) {
            std::cout << " " << trailhead::ansi::color(trailhead::ansi::BRED, "TIMEOUT") << "\n";
            ++failed_count;
        } else if (result.exit_code != 0) {
            std::cout << " " << trailhead::ansi::color(trailhead::ansi::BRED, "FAILED")
                      << " (exit " << result.exit_code << ")";
            // Show pass/fail counts from markers even on failure
            if (has_markers)
                std::cout << " (" << stdout_result.passed << "/"
                          << (stdout_result.passed + stdout_result.failed) << ")";
            std::cout << "\n";
            // Print non-marker stdout + stderr (last 5 lines)
            auto print_tail = [&](const std::string& s) {
                std::istringstream ss2(s);
                std::vector<std::string> lines;
                std::string ln;
                while (std::getline(ss2, ln)) lines.push_back(ln);
                int st = std::max(0, (int)lines.size() - 5);
                for (int i = st; i < (int)lines.size(); ++i)
                    if (!lines[i].empty())
                        std::cout << "  " << trailhead::ansi::DIM << lines[i] << trailhead::ansi::RESET << "\n";
            };
            if (!remaining_stdout.empty()) print_tail(remaining_stdout);
            if (!result.stderr_str.empty()) print_tail(result.stderr_str);
            ++failed_count;
        } else {
            // Look for a reporter.hpp JSON result first
            auto result_idx = trailhead::load_all_results(results_dir);
            const auto* r = trailhead::latest_result(result_idx, t->name);

            if (r) {
                // reporter.hpp wrote a JSON file — authoritative
                std::string badge = r->failed > 0
                    ? trailhead::ansi::color(trailhead::ansi::BRED, "FAIL")
                    : trailhead::ansi::color(trailhead::ansi::BGREEN, "PASS");
                std::cout << " " << badge
                          << " (" << r->passed << "/" << (r->passed + r->failed) << ")"
                          << " " << trailhead::fs::format_duration_ms(r->wall_ms) << "\n";
                if (r->failed > 0) ++failed_count;
            } else if (has_markers) {
                // No JSON file — synthesize result from TRAILHEAD: stdout markers
                stdout_result.wall_ms    = wall_ms;
                stdout_result.started_at = t_start;
                stdout_result.ended_at   = t_start + wall_ms;

                char hbuf[256] = {};
                gethostname(hbuf, sizeof(hbuf));
                stdout_result.host   = hbuf;
                stdout_result.run_by = "trailhead-local";

                // Write synthesized JSON so watch/show can pick it up
                std::string epoch = std::to_string(t_start);
                std::string out_path = results_dir + "/" + t->name + "_" + epoch + ".json";
                trailhead::JsonObject jobj;
                jobj.push_back({"version",    trailhead::JsonValue((int64_t)1)});
                jobj.push_back({"name",       stdout_result.name});
                jobj.push_back({"host",       stdout_result.host});
                jobj.push_back({"run_by",     stdout_result.run_by});
                jobj.push_back({"started_at", trailhead::JsonValue(stdout_result.started_at)});
                jobj.push_back({"ended_at",   trailhead::JsonValue(stdout_result.ended_at)});
                jobj.push_back({"wall_ms",    trailhead::JsonValue(stdout_result.wall_ms)});
                jobj.push_back({"exit_code",  trailhead::JsonValue((int64_t)stdout_result.exit_code)});
                jobj.push_back({"passed",     trailhead::JsonValue((int64_t)stdout_result.passed)});
                jobj.push_back({"failed",     trailhead::JsonValue((int64_t)stdout_result.failed)});
                trailhead::JsonArray timings_arr;
                for (const auto& te : stdout_result.timings) {
                    trailhead::JsonObject te_obj;
                    te_obj.push_back({"label",      te.label});
                    te_obj.push_back({"elapsed_ms", trailhead::JsonValue(te.elapsed_ms)});
                    timings_arr.push_back(trailhead::JsonValue(std::move(te_obj)));
                }
                jobj.push_back({"timings", trailhead::JsonValue(std::move(timings_arr))});
                trailhead::JsonObject meta_obj;
                for (const auto& [k, v] : stdout_result.metadata)
                    meta_obj.push_back({k, trailhead::JsonValue(v)});
                jobj.push_back({"metadata", trailhead::JsonValue(std::move(meta_obj))});
                trailhead::fs::write_file_atomic(out_path, trailhead::json_emit(trailhead::JsonValue(std::move(jobj))));

                bool failed = stdout_result.failed > 0;
                std::string badge = failed
                    ? trailhead::ansi::color(trailhead::ansi::BRED, "FAIL")
                    : trailhead::ansi::color(trailhead::ansi::BGREEN, "PASS");
                std::cout << " " << badge
                          << " (" << stdout_result.passed << "/"
                          << (stdout_result.passed + stdout_result.failed) << ")"
                          << " " << trailhead::fs::format_duration_ms(wall_ms) << "\n";
                if (failed) ++failed_count;
            } else {
                // No reporter.hpp JSON and no TRAILHEAD: markers — just exit code
                std::cout << " " << trailhead::ansi::color(trailhead::ansi::BGREEN, "OK")
                          << " " << trailhead::fs::format_duration_ms(wall_ms) << "\n";
            }
        }
    }

    std::cout << "\n";
    if (failed_count == 0)
        std::cout << trailhead::ansi::color(trailhead::ansi::BGREEN, "All passed") << "\n";
    else
        std::cout << trailhead::ansi::color(trailhead::ansi::BRED,
            std::to_string(failed_count) + " failed") << "\n";

    return failed_count > 0 ? 1 : 0;
}

// ── Subcommand: gen ───────────────────────────────────────────────────────

static int cmd_gen(int argc, char** argv) {
    Args args = Args::parse(argc, argv, 2);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);

    // Get project root (parent of .trailhead/)
    auto root = trailhead::fs::find_trailhead_root();
    std::string project_root = root ? *root : "";

    trailhead::SbatchOptions opts;
    opts.split        = args.flag("split");
    opts.project_root = args.get("root", project_root);
    opts.output_path  = args.get("out");

    auto scripts = trailhead::generate_sbatch(reg, opts);
    if (scripts.empty()) {
        std::cerr << "No tests registered.\n"; return 1;
    }

    std::string sbatch_dir = opts.output_path.empty() ? (th_dir + "/sbatch") : opts.output_path;
    trailhead::fs::mkdir_p(sbatch_dir);
    for (const auto& [name, content] : scripts) {
        std::string out_path = sbatch_dir + "/" + name;

        if (trailhead::fs::write_file_atomic(out_path, content)) {
            std::cout << trailhead::ansi::BGREEN << "Generated:" << trailhead::ansi::RESET
                      << " " << out_path << "\n";
        } else {
            std::cerr << "Failed to write: " << out_path << "\n";
        }
    }
    return 0;
}

// ── Subcommand: watch ─────────────────────────────────────────────────────

static int cmd_watch(int argc, char** argv) {
    Args args = Args::parse(argc, argv, 2);
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);
    int interval = args.get_int("interval", 1000);

    // Determine project root (parent of .trailhead/)
    auto root_opt = trailhead::fs::find_trailhead_root();
    std::string project_root = root_opt ? *root_opt : ".";

    // Check whether any build config has an rsync_dest configured
    std::optional<trailhead::RemoteDest> remote_dest;
    for (const auto& [bname, bc] : reg.builds) {
        if (!bc.rsync_dest.empty()) {
            auto d = trailhead::parse_rsync_dest(bc.rsync_dest);
            if (d) { remote_dest = d; break; }
        }
    }

    std::function<void(const std::string&, const std::string&)> run_fn;

    auto job_log = std::make_shared<trailhead::JobLog>();

    // Local runner — available whenever a GPU is present, regardless of remote config
    std::shared_ptr<trailhead::LocalRunner> local_runner;
    if (trailhead::has_local_gpu())
        local_runner = std::make_shared<trailhead::LocalRunner>(th_dir, project_root, job_log);

    // Split multi-line log messages into separate entries so the log panel renders them correctly
    auto make_log_fn = [job_log](const std::string& tname) {
        return [job_log, tname](const std::string& msg) {
            std::istringstream ss(msg);
            std::string line;
            while (std::getline(ss, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty())
                    job_log->push("[" + tname + "] " + line);
            }
        };
    };

    if (remote_dest) {
        // Query SLURM for the per-user job limit on the partitions in use
        std::vector<std::string> partitions;
        for (const auto& [nname, node] : reg.nodes)
            if (!node.partition.empty()) partitions.push_back(node.partition);
        int max_concurrent = trailhead::query_slurm_job_limit(remote_dest->remote, partitions);
        job_log->push("[trailhead] SLURM max concurrent jobs: " + std::to_string(max_concurrent));

        // Remote mode: BatchSubmitter batches rapid presses, rsyncs once, sbatches serially
        auto submitter = std::make_shared<trailhead::BatchSubmitter>(
            reg, th_dir, project_root, *remote_dest, job_log, max_concurrent);

        run_fn = [&reg = reg, job_log, make_log_fn, submitter, local_runner](
                const std::string& name, const std::string& node_name) {
            const trailhead::TestEntry* test = nullptr;
            for (const auto& t : reg.tests)
                if (t.name == name) { test = &t; break; }
            if (!test) { job_log->push("test not found: " + name); return; }

            std::string tname = test->name;
            if (node_name == "local") {
                if (!local_runner) { job_log->push("local GPU not available"); return; }
                local_runner->enqueue(*test,
                    make_log_fn(tname),
                    [job_log, tname](const std::string& status) {
                        job_log->set_live(tname, status);
                    });
            } else {
                submitter->enqueue(*test, node_name,
                    make_log_fn(tname),
                    [job_log, tname](const std::string& status) {
                        job_log->set_live(tname, status);
                    });
            }
        };
    } else {
        // No remote configured — local runner only
        run_fn = [&reg = reg, job_log, make_log_fn, local_runner](
                const std::string& name, const std::string& node_name) {
            const trailhead::TestEntry* test = nullptr;
            for (const auto& t : reg.tests)
                if (t.name == name) { test = &t; break; }
            if (!test) { job_log->push("test not found: " + name); return; }

            std::string tname = test->name;
            if (node_name == "local") {
                if (!local_runner) { job_log->push("local GPU not available"); return; }
                local_runner->enqueue(*test,
                    make_log_fn(tname),
                    [job_log, tname](const std::string& status) {
                        job_log->set_live(tname, status);
                    });
            } else {
                job_log->push("no remote configured for node: " + node_name);
            }
        };
    }

    // Resume any jobs that were in-flight when watch was last closed
    if (remote_dest) {
        auto pending = trailhead::load_pending_jobs(th_dir);
        for (const auto& pj : pending) {
            // Find the matching test entry
            const trailhead::TestEntry* test = nullptr;
            for (const auto& t : reg.tests)
                if (t.name == pj.name) { test = &t; break; }
            if (!test) {
                // Test no longer registered — clean up stale pending file
                trailhead::clear_pending_job(th_dir, pj.name);
                continue;
            }

            job_log->active++;
            auto entry   = std::make_shared<trailhead::TestEntry>(*test);
            auto pending_copy = std::make_shared<trailhead::PendingJob>(pj);
            std::thread([entry, pending_copy, reg, th_dir, job_log, make_log_fn]() {
                std::string tname = entry->name;
                job_log->push("[" + tname + "] resuming job " + pending_copy->job_id);
                trailhead::resume_job(*pending_copy, *entry, reg, th_dir,
                    make_log_fn(tname),
                    [job_log, tname](const std::string& status) {
                        job_log->set_live(tname, status);
                    });
                job_log->active--;
            }).detach();
        }
    }

    // Re-enqueue submissions that were waiting (QUEUED/RSYNC) when watch last closed
    if (remote_dest) {
        auto queued = trailhead::load_queued_submissions(th_dir);
        for (const auto& qs : queued) {
            // Check the test still exists
            bool found = false;
            for (const auto& t : reg.tests)
                if (t.name == qs.name) { found = true; break; }
            if (!found) {
                trailhead::clear_queued_submission(th_dir, qs.name);
                continue;
            }
            job_log->push("[" + qs.name + "] re-queuing from previous session");
            run_fn(qs.name, qs.node_name);
        }
    }

    bool auto_run = args.flag("run-all");
    return trailhead::run_watch(th_dir, reg, interval, job_log, run_fn, project_root, auto_run);
}

// ── Subcommand: show ──────────────────────────────────────────────────────

static int cmd_show(int argc, char** argv) {
    std::string th_dir = require_trailhead();
    auto reg = load_reg(th_dir);
    auto idx = trailhead::load_all_results(th_dir + "/results");

    std::string filter = (argc >= 3) ? argv[2] : "";

    for (const auto& t : reg.tests) {
        if (!filter.empty() && t.name != filter) continue;
        const auto* r = trailhead::latest_result(idx, t.name);
        if (!r) {
            std::cout << t.name << ": no results\n";
            continue;
        }

        using namespace trailhead;
        using namespace trailhead::ansi;

        RunStatus s = result_status(*r);
        const char* sc = (s == RunStatus::Pass) ? BGREEN : (s == RunStatus::Fail) ? BRED : GRAY;
        std::cout << BOLD << t.name << RESET << " — " << color(sc, status_str(s)) << "\n";
        std::cout << "  host:    " << r->host << "\n";
        std::cout << "  run by:  " << r->run_by << "\n";
        std::cout << "  wall:    " << fs::format_duration_ms(r->wall_ms) << "\n";
        std::cout << "  pass:    " << r->passed << "  fail: " << r->failed << "\n";
        for (const auto& te : r->timings)
            std::cout << "  timing:  " << te.label << " = " << te.elapsed_ms << "ms\n";
        for (const auto& [k, v] : r->metadata)
            std::cout << "  meta:    " << k << " = " << v << "\n";
        if (!t.node_profile.empty())
            std::cout << "  node:    " << t.node_profile << "\n";
        std::cout << "\n";
    }
    return 0;
}

// ── Subcommand: clean ─────────────────────────────────────────────────────

static int cmd_clean(int argc, char** argv) {
    Args args = Args::parse(argc, argv, 2);
    std::string th_dir = require_trailhead();
    int days    = args.get_int("days", 7);
    bool dry_run= args.flag("dry-run");
    time_t cutoff = time(nullptr) - (time_t)days * 86400;

    auto files = trailhead::fs::list_dir(th_dir + "/results", ".json");
    int removed = 0;
    for (const auto& f : files) {
        if (trailhead::fs::mtime(f) < cutoff) {
            if (dry_run) {
                std::cout << "[dry-run] would remove: " << f << "\n";
            } else {
                ::unlink(f.c_str());
                ++removed;
            }
        }
    }
    if (!dry_run)
        std::cout << "Removed " << removed << " result file(s) older than " << days << " days.\n";
    return 0;
}

// ── Usage ─────────────────────────────────────────────────────────────────

static void print_usage() {
    using namespace trailhead::ansi;
    std::cout << BOLD << "trailhead" << RESET << " — test scheduling and reporting for HPC\n\n";
    std::cout << BOLD << "Usage:" << RESET << " trailhead <command> [options]\n\n";
    std::cout << BOLD << "Commands:\n" << RESET;
    std::cout << "  init                          Initialize .trailhead/ in current directory\n";
    std::cout << "  build add --name <n> --dir <d> --build <cmd> [--configure <cmd>]\n";
    std::cout << "                                Add a build config (cmake or any build)\n";
    std::cout << "  build list                    List build configs\n";
    std::cout << "  build remove <name>           Remove a build config\n";
    std::cout << "  build run <name>              Manually run configure+build\n";
    std::cout << "  node  add --name <n> ...      Add a SLURM node profile\n";
    std::cout << "  node  list                    List node profiles\n";
    std::cout << "  node  remove <name>           Remove a node profile\n";
    std::cout << "  add   --name <n> --cmd <c>    Register a test\n";
    std::cout << "        [--build <config>]        link to a build config\n";
    std::cout << "        [--node <profile>]         link to a node profile\n";
    std::cout << "        [--label <l>] [--tag <t>] [--timeout <s>] [--args <a>]\n";
    std::cout << "  remove <name>                 Remove a registered test\n";
    std::cout << "  list                          Show test status table\n";
    std::cout << "  run   [names] [--all] [--tag <t>] [--no-build]\n";
    std::cout << "                                Build then run tests locally\n";
    std::cout << "  gen   [--split] [--out <dir>] Generate sbatch script(s)\n";
    std::cout << "  watch [--interval <ms>]       Live TUI view\n";
    std::cout << "  show  [name]                  Print latest result details\n";
    std::cout << "  clean [--days <n>] [--dry-run] Remove old result files\n";
    std::cout << "\n";
}

// ── Entry point ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        // Default: show status if .trailhead/ exists, else show usage
        auto d = trailhead::fs::find_trailhead_dir();
        if (d) {
            auto reg = trailhead::load_registry(*d);
            if (reg) {
                trailhead::print_status(*d, *reg);
                return 0;
            }
        }
        print_usage();
        return 0;
    }

    std::string cmd(argv[1]);

    if (cmd == "init")         return cmd_init(Args::parse(argc, argv, 2));
    if (cmd == "build")        return cmd_build(argc, argv);
    if (cmd == "node")         return cmd_node(argc, argv);
    if (cmd == "add")          return cmd_add(argc, argv);
    if (cmd == "remove" || cmd == "rm") return cmd_remove(argc, argv);
    if (cmd == "list" || cmd == "ls")   return cmd_list(argc, argv);
    if (cmd == "run")          return cmd_run(argc, argv);
    if (cmd == "gen")          return cmd_gen(argc, argv);
    if (cmd == "watch")        return cmd_watch(argc, argv);
    if (cmd == "show")         return cmd_show(argc, argv);
    if (cmd == "clean")        return cmd_clean(argc, argv);
    if (cmd == "--help" || cmd == "help" || cmd == "-h") { print_usage(); return 0; }

    std::cerr << "Unknown command: " << cmd << ". Run 'trailhead help' for usage.\n";
    return 1;
}
