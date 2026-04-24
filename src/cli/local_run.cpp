#include "local_run.hpp"
#include "../core/result_store.hpp"
#include "../util/file_util.hpp"
#include "../util/ansi.hpp"
#include "../util/process.hpp"
#include <sstream>
#include <deque>
#include <set>
#include <thread>
#include <mutex>
#include <chrono>

namespace trailhead {

bool has_local_gpu() {
    // nvidia-smi -L lists one line per GPU; non-empty output = GPU present
    auto r = proc::run("nvidia-smi -L", {}, {}, 5, "", nullptr, false);
    return r.exit_code == 0 && !r.stdout_str.empty();
}

std::vector<std::string> wipe_build_dirs(const Registry& reg, const std::string& project_root) {
    std::set<std::string> candidates;
    for (const auto& [bname, bc] : reg.builds) {
        std::string base = project_root;
        if (!bc.sub_dir.empty()) base += "/" + bc.sub_dir;
        std::string raw = bc.dir.empty() ? "build" : bc.dir;
        candidates.insert(base + "/" + raw);
        for (const auto& [nname, np] : reg.nodes) {
            std::string bdir = np.build_dir.empty() ? "build_" + nname : np.build_dir;
            candidates.insert(base + "/" + bdir);
        }
    }
    std::vector<std::string> removed;
    for (const auto& dir : candidates) {
        if (fs::exists(dir)) {
            proc::run("rm -rf " + dir, {}, {}, 60, "", nullptr, true);
            removed.push_back(dir);
        }
    }
    return removed;
}

// ── Shared build helpers ──────────────────────────────────────────────────

// Returns CUDA arch digits (e.g. "120"), or "" if detection failed.
static std::string detect_cuda_arch() {
    for (const char* qcmd : {
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once",
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"}) {
        auto r = proc::run(qcmd, {}, {}, 5, "", nullptr, false);
        if (r.exit_code == 0 && !r.stdout_str.empty()) {
            std::string arch;
            for (char c : r.stdout_str) {
                if (c >= '0' && c <= '9') arch += c;
                else if (c == '\n' || c == '\r') break;
            }
            if (!arch.empty()) return arch;
        }
    }
    // Fallback: parse "Compute Capability : X.Y" from nvidia-smi -q
    auto r = proc::run("nvidia-smi -q", {}, {}, 5, "", nullptr, false);
    if (r.exit_code == 0) {
        const std::string needle = "Compute Capability";
        auto p = r.stdout_str.find(needle);
        if (p != std::string::npos) {
            p = r.stdout_str.find(':', p);
            if (p != std::string::npos) {
                std::string arch;
                for (++p; p < r.stdout_str.size(); ++p) {
                    char c = r.stdout_str[p];
                    if (c >= '0' && c <= '9') arch += c;
                    else if (c == '.') continue;
                    else if (!arch.empty()) break;
                }
                return arch;
            }
        }
    }
    return "";
}

// Configure + build a single cmake target. Returns true on success.
// log_fn/log_mtx: log_mtx may be null (single-threaded callers).
static bool build_one_target(
    const BuildConfig& bc,
    const std::string& target,
    const std::string& project_root,
    const std::function<void(const std::string&)>& log_fn,
    std::mutex* log_mtx = nullptr)
{
    auto safe_log = [&](const std::string& msg) {
        if (!log_fn || msg.empty()) return;
        if (log_mtx) { std::lock_guard<std::mutex> lk(*log_mtx); log_fn(msg); }
        else log_fn(msg);
    };

    std::string raw_dir   = bc.dir.empty() ? "build" : bc.dir;
    std::string eff_dir   = bc.sub_dir.empty() ? raw_dir : bc.sub_dir + "/" + raw_dir;
    std::string base_wd   = bc.sub_dir.empty() ? project_root : project_root + "/" + bc.sub_dir;
    std::string bd        = project_root + "/" + eff_dir;

    bool needs_configure = !bc.configure_cmd.empty() &&
        !fs::exists(bd + "/Makefile") && !fs::exists(bd + "/build.ninja");
    if (needs_configure) {
        std::string configure_cmd = bc.configure_cmd;
        if (configure_cmd.find("{{arch}}") != std::string::npos) {
            std::string arch = detect_cuda_arch();
            if (arch.empty()) {
                safe_log("arch detection failed: nvidia-smi returned no compute capability");
                return false;
            }
            size_t pos = 0;
            while ((pos = configure_cmd.find("{{arch}}", pos)) != std::string::npos) {
                configure_cmd.replace(pos, 8, arch);
                pos += arch.size();
            }
        }
        bool from_build_dir = configure_cmd.size() >= 3 &&
                              configure_cmd.substr(configure_cmd.size() - 3) == " ..";
        std::string conf_wd = from_build_dir ? base_wd + "/" + raw_dir : base_wd;
        if (from_build_dir) fs::mkdir_p(conf_wd);

        safe_log("configuring: " + configure_cmd);
        auto cr = proc::run(configure_cmd, {}, {}, 300, conf_wd, [&](const std::string& line) {
            safe_log(line);
        }, true);
        if (cr.exit_code != 0) {
            if (!cr.stderr_str.empty()) safe_log(cr.stderr_str);
            safe_log("configure failed (exit=" + std::to_string(cr.exit_code) + ")");
            return false;
        }
    }

    unsigned int nj = std::max(1u, std::thread::hardware_concurrency());
    std::string build_cmd = "cmake --build " + eff_dir + " --target " + target
                          + " -j" + std::to_string(nj);  // full CPU count; called alone
    safe_log("building: " + build_cmd);
    auto br = proc::run(build_cmd, {}, {}, 300, project_root, [&](const std::string& line) {
        safe_log(line);
    }, true);
    if (br.exit_code != 0) {
        if (!br.stderr_str.empty()) safe_log(br.stderr_str);
        safe_log("build failed (exit=" + std::to_string(br.exit_code) + ")");
        return false;
    }
    return true;
}

std::vector<std::string> pre_build_all(
    const std::vector<TestEntry>& tests,
    const Registry& reg,
    const std::string& project_root,
    std::function<void(const std::string&)> log_fn)
{
    struct BuildJob { std::string build_name, target; };
    std::set<std::string> seen_jobs, seen_dirs;
    std::vector<BuildJob> jobs;
    // build dirs that need configure: eff_dir -> BuildConfig snapshot
    std::vector<std::pair<std::string, BuildConfig>> dirs_to_configure;

    for (const auto& t : tests) {
        if (t.build_name.empty() || t.target.empty()) continue;
        auto it = reg.builds.find(t.build_name);
        if (it == reg.builds.end()) continue;
        const auto& bc = it->second;
        std::string raw = bc.dir.empty() ? "build" : bc.dir;
        std::string eff = bc.sub_dir.empty() ? raw : bc.sub_dir + "/" + raw;
        std::string bd  = project_root + "/" + eff;
        if (seen_dirs.insert(eff).second && !bc.configure_cmd.empty() &&
            !fs::exists(bd + "/Makefile") && !fs::exists(bd + "/build.ninja"))
            dirs_to_configure.push_back({eff, bc});
        if (seen_jobs.insert(t.build_name + ":" + t.target).second)
            jobs.push_back({t.build_name, t.target});
    }

    if (jobs.empty()) return {};

    std::mutex log_mtx;
    auto safe_log = [&](const std::string& msg) {
        if (log_fn && !msg.empty()) {
            std::lock_guard<std::mutex> lk(log_mtx);
            log_fn(msg);
        }
    };

    std::vector<std::string> failed;
    std::mutex failed_mtx;

    // Phase 1: configure each unique build dir in parallel (fast, few dirs)
    if (!dirs_to_configure.empty()) {
        safe_log("pre-build: configuring " + std::to_string(dirs_to_configure.size()) + " dir(s)...");
        std::vector<std::thread> conf_threads;
        for (const auto& entry : dirs_to_configure) {
            const std::string eff = entry.first;
            const BuildConfig bc  = entry.second;
            conf_threads.emplace_back([&, eff, bc]() {
                std::string raw = bc.dir.empty() ? "build" : bc.dir;
                std::string base_wd = bc.sub_dir.empty()
                    ? project_root : project_root + "/" + bc.sub_dir;
                std::string configure_cmd = bc.configure_cmd;
                if (configure_cmd.find("{{arch}}") != std::string::npos) {
                    std::string arch = detect_cuda_arch();
                    if (arch.empty()) {
                        safe_log("arch detection failed for " + eff);
                        std::lock_guard<std::mutex> lk(failed_mtx);
                        failed.push_back("configure:" + eff);
                        return;
                    }
                    size_t pos = 0;
                    while ((pos = configure_cmd.find("{{arch}}", pos)) != std::string::npos) {
                        configure_cmd.replace(pos, 8, arch);
                        pos += arch.size();
                    }
                }
                bool from_build = configure_cmd.size() >= 3 &&
                                  configure_cmd.substr(configure_cmd.size() - 3) == " ..";
                std::string conf_wd = from_build ? base_wd + "/" + raw : base_wd;
                if (from_build) fs::mkdir_p(conf_wd);
                safe_log("configuring: " + configure_cmd);
                auto cr = proc::run(configure_cmd, {}, {}, 300, conf_wd,
                    [&](const std::string& line) { safe_log(line); }, true);
                if (cr.exit_code != 0) {
                    if (!cr.stderr_str.empty()) safe_log(cr.stderr_str);
                    safe_log("configure failed (exit=" + std::to_string(cr.exit_code) + ")");
                    std::lock_guard<std::mutex> lk(failed_mtx);
                    failed.push_back("configure:" + eff);
                }
            });
        }
        for (auto& th : conf_threads) th.join();
        if (!failed.empty()) return failed;
    }

    // Phase 2: build ALL unique targets in parallel, dividing CPU threads among them
    unsigned int ncpus = std::max(1u, std::thread::hardware_concurrency());
    unsigned int jobs_each = std::max(1u, ncpus / (unsigned)jobs.size());
    safe_log("pre-build: building " + std::to_string(jobs.size()) + " target(s)"
             + " (-j" + std::to_string(jobs_each) + " each, "
             + std::to_string(ncpus) + " logical CPUs)");

    std::vector<std::thread> build_threads;
    for (const auto& job : jobs) {
        build_threads.emplace_back([&, job]() {
            auto bc_it = reg.builds.find(job.build_name);
            if (bc_it == reg.builds.end()) return;
            const auto& bc = bc_it->second;
            std::string raw = bc.dir.empty() ? "build" : bc.dir;
            std::string eff = bc.sub_dir.empty() ? raw : bc.sub_dir + "/" + raw;
            std::string build_cmd = "cmake --build " + eff + " --target " + job.target
                                  + " -j" + std::to_string(jobs_each);
            safe_log("building: " + build_cmd);
            auto br = proc::run(build_cmd, {}, {}, 600, project_root,
                [&](const std::string& line) { safe_log(line); }, true);
            if (br.exit_code != 0) {
                if (!br.stderr_str.empty()) safe_log(br.stderr_str);
                safe_log("build failed: " + job.build_name + ":" + job.target);
                std::lock_guard<std::mutex> lk(failed_mtx);
                failed.push_back(job.build_name + ":" + job.target);
            }
        });
    }
    for (auto& th : build_threads) th.join();
    return failed;
}

// ── LocalRunner ───────────────────────────────────────────────────────────

LocalRunner::LocalRunner(std::string th_dir, std::string project_root,
                         std::shared_ptr<JobLog> job_log, Registry)
    : th_dir_(std::move(th_dir))
    , project_root_(std::move(project_root))
    , job_log_(std::move(job_log))
    , worker_([this] { worker_loop(); })
{}

LocalRunner::~LocalRunner() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void LocalRunner::enqueue(const TestEntry& test, const Registry& reg,
                           std::function<void(const std::string&)> log_fn,
                           std::function<void(const std::string&)> status_fn)
{
    // Snapshot the current build config so the worker sees the live state at submission time
    std::optional<BuildConfig> bc;
    if (!test.build_name.empty()) {
        auto it = reg.builds.find(test.build_name);
        if (it != reg.builds.end()) bc = it->second;
    }
    job_log_->active++;
    if (status_fn) status_fn("QUEUED");
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back({test, std::move(bc), std::move(log_fn), std::move(status_fn)});
    }
    cv_.notify_one();
}

void LocalRunner::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return stopped_ || !queue_.empty(); });
            if (stopped_ && queue_.empty()) break;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        run_task(task);
    }
}

void LocalRunner::run_task(Task& task) {
    const TestEntry& t = task.test;
    auto& log        = task.log_fn;
    auto& set_status = task.status_fn;

    if (set_status) set_status("RUNNING");
    if (log) log("running locally");

    auto now_ms = []() -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    };

    // Per-test cmake configure (once) + target build
    if (!t.build_name.empty() && !t.target.empty() && task.build_config) {
        bool ok = build_one_target(*task.build_config, t.target, project_root_, log);
        if (!ok) {
            int64_t ts = now_ms();
            TestResult res;
            res.name       = t.name;
            res.started_at = ts;
            res.ended_at   = ts;
            res.failed     = 1;
            res.exit_code  = 1;
            res.run_by     = "local";
            res.host       = "localhost";
            res.metadata["_output_tail"] = "build failed — see output above";
            std::string results_dir = th_dir_ + "/results";
            fs::mkdir_p(results_dir);
            save_result(results_dir, res);
            save_result_output(results_dir, res, res.metadata["_output_tail"]);
            if (set_status) set_status("");
            job_log_->active--;
            return;
        }
    }

    int64_t t_start = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Resolve working directory and run command.
    // When workdir is default (".") and a build is linked, run from inside the build
    // directory so that ./bin/... paths resolve correctly.
    std::string workdir = project_root_;
    std::string run_cmd = t.cmd;
    if (!t.workdir.empty() && t.workdir != ".") {
        workdir = project_root_ + "/" + t.workdir;
    } else if (!t.build_name.empty() && task.build_config) {
        const auto& bc = *task.build_config;
        std::string raw_dir = bc.dir.empty() ? "build" : bc.dir;
        std::string eff_dir = bc.sub_dir.empty() ? raw_dir : bc.sub_dir + "/" + raw_dir;
        workdir = project_root_ + "/" + eff_dir;
        // Strip "raw_dir/" prefix from cmd if present (andes-benchmarks style)
        const std::string prefix = raw_dir + "/";
        if (run_cmd.rfind(prefix, 0) == 0)
            run_cmd = "./" + run_cmd.substr(prefix.size());
    }

    // Stream stdout lines to the log panel live
    auto on_line = [&](const std::string& line) {
        if (log && !line.empty()) log(line);
    };

    auto r = proc::run(run_cmd, {}, {}, t.timeout_sec, workdir, on_line, /*use_shell=*/true);

    int64_t t_end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Parse TRAILHEAD: markers from captured stdout
    TestResult res;
    res.name       = t.name;
    res.started_at = t_start;
    res.ended_at   = t_end;
    res.wall_ms    = t_end - t_start;
    res.exit_code  = r.exit_code;
    res.run_by     = "local";
    res.host       = "localhost";

    parse_trailhead_output(r.stdout_str, res);

    // If no markers, fall back to exit code
    if (res.passed + res.failed == 0 && res.timings.empty()) {
        if (r.exit_code == 0) res.passed = 1;
        else                  res.failed = 1;
    }

    std::string full_output = r.stdout_str;
    if (!r.stderr_str.empty()) full_output += "\n--- stderr ---\n" + r.stderr_str;

    if (r.timed_out) {
        res.failed = 1;
        res.metadata["_output_tail"] = "timed out after " + std::to_string(t.timeout_sec) + "s";
    } else {
        // Store last 30 lines of combined output for the detail view inline panel
        std::deque<std::string> lines;
        std::istringstream ss(full_output);
        std::string ln;
        while (std::getline(ss, ln)) {
            lines.push_back(ln);
            if ((int)lines.size() > 30) lines.pop_front();
        }
        std::string tail;
        for (const auto& l : lines) { tail += l; tail += "\n"; }
        if (!tail.empty()) res.metadata["_output_tail"] = tail;
    }

    std::string results_dir = th_dir_ + "/results";
    fs::mkdir_p(results_dir);
    save_result(results_dir, res);
    save_result_output(results_dir, res, full_output);

    // Log outcome
    std::string badge = res.failed > 0
        ? ansi::color(ansi::BRED,   " FAIL ")
        : ansi::color(ansi::BGREEN, " PASS ");
    if (log) {
        log(badge + "  pass=" + std::to_string(res.passed)
            + "  fail=" + std::to_string(res.failed)
            + "  wall=" + fs::format_duration_ms(res.wall_ms));
        for (const auto& te : res.timings)
            log("  " + te.label + " = " + std::to_string((int)te.elapsed_ms) + "ms");
    }

    if (set_status) set_status("");
    job_log_->active--;
}

} // namespace trailhead
