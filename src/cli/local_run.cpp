#include "local_run.hpp"
#include "../core/result_store.hpp"
#include "../util/file_util.hpp"
#include "../util/ansi.hpp"
#include "../util/process.hpp"
#include <sstream>
#include <deque>
#include <chrono>

namespace trailhead {

bool has_local_gpu() {
    // nvidia-smi -L lists one line per GPU; non-empty output = GPU present
    auto r = proc::run("nvidia-smi -L", {}, {}, 5, "", nullptr, false);
    return r.exit_code == 0 && !r.stdout_str.empty();
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

    // Per-test cmake configure (once) + target build
    if (!t.build_name.empty() && !t.target.empty() && task.build_config) {
        {
            const auto& bc = *task.build_config;
            std::string raw_dir = bc.dir.empty() ? "build" : bc.dir;
            // For sub-registry builds, paths are relative to the sub-registry root
            std::string eff_build_dir = bc.sub_dir.empty()
                ? raw_dir : bc.sub_dir + "/" + raw_dir;
            std::string base_wd = bc.sub_dir.empty()
                ? project_root_ : project_root_ + "/" + bc.sub_dir;

            // Configure if build system file is absent (CMakeCache.txt alone isn't enough —
            // a partial configure may leave it without a Makefile/build.ninja).
            std::string bd = project_root_ + "/" + eff_build_dir;
            bool needs_configure = !bc.configure_cmd.empty() &&
                !fs::exists(bd + "/Makefile") && !fs::exists(bd + "/build.ninja");
            if (needs_configure) {
                std::string configure_cmd = bc.configure_cmd;

                // Substitute {{arch}} with auto-detected local GPU compute capability
                if (configure_cmd.find("{{arch}}") != std::string::npos) {
                    auto ar = proc::run(
                        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once",
                        {}, {}, 5, "", nullptr, false);
                    if (ar.exit_code == 0 && !ar.stdout_str.empty()) {
                        std::string arch;
                        for (char c : ar.stdout_str) {
                            if (c >= '0' && c <= '9') arch += c;
                            else if (c == '\n' || c == '\r') break;
                        }
                        if (!arch.empty()) {
                            size_t pos = 0;
                            while ((pos = configure_cmd.find("{{arch}}", pos)) != std::string::npos) {
                                configure_cmd.replace(pos, 8, arch);
                                pos += arch.size();
                            }
                        }
                    }
                }

                // cmake ".." form (run-from-build-dir) vs "-B build" form (run-from-source-root)
                bool from_build_dir = configure_cmd.size() >= 3 &&
                                      configure_cmd.substr(configure_cmd.size() - 3) == " ..";
                std::string conf_wd;
                if (from_build_dir) {
                    conf_wd = base_wd + "/" + raw_dir;
                    fs::mkdir_p(conf_wd);
                } else {
                    conf_wd = base_wd;
                }

                if (log) log("configuring: " + configure_cmd);
                auto cr = proc::run(configure_cmd, {}, {}, 300, conf_wd, nullptr, true);
                if (cr.exit_code != 0) {
                    if (log) {
                        if (!cr.stdout_str.empty()) log(cr.stdout_str);
                        if (!cr.stderr_str.empty()) log(cr.stderr_str);
                        log("configure failed (exit=" + std::to_string(cr.exit_code) + ")");
                    }
                    TestResult res;
                    res.name      = t.name;
                    res.failed    = 1;
                    res.exit_code = cr.exit_code;
                    res.run_by    = "local";
                    res.host      = "localhost";
                    res.metadata["_output_tail"] = cr.stdout_str + "\n" + cr.stderr_str;
                    std::string results_dir = th_dir_ + "/results";
                    fs::mkdir_p(results_dir);
                    save_result(results_dir, res);
                    save_result_output(results_dir, res, cr.stdout_str + cr.stderr_str);
                    if (set_status) set_status("");
                    job_log_->active--;
                    return;
                }
            }

            std::string build_cmd = "cmake --build " + eff_build_dir + " --target " + t.target;
            if (log) log("building: " + build_cmd);
            auto br = proc::run(build_cmd, {}, {}, 300, project_root_, nullptr, true);
            if (br.exit_code != 0) {
                if (log) {
                    if (!br.stdout_str.empty()) log(br.stdout_str);
                    if (!br.stderr_str.empty()) log(br.stderr_str);
                    log("build failed (exit=" + std::to_string(br.exit_code) + ")");
                }
                TestResult res;
                res.name      = t.name;
                res.failed    = 1;
                res.exit_code = br.exit_code;
                res.run_by    = "local";
                res.host      = "localhost";
                res.metadata["_output_tail"] = br.stdout_str + "\n" + br.stderr_str;
                std::string results_dir = th_dir_ + "/results";
                fs::mkdir_p(results_dir);
                save_result(results_dir, res);
                save_result_output(results_dir, res, br.stdout_str + br.stderr_str);
                if (set_status) set_status("");
                job_log_->active--;
                return;
            }
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
