#include "remote_run.hpp"
#include "../core/result_store.hpp"
#include "../core/json.hpp"
#include "../util/file_util.hpp"
#include "../util/ansi.hpp"
#include "../util/process.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <ctime>
#include <climits>

namespace trailhead {

// Last path component of a directory path (e.g. "/a/b/gunrock" → "gunrock")
static std::string last_component(const std::string& path) {
    auto sl = path.rfind('/');
    return (sl != std::string::npos && sl + 1 < path.size())
        ? path.substr(sl + 1) : path;
}

// ── Public helpers ────────────────────────────────────────────────────────

std::optional<RemoteDest> parse_rsync_dest(const std::string& dest) {
    auto colon = dest.find(':');
    if (colon == std::string::npos || colon == 0) return std::nullopt;
    RemoteDest d;
    d.remote      = dest.substr(0, colon);
    d.remote_path = dest.substr(colon + 1);
    if (d.remote_path.empty()) return std::nullopt;
    return d;
}

std::string node_sbatch_flags(const NodeProfile& node,
                               const SbatchDefaults& defs,
                               const std::string& job_name)
{
    std::ostringstream o;
    auto flag = [&](const std::string& k, const std::string& v) {
        if (!v.empty()) o << " --" << k << "=" << v;
    };
    auto flag_int = [&](const std::string& k, int v, int skip = 0) {
        if (v != skip) o << " --" << k << "=" << v;
    };
    flag("job-name", job_name);
    flag("partition", node.partition);
    if (!node.gpu_type.empty()) {
        o << " --gres=gpu:" << node.gpu_type;
    } else if (!node.nodelist.empty()) {
        flag("nodelist", node.nodelist);
        o << " --gres=gpu:1";
    }
    flag_int("nodes",         node.nodes,         1);
    flag_int("ntasks",        node.ntasks,        1);
    flag_int("cpus-per-task", node.cpus_per_task, 1);
    flag("time", node.time);
    flag("output", defs.output_pattern);
    flag("error",  defs.error_pattern);
    return o.str();
}

// ── Pending job persistence ───────────────────────────────────────────────

static std::string pending_dir(const std::string& th_dir) {
    return th_dir + "/pending";
}
// Replace '/' so sub-registry names ("sub/test") stay flat in the directory.
static std::string safe_filename(const std::string& name) {
    std::string f = name;
    std::replace(f.begin(), f.end(), '/', '-');
    return f;
}
static std::string pending_path(const std::string& th_dir, const std::string& name) {
    return pending_dir(th_dir) + "/" + safe_filename(name) + ".json";
}

void save_pending_job(const std::string& th_dir, const PendingJob& job) {
    fs::mkdir_p(pending_dir(th_dir));
    JsonObject obj;
    obj.push_back({"name",         job.name});
    obj.push_back({"job_id",       job.job_id});
    obj.push_back({"remote",       job.remote});
    obj.push_back({"remote_path",  job.remote_path});
    obj.push_back({"project_root", job.project_root});
    obj.push_back({"started_at",   JsonValue(job.started_at)});
    fs::write_file_atomic(pending_path(th_dir, job.name),
                          json_emit(JsonValue(std::move(obj))));
}

void clear_pending_job(const std::string& th_dir, const std::string& name) {
    ::remove(pending_path(th_dir, name).c_str());
}

std::vector<PendingJob> load_pending_jobs(const std::string& th_dir) {
    std::vector<PendingJob> out;
    for (const auto& path : fs::list_dir(pending_dir(th_dir), ".json")) {
        try {
            auto text = fs::read_file(path);
            if (!text) continue;
            auto val = json_parse(*text);
            PendingJob j;
            j.name         = val.get_str("name",         "");
            j.job_id       = val.get_str("job_id",       "");
            j.remote       = val.get_str("remote",       "");
            j.remote_path  = val.get_str("remote_path",  "");
            j.project_root = val.get_str("project_root", "");
            j.started_at   = val.get_int("started_at",   0);
            if (!j.name.empty() && !j.job_id.empty() && !j.remote.empty())
                out.push_back(std::move(j));
        } catch (...) {}
    }
    return out;
}

// ── Queued submission persistence ────────────────────────────────────────

static std::string queued_dir(const std::string& th_dir) {
    return th_dir + "/queued";
}
static std::string queued_path(const std::string& th_dir, const std::string& name) {
    return queued_dir(th_dir) + "/" + safe_filename(name) + ".json";
}

void save_queued_submission(const std::string& th_dir, const QueuedSubmission& qs) {
    fs::mkdir_p(queued_dir(th_dir));
    JsonObject obj;
    obj.push_back({"name",      qs.name});
    obj.push_back({"node_name", qs.node_name});
    fs::write_file_atomic(queued_path(th_dir, qs.name),
                          json_emit(JsonValue(std::move(obj))));
}

void clear_queued_submission(const std::string& th_dir, const std::string& name) {
    ::remove(queued_path(th_dir, name).c_str());
}

std::vector<QueuedSubmission> load_queued_submissions(const std::string& th_dir) {
    std::vector<QueuedSubmission> out;
    for (const auto& path : fs::list_dir(queued_dir(th_dir), ".json")) {
        try {
            auto text = fs::read_file(path);
            if (!text) continue;
            auto val = json_parse(*text);
            QueuedSubmission qs;
            qs.name      = val.get_str("name",      "");
            qs.node_name = val.get_str("node_name", "");
            if (!qs.name.empty())
                out.push_back(std::move(qs));
        } catch (...) {}
    }
    return out;
}

// ── Low-level SSH / process helpers ──────────────────────────────────────

static proc::RunResult ssh_run(const std::string& remote, const std::string& remote_cmd) {
    std::string cmd = "ssh -o BatchMode=yes -o ConnectTimeout=10 " + remote
                    + " " + remote_cmd;
    return proc::run(cmd, {}, {}, 120, "", nullptr, true);
}

static void write_result_json(const std::string& results_dir, const TestResult& res) {
    std::string path = results_dir + "/" + res.name + "_"
                     + std::to_string(res.started_at) + ".json";
    { auto sl = path.rfind('/'); if (sl != std::string::npos) fs::mkdir_p(path.substr(0, sl)); }
    JsonObject obj;
    obj.push_back({"version",    JsonValue((int64_t)1)});
    obj.push_back({"name",       res.name});
    obj.push_back({"host",       res.host});
    obj.push_back({"run_by",     res.run_by});
    obj.push_back({"started_at", JsonValue(res.started_at)});
    obj.push_back({"ended_at",   JsonValue(res.ended_at)});
    obj.push_back({"wall_ms",    JsonValue(res.wall_ms)});
    obj.push_back({"exit_code",  JsonValue((int64_t)res.exit_code)});
    obj.push_back({"passed",     JsonValue((int64_t)res.passed)});
    obj.push_back({"failed",     JsonValue((int64_t)res.failed)});
    JsonArray timings_arr;
    for (const auto& te : res.timings) {
        JsonObject te_obj;
        te_obj.push_back({"label",      te.label});
        te_obj.push_back({"elapsed_ms", JsonValue(te.elapsed_ms)});
        timings_arr.push_back(JsonValue(std::move(te_obj)));
    }
    obj.push_back({"timings", JsonValue(std::move(timings_arr))});
    JsonObject meta_obj;
    for (const auto& [k, v] : res.metadata)
        meta_obj.push_back({k, JsonValue(v)});
    obj.push_back({"metadata", JsonValue(std::move(meta_obj))});
    JsonArray out_arr;
    for (const auto& ln : res.output_lines) out_arr.push_back(JsonValue(ln));
    obj.push_back({"output", JsonValue(std::move(out_arr))});
    fs::write_file_atomic(path, json_emit(JsonValue(std::move(obj))));
}

// ── Step: rsync project → remote ─────────────────────────────────────────

static bool do_rsync(const std::string& project_root,
                     const RemoteDest& dest,
                     const std::function<void(const std::string&)>& log)
{
    // rsync_dest is the *parent* directory; append the project name so the
    // repo lands at remote_path/project_name/ rather than directly in remote_path/
    std::string proj_name   = last_component(project_root);
    std::string remote_dest = dest.remote_path + "/" + proj_name;

    log(ansi::BOLD + std::string("rsync") + ansi::RESET
        + "  " + project_root + " → " + dest.remote + ":" + remote_dest);

    std::string cmd = "rsync -az --exclude='.trailhead/results/' "
        + project_root + "/ "
        + dest.remote + ":" + remote_dest + "/";

    auto r = proc::run(cmd, {}, {}, 120, "", nullptr, true);
    if (r.exit_code != 0) {
        log(ansi::color(ansi::BRED, "rsync failed"));
        std::istringstream ss(r.stderr_str);
        for (std::string ln; std::getline(ss, ln); )
            if (!ln.empty()) log("  " + ln);
        return false;
    }
    return true;
}

// ── Step: submit one sbatch job → job_id ("" on failure) ─────────────────

static std::string do_sbatch(const TestEntry& test,
                              const std::string& node_name,
                              const Registry& reg,
                              const RemoteDest& dest,
                              const std::string& project_root,
                              const std::string& th_dir,
                              const std::function<void(const std::string&)>& log,
                              const std::function<void(const std::string&)>& set_status)
{
    const NodeProfile* node = nullptr;
    if (!node_name.empty()) {
        auto it = reg.nodes.find(node_name);
        if (it != reg.nodes.end()) node = &it->second;
    }

    std::string job_name = reg.sbatch_defaults.job_name_prefix + "-" + test.name;
    std::string flags    = node ? node_sbatch_flags(*node, reg.sbatch_defaults, job_name) : "";
    std::string script   = ".trailhead/sbatch/" + test.name + ".sbatch";

    set_status("SUBMIT");
    log(ansi::BOLD + std::string("sbatch") + ansi::RESET + " " + script
        + (node ? "  [" + (node->gpu_type.empty() ? node->nodelist : node->gpu_type) + "]" : ""));

    std::string remote_project = dest.remote_path + "/" + last_component(project_root);
    std::string sbatch_cmd = "\"cd " + remote_project
                           + " && sbatch" + flags + " " + script + "\"";
    std::string ssh_cmd = "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                        + dest.remote + " " + sbatch_cmd;

    auto r = proc::run(ssh_cmd, {}, {}, 30, "", nullptr, true);
    if (r.exit_code != 0) {
        log(ansi::color(ansi::BRED, "sbatch failed (exit=" + std::to_string(r.exit_code) + ")"));
        // sbatch writes errors to stdout or stderr depending on the error type — log both
        for (const std::string* s : {&r.stdout_str, &r.stderr_str}) {
            std::istringstream ss(*s);
            for (std::string ln; std::getline(ss, ln); )
                if (!ln.empty()) log("  " + ln);
        }
        // Write failure result so board reflects it
        {
            int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::string results_dir = th_dir + "/results";
            fs::mkdir_p(results_dir);
            TestResult res;
            res.name       = test.name;
            res.started_at = now;
            res.ended_at   = now;
            res.run_by     = "sbatch-failed";
            res.host       = dest.remote;
            res.failed     = 1;
            std::string output = "sbatch failed (exit=" + std::to_string(r.exit_code) + ")\n";
            if (!r.stdout_str.empty()) output += r.stdout_str;
            if (!r.stderr_str.empty()) output += r.stderr_str;
            res.metadata["_output_tail"] = output;
            write_result_json(results_dir, res);
        }
        set_status("");
        return "";
    }

    // Parse "Submitted batch job 12345"
    std::string job_id;
    {
        std::istringstream ss(r.stdout_str);
        std::string tok;
        while (ss >> tok) job_id = tok;
    }
    if (job_id.empty()) {
        log(ansi::color(ansi::BRED, "could not parse job ID: ") + r.stdout_str);
        set_status("");
        return "";
    }
    return job_id;
}

// ── Desktop notification ──────────────────────────────────────────────────

static void desktop_notify(const std::string& title, const std::string& message) {
#ifdef __APPLE__
    std::string cmd = "osascript -e 'display notification \""
        + message + "\" with title \"" + title + "\"' 2>/dev/null &";
#else
    std::string cmd = "notify-send '" + title + "' '" + message + "' 2>/dev/null &";
#endif
    (void)::system(cmd.c_str());
}

// ── Step: poll + collect result ───────────────────────────────────────────

static bool poll_and_finalize(
    const std::string& job_id,
    int64_t t_submit,
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    const RemoteDest& dest,
    const std::string& project_root,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn,
    std::function<void()> on_job_done = {})
{
    auto log = [&](const std::string& msg) {
        if (log_fn) log_fn(msg);
        else { std::cout << msg << "\n"; std::cout.flush(); }
    };
    auto set_status = [&](const std::string& s) {
        if (status_fn) status_fn(s);
    };

    std::string results_dir = th_dir + "/results";
    fs::mkdir_p(results_dir);

    std::string sacct_line;
    for (int poll = 0; ; ++poll) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::string sq_cmd = "\"squeue -j " + job_id + " -h -o '%T' 2>/dev/null\"";
        auto sq_r = ssh_run(dest.remote, sq_cmd);
        std::string sq_out = sq_r.stdout_str;
        while (!sq_out.empty() && (sq_out.back() == '\n' || sq_out.back() == '\r' || sq_out.back() == ' '))
            sq_out.pop_back();

        // SSH failed — don't interpret empty output as "job finished"
        if (sq_r.exit_code != 0 || sq_r.timed_out) {
            if (poll % 6 == 0)
                log("  job " + job_id + " " + ansi::DIM + "ssh failed, retrying..." + ansi::RESET);
            continue;
        }

        if (sq_out.empty()) {
            std::string sacct_cmd = "\"sacct -j " + job_id
                + " -o State,ExitCode,Reason -n --parsable2 2>/dev/null | head -1\"";
            auto sacct_r = ssh_run(dest.remote, sacct_cmd);

            // If sacct SSH also failed, don't assume job is done — keep polling
            if (sacct_r.exit_code != 0 || sacct_r.timed_out) {
                if (poll % 6 == 0)
                    log("  job " + job_id + " " + ansi::DIM + "ssh failed, retrying..." + ansi::RESET);
                continue;
            }

            sacct_line = sacct_r.stdout_str;
            while (!sacct_line.empty() && (sacct_line.back() == '\n' || sacct_line.back() == '\r'))
                sacct_line.pop_back();

            std::string sacct_state;
            auto pipe_pos = sacct_line.find('|');
            sacct_state = (pipe_pos != std::string::npos) ? sacct_line.substr(0, pipe_pos) : sacct_line;

            if (sacct_state == "COMPLETED" || sacct_state.empty())
                log("  job " + job_id + " finished");
            else
                log(ansi::color(ansi::BRED, "  job " + job_id + " ended: ") + sacct_line);
            break;
        }

        if (sq_out == "RUNNING" || sq_out == "COMPLETING")
            set_status("RUNNING");
        else
            set_status(sq_out);

        if (poll % 6 == 0)
            log("  job " + job_id + " " + ansi::DIM + sq_out + ansi::RESET);

        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        //watch timeout disabled - no reason to time
        // if (now - t_submit > 7200LL * 1000) {
        //     log(ansi::color(ansi::BYELLOW, "watch timeout (2h) — job may still be running"));
        //     if (on_job_done) on_job_done();
        //     set_status("");
        //     return false;
        // }
    }

    // Brief pause so SLURM's QOS accounting catches up before we submit the next job.
    // squeue going empty doesn't always mean the slot counter has decremented yet;
    // without this, rapid succession of completions triggers QOSMaxSubmitJobPerUserLimit.
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Release the SLURM slot — a new sbatch can go in now.
    // Result collection (ssh cat + rsync-back) continues below.
    if (on_job_done) on_job_done();

    int64_t t_end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    auto subst_job_id = [&](const std::string& pattern) {
        std::string s = pattern;
        auto pos = s.find("%j");
        if (pos != std::string::npos) s.replace(pos, 2, job_id);
        return s;
    };

    // rsync_dest is the parent dir; the project landed at remote_path/project_name/
    std::string remote_project = dest.remote_path + "/" + last_component(project_root);
    std::string remote_out = remote_project + "/" + subst_job_id(reg.sbatch_defaults.output_pattern);
    std::string remote_err = remote_project + "/" + subst_job_id(reg.sbatch_defaults.error_pattern);

    std::string slurm_stdout = ssh_run(dest.remote, "\"cat " + remote_out + " 2>/dev/null\"").stdout_str;
    std::string slurm_stderr = ssh_run(dest.remote, "\"cat " + remote_err + " 2>/dev/null\"").stdout_str;

    TestResult res;
    res.name       = test.name;
    res.started_at = t_submit;
    res.ended_at   = t_end;
    res.wall_ms    = t_end - t_submit;
    res.run_by     = "sbatch-" + job_id;
    res.host       = dest.remote;

    std::string remaining;
    parse_trailhead_output(slurm_stdout, res, &remaining);

    if (!sacct_line.empty())
        res.metadata["_sacct"] = sacct_line;

    {
        std::istringstream out_ss(slurm_stdout);
        std::deque<std::string> out_lines;
        for (std::string ln; std::getline(out_ss, ln); ) {
            out_lines.push_back(ln);
            if ((int)out_lines.size() > 20) out_lines.pop_front();
        }
        std::string tail;
        for (const auto& ln : out_lines) { tail += ln; tail += "\n"; }
        if (!slurm_stderr.empty() && slurm_stderr != slurm_stdout) {
            tail += "--- stderr ---\n";
            std::istringstream err_ss(slurm_stderr);
            std::deque<std::string> err_lines;
            for (std::string ln; std::getline(err_ss, ln); ) {
                err_lines.push_back(ln);
                if ((int)err_lines.size() > 10) err_lines.pop_front();
            }
            for (const auto& ln : err_lines) { tail += ln; tail += "\n"; }
        }
        if (!tail.empty()) res.metadata["_output_tail"] = tail;
    }

    if ((res.passed + res.failed == 0) && res.timings.empty()) {
        res.failed = 1;
        log(ansi::color(ansi::BYELLOW, "no TRAILHEAD: markers in output — marking as FAIL"));
    }

    write_result_json(results_dir, res);
    {
        std::string full_out = slurm_stdout;
        if (!slurm_stderr.empty() && slurm_stderr != slurm_stdout)
            full_out += "\n--- stderr ---\n" + slurm_stderr;
        save_result_output(results_dir, res, full_out);
    }
    clear_pending_job(th_dir, test.name);

    std::string badge = res.failed > 0
        ? ansi::color(ansi::BRED,   " FAIL ")
        : ansi::color(ansi::BGREEN, " PASS ");
    log(badge + "  pass=" + std::to_string(res.passed)
        + "  fail=" + std::to_string(res.failed)
        + "  wall=" + fs::format_duration_ms(res.wall_ms));
    for (const auto& te : res.timings)
        log("  " + te.label + " = " + std::to_string((int)te.elapsed_ms) + "ms");

    std::string rsync_back = "rsync -az "
        + dest.remote + ":" + dest.remote_path + "/.trailhead/results/ "
        + results_dir + "/";
    proc::run(rsync_back, {}, {}, 60, "", nullptr, true);

    desktop_notify("trailhead: " + test.name,
                   res.failed > 0 ? "FAIL" : "PASS");

    set_status("");
    return true;
}

// ── SLURM limit query ─────────────────────────────────────────────────────

int query_slurm_job_limit(const std::string& remote,
                           const std::vector<std::string>& partitions,
                           int fallback)
{
    // Step 1: resolve partition → QoS name via scontrol
    std::vector<std::string> qos_names;
    for (const auto& part : partitions) {
        if (part.empty()) continue;
        std::string cmd = "\"scontrol show partition " + part + " --oneliner 2>/dev/null\"";
        std::string out = ssh_run(remote, cmd).stdout_str;
        // Parse "QoS=<name>" from the one-liner output
        auto pos = out.find("QoS=");
        if (pos == std::string::npos) continue;
        pos += 4;
        auto end = out.find_first_of(" \t\r\n", pos);
        std::string qos = out.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (!qos.empty() && qos != "N/A") {
            // Avoid duplicates
            bool found = false;
            for (const auto& q : qos_names) if (q == qos) { found = true; break; }
            if (!found) qos_names.push_back(qos);
        }
    }

    if (qos_names.empty()) return fallback;

    // Step 2: query sacctmgr for MaxSubmitJobsPU for each QoS
    // Command: sacctmgr show qos <name> format=MaxSubmitJobsPU --parsable2 --noheader
    int limit = INT_MAX;
    for (const auto& qos : qos_names) {
        std::string cmd = "\"sacctmgr show qos " + qos
            + " format=MaxSubmitJobsPU --parsable2 --noheader 2>/dev/null\"";
        std::string out = ssh_run(remote, cmd).stdout_str;
        // Strip trailing whitespace/newlines
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
            out.pop_back();
        // Could be "10" or "UNLIMITED" or ""
        if (out.empty() || out == "UNLIMITED") continue;
        try {
            int v = std::stoi(out);
            if (v > 0 && v < limit) limit = v;
        } catch (...) {}
    }

    return (limit == INT_MAX) ? fallback : limit;
}

// ── BatchSubmitter ────────────────────────────────────────────────────────

BatchSubmitter::BatchSubmitter(Registry reg, std::string th_dir,
                               std::string project_root, RemoteDest dest,
                               std::shared_ptr<JobLog> job_log,
                               int max_concurrent)
    : reg_(std::move(reg))
    , th_dir_(std::move(th_dir))
    , project_root_(std::move(project_root))
    , dest_(std::move(dest))
    , job_log_(std::move(job_log))
    , slots_(std::make_shared<SlotTracker>(max_concurrent))
    , worker_([this] { worker_loop(); })
{}

BatchSubmitter::~BatchSubmitter() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void BatchSubmitter::enqueue(const TestEntry& test,
                              const std::string& node_name,
                              std::function<void(const std::string&)> log_fn,
                              std::function<void(const std::string&)> status_fn)
{
    job_log_->active++;
    if (status_fn) status_fn("QUEUED");
    save_queued_submission(th_dir_, {test.name, node_name});
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back({test, node_name, std::move(log_fn), std::move(status_fn)});
    }
    cv_.notify_one();
}

void BatchSubmitter::cancel(const std::string& name) {
    std::lock_guard<std::mutex> lk(mtx_);
    // Remove from in-memory queue if still waiting
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->test.name == name) {
            if (it->status_fn) it->status_fn("");
            job_log_->active--;
            queue_.erase(it);
            return;
        }
    }
    // Already picked up by worker — mark for skip in process_batch
    cancelled_.insert(name);
}

void BatchSubmitter::worker_loop() {
    while (true) {
        std::vector<Submission> batch;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return stopped_ || !queue_.empty(); });
            if (stopped_ && queue_.empty()) break;

            // Short grace period to collect additional presses before rsyncing.
            // Interruptible: wakes early if stopped_ is set during the wait.
            cv_.wait_for(lk, std::chrono::milliseconds(400), [&] { return stopped_; });
            if (stopped_ && queue_.empty()) break;

            batch = {queue_.begin(), queue_.end()};
            queue_.clear();
        }
        if (!batch.empty())
            process_batch(std::move(batch));
    }
}

void BatchSubmitter::process_batch(std::vector<Submission> batch) {
    // Refresh registry from disk so hardware edits made during the session are picked up.
    // process_batch only runs on the single worker thread, so no locking needed.
    if (auto fresh = load_registry(th_dir_)) {
        reg_ = *fresh;
        merge_sub_registries(reg_, project_root_);
    }

    // Shared log: prefix with "[batch N]" for messages that apply to all tests
    int n = (int)batch.size();
    auto blog = [&](const std::string& msg) {
        std::string prefix = n > 1 ? "[batch×" + std::to_string(n) + "] " : "";
        for (auto& s : batch)
            if (s.log_fn) s.log_fn(prefix + msg);
    };

    // Clear queued submission files — from here the job is in-flight
    for (auto& s : batch)
        clear_queued_submission(th_dir_, s.test.name);

    // Drop cancelled jobs from the batch
    {
        std::lock_guard<std::mutex> lk(mtx_);
        batch.erase(std::remove_if(batch.begin(), batch.end(), [&](const Submission& s) {
            if (cancelled_.count(s.test.name)) {
                cancelled_.erase(s.test.name);
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
                return true;
            }
            return false;
        }), batch.end());
    }
    if (batch.empty()) return;

    // Set all to RSYNC status while uploading
    for (auto& s : batch)
        if (s.status_fn) s.status_fn("RSYNC");

    // Regenerate sbatch scripts for the selected hardware before rsync uploads them
    {
        SbatchOptions script_opts;
        script_opts.split        = true;
        script_opts.project_root = project_root_;
        fs::mkdir_p(th_dir_ + "/sbatch");
        for (auto& s : batch) {
            if (s.node_name.empty()) continue;
            std::string content = generate_test_script(s.test, s.node_name, reg_, script_opts);
            if (!content.empty()) {
                std::string path = th_dir_ + "/sbatch/" + s.test.name + ".sbatch";
                auto slash = path.rfind('/');
                if (slash != std::string::npos) fs::mkdir_p(path.substr(0, slash));
                fs::write_file_atomic(path, content);
            }
        }
    }

    // Bail out early if the TUI has been closed while we were waiting.
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (stopped_) {
            for (auto& s : batch) {
                // Restore queued file so the job is re-enqueued on next startup.
                save_queued_submission(th_dir_, {s.test.name, s.node_name});
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
            }
            return;
        }
    }

    if (!do_rsync(project_root_, dest_, [&](const std::string& m){ blog(m); })) {
        // rsync failed — fail all tests in the batch
        for (auto& s : batch) {
            if (s.status_fn) s.status_fn("");
            job_log_->active--;
        }
        return;
    }

    // Submit each job serially, throttled by the slot limit.
    // Acquiring a slot blocks until SLURM has room for another job.
    for (auto& s : batch) {
        // Skip this job if the TUI was closed between submissions
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stopped_) {
                // Restore queued file so the job is re-enqueued on next startup.
                save_queued_submission(th_dir_, {s.test.name, s.node_name});
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
                continue;
            }
            if (cancelled_.erase(s.test.name)) {
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
                continue;
            }
        }

        // Wait for a free slot — show QUEUED if we have to wait
        {
            std::unique_lock<std::mutex> lk(slots_->mtx);
            if (slots_->active >= slots_->max) {
                if (s.status_fn) s.status_fn("QUEUED");
                slots_->cv.wait(lk, [&]{ return slots_->active < slots_->max; });
            }
            slots_->active++;
        }

        // Re-check cancelled after potentially long slot wait
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (cancelled_.erase(s.test.name)) {
                slots_->release();
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
                continue;
            }
        }

        int64_t t_submit = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::string job_id = do_sbatch(s.test, s.node_name, reg_, dest_, project_root_, th_dir_,
            s.log_fn ? s.log_fn : [](const std::string&){},
            s.status_fn ? s.status_fn : [](const std::string&){});

        if (job_id.empty()) {
            // sbatch failed — release the slot we just acquired
            slots_->release();
            if (s.status_fn) s.status_fn("");
            job_log_->active--;
            continue;
        }

        // If cancelled while sbatch was in flight, kill the just-submitted job
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (cancelled_.erase(s.test.name)) {
                std::string scancel_cmd = "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                    + dest_.remote + " scancel " + job_id;
                std::thread([scancel_cmd]{ proc::run(scancel_cmd, {}, {}, 15, "", nullptr, true); }).detach();
                slots_->release();
                if (s.status_fn) s.status_fn("");
                job_log_->active--;
                continue;
            }
        }

        // Save pending state so watch can resume after close
        PendingJob pj;
        pj.name         = s.test.name;
        pj.job_id       = job_id;
        pj.remote       = dest_.remote;
        pj.remote_path  = dest_.remote_path;
        pj.project_root = project_root_;
        pj.started_at   = t_submit;
        save_pending_job(th_dir_, pj);

        if (s.status_fn) s.status_fn("PENDING");
        if (s.log_fn) s.log_fn("  job " + job_id + " submitted");

        // Spawn poll thread — releases the slot when the job completes
        auto test         = s.test;
        auto log_fn       = s.log_fn;
        auto status_fn    = s.status_fn;
        auto reg          = reg_;
        auto th_dir       = th_dir_;
        auto dest         = dest_;
        auto project_root = project_root_;
        auto job_log      = job_log_;
        auto slots        = slots_;

        std::thread([=]() mutable {
            poll_and_finalize(job_id, t_submit, test, reg, th_dir, dest, project_root, log_fn, status_fn,
                              [slots]{ slots->release(); });
            job_log->active--;
        }).detach();
    }
}

// ── Single-shot submit (non-batched, for direct use) ─────────────────────

bool remote_submit_and_wait(
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    const std::string& project_root,
    const RemoteDest& dest,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn)
{
    auto log = [&](const std::string& msg) {
        if (log_fn) log_fn(msg);
        else { std::cout << msg << "\n"; std::cout.flush(); }
    };
    auto set_status = [&](const std::string& s) { if (status_fn) status_fn(s); };

    set_status("RSYNC");
    if (!do_rsync(project_root, dest, log)) { set_status(""); return false; }

    int64_t t_submit = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::string job_id = do_sbatch(test, "", reg, dest, project_root, th_dir, log, set_status);
    if (job_id.empty()) return false;

    PendingJob pj;
    pj.name         = test.name;
    pj.job_id       = job_id;
    pj.remote       = dest.remote;
    pj.remote_path  = dest.remote_path;
    pj.project_root = project_root;
    pj.started_at   = t_submit;
    save_pending_job(th_dir, pj);

    set_status("PENDING");
    log("  job " + job_id + " submitted");

    return poll_and_finalize(job_id, t_submit, test, reg, th_dir, dest, project_root, log_fn, status_fn);
}

// ── Resume a previously submitted job ────────────────────────────────────

bool resume_job(
    const PendingJob& pending,
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn)
{
    RemoteDest dest{pending.remote, pending.remote_path};
    if (log_fn) log_fn("  resuming job " + pending.job_id);
    if (status_fn) status_fn("PENDING");
    return poll_and_finalize(pending.job_id, pending.started_at,
                             test, reg, th_dir, dest, pending.project_root, log_fn, status_fn);
}

} // namespace trailhead
