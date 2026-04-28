#pragma once
#include "../core/registry.hpp"
#include "visualizer.hpp"
#include "sbatch_gen.hpp"
#include <string>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>
#include <unordered_set>

namespace trailhead {

struct RemoteDest {
    std::string remote;       // "user@host"
    std::string remote_path;  // "/absolute/path/on/host"
};

// Parse "user@host:/path" → RemoteDest. Returns nullopt if malformed.
std::optional<RemoteDest> parse_rsync_dest(const std::string& dest);

// Build sbatch command-line flags string from a node profile.
// e.g. "--partition=SaltSystemsLab --nodelist=d4067 --gres=gpu:1 --cpus-per-task=16 --time=02:00:00"
std::string node_sbatch_flags(const NodeProfile& node,
                               const SbatchDefaults& defs,
                               const std::string& job_name);

// Submit a test to a remote SLURM cluster and wait for it to finish.
//   1. rsync local project → remote (excluding .trailhead/results/)
//   2. ssh remote: sbatch <node-flags> .trailhead/sbatch/<name>.sbatch → job_id
//   3. poll: ssh remote squeue -j <job_id> until job finishes
//   4. read slurm output, parse TRAILHEAD: markers, write local result JSON
//   5. rsync remote .trailhead/results/ → local
//
// ── Pending job persistence ───────────────────────────────────────────────
// A pending job represents a SLURM job that has been submitted but not yet
// collected. Persisted to .trailhead/pending/<name>.json so that watch can
// resume polling after being closed and reopened.

struct PendingJob {
    std::string name;
    std::string job_id;
    std::string remote;
    std::string remote_path;
    std::string project_root;
    int64_t     started_at = 0;
};

void save_pending_job(const std::string& th_dir, const PendingJob& job);
void clear_pending_job(const std::string& th_dir, const std::string& name);
std::vector<PendingJob> load_pending_jobs(const std::string& th_dir);

// ── Queued submission persistence ─────────────────────────────────────────
// A queued submission is a test that has been requested but not yet submitted
// to SLURM (still waiting for rsync/sbatch). Saved immediately on enqueue so
// that watch can re-enqueue them on the next startup if trailhead exits first.

struct QueuedSubmission {
    std::string name;
    std::string node_name;
};

void save_queued_submission(const std::string& th_dir, const QueuedSubmission& qs);
void clear_queued_submission(const std::string& th_dir, const std::string& name);
std::vector<QueuedSubmission> load_queued_submissions(const std::string& th_dir);

// ── BatchSubmitter ────────────────────────────────────────────────────────
// Accepts test submissions from any thread, batches them (400ms grace period),
// runs rsync once per batch, then submits all sbatch jobs serially, and spawns
// one poll thread per submitted job.
//
// max_concurrent limits how many jobs are submitted to SLURM at once.
// Jobs beyond the limit sit in "QUEUED" status until a slot opens.

// Shared between BatchSubmitter and detached poll threads — keeps the
// slot count alive even after BatchSubmitter is destroyed.
struct SlotTracker {
    std::mutex              mtx;
    std::condition_variable cv;
    int                     active = 0;
    int                     max;
    explicit SlotTracker(int m) : max(m) {}

    // Block until a slot is free, then claim it.
    void acquire() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]{ return active < max; });
        active++;
    }

    // Release a slot and wake one waiter.
    void release() {
        { std::lock_guard<std::mutex> lk(mtx); active--; }
        cv.notify_one();
    }
};

class BatchSubmitter {
public:
    BatchSubmitter(Registry reg, std::string th_dir,
                   std::string project_root, RemoteDest dest,
                   std::shared_ptr<JobLog> job_log,
                   int max_concurrent = 5);
    ~BatchSubmitter();

    // Thread-safe. Increments job_log->active and sets initial status.
    void enqueue(const TestEntry& test,
                 const std::string& node_name,
                 std::function<void(const std::string&)> log_fn,
                 std::function<void(const std::string&)> status_fn);

    // Thread-safe. Marks a test name as cancelled so the worker skips it.
    void cancel(const std::string& name);

private:
    struct Submission {
        TestEntry test;
        std::string node_name;
        std::function<void(const std::string&)> log_fn;
        std::function<void(const std::string&)> status_fn;
    };

    void worker_loop();
    void process_batch(std::vector<Submission> batch);

    Registry                      reg_;
    std::string                   th_dir_;
    std::string                   project_root_;
    RemoteDest                    dest_;
    std::shared_ptr<JobLog>       job_log_;
    std::shared_ptr<SlotTracker>  slots_;

    std::mutex                    mtx_;
    std::condition_variable       cv_;
    std::deque<Submission>        queue_;
    std::unordered_set<std::string> cancelled_;
    bool                          stopped_ = false;
    std::thread                   worker_;
};

// ── SLURM limit query ─────────────────────────────────────────────────────
// Query the remote cluster for the per-user max-submit limit that applies to
// the given partition names.  Returns the smallest limit found, or `fallback`
// if the cluster has no configured limit / the query fails.
int query_slurm_job_limit(const std::string& remote,
                           const std::vector<std::string>& partitions,
                           int fallback = 5);

// ── Job submission / resume ───────────────────────────────────────────────

// log_fn    receives human-readable status lines to display in the log panel.
// status_fn receives short status tokens ("RSYNC","PENDING","RUNNING","") for
//           live display in the test table. Called with "" to clear on completion.

bool remote_submit_and_wait(
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    const std::string& project_root,
    const RemoteDest& dest,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn = nullptr);

// Resume polling a previously submitted job (skips rsync and sbatch).
bool resume_job(
    const PendingJob& pending,
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn = nullptr);

} // namespace trailhead
