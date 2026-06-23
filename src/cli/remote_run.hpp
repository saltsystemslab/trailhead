#pragma once
#include "../core/registry.hpp"
#include "../util/process.hpp"
#include "visualizer.hpp"
#include "sbatch_gen.hpp"
#include <string>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <map>
#include <set>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <atomic>

namespace trailhead {

// ── RemoteChannel ─────────────────────────────────────────────────────────
// Persistent SSH connection using ControlMaster multiplexing.
// All SSH operations reuse one socket, eliminating per-call connection overhead.
// A background heartbeat closes the connection after kIdleSecs of inactivity.
// Lazily connects on first use; auto-reconnects if the socket disappears.
class RemoteChannel {
public:
    static constexpr int kIdleSecs = 120;

    explicit RemoteChannel(std::string remote);
    ~RemoteChannel();

    // Run a remote command through the persistent connection.
    proc::RunResult ssh(const std::string& remote_cmd, int timeout_sec = 120);

    // Path to the ControlMaster socket (used by rsync -e to share the connection).
    std::string ctrl_path() const;

private:
    void ensure_connected();
    // Tear down a dead/zombie master and rebuild it, debounced so concurrent
    // callers don't clobber each other's freshly-established socket.
    void force_reconnect();
    void disconnect();
    void heartbeat_loop();

    std::string  remote_;
    std::string  ctrl_path_;
    std::mutex   mtx_;
    bool         connected_ = false;
    std::chrono::steady_clock::time_point last_use_;
    std::chrono::steady_clock::time_point last_reconnect_{};
    std::atomic<bool> stopped_{false};
    std::mutex              hb_mtx_;   // guards the heartbeat sleep
    std::condition_variable hb_cv_;    // signalled on stop to end the sleep early
    std::thread  heartbeat_;
};

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
    std::string node_name;   // node profile the job was submitted to
    int64_t     started_at = 0;
};

void save_pending_job(const std::string& th_dir, const PendingJob& job);
// Clears the pending record for a specific (test, node) — the same test can be
// in flight on two nodes at once, each with its own file.
void clear_pending_job(const std::string& th_dir, const std::string& name,
                       const std::string& node);
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
// Clears the queued record for a specific (test, node).
void clear_queued_submission(const std::string& th_dir, const std::string& name,
                             const std::string& node);
std::vector<QueuedSubmission> load_queued_submissions(const std::string& th_dir);

// ── BatchSubmitter ────────────────────────────────────────────────────────
// Accepts test submissions from any thread, batches them (400ms grace period),
// runs rsync once per batch, then submits all sbatch jobs serially, and spawns
// one poll thread per submitted job.
//
// max_concurrent limits how many jobs are submitted to SLURM at once.
// Jobs beyond the limit sit in "QUEUED" status until a slot opens.

// Query the number of jobs the current user has in SLURM (pending, running,
// completing).  Returns -1 on failure.  When `partition` is non-empty the
// count is restricted to that partition; when `gpu_type` is non-empty it is
// further restricted to jobs whose requested gres includes that GPU type.
// Together they scope the count to one "machine", so jobs on other partitions
// or GPU types (even within the same partition) don't count against this
// scope's slot budget.
int query_slurm_user_job_count(const std::string& remote,
                               const std::string& partition = "",
                               const std::string& gpu_type  = "");

// Thread-safe set of cancelled (test, node) submissions. Keyed by node too so
// cancelling a test on one processor doesn't cancel the same test running on
// another. Shared (via shared_ptr) between BatchSubmitter and detached poll
// threads so cancellation reaches every stage.
struct CancelSet {
    std::mutex              mtx;
    std::condition_variable cv;   // notified on every insert
    std::set<std::pair<std::string, std::string>> keys;

    void insert(const std::string& n, const std::string& node) {
        { std::lock_guard<std::mutex> lk(mtx); keys.insert({n, node}); }
        cv.notify_all();
    }
    bool erase(const std::string& n, const std::string& node) {
        std::lock_guard<std::mutex> lk(mtx); return keys.erase({n, node}) > 0;
    }
    bool contains(const std::string& n, const std::string& node) {
        std::lock_guard<std::mutex> lk(mtx); return keys.count({n, node}) > 0;
    }
};

// Shared between BatchSubmitter and detached poll threads — keeps the
// slot count alive even after BatchSubmitter is destroyed.
//
// Instead of a purely internal counter, acquire() queries the real SLURM
// queue so that interactive jobs and jobs from other tools are counted.
//
// One SlotTracker exists per scope (SLURM partition / GPU type). Each tracker
// only counts jobs in its own partition, so a backlog of pending jobs on one
// machine never throttles submissions to a different machine.
struct SlotTracker {
    std::mutex              mtx;
    std::condition_variable cv;
    int                     max;
    std::string             remote;     // ssh target for squeue queries
    std::string             partition;  // squeue scope; empty = all partitions
    std::string             gpu_type;   // gres scope; empty = any gres
    std::atomic<bool>       aborted{false};  // set on shutdown to release waiters

    SlotTracker(int m, std::string r, std::string part = "", std::string gpu = "")
        : max(m), remote(std::move(r)), partition(std::move(part)),
          gpu_type(std::move(gpu)) {}

    // Block until the real SLURM queue has room, then return true.
    // Returns false immediately if is_cancelled() fires or the tracker is
    // aborted (e.g. the TUI is closing) so shutdown doesn't wait for a SLURM
    // slot to free.
    bool acquire(std::function<bool()> is_cancelled = {}) {
        std::unique_lock<std::mutex> lk(mtx);
        while (true) {
            if (aborted.load()) return false;
            if (is_cancelled && is_cancelled()) return false;
            int count = query_slurm_user_job_count(remote, partition, gpu_type);
            if (count >= 0 && count < max) return true;
            cv.wait_for(lk, std::chrono::seconds(5));
        }
    }

    // Wake waiters so they re-check the queue.
    void release() {
        cv.notify_all();
    }

    // Permanently release all waiters (returns false from acquire).
    void abort() {
        aborted.store(true);
        cv.notify_all();
    }
};

class BatchSubmitter {
public:
    BatchSubmitter(Registry reg, std::string th_dir,
                   std::string project_root, RemoteDest dest,
                   std::shared_ptr<JobLog> job_log,
                   int max_concurrent = 4);
    ~BatchSubmitter();

    // Expose the channel so callers can issue raw SSH commands through the
    // same persistent connection (e.g. to check server status).
    std::shared_ptr<RemoteChannel> channel() const { return channel_; }

    // Thread-safe. Increments job_log->active and sets initial status.
    void enqueue(const TestEntry& test,
                 const std::string& node_name,
                 std::function<void(const std::string&)> log_fn,
                 std::function<void(const std::string&)> status_fn);

    // Thread-safe. Marks a (test, node) submission as cancelled so the worker
    // skips it — leaving the same test running on other nodes untouched.
    void cancel(const std::string& name, const std::string& node);

private:
    struct Submission {
        TestEntry test;
        std::string node_name;
        std::function<void(const std::string&)> log_fn;
        std::function<void(const std::string&)> status_fn;
    };

    void worker_loop();
    void process_batch(std::vector<Submission> batch);

    // Resolve (creating on first use) the SlotTracker for a node profile's
    // scope. Jobs in different scopes (partitions / GPU types) are throttled
    // independently, so a queue on one machine doesn't block another.
    std::shared_ptr<SlotTracker> slots_for(const std::string& node_name);

    Registry                       reg_;
    std::string                    th_dir_;
    std::string                    project_root_;
    RemoteDest                     dest_;
    std::shared_ptr<JobLog>        job_log_;
    std::atomic<int>               default_max_;   // fallback per-scope limit
    std::atomic<bool>              aborting_{false}; // set on shutdown to bail fast
    std::shared_ptr<CancelSet>     cancel_set_;
    std::shared_ptr<RemoteChannel> channel_;

    // Per-scope slot trackers, keyed by partition/GPU-type. Guarded by
    // slots_mtx_ since cancel() (UI thread) and slots_for() (worker thread)
    // both touch it.
    std::mutex                                                    slots_mtx_;
    std::map<std::string, std::shared_ptr<SlotTracker>>           slots_by_scope_;

    std::mutex                     mtx_;
    std::condition_variable        cv_;
    std::deque<Submission>         queue_;
    bool                           stopped_ = false;
    std::thread                    worker_;
};

// ── SLURM limit query ─────────────────────────────────────────────────────
// Query the remote cluster for the per-user max-submit limit that applies to
// the given partition names.  Returns the smallest limit found, or `fallback`
// if the cluster has no configured limit / the query fails.
int query_slurm_job_limit(const std::string& remote,
                           const std::vector<std::string>& partitions,
                           int fallback = 4);

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
    std::function<void(const std::string&)> status_fn = nullptr,
    std::shared_ptr<RemoteChannel> channel = nullptr);

// Resume polling a previously submitted job (skips rsync and sbatch).
bool resume_job(
    const PendingJob& pending,
    const TestEntry& test,
    const Registry& reg,
    const std::string& th_dir,
    std::function<void(const std::string&)> log_fn,
    std::function<void(const std::string&)> status_fn = nullptr,
    std::shared_ptr<RemoteChannel> channel = nullptr);

} // namespace trailhead
