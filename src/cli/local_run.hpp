#pragma once
#include "../core/registry.hpp"
#include "visualizer.hpp"
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>

namespace trailhead {

// Returns true if nvidia-smi is present and reports at least one GPU.
bool has_local_gpu();

// Runs tests sequentially on the local machine.
// One task runs at a time; others wait in the queue.
class LocalRunner {
public:
    LocalRunner(std::string th_dir, std::string project_root,
                std::shared_ptr<JobLog> job_log, Registry reg);
    ~LocalRunner();

    // Thread-safe. Increments job_log->active and sets initial status.
    // reg is read only to snapshot the current build config for the test.
    void enqueue(const TestEntry& test, const Registry& reg,
                 std::function<void(const std::string&)> log_fn,
                 std::function<void(const std::string&)> status_fn);

private:
    struct Task {
        TestEntry test;
        std::optional<BuildConfig> build_config; // snapshot at enqueue time
        std::function<void(const std::string&)> log_fn;
        std::function<void(const std::string&)> status_fn;
    };

    void worker_loop();
    void run_task(Task& task);

    std::string             th_dir_;
    std::string             project_root_;
    std::shared_ptr<JobLog> job_log_;

    std::mutex              mtx_;
    std::condition_variable cv_;
    std::deque<Task>        queue_;
    bool                    stopped_ = false;
    std::thread             worker_;
};

} // namespace trailhead
