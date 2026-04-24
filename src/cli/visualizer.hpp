#pragma once
#include "../core/registry.hpp"
#include "../core/result_store.hpp"
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <deque>
#include <vector>
#include <map>
#include <unordered_map>
#include <atomic>

namespace trailhead {

// Thread-safe log for background job output.
// Background threads push lines; the TUI render loop snapshots them.
struct JobLog {
    mutable std::mutex mtx;
    std::deque<std::string> lines;
    std::atomic<int> active{0};
    std::map<std::string, std::string> live; // test name → in-flight status
    std::unordered_map<std::string, std::vector<std::string>> live_output; // test name → recent lines
    static constexpr int MAX_LINES = 5;
    static constexpr int MAX_LIVE_OUTPUT = 500; // rolling window per test

    void push(const std::string& line) {
        std::lock_guard<std::mutex> g(mtx);
        lines.push_back(line);
        while ((int)lines.size() > MAX_LINES) lines.pop_front();
    }

    void set_live(const std::string& name, const std::string& status) {
        std::lock_guard<std::mutex> g(mtx);
        if (status.empty()) live.erase(name);
        else live[name] = status;
    }

    // Append a line to the per-test live output buffer (called from log_fn during run).
    void push_live_output(const std::string& name, const std::string& line) {
        std::lock_guard<std::mutex> g(mtx);
        auto& buf = live_output[name];
        buf.push_back(line);
        if ((int)buf.size() > MAX_LIVE_OUTPUT)
            buf.erase(buf.begin());
    }

    // Clear the per-test live output (called before starting a new run).
    void clear_live_output(const std::string& name) {
        std::lock_guard<std::mutex> g(mtx);
        live_output.erase(name);
    }

    std::vector<std::string> get_live_output(const std::string& name) const {
        std::lock_guard<std::mutex> g(mtx);
        auto it = live_output.find(name);
        return it != live_output.end() ? it->second : std::vector<std::string>{};
    }

    std::string get_live(const std::string& name) const {
        std::lock_guard<std::mutex> g(mtx);
        auto it = live.find(name);
        return it != live.end() ? it->second : "";
    }

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> g(mtx);
        return {lines.begin(), lines.end()};
    }
};

// Interactive TUI that polls .trailhead/results/ and redraws every interval_ms.
// run_fn: called (non-blocking) when user presses 's'. Should spawn a background
//         thread and return immediately; log output via job_log.
// job_log: shared log rendered in the footer. May be null.
// auto_run: skip hardware picker (forces "local"), submits all tests immediately on
//           startup, auto-exits when all finish, writes trailhead_results.csv.
// Returns 0 if all tests passed, 1 if any failed (in auto_run mode); 0 otherwise.
int run_watch(const std::string& trailhead_dir,
              Registry& reg,
              int interval_ms = 1000,
              std::shared_ptr<JobLog> job_log = nullptr,
              std::function<void(const std::string&, const std::string&)> run_fn = nullptr,
              std::string project_root = "",
              bool auto_run = false,
              int repeat = 1);

// One-shot status print (non-interactive)
void print_status(const std::string& trailhead_dir,
                  const Registry& reg);

} // namespace trailhead
