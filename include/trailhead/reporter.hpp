#pragma once
// trailhead/reporter.hpp — header-only test reporting plugin.
// Include this in any test file. No linking required.
//
// Usage:
//   #include "trailhead/reporter.hpp"
//
//   int main() {
//       trailhead::Reporter r("my_test");
//       r.meta("gpu", "H200");
//
//       r.timing_begin("phase1");
//       // ... work ...
//       r.timing_end("phase1");
//
//       r.check(result == expected, "correctness");
//       r.pass("extra_check");
//       // ~Reporter() writes .trailhead/results/<name>_<epoch>.json
//   }

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <iomanip>

// POSIX
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace trailhead {

// ── Internal helpers (not part of public API) ─────────────────────────────
namespace _detail {

inline std::string esc_json(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << (char)c;
        }
    }
    return o.str();
}

inline void mkdir_p(const std::string& path) {
    std::string p = path;
    for (size_t i = 1; i < p.size(); ++i) {
        if (p[i] == '/') {
            p[i] = '\0';
            ::mkdir(p.data(), 0755);
            p[i] = '/';
        }
    }
    ::mkdir(p.c_str(), 0755);
}

// Walk up from cwd looking for .trailhead/ directory
inline std::string find_results_dir() {
    const char* env = std::getenv("TRAILHEAD_RESULTS_DIR");
    if (env && *env) return std::string(env);

    char buf[4096];
    if (!::getcwd(buf, sizeof(buf))) return ".trailhead/results";
    std::string path(buf);

    for (int i = 0; i < 12; ++i) {
        std::string candidate = path + "/.trailhead";
        struct stat st;
        if (::stat(candidate.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            return candidate + "/results";
        size_t slash = path.rfind('/');
        if (slash == std::string::npos || path.size() <= 1) break;
        path = (slash == 0) ? "/" : path.substr(0, slash);
    }
    // Fallback: create in cwd
    return ".trailhead/results";
}

} // namespace _detail

// ── Reporter ──────────────────────────────────────────────────────────────

class Reporter {
public:
    // RAII scope for a named timing region
    class TimingScope {
    public:
        TimingScope(Reporter& r, std::string label) : r_(r), label_(std::move(label)) {
            r_.timing_begin(label_);
        }
        ~TimingScope() { r_.timing_end(label_); }
        TimingScope(const TimingScope&) = delete;
        TimingScope& operator=(const TimingScope&) = delete;
    private:
        Reporter& r_;
        std::string label_;
    };

    // name: must match the "name" field in registry.json for results to be associated
    explicit Reporter(std::string name)
        : name_(std::move(name))
        , started_at_ms_(wall_ms_now())
        , wall_start_(std::chrono::steady_clock::now())
    {
        char hbuf[256] = {};
        if (::gethostname(hbuf, sizeof(hbuf)) == 0) host_ = hbuf;
        const char* jid = std::getenv("TRAILHEAD_JOB_ID");
        run_by_ = jid ? (std::string("sbatch-") + jid) : "trailhead-local";
        results_dir_ = _detail::find_results_dir();
        _detail::mkdir_p(results_dir_);
    }

    ~Reporter() { flush(); }

    // ── Pass/fail accounting ───────────────────────────────────────────
    void pass(const std::string& /*label*/ = "") { ++passed_; }
    void fail(const std::string& /*label*/ = "") { ++failed_; }

    // Record pass if condition is true, fail otherwise
    void check(bool condition, const std::string& label = "") {
        if (condition) pass(label);
        else           fail(label);
    }

    // ── Timing ────────────────────────────────────────────────────────
    void timing_begin(const std::string& label) {
        for (auto& t : timings_) {
            if (t.label == label) { t.start = std::chrono::steady_clock::now(); t.running = true; return; }
        }
        timings_.push_back({label, {}, 0.0, true});
        timings_.back().start = std::chrono::steady_clock::now();
    }

    void timing_end(const std::string& label) {
        auto now = std::chrono::steady_clock::now();
        for (auto& t : timings_) {
            if (t.label == label && t.running) {
                t.elapsed_ms = std::chrono::duration<double, std::milli>(now - t.start).count();
                t.running = false;
                return;
            }
        }
    }

    // Returns a RAII scope that calls timing_begin/end automatically
    TimingScope time_scope(const std::string& label) {
        return TimingScope(*this, label);
    }

    // ── Metadata ──────────────────────────────────────────────────────
    void meta(const std::string& key, const std::string& value) {
        metadata_[key] = value;
    }

    // ── Flush ─────────────────────────────────────────────────────────
    // Called automatically by destructor. Safe to call multiple times.
    void flush() {
        if (flushed_) return;
        flushed_ = true;

        auto now = std::chrono::steady_clock::now();
        int64_t wall = std::chrono::duration_cast<std::chrono::milliseconds>(now - wall_start_).count();

        // Close any still-running timings
        for (auto& t : timings_) {
            if (t.running) {
                t.elapsed_ms = std::chrono::duration<double, std::milli>(now - t.start).count();
                t.running = false;
            }
        }

        std::string json  = serialize(wall);
        std::string epoch = std::to_string(started_at_ms_);
        std::string final_path = results_dir_ + "/" + name_ + "_" + epoch + ".json";
        std::string tmp_path   = final_path + ".tmp";

        {
            std::ofstream f(tmp_path);
            if (!f) return; // silent fail — don't crash test due to reporter error
            f << json;
        }
        ::rename(tmp_path.c_str(), final_path.c_str());
    }

    // Accessors (useful for in-process assertions)
    int passed() const { return passed_; }
    int failed() const { return failed_; }
    bool all_passed() const { return failed_ == 0; }

    Reporter(const Reporter&) = delete;
    Reporter& operator=(const Reporter&) = delete;

private:
    struct TimingEntry {
        std::string label;
        std::chrono::steady_clock::time_point start;
        double elapsed_ms = 0.0;
        bool   running    = false;
    };

    static int64_t wall_ms_now() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    std::string serialize(int64_t wall_ms) const {
        std::ostringstream o;
        auto q = [](const std::string& s) { return "\"" + _detail::esc_json(s) + "\""; };
        o << "{\n";
        o << "  \"version\": 1,\n";
        o << "  \"name\": "       << q(name_)    << ",\n";
        o << "  \"host\": "       << q(host_)    << ",\n";
        o << "  \"run_by\": "     << q(run_by_)  << ",\n";
        o << "  \"started_at\": " << started_at_ms_ << ",\n";
        o << "  \"ended_at\": "   << (started_at_ms_ + wall_ms) << ",\n";
        o << "  \"wall_ms\": "    << wall_ms  << ",\n";
        o << "  \"passed\": "     << passed_  << ",\n";
        o << "  \"failed\": "     << failed_  << ",\n";
        o << "  \"timings\": [\n";
        for (size_t i = 0; i < timings_.size(); ++i) {
            const auto& t = timings_[i];
            o << "    {\"label\": " << q(t.label)
              << ", \"elapsed_ms\": " << t.elapsed_ms << "}";
            if (i + 1 < timings_.size()) o << ",";
            o << "\n";
        }
        o << "  ],\n";
        o << "  \"metadata\": {\n";
        bool first = true;
        for (const auto& [k, v] : metadata_) {
            if (!first) o << ",\n";
            o << "    " << q(k) << ": " << q(v);
            first = false;
        }
        if (!metadata_.empty()) o << "\n";
        o << "  }\n";
        o << "}\n";
        return o.str();
    }

    std::string name_;
    std::string host_;
    std::string run_by_;
    std::string results_dir_;
    int64_t     started_at_ms_;
    std::chrono::steady_clock::time_point wall_start_;
    int  passed_  = 0;
    int  failed_  = 0;
    bool flushed_ = false;
    std::vector<TimingEntry> timings_;
    std::map<std::string, std::string> metadata_;
};

} // namespace trailhead
