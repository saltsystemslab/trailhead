#pragma once
#include "json.hpp"
#include "../util/file_util.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <ctime>
#include <algorithm>

namespace trailhead {

struct TimingEntry {
    std::string label;
    double elapsed_ms = 0.0;
};

struct TestResult {
    std::string name;
    std::string host;
    std::string run_by;
    int64_t started_at   = 0;  // ms epoch
    int64_t ended_at     = 0;  // ms epoch
    int64_t wall_ms      = 0;
    int     exit_code    = 0;
    int     passed       = 0;
    int     failed       = 0;
    std::vector<TimingEntry> timings;
    std::unordered_map<std::string,std::string> metadata;
    std::vector<std::string> output_lines;  // verbatim text from TH_OUTPUT markers
    std::string result_file;   // absolute path to the source JSON
    time_t  file_mtime   = 0;
};

enum class RunStatus { Unknown, Pass, Fail, BuildFail, Running };

inline RunStatus result_status(const TestResult& r) {
    if (r.metadata.count("_build_fail"))   return RunStatus::BuildFail;
    if (r.failed > 0 || r.exit_code != 0) return RunStatus::Fail;
    if (r.passed > 0 || r.wall_ms > 0)    return RunStatus::Pass;
    return RunStatus::Unknown;
}

inline std::string status_str(RunStatus s) {
    switch (s) {
        case RunStatus::Pass:      return "PASS";
        case RunStatus::Fail:      return "FAIL";
        case RunStatus::BuildFail: return "BFAIL";
        case RunStatus::Running:   return "RUNNING";
        default:                   return "---";
    }
}

// Parse a result JSON file
inline std::optional<TestResult> parse_result(const std::string& path) {
    auto content = fs::read_file(path);
    if (!content) return std::nullopt;
    try {
        auto root = json_parse(*content);
        TestResult r;
        r.name        = root.get_str("name");
        r.host        = root.get_str("host");
        r.run_by      = root.get_str("run_by");
        r.started_at  = root.get_int("started_at");
        r.ended_at    = root.get_int("ended_at");
        r.wall_ms     = root.get_int("wall_ms");
        r.exit_code   = (int)root.get_int("exit_code");
        r.passed      = (int)root.get_int("passed");
        r.failed      = (int)root.get_int("failed");
        r.result_file = path;
        r.file_mtime  = fs::mtime(path);

        const JsonValue* ta = root.get("timings");
        if (ta && ta->is_array()) {
            for (const auto& te : ta->as_array()) {
                TimingEntry e;
                e.label = te.get_str("label");
                const JsonValue* em = te.get("elapsed_ms");
                if (em) e.elapsed_ms = em->is_double() ? em->as_double() : (double)em->as_int();
                r.timings.push_back(e);
            }
        }
        const JsonValue* meta = root.get("metadata");
        if (meta && meta->is_object()) {
            for (const auto& [k, v] : meta->as_object())
                if (v.is_string()) r.metadata[k] = v.as_string();
        }
        const JsonValue* out_arr = root.get("output");
        if (out_arr && out_arr->is_array()) {
            for (const auto& v : out_arr->as_array())
                if (v.is_string()) r.output_lines.push_back(v.as_string());
        }
        return r;
    } catch (...) {
        return std::nullopt;
    }
}

// ── Stdout protocol parser ────────────────────────────────────────────────
// Parses TRAILHEAD: marker lines emitted by trailhead.h from a captured
// stdout string. Returns a partial TestResult (name/host/timing left to
// caller). Non-TRAILHEAD lines are left in `remaining_stdout`.
inline void parse_trailhead_output(
    const std::string& stdout_str,
    TestResult& out,
    std::string* remaining_stdout = nullptr)
{
    std::istringstream ss(stdout_str);
    std::string line;
    std::string remaining;
    bool capturing = false;  // inside output_start..output_stop region
    while (std::getline(ss, line)) {
        // Strip trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Check for region markers first
        if (line == "TRAILHEAD:output_start") { capturing = true; continue; }
        if (line == "TRAILHEAD:output_stop")  { capturing = false; continue; }

        // Inside a capture region, grab every line verbatim
        if (capturing) {
            out.output_lines.push_back(line);
            continue;
        }

        if (line.rfind("TRAILHEAD:", 0) != 0) {
            if (remaining_stdout) { remaining += line; remaining += '\n'; }
            continue;
        }

        // Split on ':' — up to 4 parts: TRAILHEAD, verb, field1, rest
        std::vector<std::string> parts;
        size_t pos = 0;
        for (int i = 0; i < 3; ++i) {
            size_t next = line.find(':', pos);
            if (next == std::string::npos) { parts.push_back(line.substr(pos)); pos = line.size(); break; }
            parts.push_back(line.substr(pos, next - pos));
            pos = next + 1;
        }
        if (pos <= line.size()) parts.push_back(line.substr(pos)); // rest (may contain ':')

        if (parts.size() < 2) continue;
        const std::string& verb  = parts[1];
        const std::string  field = parts.size() > 2 ? parts[2] : "";
        const std::string  rest  = parts.size() > 3 ? parts[3] : "";

        if (verb == "pass") {
            out.passed++;
        } else if (verb == "fail") {
            out.failed++;
        } else if (verb == "build_fail") {
            out.metadata["_build_fail"] = "1";
            out.failed++;
        } else if (verb == "time") {
            try {
                TimingEntry te;
                te.label = field;
                te.elapsed_ms = std::stod(rest);
                out.timings.push_back(te);
            } catch (...) {}
        } else if (verb == "meta") {
            out.metadata[field] = rest;
        } else if (verb == "output") {
            std::string text = field;
            if (!rest.empty()) { text += ':'; text += rest; }
            out.output_lines.push_back(text);
        }
    }
    if (remaining_stdout) *remaining_stdout = remaining;
}

// Return the path of the full output sidecar file for a result (may not exist).
inline std::string result_output_path(const TestResult& r) {
    if (r.result_file.empty()) return "";
    auto dot = r.result_file.rfind('.');
    return (dot != std::string::npos) ? r.result_file.substr(0, dot) + ".out"
                                      : r.result_file + ".out";
}

// ── Save ──────────────────────────────────────────────────────────────────

// Save the full stdout/stderr output alongside the result JSON as a .out sidecar file.
inline void save_result_output(const std::string& results_dir,
                                const TestResult& res,
                                const std::string& output) {
    if (output.empty()) return;
    std::string path = results_dir + "/" + res.name + "_"
                     + std::to_string(res.started_at) + ".out";
    auto slash = path.rfind('/');
    if (slash != std::string::npos) fs::mkdir_p(path.substr(0, slash));
    fs::write_file_atomic(path, output);
}

inline void save_result(const std::string& results_dir, const TestResult& res) {
    std::string path = results_dir + "/" + res.name + "_"
                     + std::to_string(res.started_at) + ".json";
    auto slash = path.rfind('/');
    if (slash != std::string::npos) fs::mkdir_p(path.substr(0, slash));
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

// ── Index ─────────────────────────────────────────────────────────────────

// Index: test name → list of results sorted oldest→newest
using ResultIndex = std::unordered_map<std::string, std::vector<TestResult>>;

// Load all results from .trailhead/results/
inline ResultIndex load_all_results(const std::string& results_dir) {
    ResultIndex idx;
    for (const auto& path : fs::list_files_recursive(results_dir, ".json")) {
        auto r = parse_result(path);
        if (!r) continue;
        idx[r->name].push_back(*r);
    }
    for (auto& [name, vec] : idx) {
        std::sort(vec.begin(), vec.end(),
            [](const TestResult& a, const TestResult& b) { return a.started_at < b.started_at; });
    }
    return idx;
}

// Incremental refresh: only parse files newer than last scan.
// Keeps a static cache of known files and their mtimes.
inline void refresh_results(const std::string& results_dir, ResultIndex& idx) {
    static ::std::unordered_map<::std::string, time_t> seen;
    for (const auto& path : fs::list_files_recursive(results_dir, ".json")) {
        time_t mt = fs::mtime(path);
        auto it = seen.find(path);
        if (it != seen.end() && it->second == mt) continue;
        seen[path] = mt;
        auto r = parse_result(path);
        if (!r) continue;
        auto& vec = idx[r->name];
        vec.erase(::std::remove_if(vec.begin(), vec.end(),
            [&](const TestResult& t) { return t.started_at == r->started_at; }), vec.end());
        vec.push_back(*r);
        ::std::sort(vec.begin(), vec.end(),
            [](const TestResult& a, const TestResult& b) { return a.started_at < b.started_at; });
    }
}

// Get the most recent result for a test name
inline const TestResult* latest_result(const ResultIndex& idx, const std::string& name) {
    auto it = idx.find(name);
    if (it == idx.end() || it->second.empty()) return nullptr;
    return &it->second.back();
}

} // namespace trailhead
