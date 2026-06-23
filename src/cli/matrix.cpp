#include "matrix.hpp"
#include "../core/result_store.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace trailhead {

// Resolve a "tag:N" / "name-prefix" / "name-suffix" / "dataset" axis spec
// against a TestEntry → axis label, or empty string to skip the test.
static std::string axis_value(const std::string& spec, const TestEntry& t) {
    // merge_sub_registries prefixes test names and dataset references with
    // the sub-registry's display name (e.g. "Ocean-SpGEMM/"). Strip it so
    // sub-registry tests share rows/columns with parent tests on common ids.
    auto strip_sub_prefix = [&](std::string s) {
        if (t.sub_dir.empty()) return s;
        std::string sub_name = t.sub_dir;
        auto slash = sub_name.rfind('/');
        if (slash != std::string::npos) sub_name = sub_name.substr(slash + 1);
        std::string prefix = sub_name + "/";
        if (s.rfind(prefix, 0) == 0) s.erase(0, prefix.size());
        return s;
    };

    if (spec == "dataset") {
        return t.datasets.empty() ? "" : strip_sub_prefix(t.datasets.front());
    }
    if (spec == "name-prefix") {
        std::string n = strip_sub_prefix(t.name);
        auto p = n.find("__");
        return (p == std::string::npos) ? n : n.substr(0, p);
    }
    if (spec == "name-suffix") {
        std::string n = strip_sub_prefix(t.name);
        auto p = n.find("__");
        return (p == std::string::npos) ? "" : n.substr(p + 2);
    }
    if (spec.rfind("tag:", 0) == 0) {
        try {
            int idx = std::stoi(spec.substr(4));
            if (idx < 0 || (size_t)idx >= t.tags.size()) return "";
            return t.tags[idx];
        } catch (...) { return ""; }
    }
    return "";
}

// Render a numeric cell with sane defaults for the chosen metric.
static std::string cell_string(const std::string& metric, const TestResult* r) {
    if (!r) return "—";
    bool failed = (r->failed > 0 || r->exit_code != 0);
    if (metric == "pass") return failed ? "FAIL" : "PASS";
    if (failed)            return "FAIL";

    auto fmt = [](double v) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(3) << v;
        return o.str();
    };

    if (metric == "wall_ms")     return fmt((double)r->wall_ms);
    if (metric == "reported_ms") {
        double s = 0;
        for (const auto& te : r->timings) s += te.elapsed_ms;
        return fmt(s);
    }
    // Otherwise treat metric as a specific timing label.
    for (const auto& te : r->timings)
        if (te.label == metric) return fmt(te.elapsed_ms);
    return "—";
}

int cmd_matrix(const Registry& reg,
               const std::string& th_dir,
               const MatrixOptions& opts)
{
    ResultIndex idx = load_all_results(th_dir + "/results");

    // Build the (row, col) → cell map. Tests are skipped when either axis
    // resolves to empty (e.g. test has no dataset and row_by=="dataset").
    std::map<std::string, std::map<std::string, std::string>> cells;
    std::set<std::string> rows, cols;
    int matched = 0;
    for (const auto& t : reg.tests) {
        if (!opts.filter_tag.empty() &&
            std::find(t.tags.begin(), t.tags.end(), opts.filter_tag) == t.tags.end())
            continue;
        std::string r = axis_value(opts.row_by, t);
        std::string c = axis_value(opts.col_by, t);
        if (r.empty() || c.empty()) continue;
        const TestResult* res = latest_result(idx, t.name);
        cells[r][c] = cell_string(opts.metric, res);
        rows.insert(r);
        cols.insert(c);
        ++matched;
    }
    if (matched == 0) {
        std::cerr << "No tests matched current axis specs (row="
                  << opts.row_by << ", col=" << opts.col_by << ").\n";
        return 1;
    }

    // Choose stream
    std::ofstream file_stream;
    std::ostream* out = &std::cout;
    if (!opts.out_path.empty()) {
        file_stream.open(opts.out_path);
        if (!file_stream) {
            std::cerr << "Cannot open " << opts.out_path << " for writing\n";
            return 1;
        }
        out = &file_stream;
    }

    auto cell_at = [&](const std::string& r, const std::string& c) -> std::string {
        auto it = cells.find(r);
        if (it == cells.end()) return "—";
        auto jt = it->second.find(c);
        return jt == it->second.end() ? "—" : jt->second;
    };

    if (opts.format == "csv") {
        *out << opts.row_by;
        for (const auto& c : cols) *out << "," << c;
        *out << "\n";
        for (const auto& r : rows) {
            *out << r;
            for (const auto& c : cols) *out << "," << cell_at(r, c);
            *out << "\n";
        }
    } else if (opts.format == "md") {
        *out << "| " << opts.row_by;
        for (const auto& c : cols) *out << " | " << c;
        *out << " |\n";
        *out << "|---";
        for (size_t i = 0; i < cols.size(); ++i) *out << "|---";
        *out << "|\n";
        for (const auto& r : rows) {
            *out << "| " << r;
            for (const auto& c : cols) *out << " | " << cell_at(r, c);
            *out << " |\n";
        }
    } else {
        // table: aligned, monospace-friendly
        size_t row_w = std::max((size_t)8, opts.row_by.size());
        for (const auto& r : rows) row_w = std::max(row_w, r.size());
        std::vector<size_t> col_w;
        col_w.reserve(cols.size());
        for (const auto& c : cols) {
            size_t w = c.size();
            for (const auto& r : rows) w = std::max(w, cell_at(r, c).size());
            col_w.push_back(w + 2);
        }
        // Header
        *out << std::left << std::setw((int)row_w) << opts.row_by;
        size_t i = 0;
        for (const auto& c : cols) *out << "  " << std::setw((int)col_w[i++]) << c;
        *out << "\n";
        size_t total = row_w;
        for (size_t w : col_w) total += 2 + w;
        *out << std::string(total, '-') << "\n";
        // Rows
        for (const auto& r : rows) {
            *out << std::left << std::setw((int)row_w) << r;
            i = 0;
            for (const auto& c : cols)
                *out << "  " << std::setw((int)col_w[i++]) << cell_at(r, c);
            *out << "\n";
        }
    }

    if (out == &file_stream) {
        file_stream.close();
        std::cout << "Wrote matrix (" << rows.size() << " rows × "
                  << cols.size() << " cols) to " << opts.out_path << "\n";
    }
    return 0;
}

} // namespace trailhead
