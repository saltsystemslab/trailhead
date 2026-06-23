#pragma once
#include "../core/registry.hpp"
#include <string>

namespace trailhead {

struct MatrixOptions {
    std::string metric  = "reported_ms"; // "pass" | "wall_ms" | "reported_ms" | <label>
    std::string format  = "table";       // "table" | "csv" | "md"
    std::string row_by  = "dataset";     // "dataset" | "tag:N" | "name-suffix"
    std::string col_by  = "tag:0";       // "tag:N" | "name-prefix"
    std::string out_path;                // empty = stdout
    std::string filter_tag;              // optional --tag <t>
};

// Build and emit a row × col matrix where each cell is the latest result's
// metric for the test that links that row to that column. Returns 0 on
// success, 1 on usage error (no matching tests).
int cmd_matrix(const Registry& reg,
               const std::string& th_dir,
               const MatrixOptions& opts);

} // namespace trailhead
