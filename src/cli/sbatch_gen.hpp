#pragma once
#include "../core/registry.hpp"
#include <string>
#include <utility>
#include <vector>

namespace trailhead {

// Group setup steps into ordered execution stages. Items within a stage may run
// concurrently; stages run one after another. Each returned item carries its
// original index among the non-barrier steps so callers can build stable
// per-item state (sentinels/locks) across runs. Rules:
//   • If explicit "---" barriers are present, split there (manual control).
//   • Otherwise, if every step is a recognised dataset-prep verb, group by
//     phase — mkdir → download → extract → move → cleanup — so independent
//     downloads/extractions run in parallel while ordering dependencies hold.
//   • If any step is unrecognised, fall back to one item per stage (fully
//     sequential), preserving authoring order.
std::vector<std::vector<std::pair<int, std::string>>>
plan_setup_stages(const std::vector<std::string>& setup);

struct SbatchOptions {
    std::string output_path;  // where to write; empty = print to stdout
    bool split = false;       // one file per test
    std::string project_root; // for resolving workdir and output paths
    std::string node_name;    // hardware target for headers (empty = no hardware headers)
};

// Generate sbatch scripts from registry.
// Returns list of (filename, content) pairs.
std::vector<std::pair<std::string,std::string>>
generate_sbatch(const Registry& reg, const SbatchOptions& opts);

// Write generated scripts to disk (inside .trailhead/)
bool write_sbatch(const std::string& trailhead_dir,
                  const Registry& reg,
                  const SbatchOptions& opts);

// Generate a single test's sbatch script for the given node.
std::string generate_test_script(const TestEntry& test,
                                  const std::string& node_name,
                                  const Registry& reg,
                                  const SbatchOptions& opts);

} // namespace trailhead
