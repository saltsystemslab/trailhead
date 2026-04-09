#pragma once
#include "../core/registry.hpp"
#include <string>
#include <vector>

namespace trailhead {

struct SbatchOptions {
    std::string output_path;  // where to write; empty = print to stdout
    bool split = false;       // one file per test
    std::string project_root; // for resolving workdir and output paths
};

// Generate sbatch scripts from registry.
// Returns list of (filename, content) pairs.
std::vector<std::pair<std::string,std::string>>
generate_sbatch(const Registry& reg, const SbatchOptions& opts);

// Write generated scripts to disk (inside .trailhead/)
bool write_sbatch(const std::string& trailhead_dir,
                  const Registry& reg,
                  const SbatchOptions& opts);

} // namespace trailhead
