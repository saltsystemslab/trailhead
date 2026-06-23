#pragma once
#include "../core/registry.hpp"
#include "remote_run.hpp"
#include <string>
#include <vector>

namespace trailhead {

struct BatchRunOptions {
    std::string node_name;            // required: SLURM node profile
    int  batch_size      = 50;        // tests per SLURM job
    int  max_concurrent  = 0;         // 0 = auto-detect via query_slurm_job_limit
    bool no_build        = false;     // skip configure + per-target rebuild
    bool run_all         = false;
    std::string filter_tag;
    std::vector<std::string> test_names;
};

// Pack filtered tests into chunks of `batch_size` and submit each chunk as a
// single SLURM job. Each test inside a chunk is wrapped so its non-zero exit
// does not abort the rest of the chunk; per-test stdout is captured separately
// and parsed into TestResult JSON locally after the chunk completes.
//
// Returns 0 if every selected test passed, 1 otherwise (any failure or
// submission error). Blocks until all chunks finish.
int batch_run(const Registry& reg,
              const std::string& th_dir,
              const std::string& project_root,
              const RemoteDest& dest,
              const BatchRunOptions& opts);

} // namespace trailhead
