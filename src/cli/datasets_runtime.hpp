#pragma once
#include "../core/registry.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace trailhead {

// Bash helpers (`th_ds_ensure`, `th_ds_finish`) emitted as a sourceable
// library. Same text used by every code path that runs tests so dataset
// refcounting behaves identically locally and on the remote.
std::string dataset_lib_sh();

// Write `<th_dir>/lib/datasets.sh` if absent or stale.
bool write_dataset_lib(const std::string& th_dir);

// Build per-dataset state files used by th_ds_{ensure,finish} for a run.
// `selected` is the set of test names that will execute in this invocation —
// the expected refcount for each dataset is the count of selected tests that
// list it. Returns the ordered list of datasets touched (so callers can
// sweep them at end-of-run).
//
// Layout written under th_dir + "/datasets/<name>/":
//   path           single line: ds.path
//   fetch          ds.fetch_cmd
//   cache_paths    one cache path per line (may be empty)
//   expected       integer: number of consumers in this run
//   finished.txt   truncated to empty
//   cleaned        removed if it exists (so refcount can re-trigger)
// `done` is preserved if present — the cached fetch survives across runs.
std::vector<std::string> init_dataset_state(const Registry& reg,
                                              const std::vector<std::string>& selected_test_names,
                                              const std::string& th_dir);

// Map of dataset name → expected consumer count for the given selection.
std::unordered_map<std::string, int> dataset_expected_counts(
        const Registry& reg,
        const std::vector<std::string>& selected_test_names);

// Union of `requires_targets` across the transitive closure of datasets used
// by `selected_test_names`. Caller appends these to the cmake target list so
// converters/preprocessors are built before any fetch_cmd runs.
std::vector<std::string> required_build_targets(
        const Registry& reg,
        const std::vector<std::string>& selected_test_names);

} // namespace trailhead
