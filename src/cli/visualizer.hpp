#pragma once
#include "../core/registry.hpp"
#include "../core/result_store.hpp"
#include <string>

namespace trailhead {

// Interactive TUI that polls .trailhead/results/ and redraws every interval_ms.
// Press 'q' to quit, arrow keys to select a test, Enter for detail view.
// Returns 0 on clean exit.
int run_watch(const std::string& trailhead_dir,
              const Registry& reg,
              int interval_ms = 1000);

// One-shot status print (non-interactive)
void print_status(const std::string& trailhead_dir,
                  const Registry& reg);

} // namespace trailhead
