#!/usr/bin/env bash
# trailhead_run_tests.sh — Build trailhead and run all registered tests locally.
#
# Usage:
#   ./ci_run.sh            # build + run all tests
#   ./ci_run.sh --no-build # skip cmake (use existing build)
#
# Exit code: 0 if all tests pass, 1 if any fail or if trailhead can't build.
# Results are written to trailhead_results.csv in this directory.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NO_BUILD=0
for arg in "$@"; do
    [[ "$arg" == "--no-build" ]] && NO_BUILD=1
done

if [[ $NO_BUILD -eq 0 ]]; then
    # Detect parallelism
    NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

    echo "==> Configuring..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release -S .

    echo "==> Building (${NCPU} jobs)..."
    cmake --build build -j"$NCPU"
fi

TRAILHEAD=./build/trailhead
if [[ ! -x "$TRAILHEAD" ]]; then
    echo "Error: $TRAILHEAD not found. Run without --no-build first." >&2
    exit 1
fi

echo "==> Running tests..."
"$TRAILHEAD" watch --run-all
