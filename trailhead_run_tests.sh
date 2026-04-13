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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"   # directory the user invoked the script from

NO_BUILD=0
for arg in "$@"; do
    [[ "$arg" == "--no-build" ]] && NO_BUILD=1
done

if [[ $NO_BUILD -eq 0 ]]; then
    # Detect parallelism
    NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

    echo "==> Configuring..."
    cmake -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release -S "$SCRIPT_DIR"

    echo "==> Building (${NCPU} jobs)..."
    cmake --build "$SCRIPT_DIR/build" -j"$NCPU"
fi

TRAILHEAD="$SCRIPT_DIR/build/trailhead"
if [[ ! -x "$TRAILHEAD" ]]; then
    echo "Error: $TRAILHEAD not found. Run without --no-build first." >&2
    exit 1
fi

# Run from the project directory so trailhead finds the right .trailhead/
cd "$PROJECT_DIR"
echo "==> Running tests in $PROJECT_DIR..."
exec "$TRAILHEAD" watch --run-all
