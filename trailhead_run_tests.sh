#!/usr/bin/env bash
# trailhead_run_tests.sh — Build trailhead and run all registered tests locally.
#
# Usage:
#   ./trailhead_run_tests.sh                  # build, setup, run all tests 5x, report median
#   ./trailhead_run_tests.sh --wipe           # wipe build directories before running
#   ./trailhead_run_tests.sh --no-build       # skip trailhead cmake (use existing build)
#   ./trailhead_run_tests.sh --no-setup       # skip 'trailhead setup run'
#   ./trailhead_run_tests.sh --repeat <N>     # run each test N times (default 5)
#
# Exit code: 0 if all tests pass, 1 if any fail or if trailhead can't build.
# Results are written to .trailhead/results/ and trailhead_results.csv.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"   # directory the user invoked the script from

# ── Self-update from GitHub ───────────────────────────────────────────────
_SELF="${BASH_SOURCE[0]}"
_UPDATE_URL="https://raw.githubusercontent.com/saltsystemslab/trailhead/main/trailhead_run_tests.sh"
if command -v curl &>/dev/null; then
    _TMP=$(mktemp)
    if curl -sf --max-time 5 -o "$_TMP" "$_UPDATE_URL" && [[ -s "$_TMP" ]]; then
        if ! cmp -s "$_SELF" "$_TMP"; then
            echo "==> Updating trailhead_run_tests.sh from GitHub..."
            chmod +x "$_TMP"
            mv "$_TMP" "$_SELF"
            exec "$_SELF" "$@"
        fi
    fi
    rm -f "$_TMP"
fi

NO_BUILD=0
NO_SETUP=0
WIPE=0
REPEAT=5
args=("$@")
for ((idx=0; idx<${#args[@]}; idx++)); do
    [[ "${args[$idx]}" == "--no-build" ]] && NO_BUILD=1
    [[ "${args[$idx]}" == "--no-setup" ]] && NO_SETUP=1
    [[ "${args[$idx]}" == "--wipe"     ]] && WIPE=1
    if [[ "${args[$idx]}" == "--repeat" ]]; then
        REPEAT="${args[$((idx+1))]}"
        idx=$((idx+1))
    fi
done

if [[ $NO_BUILD -eq 0 ]]; then
    # Detect parallelism
    NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

    echo "==> Configuring trailhead..."
    cmake -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release -S "$SCRIPT_DIR"

    echo "==> Building trailhead (${NCPU} jobs)..."
    cmake --build "$SCRIPT_DIR/build" -j"$NCPU"
fi

TRAILHEAD="$SCRIPT_DIR/build/trailhead"
if [[ ! -x "$TRAILHEAD" ]]; then
    echo "Error: $TRAILHEAD not found. Run without --no-build first." >&2
    exit 1
fi

# Run from the project directory so trailhead finds the right .trailhead/
cd "$PROJECT_DIR"

if [[ $NO_SETUP -eq 0 ]]; then
    echo "==> Running project setup steps..."
    "$TRAILHEAD" setup run || true   # setup run prints its own errors; don't abort if no steps
fi

WIPE_FLAG=""
[[ $WIPE -eq 1 ]] && WIPE_FLAG="--wipe"

echo "==> Running tests in $PROJECT_DIR (${REPEAT}x each)..."
exec "$TRAILHEAD" watch --run-all $WIPE_FLAG --repeat "$REPEAT"
