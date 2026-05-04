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

_BOLD=$'\033[1m'; _CYAN=$'\033[36m'; _GREEN=$'\033[32m'; _RESET=$'\033[0m'
echo "${_BOLD}${_CYAN}TRAILHEAD${_RESET}  ${_BOLD}run-tests${_RESET}  $(git -C "$SCRIPT_DIR" describe --tags --always 2>/dev/null || true)"

# ── Self-update from GitHub ───────────────────────────────────────────────
# Try git pull; if HEAD changed, re-exec so the new script and binary are used.
_OLD_HEAD=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)
echo "==> Checking for updates..."
if ! git -C "$SCRIPT_DIR" pull --ff-only https://github.com/saltsystemslab/trailhead.git main; then
    echo "==> Could not pull updates (see above). Continuing with current version."
fi
_NEW_HEAD=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ -n "$_OLD_HEAD" && "$_OLD_HEAD" != "$_NEW_HEAD" ]]; then
    echo "${_GREEN}==> Updated to: $(git -C "$SCRIPT_DIR" log --oneline -1 2>/dev/null)${_RESET}"
    echo "==> Re-running with new version..."
    exec "$SCRIPT_DIR/trailhead_run_tests.sh" "$@"
else
    echo "==> Up to date."
fi
unset _OLD_HEAD _NEW_HEAD

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

    # Skip configure if cache exists and CMakeLists.txt hasn't changed since.
    if [[ ! -f "$SCRIPT_DIR/build/CMakeCache.txt" ]] || \
       [[ "$SCRIPT_DIR/CMakeLists.txt" -nt "$SCRIPT_DIR/build/CMakeCache.txt" ]]; then
        echo "==> Configuring trailhead..."
        cmake -B "$SCRIPT_DIR/build" -DCMAKE_BUILD_TYPE=Release -S "$SCRIPT_DIR"
    fi

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
