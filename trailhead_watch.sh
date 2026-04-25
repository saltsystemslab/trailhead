#!/usr/bin/env bash
# trailhead_watch.sh — Build trailhead and open the interactive TUI.
#
# Usage:
#   ./trailhead_watch.sh                  # build and open watch TUI
#   ./trailhead_watch.sh --no-build       # skip trailhead cmake (use existing build)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"   # directory the user invoked the script from

_BOLD=$'\033[1m'; _CYAN=$'\033[36m'; _GREEN=$'\033[32m'; _RESET=$'\033[0m'
echo "${_BOLD}${_CYAN}TRAILHEAD${_RESET}  ${_BOLD}watch${_RESET}  $(git -C "$SCRIPT_DIR" describe --tags --always 2>/dev/null || true)"

# ── Self-update from GitHub ───────────────────────────────────────────────
_OLD_HEAD=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)
echo "==> Checking for updates..."
if ! git -C "$SCRIPT_DIR" pull --ff-only https://github.com/saltsystemslab/trailhead.git main; then
    echo "==> Could not pull updates (see above). Continuing with current version."
fi
_NEW_HEAD=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ -n "$_OLD_HEAD" && "$_OLD_HEAD" != "$_NEW_HEAD" ]]; then
    echo "${_GREEN}==> Updated to: $(git -C "$SCRIPT_DIR" log --oneline -1 2>/dev/null)${_RESET}"
    echo "==> Re-running with new version..."
    exec "$SCRIPT_DIR/trailhead_watch.sh" "$@"
else
    echo "==> Up to date."
fi
unset _OLD_HEAD _NEW_HEAD

NO_BUILD=0
args=("$@")
for ((idx=0; idx<${#args[@]}; idx++)); do
    [[ "${args[$idx]}" == "--no-build" ]] && NO_BUILD=1
done

if [[ $NO_BUILD -eq 0 ]]; then
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

cd "$PROJECT_DIR"
exec "$TRAILHEAD" watch
