# trailhead

A lightweight CLI for scheduling, running, and reporting tests on HPC/SLURM clusters. Zero external dependencies — pure C++17 + POSIX.

The `.trailhead/` directory travels with your project. Run `trailhead init` once, then register tests and hardware profiles. From there, `trailhead watch` gives you a live TUI to submit and monitor jobs locally or on remote nodes.

---

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# optional: install to PATH
cmake --install build
```

Requires CMake 3.16+ and a C++17 compiler.

---

## Quick start

```bash
# 1. Initialize in your project root
trailhead init

# 2. Register a SLURM node profile
trailhead node add --name h200 --partition gpu-interactive --gpu-type h200 \
                   --cpus 8 --time 02:00:00

# 3. Open the live TUI — press [a] to add a test, [s] to submit
trailhead watch
```

---

## Concepts

| Concept | What it is |
|---|---|
| **NodeProfile** | A SLURM node description: partition, GPU target, CPUs, time limit |
| **TestEntry** | A generic shell command. Hardware is chosen at submit time, not at definition time. |
| **BuildConfig** | An optional cmake build step that runs before linked tests |
| **Results** | JSON files written to `.trailhead/results/` — parsed by `watch` and `show` |

---

## Node profiles

Node profiles describe SLURM submission targets. Three hardware modes:

```bash
# GPU model — requests any node with that GPU type
trailhead node add --name h200 --partition gpu-interactive --gpu-type h200

# Specific node — pins to an exact node, allocates 1 GPU
trailhead node add --name rtx6000 --partition SaltSystemsLab --nodelist d4067

# CPU-only — no --gres flag, pure CPU allocation
trailhead node add --name cpu --partition normal --cpus 32 --time 04:00:00

trailhead node list
trailhead node remove <name>
```

You can also **add, edit, and delete node profiles interactively** from within `trailhead watch` by pressing `[h]` to open the hardware picker, then `[a]` to add, `[e]` to edit, or `[x]` to delete.

---

## Managing tests

### From the TUI (recommended)

In `trailhead watch`, press `[a]` to add a test:

1. Enter a task name
2. Your `$EDITOR` (or vim) opens — write the commands to run, one per line
3. Optionally link a build config (ensures cmake runs before the test)
4. The test is saved

Tests are **hardware-agnostic** — the same test can be submitted to any node. Select the target hardware with `[h]` and submit with `[s]`.

Press `[e]` to edit an existing test, `[d]` to delete.

### From the CLI

```bash
trailhead add --name my_test --cmd "./build/my_test --verbose"
trailhead remove my_test
trailhead list
```

---

## Build configs

Build configs let you define a cmake build that runs before tests. Useful for keeping remote nodes' build directories in sync.

```bash
trailhead build add --name release \
    --dir build \
    --configure "cmake -B build -DCMAKE_BUILD_TYPE=Release" \
    --build "cmake --build build -j8" \
    --rsync-dest user@cluster:/scratch/myproject

trailhead build list
trailhead build run release   # manually trigger configure+build
```

The `--rsync-dest` field is also how `trailhead watch` knows to use SLURM submission — any build config with a remote destination enables remote mode.

Per-node build directories (for GPU arch auto-detection):

```bash
trailhead node add --name h200 --partition gpu-interactive --gpu-type h200 \
                   --build-dir build_h200
```

When a node has a `build_dir`, new tests added via the TUI will automatically run from that directory.

---

## Running tests

### Live TUI

```bash
trailhead watch
```

On startup you select a hardware target. From there:

| Key | Action |
|---|---|
| `s` | Submit selected test |
| `R` | Submit all visible tests |
| `r` | Refresh results from disk |
| `enter` | Open detail view (live output, pipeline, metadata) |
| `a` | Add new test |
| `e` | Edit selected test |
| `d` | Delete selected test |
| `h` | Change hardware / add, edit, or delete node profiles |
| `j` / `k` / `↑` / `↓` | Navigate |
| `q` | Quit |

**Jobs persist across restarts.** Submitted SLURM jobs resume polling when you reopen `trailhead watch`. Locally queued jobs are re-enqueued.

### Local batch run

```bash
trailhead run --all              # run all tests locally
trailhead run my_test            # run specific test
trailhead run --tag perf         # run tests with a tag
trailhead run --all --no-build   # skip cmake build step
```

### CI / automated

```bash
./trailhead_run_tests.sh         # build trailhead + run all tests, writes trailhead_results.csv
./trailhead_run_tests.sh --no-build
```

Or use the `--run-all` flag directly:

```bash
trailhead watch --run-all   # non-interactive: submit all, exit when done, write CSV
```

Exit code is 0 if all tests passed, 1 if any failed.

---

## Instrumenting tests

Include `trailhead/trailhead.h` (C or C++) or `trailhead/reporter.hpp` (C++ only) in your test binary. No linking required.

### trailhead.h — macro API (C and C++)

```c
#include "trailhead/trailhead.h"

int main() {
    TH_META("gpu", "H200");

    TH_TIME_BEGIN(construction);
    build_data_structure();
    TH_TIME_END(construction);

    TH_CHECK(result == expected, "correctness");
    TH_PASS("extra_check");
}
```

Macros emit `TRAILHEAD:` lines to stdout; the CLI strips and parses them automatically.

| Macro | Effect |
|---|---|
| `TH_PASS(label)` | Record a pass |
| `TH_FAIL(label)` | Record a fail |
| `TH_CHECK(cond, label)` | Pass if cond true, fail otherwise |
| `TH_META(key, value)` | Attach metadata to the result |
| `TH_TIME_BEGIN(label)` | Start a timing region |
| `TH_TIME_END(label)` | End timing, emit elapsed ms |
| `TH_SCOPE("label")` | C++ RAII timing scope |
| `TH_TIME_EMIT(label, ms)` | Emit a pre-computed duration |

### reporter.hpp — class API (C++ only)

```cpp
#include "trailhead/reporter.hpp"

int main() {
    trailhead::Reporter r("my_test");  // name must match registry
    r.meta("data_size", "1000000");

    {
        auto s = r.time_scope("sort");
        std::sort(v.begin(), v.end());
    }

    r.check(v.front() == 0, "sorted");
    // ~Reporter() writes .trailhead/results/my_test_<epoch>.json
}
```

`reporter.hpp` writes a JSON result file directly; `trailhead.h` emits stdout markers that the CLI parses. Use `reporter.hpp` when you want authoritative JSON results even outside of trailhead's process supervision.

---

## Viewing results

```bash
trailhead list            # table of all tests with latest status
trailhead show            # detailed view of all latest results
trailhead show my_test    # detailed view of one test
trailhead clean --days 7  # remove result files older than 7 days
```

---

## .trailhead/ layout

```
.trailhead/
  registry.json      # all config: nodes, builds, tests
  results/           # one JSON file per test run
  sbatch/            # generated sbatch scripts (auto-regenerated)
  pending/           # in-flight SLURM jobs (resumed on watch restart)
  queued/            # jobs waiting to be submitted (resumed on watch restart)
```

`registry.json` is the only file you'd ever edit manually. Everything else is managed by the CLI.

---

## Project setup steps

One-time commands (submodule init, dataset downloads, etc.) can be registered as setup steps. They run before cmake configure in every generated sbatch script, and can be run locally with `trailhead setup run`.

```bash
trailhead setup add "git submodule update --init --recursive"
trailhead setup add "python3 scripts/download_data.py"
trailhead setup list          # show all steps with indices
trailhead setup remove 1      # remove step at index 1
trailhead setup run           # run all steps locally
```

Setup steps are stored in `registry.json` and emitted into sbatch scripts in order, after `cd` to the project root and before the cmake configure step.

## Generating sbatch scripts manually

Since tests are hardware-agnostic, use `--node` to target a specific profile:

```bash
trailhead gen --node h200            # combined script for all tests on h200
trailhead gen --node h200 --split    # one script per test → .trailhead/sbatch/
trailhead gen --node rtx6000 --out ./jobs
```

When submitting via the TUI (`[s]`), the sbatch script is generated on the fly using the currently selected hardware — no pre-generation needed.
