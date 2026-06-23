#include "batch_run.hpp"
#include "datasets_runtime.hpp"
#include "../core/result_store.hpp"
#include "../util/file_util.hpp"
#include "../util/process.hpp"
#include "../util/ansi.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace trailhead {

// Terminal size (for the live grid); falls back if not a TTY / query fails.
static int bt_term_cols() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return (int)ws.ws_col;
    return 80;
}
static int bt_term_rows() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return (int)ws.ws_row;
    return 24;
}

// Internal counting semaphore used to throttle chunk submissions. Pure local
// state — does NOT query the real SLURM queue, so unrelated jobs (interactive
// shells, leftovers from a prior run) don't block us. Each chunk thread does:
//   1. acquire()           — wait for an in-flight slot
//   2. sbatch + poll
//   3. release()           — once the SLURM job has actually exited
// SLURM's own QOS / max-submit limits are enforced server-side; our sbatch
// retry logic backs off and tries again if SLURM rejects a submission.
class ChunkSemaphore {
public:
    explicit ChunkSemaphore(int n) : avail_(n) {}
    // Returns false if `abort` fired before a slot was acquired.
    bool acquire(const std::atomic<bool>& abort) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&]{ return avail_ > 0 || abort.load(); });
        if (abort.load()) return false;
        --avail_;
        return true;
    }
    void release() {
        { std::lock_guard<std::mutex> lk(mtx_); ++avail_; }
        cv_.notify_all();
    }
    // Wake every waiter so they re-check the abort flag.
    void wake() { cv_.notify_all(); }
private:
    std::mutex              mtx_;
    std::condition_variable cv_;
    int                     avail_;
};

// ── Small string helpers (private to this TU) ─────────────────────────────

static std::string last_component(const std::string& p) {
    auto sl = p.rfind('/');
    return (sl != std::string::npos && sl + 1 < p.size()) ? p.substr(sl + 1) : p;
}

// Flatten a name into a single safe filename component ('/' → '-').
static std::string safe_name(std::string s) {
    for (auto& c : s) if (c == '/') c = '-';
    return s;
}

// Run an rsync command with retries (transient SSH/network drops shouldn't lose
// a chunk's results). Returns true on eventual success.
static bool rsync_with_retry(const std::string& cmd, int tries = 5) {
    proc::RunResult r;
    for (int attempt = 1; attempt <= tries; ++attempt) {
        r = proc::run(cmd, {}, {}, 300, "", nullptr, true);
        if (r.exit_code == 0) return true;
        if (attempt < tries)
            std::this_thread::sleep_for(std::chrono::seconds(3 * attempt));
    }
    return false;
}

static std::string str_replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Single-quote a string for embedding inside a bash command.
static std::string bash_sq(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// ── #SBATCH header emission ───────────────────────────────────────────────

static std::string sbatch_headers(const NodeProfile& node,
                                   const SbatchDefaults& defs,
                                   const std::string& job_name)
{
    std::ostringstream o;
    auto line = [&](const std::string& k, const std::string& v) {
        if (!v.empty()) o << "#SBATCH --" << k << "=" << v << "\n";
    };
    auto line_int = [&](const std::string& k, int v, int skip = 0) {
        if (v != skip) o << "#SBATCH --" << k << "=" << v << "\n";
    };
    o << "#SBATCH --job-name=" << job_name << "\n";
    line("partition", node.partition);
    if (!node.gpu_type.empty()) {
        o << "#SBATCH --gres=gpu:" << node.gpu_type << "\n";
    } else if (!node.nodelist.empty()) {
        line("nodelist", node.nodelist);
        o << "#SBATCH --gres=gpu:1\n";
    }
    line_int("nodes",         node.nodes,         0);
    line_int("ntasks",        node.ntasks,        0);
    line_int("cpus-per-task", node.cpus_per_task, 0);
    // Always emit --time. If the profile has none, default to 1h rather than
    // letting the job silently inherit the QOS DefaultTime (often just minutes).
    o << "#SBATCH --time=" << (node.time.empty() ? "01:00:00" : node.time) << "\n";
    if (!node.account.empty()) line("account", node.account);
    line("output", defs.output_pattern);
    line("error",  defs.error_pattern);
    for (const auto& [k, v] : node.extra) line(k, v);
    return o.str();
}

// ── Chunk-script construction ────────────────────────────────────────────

struct ChunkScriptInputs {
    std::vector<const TestEntry*> tests;
    const NodeProfile* node = nullptr;
    const Registry*    reg  = nullptr;
    std::string        node_name;
    std::string        remote_project;     // absolute path on remote (where rsync landed)
    std::string        chunk_label;        // e.g. "001"
    std::string        sub_dir;            // empty for parent; e.g. "Ocean-SpGEMM"
    std::string        build_dir;          // e.g. "build_H200"  (the bare dir name)
    std::string        configure_cmd;      // {{arch}} already substituted
    bool               no_build = false;
    std::vector<std::string> unique_targets;
};

// Build a chunk's sbatch script. Each test is wrapped in a per-test capture
// function so its non-zero exit doesn't abort the rest of the chunk; per-test
// stdout goes to .trailhead/batch_results/<name>__<ts>.out + a .meta sidecar.
static std::string build_chunk_script(const ChunkScriptInputs& in) {
    std::ostringstream s;
    s << "#!/bin/bash\n";
    const auto& defs = in.reg->sbatch_defaults;
    std::string job_name = defs.job_name_prefix + "-batch-" + in.chunk_label;
    s << sbatch_headers(*in.node, defs, job_name);
    s << "\n";

    // Per-submodule environment. A sub-registry chunk gets the parent project's
    // shared base (modules/preamble — i.e. the build tooling: cmake, compiler,
    // CUDA, …) AND, LAYERED on top, the sub-registry's own preamble/setup (e.g.
    // its python env / conda activate). Layering rather than replacing means a
    // submodule keeps the shared build environment while still activating its
    // own — so competing envs don't clash, but builds don't lose `module load`s.
    const SbatchDefaults* sub_defs  = nullptr;
    const std::vector<std::string>* sub_setup = nullptr;
    if (!in.sub_dir.empty()) {
        auto dit = in.reg->sub_sbatch_defaults.find(in.sub_dir);
        if (dit != in.reg->sub_sbatch_defaults.end()) sub_defs = &dit->second;
        auto sit = in.reg->sub_setups.find(in.sub_dir);
        if (sit != in.reg->sub_setups.end() && !sit->second.empty()) sub_setup = &sit->second;
    }

    // Base environment: node preamble takes precedence over the parent project's
    // modules+preamble (mirrors sbatch_gen.cpp:sbatch_body behaviour).
    if (!in.node->preamble.empty()) {
        for (const auto& l : in.node->preamble) s << l << "\n";
    } else {
        for (const auto& m : defs.modules)  s << "module load " << m << "\n";
        for (const auto& l : defs.preamble) s << l << "\n";
    }
    // Submodule layer on top of the shared base.
    if (sub_defs) {
        for (const auto& m : sub_defs->modules)  s << "module load " << m << "\n";
        for (const auto& l : sub_defs->preamble) s << l << "\n";
    }
    s << "\n";

    // Per-test captures go under a NODE-scoped subdir so two batch-run instances
    // sharing this remote workspace (e.g. one on h200, one on rtx6000) don't
    // write to the same files — otherwise ingest, which matches metas by test
    // name, would pull the other node's results and conflate them.
    std::string snode = safe_name(in.node_name);
    s << "export TRAILHEAD_JOB_ID=$SLURM_JOB_ID\n";
    s << "export TRAILHEAD_ENABLED=1\n";
    s << "export TRAILHEAD_BUILD_DIR=" << in.build_dir << "\n";
    s << "cd " << in.remote_project << "\n";
    // Anchor all capture paths to an ABSOLUTE root captured here. A setup/build/
    // test step that leaves the cwd changed would otherwise make the relative
    // ".trailhead/..." capture paths resolve in the wrong place, so the per-test
    // .out/.err/.meta writes silently fail ("No such file") and every test in the
    // chunk is falsely reported as "did not run".
    s << "export TRAILHEAD_ROOT=\"$(pwd)\"\n";
    s << "mkdir -p \"$TRAILHEAD_ROOT/.trailhead\" \"$TRAILHEAD_ROOT/.trailhead/batch_results/" << snode << "\"\n";

    // Resolve the qualified build dir (parent vs sub-registry) once. Both
    // configure and per-target build steps run "from" this path; tests cd
    // into it via $wd. For parent tests sub_dir is empty so qbd == build_dir.
    std::string qbd = in.sub_dir.empty() ? in.build_dir
                                          : in.sub_dir + "/" + in.build_dir;

    // Source dataset helpers if any selected test in this chunk uses datasets.
    // The library + per-dataset state are populated locally before rsync.
    bool any_datasets = false;
    for (const auto* t : in.tests)
        if (!t->datasets.empty()) { any_datasets = true; break; }
    if (any_datasets)
        s << "[ -f .trailhead/lib/datasets.sh ] && source .trailhead/lib/datasets.sh\n";
    s << "\n";

    bool has_parent_setup = !in.reg->setup.empty();
    bool has_sub_setup    = (sub_setup != nullptr);
    if (has_parent_setup || has_sub_setup) {
        // Per-item locking so chunks landing on different compute nodes can
        // contribute different items in parallel without re-running anything.
        // Sentinels/locks are scoped (1st arg) — "root" for the shared parent
        // setup, the submodule name for a sub-registry's own — so two submodules
        // with independent setups don't collide on setup_0.
        //
        // For each item i: subshell opens the lock and tries flock -n. If it
        // acquires and the sentinel is absent → run the item, touch sentinel,
        // release. If it can't acquire, another node owns the lock → skip
        // without blocking. th_setup_stage (below) re-attempts each pass, so a
        // skipped item is retried until its owner finishes — or, if that owner
        // died, the freed lock lets a survivor take over.
        s << "th_setup_item() {\n";
        s << "  local scope=\"$1\" idx=\"$2\" cmd=\"$3\"\n";
        s << "  local done_f=\".trailhead/setup_${scope}_${idx}.done\"\n";
        s << "  local lock_f=\".trailhead/setup_${scope}_${idx}.lock\"\n";
        s << "  if [ -f \"$done_f\" ]; then return 0; fi\n";
        s << "  ( exec 200>\"$lock_f\"\n";
        s << "    if flock -n 200; then\n";
        s << "      if [ ! -f \"$done_f\" ]; then\n";
        s << "        echo \"[trailhead] setup[${scope}/${idx}]: ${cmd}\"\n";
        // Always mark the item done once it has been attempted, even if the
        // command exits non-zero. Many setup steps are non-idempotent across
        // re-runs (e.g. `mkdir datasets` errors when the dir already exists);
        // gating the sentinel on success would leave it un-touched and make a
        // later th_setup_wait poll silently until the SLURM job is killed. We
        // log a warning instead so a real failure still surfaces, mirroring the
        // single-test runner which runs setup best-effort and continues.
        s << "        bash -c \"$cmd\"; rc=$?\n";
        s << "        if [ \"$rc\" -ne 0 ]; then echo \"[trailhead] setup[${scope}/${idx}] exited ${rc} (continuing)\"; fi\n";
        s << "        touch \"$done_f\"\n";
        s << "      fi\n";
        s << "      flock -u 200\n";
        s << "    fi\n";
        s << "  )\n";
        s << "}\n";
        // Run a stage to completion: each pass re-attempts every item, then
        // checks sentinels. th_setup_item is idempotent (sentinel + flock guard),
        // so a done item is a no-op and a live owner's item is skipped — but if
        // the owner DIED (killed at its time limit) it freed the lock without
        // touching the sentinel, and the next pass re-acquires and runs it here.
        // That breaks the cascade where one chunk's death stalls all the others
        // until they hit their own time limits. Args: scope timeout idx cmd ...
        s << "th_setup_stage() {\n";
        s << "  local scope=\"$1\" tmo=\"$2\"; shift 2\n";
        s << "  local pairs=(\"$@\") start; start=$(date +%s)\n";
        s << "  while true; do\n";
        s << "    local i=0\n";
        s << "    while [ $i -lt ${#pairs[@]} ]; do\n";
        s << "      th_setup_item \"$scope\" \"${pairs[$i]}\" \"${pairs[$((i+1))]}\" &\n";
        s << "      i=$((i+2))\n";
        s << "    done\n";
        s << "    wait\n";
        s << "    local missing=0 j=0\n";
        s << "    while [ $j -lt ${#pairs[@]} ]; do\n";
        s << "      [ -f \".trailhead/setup_${scope}_${pairs[$j]}.done\" ] || { missing=1; break; }\n";
        s << "      j=$((j+2))\n";
        s << "    done\n";
        s << "    [ \"$missing\" = 0 ] && return 0\n";
        s << "    if [ $(( $(date +%s) - start )) -gt \"$tmo\" ]; then\n";
        s << "      echo \"TRAILHEAD: setup timeout (scope=$scope)\"; return 1\n";
        s << "    fi\n";
        s << "    sleep 2\n";
        s << "  done\n";
        s << "}\n\n";

        // Emit one setup group (scope-tagged), grouped into stages via the shared
        // planner so batch-run matches single-test runs: explicit "---" barriers
        // split manually, otherwise dataset-prep verbs are phased (downloads in
        // parallel, then extracts, then moves) and unknown commands stay
        // sequential. `cd_prefix` runs a submodule's setup from its own dir.
        auto emit_setup = [&](const std::vector<std::string>& items,
                              const std::string& scope, const std::string& cd_prefix) {
            for (const auto& stage : plan_setup_stages(items)) {
                if (stage.empty()) continue;
                s << "th_setup_stage " << scope << " \"${TRAILHEAD_SETUP_TIMEOUT_SEC:-1800}\"";
                for (const auto& [i, cmd] : stage)
                    s << " " << i << " " << bash_sq(cd_prefix + cmd);
                s << " || { echo \"TRAILHEAD:build_fail\"; exit 1; }\n";
            }
        };

        // Shared parent setup first (so submodule setup can rely on it), then the
        // submodule's own setup from its directory.
        if (has_parent_setup) emit_setup(in.reg->setup, "root", "");
        if (has_sub_setup) {
            std::string sub_scope = in.sub_dir;
            for (auto& c : sub_scope) if (c == '/') c = '_';
            emit_setup(*sub_setup, sub_scope, "cd " + in.sub_dir + " && ");
        }
        s << "\n";
    }

    if (!in.no_build) {
        // Build barrier: chunks may land on different compute nodes that share
        // the rsync_dest filesystem. flock serialises configure + per-target
        // rebuilds so two chunks don't race on the same build tree. Lock
        // filename includes sub_dir so a sub-registry's build doesn't block
        // a parallel parent build.
        std::string lock_key = in.sub_dir.empty()
            ? in.build_dir
            : in.sub_dir + "_" + in.build_dir;
        // sanitise — sub_dir may contain '/'
        for (auto& c : lock_key) if (c == '/') c = '_';
        s << "exec 200>\".trailhead/build_" << lock_key << ".lock\"\n";
        // Bounded wait so a stale lock (a prior holder killed at the time limit,
        // or flock-over-NFS not releasing) can't make this chunk block forever
        // and hit its own time limit. Proceed on timeout — configure is guarded
        // and the build is incremental.
        s << "flock -w \"${TRAILHEAD_BUILD_LOCK_WAIT:-600}\" 200 "
             "|| echo \"[trailhead] build lock wait timed out — proceeding\"\n";
        bool have_cfg = !in.configure_cmd.empty();
        if (have_cfg) {
            std::string cfg = str_replace_all(in.configure_cmd, "-B build", "-B " + in.build_dir);
            bool from_build = cfg.size() >= 3 && cfg.substr(cfg.size() - 3) == " ..";
            // The configure invocation, wrapped so it can be re-run on demand:
            // a target that's missing from a stale Makefile (added/renamed since
            // the build dir was last configured) needs the cache regenerated
            // before `make <target>` can find a rule for it.
            std::string cfg_invoke;
            if (from_build) {
                cfg_invoke = in.sub_dir.empty()
                    ? "mkdir -p " + in.build_dir + " && ( cd " + in.build_dir + " && " + cfg + " )"
                    : "( cd " + in.sub_dir + " && mkdir -p " + in.build_dir
                      + " && ( cd " + in.build_dir + " && " + cfg + " ) )";
            } else {
                cfg_invoke = in.sub_dir.empty() ? cfg
                                                : "( cd " + in.sub_dir + " && " + cfg + " )";
            }
            s << "th_configure() { " << cfg_invoke << "; }\n";
            // First-time configure only when there's no build system yet.
            s << "if [ ! -f " << qbd << "/Makefile ] && [ ! -f " << qbd << "/build.ninja ]; then\n";
            s << "  th_configure || { echo \"[trailhead] configure failed\"; "
                 "echo \"TRAILHEAD:build_fail\"; flock -u 200; exit 1; }\n";
            s << "fi\n";
        }
        int j = std::max(1, in.node->cpus_per_task);
        // Build each target in its own cmake invocation. Echo a clear marker
        // around each so that on failure the slurm log identifies which target
        // broke (the "Built target …" success spam otherwise buries the error,
        // and the failing target's name isn't obvious from cmake output alone).
        // On failure, if we have a configure command, reconfigure once and retry
        // — a stale Makefile that predates this target reports "No rule to make
        // target" before CMake's own regenerate step can run, so an explicit
        // reconfigure is what actually picks up a newly-added target.
        for (const auto& tgt : in.unique_targets) {
            s << "echo \"[trailhead] building target: " << tgt << "\"\n";
            s << "if ! cmake --build " << qbd << " --target " << tgt << " -j " << j << "; then\n";
            if (have_cfg) {
                s << "  echo \"[trailhead] target '" << tgt << "' failed — reconfiguring and retrying\"\n";
                s << "  th_configure || true\n";
                s << "  cmake --build " << qbd << " --target " << tgt << " -j " << j
                  << " || { echo \"[trailhead] target '" << tgt << "' failed to build\"; "
                     "echo \"TRAILHEAD:build_fail\"; flock -u 200; exit 1; }\n";
            } else {
                s << "  echo \"[trailhead] target '" << tgt << "' failed to build\"; "
                     "echo \"TRAILHEAD:build_fail\"; flock -u 200; exit 1\n";
            }
            s << "fi\n";
        }
        s << "flock -u 200\n\n";
    }

    // Per-test capture wrapper. Runs without `set -e` so failures don't abort.
    // Variadic args after the fixed three are dataset names — fetch before run,
    // refcount-finish after (so cleanup fires when the last consumer exits).
    s << "th_run_one() {\n";
    s << "  local name=\"$1\" cmd=\"$2\" wd=\"$3\"; shift 3\n";
    s << "  local datasets=(\"$@\")\n";
    s << "  for ds in \"${datasets[@]}\"; do\n";
    s << "    if declare -F th_ds_ensure >/dev/null; then th_ds_ensure \"$ds\" || true; fi\n";
    s << "  done\n";
    s << "  local started=$(date +%s%3N)\n";
    s << "  local stem=\"$TRAILHEAD_ROOT/.trailhead/batch_results/" << snode << "/${name}__${started}\"\n";
    // Sub-registry test names contain '/' (e.g. Ocean-SpGEMM/ocean__foo), so
    // the .out / .err / .meta paths land in a nested directory. Ensure that
    // directory exists or the redirects below fail with "No such file" and the
    // wrapper falsely reports exit=1. Use bash's ${stem%/*} (strip last /comp)
    // rather than $(dirname …) so we don't depend on dirname being on PATH on
    // the compute node — if it weren't, the mkdir would no-op and every capture
    // would fail.
    s << "  mkdir -p \"${stem%/*}\"\n";
    // stdout and stderr captured to separate files so error diagnostics aren't
    // interleaved with TRAILHEAD: markers in the parsed stdout. Both are
    // rsync'd back; ingest_chunk_results combines them in the saved result.
    // Capture the test's exit with `|| exit_code=$?` so a non-zero exit is
    // RECORDED rather than aborting the chunk: the project preamble usually sets
    // `set -e`, under which a bare failing command would kill the whole script
    // (taking every later test in the chunk down with it, uncaptured).
    s << "  local exit_code=0\n";
    s << "  if [ -n \"$wd\" ] && [ \"$wd\" != \".\" ]; then\n";
    s << "    ( cd \"$wd\" && bash -c \"$cmd\" ) > \"${stem}.out\" 2> \"${stem}.err\" || exit_code=$?\n";
    s << "  else\n";
    s << "    bash -c \"$cmd\" > \"${stem}.out\" 2> \"${stem}.err\" || exit_code=$?\n";
    s << "  fi\n";
    s << "  local ended=$(date +%s%3N)\n";
    s << "  printf 'name=%s\\nstarted_at=%s\\nended_at=%s\\nexit_code=%s\\nhost=%s\\njob_id=%s\\n' "
         "\"$name\" \"$started\" \"$ended\" \"$exit_code\" \"$(hostname)\" \"${SLURM_JOB_ID}\" > \"${stem}.meta\"\n";
    s << "  for ds in \"${datasets[@]}\"; do\n";
    s << "    if declare -F th_ds_finish >/dev/null; then th_ds_finish \"$ds\" \"$name\" || true; fi\n";
    s << "  done\n";
    s << "  echo \"[trailhead] ${name} exit=${exit_code}\"\n";
    s << "}\n\n";

    for (const TestEntry* t : in.tests) {
        std::string cmd = str_replace_all(t->cmd, "build/", in.build_dir + "/");
        std::string wd  = t->workdir;
        // Mirror sbatch_gen: when workdir is "."/empty AND a build is linked,
        // run from inside the build dir and strip build_dir/ prefixes. For
        // sub-registry tests the wd is qualified with sub_dir so cd lands in
        // the right tree (e.g. Ocean-SpGEMM/build_H200).
        if ((wd.empty() || wd == ".") && !t->build_name.empty() && !in.build_dir.empty()) {
            wd  = qbd;
            cmd = str_replace_all(cmd, "./" + in.build_dir + "/", "./");
            cmd = str_replace_all(cmd, in.build_dir + "/", "./");
        }
        if (wd.empty()) wd = ".";
        s << "th_run_one " << bash_sq(t->name) << " " << bash_sq(cmd) << " " << bash_sq(wd);
        for (const auto& ds : t->datasets)
            s << " " << bash_sq(ds);
        s << "\n";
    }
    s << "\necho \"[trailhead] chunk done\"\n";
    return s.str();
}

// ── Result ingestion ──────────────────────────────────────────────────────

static std::unordered_map<std::string,std::string> parse_meta(const std::string& path) {
    std::unordered_map<std::string,std::string> out;
    auto content = fs::read_file(path);
    if (!content) return out;
    std::istringstream ss(*content);
    std::string ln;
    while (std::getline(ss, ln)) {
        auto eq = ln.find('=');
        if (eq == std::string::npos) continue;
        out[ln.substr(0, eq)] = ln.substr(eq + 1);
    }
    return out;
}

// After a chunk job completes, pull batch_results + results back from the
// remote, then convert each <name>__<ts>.{meta,out} pair into a TestResult
// JSON in .trailhead/results/. Returns the number of failed tests, and appends
// a human-readable "name — reason" line to `fail_detail` for each one so the
// caller can show WHY a chunk was flagged (real failure vs. result not pulled).
static int ingest_chunk_results(const std::string& th_dir,
                                 const RemoteDest& dest,
                                 const std::string& project_root,
                                 const std::vector<const TestEntry*>& chunk_tests,
                                 const std::string& job_id,
                                 const std::string& node_name,
                                 std::vector<std::string>& fail_detail)
{
    std::string remote_proj = dest.remote_path + "/" + last_component(project_root);
    std::string snode = safe_name(node_name);

    // Pull this NODE's captures only (batch_results/<node>/) — sharing the bare
    // batch_results/ across two instances would let us match the other node's
    // metas by test name and conflate results. Retry on a dropped rsync so a
    // transient blip doesn't lose the chunk (which would mark every test failed).
    rsync_with_retry("rsync -az "
        + dest.remote + ":" + remote_proj + "/.trailhead/batch_results/" + snode + "/ "
        + th_dir + "/batch_results/" + snode + "/");
    rsync_with_retry("rsync -az "
        + dest.remote + ":" + remote_proj + "/.trailhead/results/ "
        + th_dir + "/results/");

    std::set<std::string> expected;
    for (const auto* t : chunk_tests) expected.insert(t->name);

    std::string br_dir      = th_dir + "/batch_results/" + snode;
    std::string results_dir = th_dir + "/results";
    fs::mkdir_p(results_dir);

    auto meta_files = fs::list_files_recursive(br_dir, ".meta");
    // Sort ascending by path → the last entry per name (highest timestamp) wins
    std::sort(meta_files.begin(), meta_files.end());

    int failed = 0;
    std::set<std::string> seen;

    // Walk newest-first so we keep the most recent run per test name.
    for (auto it = meta_files.rbegin(); it != meta_files.rend(); ++it) {
        const std::string& mpath = *it;
        auto kv = parse_meta(mpath);
        auto it_name = kv.find("name");
        if (it_name == kv.end()) continue;
        const std::string& name = it_name->second;
        if (!expected.count(name)) continue;
        if (!seen.insert(name).second) continue;

        std::string stem     = mpath.substr(0, mpath.size() - 5); // strip ".meta"
        std::string out_path = stem + ".out";
        std::string err_path = stem + ".err";
        std::string stdout_str = fs::read_file(out_path).value_or("");
        std::string stderr_str = fs::read_file(err_path).value_or("");

        TestResult res;
        res.name   = name;
        res.run_by = "sbatch-" + job_id;
        try { res.started_at = std::stoll(kv["started_at"]); } catch (...) {}
        try { res.ended_at   = std::stoll(kv["ended_at"]);   } catch (...) {}
        if (res.ended_at >= res.started_at)
            res.wall_ms = res.ended_at - res.started_at;
        try { res.exit_code  = std::stoi(kv["exit_code"]);   } catch (...) {}
        res.host = kv["host"];
        res.node = node_name;

        std::string remaining;
        parse_trailhead_output(stdout_str, res, &remaining);

        if ((res.passed + res.failed == 0) && res.timings.empty() && res.output_lines.empty()) {
            if (res.exit_code == 0) res.passed = 1;
            else                    res.failed = 1;
        }

        // Combined stdout + stderr for the detail-view tail. stderr (where
        // real diagnostics usually land) is appended after stdout with a
        // separator so it's easy to tell apart.
        std::string combined = stdout_str;
        if (!stderr_str.empty()) {
            if (!combined.empty() && combined.back() != '\n') combined += '\n';
            combined += "--- stderr ---\n";
            combined += stderr_str;
        }
        {
            std::deque<std::string> tail_lines;
            std::istringstream ss(combined);
            for (std::string ln; std::getline(ss, ln); ) {
                tail_lines.push_back(ln);
                if ((int)tail_lines.size() > 30) tail_lines.pop_front();
            }
            std::string tail;
            for (const auto& l : tail_lines) { tail += l; tail += "\n"; }
            if (!tail.empty()) res.metadata["_output_tail"] = tail;
        }

        save_result(results_dir, res);
        save_result_output(results_dir, res, combined);

        if (res.failed > 0 || res.exit_code != 0) {
            ++failed;
            std::string reason;
            if (res.metadata.count("_build_fail"))
                reason = "build failed (TRAILHEAD:build_fail)";
            else if (res.failed > 0 && res.exit_code == 0)
                reason = "reported FAIL via TRAILHEAD markers (exit 0)";
            else if (res.failed > 0)
                reason = "reported FAIL via markers; exit " + std::to_string(res.exit_code);
            else
                reason = "exit code " + std::to_string(res.exit_code);
            fail_detail.push_back(name + " — " + reason);
        }
    }

    // Tests that have no .meta on disk — chunk crashed before reaching them.
    for (const auto* t : chunk_tests) {
        if (seen.count(t->name)) continue;
        TestResult res;
        res.name      = t->name;
        res.run_by    = "sbatch-" + job_id;
        res.host      = dest.remote;
        res.node      = node_name;
        res.exit_code = 1;
        res.failed    = 1;
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        res.started_at = now;
        res.ended_at   = now;
        res.metadata["_output_tail"] = "test did not run (chunk job did not produce a result)";
        save_result(results_dir, res);
        ++failed;
        // No .meta for an expected test means either the chunk died before
        // reaching it, or its capture wasn't pulled back (dropped rsync). This
        // is the one "failure" that may NOT be real — flag it as such.
        fail_detail.push_back(t->name +
            " — no result captured (chunk crashed before this test, or its output was not pulled back)");
    }
    return failed;
}

// ── Filter selected tests (mirrors cmd_run semantics) ─────────────────────

static std::vector<const TestEntry*> filter_tests(const Registry& reg,
                                                    const BatchRunOptions& opts)
{
    std::vector<const TestEntry*> out;
    for (const auto& t : reg.tests) {
        bool match = false;
        if (!opts.test_names.empty()) {
            match = std::find(opts.test_names.begin(), opts.test_names.end(), t.name)
                  != opts.test_names.end();
        } else if (opts.run_all || !opts.filter_tag.empty()) {
            match = true;
        }
        if (match && !opts.filter_tag.empty()) {
            if (std::find(t.tags.begin(), t.tags.end(), opts.filter_tag) == t.tags.end())
                match = false;
        }
        if (match) out.push_back(&t);
    }
    return out;
}

// ── Build context: each chunk has exactly one (sub_dir + build_dir) so the
// chunk script's configure + per-target build phase is a single flock + cmake
// sequence. Tests are grouped by their context before chunking.
struct BuildCtx {
    std::string sub_dir;        // empty for parent
    std::string build_dir;      // resolved per-node (e.g. "build_H200")
    std::string configure_cmd;  // {{arch}} substituted

    bool operator<(const BuildCtx& o) const {
        return std::tie(sub_dir, build_dir, configure_cmd)
             < std::tie(o.sub_dir, o.build_dir, o.configure_cmd);
    }
};

static BuildCtx resolve_ctx(const TestEntry& t,
                              const Registry& reg,
                              const NodeProfile& node,
                              const std::string& node_name)
{
    BuildCtx ctx;
    ctx.sub_dir = t.sub_dir;

    // Pick the test's linked build, or fall back to any registered build in
    // the same sub_dir scope.
    const BuildConfig* bc = nullptr;
    if (!t.build_name.empty()) {
        auto it = reg.builds.find(t.build_name);
        if (it != reg.builds.end()) bc = &it->second;
    }
    if (!bc) {
        for (const auto& [_, b] : reg.builds)
            if (b.sub_dir == t.sub_dir && !b.configure_cmd.empty()) { bc = &b; break; }
    }

    std::string raw_dir = "build";
    if (bc) {
        if (!bc->dir.empty())            raw_dir = bc->dir;
        if (!bc->configure_cmd.empty())  ctx.configure_cmd = bc->configure_cmd;
    }
    // Per-node build dir: explicit setting wins; otherwise append node name.
    if (!node.build_dir.empty()) ctx.build_dir = node.build_dir;
    else                          ctx.build_dir = raw_dir + "_" + node_name;

    if (!node.cuda_arch.empty()) {
        ctx.configure_cmd = str_replace_all(ctx.configure_cmd, "{{arch}}", node.cuda_arch);
    } else if (ctx.configure_cmd.find("{{arch}}") != std::string::npos) {
        ctx.configure_cmd = str_replace_all(ctx.configure_cmd, "{{arch}}",
            "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')");
    }
    return ctx;
}

// ── Entry point ──────────────────────────────────────────────────────────

int batch_run(const Registry& reg,
              const std::string& th_dir,
              const std::string& project_root,
              const RemoteDest& dest,
              const BatchRunOptions& opts)
{
    auto node_it = reg.nodes.find(opts.node_name);
    if (node_it == reg.nodes.end()) {
        std::cerr << ansi::RED << "Error:" << ansi::RESET
                  << " node profile '" << opts.node_name << "' not found\n";
        return 1;
    }
    const NodeProfile& node = node_it->second;

    auto selected = filter_tests(reg, opts);
    if (selected.empty()) {
        std::cerr << "No tests matched. Use --all, --tag, or specify test names.\n";
        return 1;
    }

    // Run-start timestamp: anchors which results count as "this session" for
    // the CSV emitted at end-of-run. Captured before any sbatch fires so the
    // first test's started_at is guaranteed >= this value.
    int64_t session_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // ── Group selected tests by (sub_dir, build_dir, configure_cmd) so each
    // chunk has a single build context. Sub-registry tests get their own
    // groups; the parent's tests group together.
    std::map<BuildCtx, std::vector<const TestEntry*>> ctx_groups;
    for (const auto* t : selected)
        ctx_groups[resolve_ctx(*t, reg, node, opts.node_name)].push_back(t);

    fs::mkdir_p(th_dir + "/sbatch");
    // Node-scope all chunk scripts (batch_<node>_NNN.sbatch). Two batch-run
    // instances sharing this workspace (one per processor) would otherwise both
    // write batch_NNN.sbatch and submit each other's script for a given chunk.
    std::string snode = safe_name(opts.node_name);
    std::string chunk_prefix = "batch_" + snode + "_";
    // Drop THIS node's chunk scripts from prior runs so a smaller selection
    // doesn't leave stale files lying around — but leave another instance's
    // (other-node) scripts and sbatch_gen's per-test scripts alone.
    {
        auto stale = fs::list_files_recursive(th_dir + "/sbatch", ".sbatch");
        for (const auto& p : stale) {
            auto sl = p.rfind('/');
            std::string base = (sl == std::string::npos) ? p : p.substr(sl + 1);
            if (base.rfind(chunk_prefix, 0) == 0) std::remove(p.c_str());
        }
    }
    std::string remote_proj = dest.remote_path + "/" + last_component(project_root);

    // ── Datasets: emit helper lib + per-dataset state, then rsync carries
    //    .trailhead/lib/ and .trailhead/datasets/ up with the rest of the project.
    std::vector<std::string> selected_names;
    for (const auto* t : selected) selected_names.push_back(t->name);
    std::vector<std::string> touched_datasets =
        init_dataset_state(reg, selected_names, th_dir);
    if (!touched_datasets.empty()) {
        write_dataset_lib(th_dir);
    }

    int bs = std::max(1, opts.batch_size);

    int total_chunks = 0;
    for (const auto& [_, gtests] : ctx_groups)
        total_chunks += (int)((gtests.size() + bs - 1) / bs);

    std::cout << ansi::BOLD << ansi::BCYAN << "batch-run" << ansi::RESET
              << "  " << ansi::BOLD << selected.size() << ansi::RESET << " tests "
              << ansi::DIM << "→" << ansi::RESET << " "
              << ansi::BOLD << total_chunks << ansi::RESET << " chunks "
              << ansi::DIM << "(" << bs << "/chunk)" << ansi::RESET << " across "
              << ansi::BOLD << ctx_groups.size() << ansi::RESET << " build context(s) on "
              << ansi::color(ansi::BYELLOW, opts.node_name) << "\n";
    // Surface the per-chunk wall limit each job will request — the usual cause
    // of a surprise TIMEOUT is a too-small (or empty → defaulted) profile time.
    {
        std::string t = node.time.empty() ? "01:00:00 (default — profile has no time)" : node.time;
        std::cout << "  " << ansi::DIM << "per-chunk time limit: " << ansi::RESET
                  << ansi::color(ansi::BCYAN, t)
                  << ansi::DIM << "  (#SBATCH --time; subject to your QOS/partition cap)"
                  << ansi::RESET << "\n";
    }
    if (opts.no_build)
        std::cout << "  " << ansi::color(ansi::YELLOW, "--no-build set")
                  << ansi::DIM << ": skipping configure + per-target rebuild" << ansi::RESET << "\n";
    if (!touched_datasets.empty())
        std::cout << "  " << ansi::DIM << "datasets in flight: "
                  << touched_datasets.size() << ansi::RESET << "\n";

    struct ChunkJob {
        std::string label;
        std::string sbatch_rel;
        std::vector<const TestEntry*> tests;
    };
    // Build chunks per context, then interleave them proportionally so dataset
    // lifetime is bounded by a single chunk's worth of work — Ocean chunks
    // (when there are fewer of them) get sprinkled throughout the parent
    // chunks instead of all firing at the end. Each registry refcount-cleans
    // its own copy of a matrix as soon as the chunk that touched it finishes,
    // so spreading Ocean across the timeline keeps peak disk roughly constant.
    std::vector<std::vector<ChunkJob>> per_ctx_jobs;
    per_ctx_jobs.reserve(ctx_groups.size());
    int chunk_idx = 0;

    for (const auto& [ctx, gtests] : ctx_groups) {
        per_ctx_jobs.emplace_back();
        auto& cur_ctx_jobs = per_ctx_jobs.back();
        // Targets for this context: per-test cmake target + datasets'
        // requires_targets (filtered to datasets in this context's sub_dir,
        // which is the normal authoring pattern).
        std::vector<std::string> sel_names;
        for (const auto* t : gtests) sel_names.push_back(t->name);
        std::vector<std::string> ctx_targets;
        std::set<std::string> tgt_seen;
        for (const auto* t : gtests)
            if (!t->target.empty() && tgt_seen.insert(t->target).second)
                ctx_targets.push_back(t->target);
        // required_build_targets walks the dataset closure for sel_names; we
        // include every target it returns. requires_targets are raw cmake
        // names that build in the dataset's own context (which equals the
        // test's context under normal usage).
        for (const auto& tgt : required_build_targets(reg, sel_names))
            if (tgt_seen.insert(tgt).second) ctx_targets.push_back(tgt);

        std::cout << "  " << ansi::DIM << "context:" << ansi::RESET << " "
                  << ansi::color(ansi::BGREEN, ctx.sub_dir.empty() ? "<parent>" : ctx.sub_dir)
                  << "  " << ansi::DIM << "build_dir=" << ansi::RESET << ctx.build_dir
                  << "  " << ansi::DIM << "tests=" << ansi::RESET
                  << ansi::BOLD << gtests.size() << ansi::RESET;
        if (!ctx_targets.empty()) {
            std::cout << "  " << ansi::DIM << "targets:" << ansi::RESET;
            for (const auto& t : ctx_targets) std::cout << " " << ansi::color(ansi::CYAN, t);
        }
        std::cout << "\n";

        // Chunk this group's tests.
        for (size_t i = 0; i < gtests.size(); i += bs) {
            std::vector<const TestEntry*> chunk_tests(
                gtests.begin() + i,
                gtests.begin() + std::min(gtests.size(), i + bs));
            ++chunk_idx;
            char buf[16]; std::snprintf(buf, sizeof(buf), "%03d", chunk_idx);

            ChunkScriptInputs in;
            in.tests          = chunk_tests;
            in.node           = &node;
            in.reg            = &reg;
            in.node_name      = opts.node_name;
            in.remote_project = remote_proj;
            in.chunk_label    = buf;
            in.sub_dir        = ctx.sub_dir;
            in.build_dir      = ctx.build_dir;
            in.configure_cmd  = ctx.configure_cmd;
            in.no_build       = opts.no_build;
            in.unique_targets = ctx_targets;

            std::string content = build_chunk_script(in);
            std::string fname   = chunk_prefix + buf + ".sbatch";
            fs::write_file_atomic(th_dir + "/sbatch/" + fname, content);

            ChunkJob cj;
            cj.label      = buf;
            cj.sbatch_rel = ".trailhead/sbatch/" + fname;
            cj.tests      = std::move(chunk_tests);
            cur_ctx_jobs.push_back(std::move(cj));
        }
    }

    // Proportional interleave: at each step pick the context whose next chunk
    // is "most overdue". We rank a context's j-th chunk by the MIDPOINT of its
    // slot, (j + 0.5) / size, and always take the smallest. Using the midpoint
    // (rather than j / size) is what spreads each context evenly: a 6-chunk
    // context's first rank is 0.5/6 ≈ 0.08 so it goes early and often, while a
    // 1-chunk context ranks 0.5 and lands mid-timeline. Plain j/size instead
    // starts every context at 0, so a whole "one each" round fires first
    // (A,B,C,…) and the big contexts' remaining chunks all pile up at the end.
    // With per-chunk refcount cleanup, even spacing keeps each registry's
    // matrix lifetime bounded by ~one chunk's worth of work.
    std::vector<ChunkJob> jobs;
    std::vector<int>      jobs_ctx;   // ctx index per job, parallel to `jobs`
    {
        size_t total = 0;
        for (const auto& v : per_ctx_jobs) total += v.size();
        jobs.reserve(total);
        jobs_ctx.reserve(total);
        std::vector<size_t> idx(per_ctx_jobs.size(), 0);
        while (jobs.size() < total) {
            int    best = -1;
            double best_rank = 2.0;
            for (size_t i = 0; i < per_ctx_jobs.size(); ++i) {
                if (idx[i] >= per_ctx_jobs[i].size()) continue;
                double rank = (idx[i] + 0.5) / (double)per_ctx_jobs[i].size();
                if (rank < best_rank) { best = (int)i; best_rank = rank; }
            }
            if (best < 0) break;
            jobs.push_back(std::move(per_ctx_jobs[best][idx[best]]));
            jobs_ctx.push_back(best);
            idx[best]++;
        }
    }
    // Show the interleaved submission order so the user can confirm contexts
    // are spread across the timeline (one letter per context, by enumeration
    // order — first context = 'A', second = 'B', etc.).
    if (per_ctx_jobs.size() > 1) {
        std::cout << "  submission order: ";
        for (int c : jobs_ctx) std::cout << (char)('A' + c);
        std::cout << "\n";
    }

    // Single rsync up before any submission.
    auto channel = std::make_shared<RemoteChannel>(dest.remote);
    std::cout << ansi::BOLD << "rsync" << ansi::RESET
              << "  " << project_root << " → " << dest.remote << ":" << remote_proj << "\n";
    {
        // BatchMode + ConnectTimeout so SSH fails fast instead of waiting for
        // a passphrase or host-key prompt. --info=progress2 streams progress so
        // the user can see whether it's actually stuck or just slow. Excludes
        // mirror what the dataset/build pipeline produces remotely so we never
        // upload matrices, build trees, CPM caches, or VCS metadata that the
        // remote will rebuild/refetch.
        // The dataset state files (path/fetch/expected/finished.txt) MUST
        // upload so refcount starts fresh against the locally-written state
        // (init_dataset_state truncates finished.txt and removes `cleaned`).
        // We do NOT pass --delete, so a stale `cleaned` sentinel left on the
        // remote from a prior run is harmless: th_ds_ensure detects it and
        // refetches transparently.
        //
        // Generic excludes only — no project-specific paths. `build/` and
        // `build_*/` are excluded because they're produced remotely by
        // configure+build steps; uploading local artifacts would clobber a
        // node's arch-specific build. Dataset paths (matrices, etc.) are
        // user-defined and live wherever the user wants — those naturally
        // don't exist locally if you're using datasets, so no exclude needed.
        // No --info=progress2 / --progress: macOS still ships rsync 2.6.9 which
        // predates them. The `-v` flag is universally supported and gives
        // per-file progress that streams through on_line so the user can see
        // it isn't stuck.
        std::string cmd =
            "rsync -azv "
            "-e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30' "
            "--exclude='/.trailhead/results/' "
            "--exclude='/.trailhead/batch_results/' "
            "--exclude='/build/' --exclude='/build_*/' "
            "--exclude='/.git/' "
            "--exclude='core' --exclude='core.*' "
            + project_root + "/ " + dest.remote + ":" + remote_proj + "/";
        auto on_line = [](const std::string& line) {
            if (line.empty()) return;
            std::cout << "  " << line << std::endl;
        };
        // Retry on failure — a transient network/SSH hiccup shouldn't sink the
        // whole batch before any chunk is submitted. Up to 5 attempts with
        // increasing backoff.
        const int kRsyncTries = 5;
        proc::RunResult r;
        for (int attempt = 1; attempt <= kRsyncTries; ++attempt) {
            r = proc::run(cmd, {}, {}, 600, "", on_line, true);
            if (r.exit_code == 0) break;
            if (attempt < kRsyncTries) {
                int backoff = 5 * attempt;   // 5s, 10s, 15s, 20s
                std::cerr << ansi::color(ansi::BYELLOW,
                    "rsync failed (exit=" + std::to_string(r.exit_code)
                    + "), retrying in " + std::to_string(backoff) + "s ("
                    + std::to_string(attempt) + "/" + std::to_string(kRsyncTries - 1)
                    + ")\n");
                std::this_thread::sleep_for(std::chrono::seconds(backoff));
            }
        }
        if (r.exit_code != 0) {
            std::cerr << ansi::color(ansi::BRED, "rsync failed (exit="
                + std::to_string(r.exit_code) + ") after "
                + std::to_string(kRsyncTries) + " attempts\n");
            if (!r.stderr_str.empty()) std::cerr << r.stderr_str;
            return 1;
        }
    }

    int max_concurrent = opts.max_concurrent;
    bool auto_concurrency = (max_concurrent <= 0);
    if (auto_concurrency) {
        std::vector<std::string> parts;
        if (!node.partition.empty()) parts.push_back(node.partition);
        max_concurrent = query_slurm_job_limit(dest.remote, parts, 4);
    }
    std::cout << "  max concurrent chunks: " << max_concurrent
              << (auto_concurrency ? " (auto)" : " (--max-concurrent)") << "\n";
    auto slots = std::make_shared<ChunkSemaphore>(max_concurrent);

    std::atomic<int> total_passed{0}, total_failed{0}, chunks_done{0};
    std::mutex log_mtx;

    // ── Live grid state ────────────────────────────────────────────────────
    // When stdout is a TTY we draw a chunk-status matrix + progress bar on the
    // alternate screen; otherwise we keep plain line logging (pipes/CI).
    bool tty = isatty(STDOUT_FILENO);
    enum ChunkState { CS_WAIT = 0, CS_SUBMIT, CS_PEND, CS_RUN, CS_COMPL, CS_PASS, CS_FAIL };
    std::vector<std::atomic<int>> board(jobs.size());     // per-chunk state
    std::vector<std::string>      labels;                 // per-chunk label (jobs get moved away)
    labels.reserve(jobs.size());
    for (const auto& j : jobs) labels.push_back(j.label);
    std::vector<std::string> job_ids(jobs.size());        // SLURM job id per chunk
    std::vector<std::string> chunk_logs(jobs.size());     // local log path per chunk (once finished)
    std::vector<std::string> chunk_verdict(jobs.size());  // SLURM end verdict (for inspect header)
    std::mutex jid_mtx;                                   // guards job_ids + chunk_logs
    std::deque<std::string> rlog;                         // recent log lines (grid mode)
    std::atomic<bool> abort{false};                       // [q]/Ctrl-C mid-run → cancel in-flight
    std::atomic<bool> all_done{false};                    // every chunk finished (set after join)
    std::atomic<bool> quit{false};                        // user dismissed the grid

    // Render-thread-owned UI state (touched only by the render loop / final draw).
    int  sel = 0;                       // selected chunk cell
    int  grid_scroll = 0;               // matrix row offset (auto-follows sel)
    bool detail_mode = false;           // inspecting one chunk's slurm-out
    std::vector<std::string> detail_lines;
    int  detail_scroll = 0;
    int  detail_chunk = 0;
    int  last_render_mode = -1;         // -1 none, 0 grid, 1 detail (for full-clear on switch)

    auto log = [&](const std::string& m) {
        std::lock_guard<std::mutex> lk(log_mtx);
        if (tty) {
            rlog.push_back(m);
            while (rlog.size() > 8) rlog.pop_front();
        } else {
            std::cout << m << std::endl;
        }
    };

    auto state_color = [](int st) -> const char* {
        switch (st) {
            case CS_SUBMIT: return ansi::BYELLOW;
            case CS_PEND:   return ansi::YELLOW;
            case CS_RUN:    return ansi::BCYAN;
            case CS_COMPL:  return ansi::CYAN;
            case CS_PASS:   return ansi::BGREEN;
            case CS_FAIL:   return ansi::BRED;
            default:        return ansi::GRAY;   // CS_WAIT
        }
    };

    auto render_grid = [&]() {
        int tot  = (int)jobs.size();           // chunks (the matrix cells)
        int cols_t = bt_term_cols();
        int rows_t = bt_term_rows();

        // On a grid↔detail switch, wipe the whole screen once. The per-frame
        // CURSOR_HOME + ERASE_DOWN only clears below the cursor, so if the new
        // view is shorter (or the terminal scrolled), the old view's lines can
        // bleed through — e.g. detail-view text lingering on the grid.
        int mode = detail_mode ? 1 : 0;
        std::string clear_pfx = (mode != last_render_mode) ? "\033[2J" : "";
        last_render_mode = mode;

        // ── Detail view: a single chunk's slurm-out, scrollable. ────────────
        if (detail_mode) {
            std::ostringstream o;
            o << clear_pfx << ansi::CURSOR_HOME;
            std::string jid, verdict;
            { std::lock_guard<std::mutex> lk(jid_mtx);
              if (detail_chunk < (int)job_ids.size())       jid     = job_ids[detail_chunk];
              if (detail_chunk < (int)chunk_verdict.size())  verdict = chunk_verdict[detail_chunk]; }
            const char* sc = state_color(board[detail_chunk].load());
            o << ansi::BOLD << "chunk " << ansi::color(sc, "■" + labels[detail_chunk])
              << ansi::RESET << ansi::BOLD << "  job " << (jid.empty() ? "—" : jid)
              << ansi::RESET << ansi::ERASE_EOL << "\n";
            // SLURM verdict line (TIMEOUT / OOM / FAILED / …) when known.
            if (!verdict.empty())
                o << ansi::color(ansi::BYELLOW, "  " + verdict) << ansi::ERASE_EOL << "\n";
            o << ansi::DIM << std::string(std::min(cols_t, 60), '-') << ansi::RESET
              << ansi::ERASE_EOL << "\n";
            int view = std::max(3, rows_t - 5);
            int n = (int)detail_lines.size();
            if (detail_scroll > std::max(0, n - view)) detail_scroll = std::max(0, n - view);
            if (detail_scroll < 0) detail_scroll = 0;
            for (int i = detail_scroll; i < std::min(n, detail_scroll + view); ++i)
                o << "  " << detail_lines[i] << ansi::ERASE_EOL << "\n";
            o << ansi::DIM << "  [b/esc] back   [j/k ↑/↓] scroll"
              << (n > view ? "   (" + std::to_string(detail_scroll + 1) + "-"
                             + std::to_string(std::min(n, detail_scroll + view))
                             + "/" + std::to_string(n) + ")" : "")
              << ansi::RESET << ansi::ERASE_EOL << "\n";
            o << ansi::ERASE_DOWN;
            std::cout << o.str(); std::cout.flush();
            return;
        }

        int done = chunks_done.load();
        int pass = total_passed.load(), fail = total_failed.load();
        int total_tests = (int)selected.size();  // individual jobs

        // Matrix geometry: how many cells per row, how many rows total.
        int cell_w = (int)(labels.empty() ? 3 : labels[0].size()) + 2; // ■ + label + space
        int cols   = std::max(1, (cols_t - 2) / cell_w);
        int mat_rows_total = (tot + cols - 1) / cols;

        // Vertical budget: everything that isn't the matrix viewport. title(2) +
        // bar(2) + up/down indicators(2) + legend(2) + footer(1) + log + 1 safety.
        int log_lines;
        { std::lock_guard<std::mutex> lk(log_mtx); log_lines = (int)rlog.size(); }
        int nonmat   = 10 + log_lines;
        int mat_view = std::min(mat_rows_total, std::max(1, rows_t - nonmat));

        // Auto-scroll so the selected cell's row stays in view.
        if (sel < 0) sel = 0;
        if (sel >= tot) sel = tot - 1;
        int sel_row = (sel < 0 ? 0 : sel / cols);
        if (sel_row < grid_scroll)               grid_scroll = sel_row;
        if (sel_row >= grid_scroll + mat_view)   grid_scroll = sel_row - mat_view + 1;
        int max_scroll = std::max(0, mat_rows_total - mat_view);
        if (grid_scroll > max_scroll) grid_scroll = max_scroll;
        if (grid_scroll < 0)          grid_scroll = 0;

        std::ostringstream o;
        o << clear_pfx << ansi::CURSOR_HOME;
        o << ansi::BOLD << "batch-run" << ansi::RESET << "  " << opts.node_name
          << "  " << done << "/" << tot << " chunks"
          << (abort.load() ? std::string("  ") + ansi::BRED + "(stopping…)" + ansi::RESET : "")
          << ansi::ERASE_EOL << "\n\n";

        // Progress bar over individual jobs (not chunks). The filled portion is
        // split into a green (passed) segment and a red (failed) segment so the
        // bar's colour conveys the success/fail split; the rest is still to run.
        int done_tests = pass + fail;
        int barw = std::max(10, std::min(40, cols_t - 40));
        int gw = total_tests > 0 ? pass * barw / total_tests : 0;
        int rw = total_tests > 0 ? fail * barw / total_tests : 0;
        if (gw + rw > barw) rw = barw - gw;       // clamp rounding overflow
        int rest = barw - gw - rw;
        o << "  [" << ansi::BGREEN;
        for (int i = 0; i < gw;   ++i) o << "█";
        o << ansi::BRED;
        for (int i = 0; i < rw;   ++i) o << "█";
        o << ansi::DIM;
        for (int i = 0; i < rest; ++i) o << "░";
        o << ansi::RESET << "] " << done_tests << "/" << total_tests << " jobs  "
          << ansi::color(ansi::BGREEN, "pass " + std::to_string(pass)) << " "
          << ansi::color(ansi::BRED,   "fail " + std::to_string(fail))
          << ansi::ERASE_EOL << "\n\n";

        // Up-scroll indicator (always one line so the layout height is stable).
        if (grid_scroll > 0)
            o << ansi::DIM << "  ↑ " << grid_scroll << " more row(s)" << ansi::RESET;
        o << ansi::ERASE_EOL << "\n";

        // Chunk matrix window — one colored "■NNN" cell per chunk; selected cell
        // shown in reverse video.
        int first = grid_scroll * cols;
        int last  = std::min(tot, (grid_scroll + mat_view) * cols);
        for (int i = first; i < last; ++i) {
            if (i == sel) o << ansi::REVERSE;
            o << ansi::color(state_color(board[i].load()), "■" + labels[i]);
            if (i == sel) o << ansi::RESET;
            o << " ";
            if ((i + 1) % cols == 0) o << ansi::ERASE_EOL << "\n";
        }
        if (last % cols != 0) o << ansi::ERASE_EOL << "\n";

        // Down-scroll indicator (always one line).
        if (grid_scroll < max_scroll)
            o << ansi::DIM << "  ↓ " << (mat_rows_total - grid_scroll - mat_view)
              << " more row(s)" << ansi::RESET;
        o << ansi::ERASE_EOL << "\n";

        o << ansi::DIM << "  " << ansi::RESET
          << ansi::color(ansi::GRAY,    "■wait")    << " "
          << ansi::color(ansi::BYELLOW, "■queued")  << " "
          << ansi::color(ansi::BCYAN,   "■running") << " "
          << ansi::color(ansi::BGREEN,  "■pass")    << " "
          << ansi::color(ansi::BRED,    "■fail")    << ansi::ERASE_EOL << "\n\n";

        { std::lock_guard<std::mutex> lk(log_mtx);
          for (const auto& l : rlog)
              o << ansi::DIM << "  " << l << ansi::RESET << ansi::ERASE_EOL << "\n"; }

        o << ansi::DIM << "  "
          << (all_done.load() ? "[q] return" : "[q] stop & return")
          << "   [↑/↓/←/→ hjkl] select   [enter] inspect chunk"
          << ansi::RESET << ansi::ERASE_EOL << "\n";
        o << ansi::ERASE_DOWN;
        std::cout << o.str();
        std::cout.flush();
    };

    // Fetch the slurm-out tail for the chunk at `idx` (blocking SSH) into
    // detail_lines, then switch to the detail view.
    // Inspect a chunk's log. NEVER does network I/O — this runs on the render/
    // input thread, so a blocking SSH here would freeze the whole UI (especially
    // for a queued chunk on a slow/stalled connection). A finished chunk reads
    // its local log; an unfinished one shows its current state instantly. The log
    // for a chunk is pulled local when it finishes, so it's always available then.
    auto inspect_chunk = [&](int idx) {
        if (idx < 0 || idx >= (int)jobs.size()) return;
        std::string local;
        { std::lock_guard<std::mutex> lk(jid_mtx);
          if (idx < (int)chunk_logs.size()) local = chunk_logs[idx]; }
        detail_lines.clear();
        detail_scroll = 0;
        detail_chunk  = idx;
        if (!local.empty()) {                     // finished — read local copy
            auto c = fs::read_file(local);
            if (c) {
                std::istringstream iss(*c);
                for (std::string ln; std::getline(iss, ln); ) detail_lines.push_back(ln);
            }
            if (detail_lines.empty())
                detail_lines.push_back("(local log is empty: " + local + ")");
        } else {                                  // not finished — describe state
            int st = board[idx].load();
            const char* msg =
                st == CS_SUBMIT ? "submitted"          :
                st == CS_PEND   ? "queued (PENDING)"   :
                st == CS_RUN    ? "running"            :
                st == CS_COMPL  ? "completing"         :
                                  "waiting for a slot";
            detail_lines.push_back("chunk " + labels[idx] + " is " + msg + ".");
            detail_lines.push_back("");
            detail_lines.push_back("Its build/run log is pulled here and shown once the chunk finishes.");
        }
        detail_mode = true;
    };

    auto submit_and_wait = [&, total = jobs.size()](size_t idx, ChunkJob cj) {
        board[idx] = CS_WAIT;
        if (!slots->acquire(abort)) return;   // aborted before this chunk submitted
        if (abort.load()) { slots->release(); return; }

        // sbatch with light retry on SSH transport / QOS errors.
        std::string sbatch_cmd = "\"cd " + remote_proj
                               + " && sbatch " + cj.sbatch_rel + "\"";
        proc::RunResult r;
        for (int attempt = 0; attempt < 4 && !abort.load(); ++attempt) {
            if (attempt > 0)
                std::this_thread::sleep_for(std::chrono::seconds(15 * attempt));
            r = channel->ssh(sbatch_cmd, 90);
            if (r.exit_code == 0) break;
        }
        if (abort.load()) { slots->release(); return; }
        if (r.exit_code != 0) {
            log(ansi::color(ansi::BRED, "[chunk " + cj.label + "] sbatch failed: ")
                + (r.stderr_str.empty() ? r.stdout_str : r.stderr_str));
            board[idx] = CS_FAIL;
            slots->release();
            chunks_done++;
            total_failed += (int)cj.tests.size();
            return;
        }

        std::string job_id;
        {
            std::istringstream ss(r.stdout_str);
            std::string tok;
            while (ss >> tok) job_id = tok;
        }
        if (job_id.empty()) {
            log(ansi::color(ansi::BRED, "[chunk " + cj.label + "] could not parse job_id: ")
                + r.stdout_str);
            board[idx] = CS_FAIL;
            slots->release();
            chunks_done++;
            total_failed += (int)cj.tests.size();
            return;
        }
        board[idx] = CS_SUBMIT;
        { std::lock_guard<std::mutex> lk(jid_mtx); job_ids[idx] = job_id; }
        log("[chunk " + cj.label + "] submitted job " + job_id
            + " (" + std::to_string(cj.tests.size()) + " tests)");

        // Poll until squeue empties + sacct reports a final state. Log the
        // first observed state immediately and any transition after that, plus
        // a heartbeat every minute, so the run never sits silent for >60s.
        //
        // Race we have to defend against: squeue can return empty *before* the
        // job is visible (very brief window after sbatch) AND sacct also
        // returns empty until the accounting DB catches up. If both are empty
        // and we'd never seen the job in squeue, we'd wrongly conclude
        // "finished" and ingest before the chunk actually wrote any results.
        // Track `seen_in_queue` and require it before treating empty squeue as
        // terminal; also require non-empty sacct content.
        std::string last_state;
        bool        seen_in_queue = false;
        bool        aborted_here  = false;
        bool        conn_lost     = false;   // SSH currently failing → reconnecting
        std::string sacct_line;   // "State|ExitCode|Elapsed|Timelimit" of the finished job
        for (int p = 0; ; ++p) {
            // Abort-aware 10s poll interval: wake within ~250ms of [q].
            for (int w = 0; w < 40 && !abort.load(); ++w)
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (abort.load()) {
                // Cancel the running job so we don't leave it consuming the
                // cluster, then bail — already-finished chunks keep their results.
                channel->ssh("\"scancel " + job_id + " 2>/dev/null\"", 15);
                aborted_here = true;
                break;
            }
            auto sq = channel->ssh("\"squeue -j " + job_id + " -h -o '%T' 2>/dev/null\"", 30);
            std::string sq_out = sq.stdout_str;
            while (!sq_out.empty() && (sq_out.back() == '\n' || sq_out.back() == '\r' || sq_out.back() == ' '))
                sq_out.pop_back();
            if (sq.exit_code != 0 || sq.timed_out) {
                // SSH/transport error (channel->ssh already retried with a
                // reconnect). Keep polling — when the link comes back, squeue
                // tells us whether the job is still running or already done — but
                // surface the outage once so a long run doesn't look hung.
                if (!conn_lost) {
                    conn_lost = true;
                    log(ansi::color(ansi::BYELLOW, "[chunk " + cj.label
                        + "] connection lost — reconnecting (job " + job_id + " still tracked)"));
                }
                continue;
            }
            if (conn_lost) {
                conn_lost = false;
                log(ansi::color(ansi::BGREEN, "[chunk " + cj.label + "] reconnected"));
            }
            if (!sq_out.empty()) {
                seen_in_queue = true;
                board[idx] = (sq_out.find("RUN") != std::string::npos)   ? CS_RUN
                           : (sq_out.find("COMPL") != std::string::npos) ? CS_COMPL
                           : CS_PEND;
                bool changed   = (sq_out != last_state);
                bool heartbeat = (p % 6 == 0);
                if (changed || heartbeat)
                    log("[chunk " + cj.label + "] "
                        + std::string(ansi::DIM) + sq_out + ansi::RESET);
                last_state = sq_out;
                continue;
            }
            // squeue empty: could be "not yet visible" or "already done".
            // Hold off declaring done until we've previously seen the job
            // in squeue, OR we've waited long enough that submission must
            // have failed silently (~5 min).
            if (!seen_in_queue) {
                if (p >= 30) {
                    log("[chunk " + cj.label + "] job " + job_id
                        + " never appeared in queue after 5min — giving up");
                    break;
                }
                continue;
            }
            // sacct must produce a terminal-state line; an empty response
            // means the accounting DB hasn't caught up yet, even though the
            // job has left squeue. Keep polling.
            auto sa = channel->ssh("\"sacct -j " + job_id
                + " -o State,ExitCode,Elapsed,Timelimit -n --parsable2 2>/dev/null | head -1\"", 30);
            if (sa.exit_code != 0 || sa.timed_out) continue;
            std::string sa_out = sa.stdout_str;
            while (!sa_out.empty() && (sa_out.back() == '\n' || sa_out.back() == '\r' || sa_out.back() == ' '))
                sa_out.pop_back();
            if (sa_out.empty()) continue;
            sacct_line = sa_out;
            break;
        }
        slots->release();

        if (aborted_here) {                 // [q] pressed mid-flight — don't ingest
            board[idx] = CS_WAIT;
            log("[chunk " + cj.label + "] cancelled");
            return;
        }

        // Translate SLURM accounting into a one-line verdict — distinguishes a
        // real test failure from a TIMEOUT / OUT_OF_MEMORY / node failure, and
        // shows the GRANTED time limit (which differs from the requested one if a
        // QOS/partition capped it) alongside how long the job actually ran.
        std::string verdict;
        if (!sacct_line.empty()) {
            std::vector<std::string> f;
            { size_t p = 0, q; std::string s = sacct_line;
              while ((q = s.find('|', p)) != std::string::npos) { f.push_back(s.substr(p, q - p)); p = q + 1; }
              f.push_back(s.substr(p)); }
            std::string state = f.size() > 0 ? f[0] : "";
            std::string exit  = f.size() > 1 ? f[1] : "";
            std::string elapsed = f.size() > 2 ? f[2] : "";
            std::string tl      = f.size() > 3 ? f[3] : "";
            if      (state.find("TIMEOUT") != std::string::npos)
                verdict = "TIMEOUT — ran " + elapsed + " of granted limit " + tl;
            else if (state.find("OUT_OF_ME") != std::string::npos || state.find("OOM") != std::string::npos)
                verdict = "OUT_OF_MEMORY — ran " + elapsed;
            else if (state.rfind("CANCELLED", 0) == 0)
                verdict = "CANCELLED — ran " + elapsed + " of granted limit " + tl;
            else if (state.find("NODE_FAIL") != std::string::npos)
                verdict = "NODE_FAIL — ran " + elapsed;
            else if (state.find("FAILED") != std::string::npos)
                verdict = "FAILED (exit " + exit + ") — ran " + elapsed;
            else if (!state.empty())
                verdict = state + " — ran " + elapsed + " of limit " + tl;   // COMPLETED, etc.
        } else if (!seen_in_queue) {
            verdict = "job never appeared in the queue (submission likely rejected)";
        }
        { std::lock_guard<std::mutex> lk(jid_mtx); chunk_verdict[idx] = verdict; }

        log("[chunk " + cj.label + "] finished — collecting results");
        std::vector<std::string> fail_detail;
        int chunk_failed = ingest_chunk_results(th_dir, dest, project_root, cj.tests, job_id, opts.node_name, fail_detail);
        int chunk_passed = (int)cj.tests.size() - chunk_failed;
        total_passed += chunk_passed;
        total_failed += chunk_failed;
        board[idx] = chunk_failed > 0 ? CS_FAIL : CS_PASS;
        chunks_done++;
        log("[chunk " + cj.label + "] "
            + ansi::color(chunk_failed > 0 ? ansi::BRED : ansi::BGREEN,
                std::to_string(chunk_passed) + " pass, " + std::to_string(chunk_failed) + " fail")
            + "  (" + std::to_string(chunks_done.load()) + "/" + std::to_string(total) + " chunks)"
            + (chunk_failed > 0 && !verdict.empty()
                 ? "  " + std::string(ansi::BYELLOW) + verdict + ansi::RESET : ""));
        // Spell out which tests failed and why, so a flagged chunk is never a
        // mystery — distinguishes a real failure (exit code / FAIL marker) from
        // a result that simply wasn't pulled back.
        for (const auto& fd : fail_detail)
            log("[chunk " + cj.label + "]   " + ansi::color(ansi::BRED, "\xe2\x9c\x97") + " " + fd);

        // If every test in the chunk failed, the chunk script almost certainly
        // exited before the per-test wrappers ran (build error, missing dep,
        // SLURM time/oom kill). Dump the tail of the slurm-out so the user
        // doesn't have to ssh in to figure out what went wrong. Skipped in
        // grid (TTY) mode — it would scribble over the matrix, and the error
        // tail is still captured per-test in _output_tail for the board's
        // detail view.
        // Bring this chunk's log (build + run output) local, store one copy per
        // (test, machine) — overwriting any previous so logs don't accumulate
        // across runs — then delete the remote artifacts so the cluster stays
        // clean. The local copy is what the [enter] inspect view reads.
        {
            auto subst = [&](std::string p) {
                auto q = p.find("%j");
                if (q != std::string::npos) p.replace(q, 2, job_id);
                return p;
            };
            std::string out_rel = subst(reg.sbatch_defaults.output_pattern);
            std::string err_rel = subst(reg.sbatch_defaults.error_pattern);

            // Fetch full stdout + stderr.
            std::string log_text =
                channel->ssh("\"cat " + remote_proj + "/" + out_rel + " 2>/dev/null\"", 60).stdout_str;
            std::string errc =
                channel->ssh("\"cat " + remote_proj + "/" + err_rel + " 2>/dev/null\"", 60).stdout_str;
            if (!errc.empty() && errc != log_text)
                log_text += "\n--- stderr ---\n" + errc;

            // Prepend a header (job id + SLURM verdict) so the log is self-
            // describing when inspected — you see WHY it ended before the output.
            std::string header = "== chunk " + cj.label + "  job " + job_id + " ==\n";
            if (!verdict.empty()) header += "SLURM: " + verdict + "\n";
            header += std::string(40, '-') + "\n";
            log_text = header + log_text;

            // One log per chunk per machine (no per-test redundancy), overwriting
            // any previous so logs don't pile up across runs.
            std::string log_dir = th_dir + "/logs/" + safe_name(opts.node_name);
            fs::mkdir_p(log_dir);
            std::string log_path = log_dir + "/chunk_" + cj.label + ".log";
            fs::write_file_atomic(log_path, log_text);
            { std::lock_guard<std::mutex> lk(jid_mtx); chunk_logs[idx] = log_path; }

            // Non-TTY: surface the failure tail (we just have it in log_text).
            if (!tty && chunk_passed == 0 && chunk_failed > 0 && !log_text.empty()) {
                std::lock_guard<std::mutex> lk(log_mtx);
                std::cout << ansi::DIM << "  ── log tail for chunk " << cj.label
                          << " (job " << job_id << ") ──" << ansi::RESET << "\n";
                std::deque<std::string> tail;
                std::istringstream iss(log_text);
                for (std::string ln; std::getline(iss, ln); ) {
                    tail.push_back(ln);
                    if (tail.size() > 120) tail.pop_front();
                }
                for (const auto& ln : tail) std::cout << "    " << ln << "\n";
                std::cout << ansi::DIM << "  ── end log ──" << ansi::RESET << std::endl;
            }

            // Delete remote artifacts: SLURM logs + this chunk's per-test captures.
            std::ostringstream rm;
            rm << "\"cd " << remote_proj << " && rm -f " << out_rel << " " << err_rel;
            for (const auto* t : cj.tests)
                rm << " .trailhead/batch_results/" << snode << "/" << t->name << "__*";
            rm << " 2>/dev/null\"";
            channel->ssh(rm.str(), 30);
        }
    };

    // Live grid render loop (TTY only): redraws the chunk matrix + progress bar
    // on the alternate screen every ~250ms while chunks run, and reads keys —
    // [q]/Ctrl-C aborts (cancels in-flight jobs, keeps completed results),
    // [j]/[k]/arrows scroll the matrix.
    std::atomic<bool> render_stop{false};
    std::thread render_thread;
    struct termios orig_term;
    bool raw_set = false;
    if (tty) {
        if (tcgetattr(STDIN_FILENO, &orig_term) == 0) {
            struct termios raw = orig_term;
            raw.c_lflag &= ~(ICANON | ECHO | ISIG);   // ISIG off → Ctrl-C arrives as a byte
            raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
            raw_set = true;
        }
        std::cout << ansi::ALT_SCREEN_ON << ansi::CURSOR_HIDE;
        std::cout.flush();
        render_thread = std::thread([&]{
            int tot = (int)jobs.size();
            // Cells per row, recomputed each loop (terminal may resize).
            auto cols_now = [&]{
                int cw = (int)(labels.empty() ? 3 : labels[0].size()) + 2;
                return std::max(1, (bt_term_cols() - 2) / cw);
            };
            while (!render_stop.load()) {
                char b[16];
                int n = (int)read(STDIN_FILENO, b, sizeof(b));
                int cols = cols_now();
                for (int i = 0; i < n; ++i) {
                    char c = b[i];
                    bool arrow = false; char dir = 0;
                    if (c == 0x1b && i + 2 < n && b[i+1] == '[') { arrow = true; dir = b[i+2]; i += 2; }

                    // [q] dismisses the grid; mid-run it also cancels in-flight jobs.
                    auto do_quit = [&]{
                        quit.store(true);
                        if (!all_done.load()) { abort.store(true); slots->wake(); }
                    };

                    if (detail_mode) {
                        // Note: arrow keys start with ESC (0x1b == 27), so only a
                        // LONE ESC (not an arrow sequence) means "back" — otherwise
                        // ↓/↑ would be misread as ESC and bounce out to the grid.
                        if (c == 'b' || (c == 27 && !arrow)) detail_mode = false;
                        else if (c == 'q' || c == 'Q' || c == 3) { detail_mode = false; do_quit(); }
                        else if (c == 'j' || (arrow && dir == 'B')) ++detail_scroll;
                        else if (c == 'k' || (arrow && dir == 'A')) --detail_scroll;
                        else if (c == 'd') detail_scroll += 10;
                        else if (c == 'u') detail_scroll -= 10;
                        continue;
                    }
                    // Grid mode
                    if (c == 'q' || c == 'Q' || c == 3 /*Ctrl-C*/) do_quit();
                    else if (c == '\r' || c == '\n') inspect_chunk(sel);
                    else if (c == 'j' || (arrow && dir == 'B')) { if (sel + cols < tot) sel += cols; }
                    else if (c == 'k' || (arrow && dir == 'A')) { if (sel - cols >= 0)   sel -= cols; }
                    else if (c == 'l' || (arrow && dir == 'C')) { if (sel + 1 < tot)     ++sel; }
                    else if (c == 'h' || (arrow && dir == 'D')) { if (sel - 1 >= 0)      --sel; }
                }
                render_grid();
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        });
    }

    std::vector<std::thread> threads;
    for (size_t i = 0; i < jobs.size(); ++i)
        threads.emplace_back(submit_and_wait, i, std::move(jobs[i]));
    for (auto& th : threads) th.join();

    if (tty) {
        // All chunks finished — keep the grid up (so failures can still be
        // inspected) until the user presses [q]. If they already quit mid-run,
        // `quit` is set and we fall through immediately.
        all_done.store(true);
        while (!quit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        render_stop = true;
        if (render_thread.joinable()) render_thread.join();
        std::cout << ansi::CURSOR_SHOW << ansi::ALT_SCREEN_OFF;
        std::cout.flush();
        if (raw_set) tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);
        if (abort.load())
            std::cout << ansi::color(ansi::BYELLOW,
                "batch-run stopped — " + std::to_string(chunks_done.load())
                + "/" + std::to_string(jobs.size()) + " chunks completed\n");
    }

    // ── Best-effort dataset sweep on remote ───────────────────────────────
    // Per-test refcounted cleanup should already have removed datasets whose
    // last consumer ran. Anything left over (chunk crashed mid-test, fetch
    // failed, etc.) gets force-removed here so disk usage doesn't grow
    // across runs. Skip if no datasets were in flight.
    if (!touched_datasets.empty()) {
        std::ostringstream rm;
        rm << "\"cd " << remote_proj << " && rm -rf";
        for (const auto& name : touched_datasets) {
            auto it = reg.datasets.find(name);
            if (it == reg.datasets.end()) continue;
            const auto& ds = it->second;
            if (!ds.path.empty()) rm << " " << ds.path;
            for (const auto& cp : ds.cache_paths)
                if (!cp.empty()) rm << " " << cp;
            rm << " .trailhead/datasets/" << name;
        }
        rm << "\"";
        std::cout << ansi::DIM << "  end-of-run sweep on " << dest.remote
                  << ansi::RESET << "\n";
        channel->ssh(rm.str(), 120);
    }

    // Clear THIS node's per-test capture dir. All of our chunks have joined and
    // every result has been pulled local, so it's safe to wipe our own subtree.
    // We must NOT touch .trailhead/results or the bare batch_results/ dir: a
    // second batch-run instance on another processor may share this workspace
    // and still be writing there — wiping wholesale would destroy its results.
    channel->ssh("\"cd " + remote_proj
                 + " && rm -rf .trailhead/batch_results/" + snode + " 2>/dev/null\"", 60);

    std::cout << "\n" << ansi::BOLD << "summary" << ansi::RESET
              << "  pass=" << total_passed.load()
              << "  fail=" << total_failed.load()
              << "  total=" << selected.size() << "\n";

    // ── CSV export ───────────────────────────────────────────────────────
    // Mirrors trailhead watch --run-all (single-run mode) so any tooling that
    // consumes trailhead_results.csv keeps working with batch-run.
    {
        ResultIndex idx = load_all_results(th_dir + "/results");
        std::string csv_path = (project_root.empty() ? "." : project_root)
                             + "/trailhead_results.csv";

        // For each selected test, the most recent result from this session.
        auto session_latest = [&](const std::string& name) -> const TestResult* {
            auto it = idx.find(name);
            if (it == idx.end()) return nullptr;
            const TestResult* best = nullptr;
            for (const auto& r : it->second) {
                if (r.started_at < session_start_ms) continue;
                if (!best || r.started_at > best->started_at) best = &r;
            }
            return best;
        };

        // Collect all timing labels reported by tests with 2+ labels — those
        // tests use per-label columns and suppress reported_ms (sum is
        // meaningless when the labels measure overlapping/nested regions).
        std::unordered_map<std::string, std::vector<std::string>> per_test_labels;
        std::vector<std::string> all_labels;
        std::unordered_set<std::string> seen_labels;
        for (const auto* t : selected) {
            const TestResult* r = session_latest(t->name);
            if (!r) continue;
            std::vector<std::string> ordered;
            std::unordered_set<std::string> seen;
            for (const auto& te : r->timings)
                if (seen.insert(te.label).second) ordered.push_back(te.label);
            per_test_labels[t->name] = ordered;
            if (ordered.size() >= 2)
                for (const auto& lbl : ordered)
                    if (seen_labels.insert(lbl).second) all_labels.push_back(lbl);
        }
        auto is_multi_label = [&](const std::string& name) {
            auto it = per_test_labels.find(name);
            return it != per_test_labels.end() && it->second.size() >= 2;
        };
        auto reported_ms_of = [](const TestResult* r) {
            double s = 0; for (const auto& te : r->timings) s += te.elapsed_ms; return s;
        };
        auto timing_val = [](const TestResult* r, const std::string& lbl) -> double {
            for (const auto& te : r->timings) if (te.label == lbl) return te.elapsed_ms;
            return 0.0;
        };

        std::ofstream csv(csv_path);
        csv << "test_name,status,reported_ms";
        for (const auto& lbl : all_labels) csv << "," << lbl;
        csv << "\n";
        csv << std::fixed << std::setprecision(3);

        for (const auto* t : selected) {
            const TestResult* r = session_latest(t->name);
            if (!r) { csv << t->name << ",NO_RESULT\n"; continue; }
            std::string status = (r->failed > 0 || r->exit_code != 0) ? "FAIL" : "PASS";
            csv << t->name << "," << status << ",";
            if (!is_multi_label(t->name)) csv << reported_ms_of(r);
            for (const auto& lbl : all_labels) {
                double v = timing_val(r, lbl);
                csv << "," << (v == 0.0 ? "" : std::to_string(v));
            }
            csv << "\n";
        }
        csv.close();
        std::cout << "Results written to: " << csv_path << "\n";
    }

    return total_failed.load() == 0 ? 0 : 1;
}

} // namespace trailhead
