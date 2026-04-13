#include "visualizer.hpp"
#include "sbatch_gen.hpp"
#include "local_run.hpp"
#include "../core/registry.hpp"
#include "../util/ansi.hpp"
#include "../util/file_util.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <functional>

// POSIX terminal control
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/ioctl.h>

namespace trailhead {

// ── Terminal helpers ──────────────────────────────────────────────────────

static struct termios s_orig_term;
static bool s_raw_mode = false;

static void enter_raw_mode() {
    if (!isatty(STDIN_FILENO)) return;
    tcgetattr(STDIN_FILENO, &s_orig_term);
    struct termios raw = s_orig_term;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    s_raw_mode = true;
}

static void restore_terminal() {
    if (s_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &s_orig_term);
        std::cout << ansi::CURSOR_SHOW;
        s_raw_mode = false;
    }
}

// Read a key with a timeout (ms). Returns -1 on timeout, char otherwise.
// Handles ESC sequences for arrow keys: returns 1000+n for arrows (U=0,D=1,R=2,L=3).
static int read_key(int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int n = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
    if (n <= 0) return -1;

    char buf[8] = {};
    ssize_t r = read(STDIN_FILENO, buf, sizeof(buf));
    if (r <= 0) return -1;
    if (r >= 3 && buf[0] == 0x1b && buf[1] == '[') {
        switch (buf[2]) {
            case 'A': return 1000; // Up
            case 'B': return 1001; // Down
            case 'C': return 1002; // Right
            case 'D': return 1003; // Left
        }
    }
    return (unsigned char)buf[0];
}

// ── Rendering helpers ─────────────────────────────────────────────────────

static std::string now_str() {
    time_t t = time(nullptr);
    char buf[32];
    struct tm* tm = localtime(&t);
    strftime(buf, sizeof(buf), "%H:%M:%S", tm);
    return buf;
}

static std::string status_badge(RunStatus s) {
    switch (s) {
        case RunStatus::Pass:    return ansi::color(ansi::BGREEN,  " PASS ");
        case RunStatus::Fail:    return ansi::color(ansi::BRED,    " FAIL ");
        case RunStatus::Running: return ansi::color(ansi::BYELLOW, " RUN  ");
        default:                 return ansi::color(ansi::GRAY,    "  --- ");
    }
}

static const int COL_NAME   = 24;
static const int COL_NODE   = 14;
static const int COL_STATUS =  8;
static const int COL_PASS   =  6;
static const int COL_FAIL   =  6;
static const int COL_TIME   = 14;
static const int COL_WALL   =  9;
static const int COL_WHEN   = 10;
static const int TOTAL_WIDTH = COL_NAME + COL_NODE + COL_STATUS + COL_PASS + COL_FAIL + COL_TIME + COL_WALL + COL_WHEN + 5;

// Format a timing value in ms: integers stay as "NNNms", large values as "N.Ns"
static std::string fmt_ms(double ms) {
    if (ms < 1000.0) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(0) << ms << "ms";
        return o.str();
    }
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << (ms / 1000.0) << "s";
    return o.str();
}

// Build a compact timing string from a result's timings, e.g. "123ms" or "12ms/456ms"
static std::string timing_str(const TestResult* r, int max_width) {
    if (!r || r->timings.empty()) return "-";
    std::string out;
    for (const auto& te : r->timings) {
        if (!out.empty()) out += "/";
        out += fmt_ms(te.elapsed_ms);
        if ((int)out.size() >= max_width - 1) { out = out.substr(0, max_width - 1); break; }
    }
    return out;
}

static std::string header_row() {
    using namespace ansi;
    std::ostringstream o;
    o << BOLD
      << pad("NAME",   COL_NAME) << " "
      << pad("NODE",   COL_NODE) << " "
      << pad("STATUS", COL_STATUS) << " "
      << rpad("PASS", COL_PASS) << " "
      << rpad("FAIL", COL_FAIL) << " "
      << rpad("TIME",  COL_TIME) << " "
      << rpad("WALL",  COL_WALL) << " "
      << pad("LAST RUN", COL_WHEN)
      << RESET;
    return o.str();
}

static std::string test_row(const TestEntry& t, const TestResult* r,
                              const Registry& reg, bool selected,
                              const std::string& live_status = "")
{
    using namespace ansi;
    RunStatus status = r ? result_status(*r) : RunStatus::Unknown;

    std::string name_col = pad(t.name, COL_NAME);
    std::string node_col = pad(t.node_profile, COL_NODE);
    std::string pass_col = rpad(r ? std::to_string(r->passed) : "-", COL_PASS);
    std::string fail_col = rpad(r ? std::to_string(r->failed) : "-", COL_FAIL);
    std::string time_col = rpad(timing_str(r, COL_TIME), COL_TIME);
    std::string wall_col = rpad(r ? fs::format_duration_ms(r->wall_ms) : "-", COL_WALL);
    std::string when_col = pad(r ? fs::relative_time((time_t)(r->ended_at / 1000)) : "never", COL_WHEN);

    // Live status overrides persisted status display
    std::string stat_col;
    const char* stat_color;
    if (!live_status.empty()) {
        stat_col   = pad(live_status, COL_STATUS);
        stat_color = BYELLOW;
    } else {
        stat_col   = pad(status_str(status), COL_STATUS);
        switch (status) {
            case RunStatus::Pass: stat_color = BGREEN; break;
            case RunStatus::Fail: stat_color = BRED;   break;
            default:              stat_color = GRAY;   break;
        }
    }

    std::ostringstream row;
    if (selected) row << CYAN << BOLD;
    row << name_col << " " << dim(node_col) << " "
        << color(stat_color, stat_col) << " "
        << pass_col << " ";
    if (r && r->failed > 0) row << color(RED, fail_col);
    else row << fail_col;
    row << " " << time_col << " " << dim(wall_col) << " " << dim(when_col);
    if (selected) row << RESET;
    return row.str();
}

// ── Detail view ───────────────────────────────────────────────────────────

static void render_detail(const TestEntry& t, const TestResult& r) {
    using namespace ansi;
    std::ostringstream o;

    o << CURSOR_HOME;
    o << BOLD << "TRAILHEAD" << RESET << " — " << CYAN << t.name << RESET;
    if (!t.label.empty()) o << "  " << DIM << t.label << RESET;
    o << "\n" << hline(TOTAL_WIDTH) << "\n\n";

    RunStatus s = result_status(r);
    o << "  Status:   " << status_badge(s) << "\n";
    o << "  Host:     " << r.host << "\n";
    o << "  Run by:   " << r.run_by << "\n";

    // Format timestamps
    time_t started = (time_t)(r.started_at / 1000);
    char tbuf[32];
    struct tm* tm = localtime(&started);
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", tm);
    o << "  Started:  " << tbuf << "\n";
    o << "  Wall:     " << fs::format_duration_ms(r.wall_ms) << "\n";
    o << "  Pass:     " << color(BGREEN, std::to_string(r.passed))
      << "   Fail: " << (r.failed > 0 ? color(BRED, std::to_string(r.failed)) : "0")
      << "\n";

    if (!r.timings.empty()) {
        o << "\n  " << BOLD << "Timings:" << RESET << "\n";
        for (const auto& te : r.timings) {
            std::ostringstream dur;
            dur << std::fixed << std::setprecision(1) << te.elapsed_ms << "ms";
            o << "    " << pad(te.label, 20) << " " << dur.str() << "\n";
        }
    }

    // Metadata — exclude internal keys shown elsewhere
    {
        bool any = false;
        for (const auto& [k, v] : r.metadata)
            if (k.empty() || k[0] != '_') any = true;
        if (any) {
            o << "\n  " << BOLD << "Metadata:" << RESET << "\n";
            for (const auto& [k, v] : r.metadata)
                if (k.empty() || k[0] != '_')
                    o << "    " << k << ": " << v << "\n";
        }
    }

    // SLURM terminal state from sacct (shown on non-COMPLETED jobs)
    {
        auto it = r.metadata.find("_sacct");
        if (it != r.metadata.end() && !it->second.empty()) {
            std::string raw = it->second;
            std::string state, rest;
            auto p = raw.find('|');
            if (p != std::string::npos) { state = raw.substr(0, p); rest = raw.substr(p+1); }
            else { state = raw; }
            bool bad = (state != "COMPLETED");
            const char* sc = bad ? BRED : BGREEN;
            o << "\n  " << BOLD << "SLURM state:" << RESET
              << "  " << color(sc, state);
            if (!rest.empty()) o << "  " << DIM << rest << RESET;
            o << "\n";
        }
    }

    // Output tail
    {
        auto it = r.metadata.find("_output_tail");
        if (it != r.metadata.end() && !it->second.empty()) {
            RunStatus s2 = result_status(r);
            const char* hdr_color = (s2 == RunStatus::Fail) ? BRED : GRAY;
            o << "\n  " << color(hdr_color, BOLD + std::string("Output:") + RESET) << "\n";
            std::istringstream ss(it->second);
            for (std::string ln; std::getline(ss, ln); )
                o << "    " << DIM << ln << RESET << "\n";
        }
    }

    if (!t.node_profile.empty())
        o << "\n  " << BOLD << "Node profile:" << RESET << " " << t.node_profile << "\n";

    o << "\n" << hline(TOTAL_WIDTH) << "\n";
    o << DIM << "[b] back  [q] quit" << RESET << "\n";
    o << ERASE_DOWN;

    std::cout << o.str();
    std::cout.flush();
}

// ── Terminal size ─────────────────────────────────────────────────────────

static int term_rows() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return (int)ws.ws_row;
    return 24; // fallback
}

// ── Wizard helpers ────────────────────────────────────────────────────────

// Open $VISUAL/$EDITOR/vim with a temp file pre-populated with `existing_cmd`.
// Comment lines (starting with '#') are stripped from the result.
// Caller must restore the terminal BEFORE calling and re-enter raw mode AFTER.
static std::string wizard_open_editor(const std::string& task_name,
                                       const std::string& existing_cmd = "")
{
    std::string tmpfile = "/tmp/trailhead_edit_" + std::to_string(getpid()) + ".sh";
    {
        std::ofstream f(tmpfile);
        f << "# Trailhead task: " << task_name << "\n"
          << "# Lines starting with '#' are ignored.\n"
          << "# Enter the command to run (passed to sh -c).\n"
          << "# Newlines, &&, pipes, etc. are all supported.\n"
          << "#\n";
        if (!existing_cmd.empty()) {
            f << existing_cmd;
            if (existing_cmd.back() != '\n') f << "\n";
        }
    }
    const char* ev = getenv("VISUAL");
    if (!ev) ev = getenv("EDITOR");
    system((std::string(ev ? ev : "vim") + " " + tmpfile).c_str());

    std::string cmd;
    {
        std::ifstream f(tmpfile);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] == '#') continue;
            if (!cmd.empty()) cmd += "\n";
            cmd += line;
        }
    }
    ::unlink(tmpfile.c_str());
    while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' '))
        cmd.pop_back();
    return cmd;
}

// ── Add-task wizard ───────────────────────────────────────────────────────

// Single-select hardware picker. Returns selected node name, or "" if cancelled/empty.
static std::string wizard_select_hardware(const Registry& reg) {
    std::vector<std::string> names;
    if (has_local_gpu()) names.push_back("local");
    for (const auto& [k, v] : reg.nodes) names.push_back(k);
    std::sort(names.begin() + (names.empty() || names[0] != "local" ? 0 : 1), names.end());
    if (names.empty()) return "";
    if (names.size() == 1) return names[0]; // auto-select if only one

    int cursor = 0;
    while (true) {
        using namespace ansi;
        std::cout << CLEAR;
        std::cout << BOLD << "TRAILHEAD" << RESET << " — Select hardware\n";
        std::cout << hline(50) << "\n\n";
        for (int i = 0; i < (int)names.size(); ++i) {
            if (i == cursor) std::cout << CYAN << BOLD;
            std::cout << (i == cursor ? " > " : "   ") << names[i];
            if (names[i] == "local") {
                std::cout << DIM << "  (this machine, sequential)" << RESET;
            } else {
                auto it = reg.nodes.find(names[i]);
                if (it != reg.nodes.end()) {
                    std::cout << DIM << "  (" << it->second.partition;
                    if (!it->second.gpu_type.empty())  std::cout << ", " << it->second.gpu_type;
                    if (!it->second.nodelist.empty())   std::cout << ", " << it->second.nodelist;
                    std::cout << ")" << RESET;
                }
            }
            if (i == cursor) std::cout << RESET;
            std::cout << "\n";
        }
        std::cout << "\n" << hline(50) << "\n";
        std::cout << DIM << "[↑/k] up  [↓/j] down  [enter] select  [ESC] cancel"
                  << RESET << "\n";
        std::cout.flush();

        int k = read_key(30000);
        if (k == 27 || k == 'q') return "";
        if (k == '\r' || k == '\n') return names[cursor];
        if (k == 1000 || k == 'k') cursor = (cursor - 1 + (int)names.size()) % (int)names.size();
        if (k == 1001 || k == 'j') cursor = (cursor + 1) % (int)names.size();
    }
}

// Multi-select node picker. Returns selected node names; empty = cancelled.
static std::vector<std::string> wizard_select_nodes(const Registry& reg) {
    std::vector<std::string> names;
    for (const auto& [k, v] : reg.nodes) names.push_back(k);
    std::sort(names.begin(), names.end());
    if (names.empty()) return {};

    std::vector<bool> checked(names.size(), true); // all selected by default
    int cursor = 0;

    while (true) {
        using namespace ansi;
        std::cout << CLEAR;
        std::cout << BOLD << "TRAILHEAD" << RESET << " — Select nodes\n";
        std::cout << hline(50) << "\n\n";
        for (int i = 0; i < (int)names.size(); ++i) {
            if (i == cursor) std::cout << CYAN << BOLD;
            std::cout << (i == cursor ? " > " : "   ");
            std::cout << (checked[i] ? "[x] " : "[ ] ");
            std::cout << names[i];
            if (i == cursor) std::cout << RESET;
            std::cout << "\n";
        }
        std::cout << "\n" << hline(50) << "\n";
        std::cout << DIM << "[↑/k] up  [↓/j] down  [space] toggle  [enter] confirm  [ESC] cancel"
                  << RESET << "\n";
        std::cout.flush();

        int k = read_key(30000);
        if (k == 27 || k == 'q') return {};
        if (k == '\r' || k == '\n') {
            std::vector<std::string> result;
            for (int i = 0; i < (int)names.size(); ++i)
                if (checked[i]) result.push_back(names[i]);
            return result;
        }
        if (k == 1000 || k == 'k') cursor = (cursor - 1 + (int)names.size()) % (int)names.size();
        if (k == 1001 || k == 'j') cursor = (cursor + 1) % (int)names.size();
        if (k == ' ') checked[cursor] = !checked[cursor];
    }
}

// Orchestrates the three-step wizard: name → editor → node select → save.
// Leaves the terminal in raw+cursor-hidden state when it returns.
static bool run_add_wizard(Registry& reg,
                            const std::string& th_dir,
                            const std::string& project_root)
{
    // ── Step 1: name ─────────────────────────────────────────────────────
    restore_terminal(); // restore canonical mode + echo for normal text input
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Add task\n\n";
    std::cout << "  Task name: ";
    std::cout.flush();

    std::string name;
    std::getline(std::cin, name);
    // Trim and sanitize
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back()  == ' ') name.pop_back();
    for (auto& c : name) if (c == ' ') c = '_';

    if (name.empty()) {
        enter_raw_mode();
        std::cout << ansi::CURSOR_HIDE;
        return false;
    }

    // ── Step 2: editor ───────────────────────────────────────────────────
    std::string cmd = wizard_open_editor(name);

    enter_raw_mode();
    std::cout << ansi::CURSOR_HIDE;

    if (cmd.empty()) {
        std::cout << ansi::CLEAR;
        std::cout << ansi::color(ansi::BRED, "No command entered — cancelled.") << "\n\n";
        std::cout << ansi::DIM << "Press any key to return..." << ansi::RESET;
        std::cout.flush();
        read_key(10000);
        return false;
    }

    // ── Step 3: node selection ────────────────────────────────────────────
    auto selected_nodes = wizard_select_nodes(reg);
    if (selected_nodes.empty()) return false;

    // ── Step 4: create entries and save ──────────────────────────────────
    // No cmake build step by default — the cmd is self-contained.
    // build_name and target stay empty so sbatch_gen skips the cmake --build line.
    for (const auto& node : selected_nodes) {
        std::string entry_name = name + "_" + node;
        // Replace if already exists
        reg.tests.erase(std::remove_if(reg.tests.begin(), reg.tests.end(),
            [&](const TestEntry& t){ return t.name == entry_name; }),
            reg.tests.end());

        TestEntry t;
        t.name         = entry_name;
        t.cmd          = cmd;
        t.node_profile = node;
        reg.tests.push_back(t);
    }

    save_registry(th_dir, reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = project_root;
    write_sbatch(th_dir, reg, opts);

    return true;
}

// ── Edit-task wizard ──────────────────────────────────────────────────────

// Edit an existing test in-place: name (pre-filled) → editor (pre-filled) → save.
static bool run_edit_wizard(Registry& reg, int test_idx,
                             const std::string& th_dir,
                             const std::string& project_root)
{
    TestEntry& test = reg.tests[test_idx];

    // ── Step 1: name (pre-filled, blank = keep current) ──────────────────
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Edit task\n\n";
    std::cout << "  Task name [" << test.name << "]: ";
    std::cout.flush();

    std::string name;
    std::getline(std::cin, name);
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back()  == ' ') name.pop_back();
    for (auto& c : name) if (c == ' ') c = '_';
    if (name.empty()) name = test.name;

    // ── Step 2: editor (pre-filled with existing cmd) ────────────────────
    std::string cmd = wizard_open_editor(name, test.cmd);

    enter_raw_mode();
    std::cout << ansi::CURSOR_HIDE;

    if (cmd.empty()) {
        std::cout << ansi::CLEAR;
        std::cout << ansi::color(ansi::BRED, "No command entered — cancelled.") << "\n\n";
        std::cout << ansi::DIM << "Press any key to return..." << ansi::RESET;
        std::cout.flush();
        read_key(10000);
        return false;
    }

    // Remove old sbatch file if name changed
    if (name != test.name) {
        std::string old_sbatch = th_dir + "/sbatch/" + test.name + ".sbatch";
        ::unlink(old_sbatch.c_str());
    }

    // ── Update entry and save ─────────────────────────────────────────────
    test.name = name;
    test.cmd  = cmd;

    save_registry(th_dir, reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = project_root;
    write_sbatch(th_dir, reg, opts);
    return true;
}

// ── Delete confirmation ────────────────────────────────────────────────────

static bool wizard_confirm_delete(Registry& reg, int test_idx,
                                   const std::string& th_dir,
                                   const std::string& project_root)
{
    const TestEntry& test = reg.tests[test_idx];

    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Delete task\n\n";
    std::cout << "  Delete " << ansi::color(ansi::BRED, test.name) << "?\n\n";
    std::cout << ansi::DIM << "  [y] Yes, delete    [n / ESC] Cancel" << ansi::RESET << "\n";
    std::cout.flush();

    while (true) {
        int k = read_key(30000);
        if (k == 'y' || k == 'Y') {
            // Remove sbatch file
            std::string sbatch = th_dir + "/sbatch/" + test.name + ".sbatch";
            ::unlink(sbatch.c_str());
            reg.tests.erase(reg.tests.begin() + test_idx);
            save_registry(th_dir, reg);
            SbatchOptions opts;
            opts.split        = true;
            opts.project_root = project_root;
            write_sbatch(th_dir, reg, opts);
            return true;
        }
        if (k == 'n' || k == 'N' || k == 27 || k == 'q' || k < 0) return false;
    }
}

// ── Main watch loop ───────────────────────────────────────────────────────

int run_watch(const std::string& trailhead_dir, Registry& reg, int interval_ms,
              std::shared_ptr<JobLog> job_log,
              std::function<void(const std::string&, const std::string&)> run_fn,
              std::string project_root, bool auto_run) {
    std::string results_dir = trailhead_dir + "/results";
    fs::mkdir_p(results_dir);

    enter_raw_mode();
    std::cout << ansi::CURSOR_HIDE;

    // Select hardware at startup if nodes are available
    std::string selected_hw;
    if (auto_run) {
        selected_hw = "local"; // non-interactive: always run locally
    } else if (!reg.nodes.empty() && run_fn) {
        selected_hw = wizard_select_hardware(reg);
    }

    int selected = 0;
    int scroll   = 0;
    bool detail_mode = false;
    int total = 0;
    std::vector<int> filtered; // indices into reg.tests, filtered by selected_hw

    // Rebuild the filtered list and update total. Call after hardware change or test add/delete.
    auto rebuild_filtered = [&]() {
        filtered.clear();
        std::unordered_set<std::string> seen_targets; // for local dedup
        for (int i = 0; i < (int)reg.tests.size(); ++i) {
            const auto& t = reg.tests[i];
            if (selected_hw == "local") {
                // Local runs each unique cmake target once — skip node-specific duplicates.
                if (!t.target.empty()) {
                    if (seen_targets.count(t.target)) continue;
                    seen_targets.insert(t.target);
                }
                filtered.push_back(i);
            } else if (selected_hw.empty() || t.node_profile.empty() ||
                       t.node_profile == selected_hw) {
                filtered.push_back(i);
            }
        }
        total = (int)filtered.size();
        if (selected >= total) selected = std::max(0, total - 1);
        if (scroll  >= total) scroll   = std::max(0, total - 1);
    };
    rebuild_filtered();

    // Fixed chrome: title(1) + hline(1) + header(1) + hline(1) + hline(1) + keys(1) = 6
    static constexpr int FIXED_ROWS = 6;
    // Log panel: blank(1) + up to MAX_LINES rows (only reserved when non-empty)
    static constexpr int LOG_PANEL_MAX = 1 + JobLog::MAX_LINES;

    auto clamp_scroll = [&](int v) {
        if (selected < scroll) scroll = selected;
        if (selected >= scroll + v) scroll = selected - v + 1;
        if (total > v && scroll > total - v) scroll = total - v;
        if (scroll < 0) scroll = 0;
    };

    auto render_main = [&](const ResultIndex& idx) {
        using namespace ansi;

        // Snapshot log before computing layout so height is stable during render
        std::vector<std::string> snap;
        if (job_log) snap = job_log->snapshot();
        int log_rows = snap.empty() ? 0 : (int)snap.size() + 1; // +1 for blank line
        // Cap log panel so tests always get at least a few rows
        log_rows = std::min(log_rows, LOG_PANEL_MAX);

        // Reserve rows for scroll indicators so clamp_scroll uses the real visible count
        int vis = std::max(term_rows() - FIXED_ROWS - log_rows - 1, 3);
        bool need_up   = scroll > 0;
        bool need_down = (scroll + vis) < total;  // preliminary check; refined below
        // Account for indicator rows in the visible count before clamping
        int vis_tests = vis - (need_up ? 1 : 0) - (need_down ? 1 : 0);
        if (vis_tests < 1) vis_tests = 1;
        clamp_scroll(vis_tests);
        // Recompute after clamping
        need_up   = scroll > 0;
        need_down = (scroll + vis_tests) < total;
        vis_tests = vis - (need_up ? 1 : 0) - (need_down ? 1 : 0);
        if (vis_tests < 1) vis_tests = 1;

        // Buffer the entire frame and write it in one shot to minimise
        // partial-frame flicker over high-latency connections (SSH).
        std::ostringstream o;

        // Move to top-left without clearing — old content is overwritten in place
        o << CURSOR_HOME;
        o << BOLD << "TRAILHEAD" << RESET
          << "  " << DIM << trailhead_dir << RESET
          << "  " << now_str();
        if (job_log && job_log->active > 0)
            o << "  " << color(BYELLOW, std::to_string(job_log->active.load())
                               + " job(s) running");
        if (!selected_hw.empty())
            o << "  " << color(CYAN, "hw:" + selected_hw);
        o << "\n";
        o << hline(TOTAL_WIDTH) << "\n";
        o << header_row() << "\n";
        o << hline(TOTAL_WIDTH) << "\n";

        if (need_up)
            o << DIM << "  ↑ " << scroll << " more above" << RESET << "\n";

        int end = std::min(scroll + vis_tests, total);
        for (int i = scroll; i < end; ++i) {
            const auto& t = reg.tests[filtered[i]];
            const TestResult* r = latest_result(idx, t.name);
            std::string live = job_log ? job_log->get_live(t.name) : "";
            o << test_row(t, r, reg, i == selected, live) << "\n";
        }

        if (need_down)
            o << DIM << "  ↓ " << (total - end) << " more below" << RESET << "\n";

        o << hline(TOTAL_WIDTH) << "\n";
        o << DIM
          << "[q] quit  [↑/k/↓/j] nav  [enter] detail  [s] submit  [R] run all  [a] add  [e] edit  [d] delete  [h] hw"
          << RESET << "\n";

        // Log panel — fixed at snapshot taken above
        if (!snap.empty()) {
            o << "\n";
            for (const auto& line : snap)
                o << DIM << "  " << line << RESET << "\n";
        }

        // Erase everything below the current frame (handles shrinking content)
        o << ERASE_DOWN;

        std::cout << o.str();
        std::cout.flush();
    };

    ResultIndex idx = load_all_results(results_dir);
    render_main(idx);

    // auto_run: submit all visible tests immediately, then watch until done
    bool submitted_all = false;
    if (auto_run && run_fn && total > 0) {
        job_log->push("auto-run: queuing " + std::to_string(total) + " tests...");
        for (int i = 0; i < total; ++i)
            run_fn(reg.tests[filtered[i]].name, selected_hw);
        submitted_all = true;
    }

    // Ticks elapsed since last full refresh
    int ticks = 0;
    int tick_ms = 100; // poll input every 100ms
    int refresh_every = interval_ms / tick_ms;

    while (true) {
        int key = read_key(tick_ms);

        if (!detail_mode) {
            bool redraw = false;
            if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) break;
            if (key == 'j' || key == 1001 /*down*/) { selected = (selected + 1) % std::max(total, 1); redraw = true; }
            if (key == 'k' || key == 1000 /*up*/)   { selected = (selected - 1 + std::max(total, 1)) % std::max(total, 1); redraw = true; }
            if (key == 'r') { idx = load_all_results(results_dir); ticks = 0; redraw = true; }
            if ((key == 'R') && total > 0 && run_fn) {
                if (selected_hw.empty() && !reg.nodes.empty()) {
                    job_log->push("No hardware selected — press [h] to choose");
                } else {
                    job_log->push("Submitting all " + std::to_string(total) + " tests...");
                    for (int i = 0; i < total; ++i)
                        run_fn(reg.tests[filtered[i]].name, selected_hw);
                }
                redraw = true;
            }
            if ((key == 'h' || key == 'H') && !reg.nodes.empty()) {
                std::string hw = wizard_select_hardware(reg);
                if (!hw.empty()) { selected_hw = hw; selected = 0; scroll = 0; }
                rebuild_filtered();
                redraw = true;
            }
            if ((key == 's' || key == 'S') && total > 0 && run_fn) {
                if (selected_hw.empty() && !reg.nodes.empty()) {
                    job_log->push("No hardware selected — press [h] to choose");
                } else {
                    run_fn(reg.tests[filtered[selected]].name, selected_hw);
                }
                redraw = true;
            }
            if (key == 'a' || key == 'A') {
                run_add_wizard(reg, trailhead_dir, project_root);
                rebuild_filtered();
                idx = load_all_results(results_dir);
                redraw = true;
            }
            if ((key == 'e' || key == 'E') && total > 0) {
                run_edit_wizard(reg, filtered[selected], trailhead_dir, project_root);
                rebuild_filtered();
                idx = load_all_results(results_dir);
                redraw = true;
            }
            if ((key == 'd' || key == 'D' || key == 127 /*backspace*/) && total > 0) {
                if (wizard_confirm_delete(reg, filtered[selected], trailhead_dir, project_root)) {
                    rebuild_filtered();
                    idx = load_all_results(results_dir);
                }
                redraw = true;
            }
            if (key == '\r' || key == '\n') {
                if (total > 0) {
                    detail_mode = true;
                    const auto& t = reg.tests[filtered[selected]];
                    const TestResult* r = latest_result(idx, t.name);
                    if (r) render_detail(t, *r);
                    // don't set redraw — detail was just rendered
                }
            }
            // Auto-refresh
            ++ticks;
            if (ticks >= refresh_every) {
                idx = load_all_results(results_dir);
                ticks = 0;
                redraw = true;
            }
            if (redraw) render_main(idx);

            // auto_run: exit once all submitted jobs have finished
            if (auto_run && submitted_all && job_log && job_log->active.load() == 0)
                break;
        } else {
            // Detail mode
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                detail_mode = false;
                render_main(idx);
            } else if (key == 'q' || key == 'Q') {
                break;
            } else {
                // Refresh detail
                idx = load_all_results(results_dir);
                const auto& t = reg.tests[filtered[selected]];
                const TestResult* r = latest_result(idx, t.name);
                if (r) render_detail(t, *r);
            }
        }
    }

    restore_terminal();
    std::cout << "\n";

    if (!auto_run) return 0;

    // ── CSV export ────────────────────────────────────────────────────────────
    // Load final results and write trailhead_results.csv in the project root.
    ResultIndex final_idx = load_all_results(results_dir);
    std::string csv_path = (project_root.empty() ? "." : project_root) + "/trailhead_results.csv";

    std::ofstream csv(csv_path);
    csv << "test_name,status,passed,failed,reported_ms,exit_code\n";
    int any_failed = 0;
    for (int i : filtered) {
        const auto& t = reg.tests[i];
        const TestResult* r = latest_result(final_idx, t.name);
        if (!r) {
            csv << t.name << ",NO_RESULT,0,0,0,0\n";
            any_failed = 1;
            continue;
        }
        std::string status = (r->failed > 0 || r->exit_code != 0) ? "FAIL" : "PASS";
        if (r->failed > 0 || r->exit_code != 0) any_failed = 1;
        double reported_ms = 0.0;
        for (const auto& te : r->timings) reported_ms += te.elapsed_ms;
        csv << std::fixed << std::setprecision(3)
            << t.name << "," << status << "," << r->passed << "," << r->failed
            << "," << reported_ms << "," << r->exit_code << "\n";
    }
    csv.close();

    std::cout << "Results written to: " << csv_path << "\n";
    // Print a quick summary table
    std::cout << "\n";
    std::cout << std::left << std::setw(32) << "TEST" << std::setw(8) << "STATUS"
              << std::setw(8) << "PASS" << std::setw(8) << "FAIL"
              << std::setw(14) << "REPORTED_MS" << "\n";
    std::cout << std::string(70, '-') << "\n";
    for (int i : filtered) {
        const auto& t = reg.tests[i];
        const TestResult* r = latest_result(final_idx, t.name);
        if (!r) {
            std::cout << std::left << std::setw(32) << t.name << std::setw(8) << "NO_RESULT\n";
            continue;
        }
        std::string status = (r->failed > 0 || r->exit_code != 0) ? "FAIL" : "PASS";
        double reported_ms = 0.0;
        for (const auto& te : r->timings) reported_ms += te.elapsed_ms;
        std::cout << std::left << std::setw(32) << t.name
                  << std::setw(8) << status
                  << std::setw(8) << r->passed
                  << std::setw(8) << r->failed
                  << std::fixed << std::setprecision(3) << reported_ms << "\n";
    }
    std::cout << "\n";

    return any_failed;
}

// ── Non-interactive status print ──────────────────────────────────────────

void print_status(const std::string& trailhead_dir, const Registry& reg) {
    using namespace ansi;
    std::string results_dir = trailhead_dir + "/results";
    auto idx = load_all_results(results_dir);

    std::cout << BOLD << "TRAILHEAD" << RESET << "  " << trailhead_dir << "\n";
    std::cout << hline(TOTAL_WIDTH) << "\n";
    std::cout << header_row() << "\n";
    std::cout << hline(TOTAL_WIDTH) << "\n";

    if (reg.tests.empty()) {
        std::cout << DIM << "  No tests registered. Use: trailhead add\n" << RESET;
    }
    for (const auto& t : reg.tests) {
        const TestResult* r = latest_result(idx, t.name);
        std::cout << test_row(t, r, reg, false) << "\n";
    }
    std::cout << hline(TOTAL_WIDTH) << "\n";
}

} // namespace trailhead
