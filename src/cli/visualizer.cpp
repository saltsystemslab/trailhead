#include "visualizer.hpp"
#include "../util/ansi.hpp"
#include "../util/file_util.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>

// POSIX terminal control
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

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
static const int COL_WALL   =  9;
static const int COL_WHEN   = 10;
static const int TOTAL_WIDTH = COL_NAME + COL_NODE + COL_STATUS + COL_PASS + COL_FAIL + COL_WALL + COL_WHEN + 4;

static std::string header_row() {
    using namespace ansi;
    std::ostringstream o;
    o << BOLD
      << pad("NAME",   COL_NAME) << " "
      << pad("NODE",   COL_NODE) << " "
      << pad("STATUS", COL_STATUS) << " "
      << rpad("PASS", COL_PASS) << " "
      << rpad("FAIL", COL_FAIL) << " "
      << rpad("WALL",  COL_WALL) << " "
      << pad("LAST RUN", COL_WHEN)
      << RESET;
    return o.str();
}

static std::string test_row(const TestEntry& t, const TestResult* r,
                              const Registry& reg, bool selected)
{
    using namespace ansi;
    RunStatus status = r ? result_status(*r) : RunStatus::Unknown;

    std::string name_col = pad(t.name, COL_NAME);
    std::string node_col = pad(t.node_profile, COL_NODE);
    std::string stat_col = pad(status_str(status), COL_STATUS);
    std::string pass_col = rpad(r ? std::to_string(r->passed) : "-", COL_PASS);
    std::string fail_col = rpad(r ? std::to_string(r->failed) : "-", COL_FAIL);
    std::string wall_col = rpad(r ? fs::format_duration_ms(r->wall_ms) : "-", COL_WALL);
    std::string when_col = pad(r ? fs::relative_time((time_t)(r->ended_at / 1000)) : "never", COL_WHEN);

    // Color status
    const char* stat_color = GRAY;
    switch (status) {
        case RunStatus::Pass: stat_color = BGREEN;  break;
        case RunStatus::Fail: stat_color = BRED;    break;
        default: break;
    }

    std::ostringstream row;
    if (selected) row << CYAN << BOLD;
    row << name_col << " " << dim(node_col) << " "
        << color(stat_color, stat_col) << " "
        << pass_col << " ";
    if (r && r->failed > 0) row << color(RED, fail_col);
    else row << fail_col;
    row << " " << wall_col << " " << dim(when_col);
    if (selected) row << RESET;
    return row.str();
}

// ── Detail view ───────────────────────────────────────────────────────────

static void render_detail(const TestEntry& t, const TestResult& r) {
    using namespace ansi;
    std::cout << CLEAR;
    std::cout << BOLD << "TRAILHEAD" << RESET << " — " << CYAN << t.name << RESET;
    if (!t.label.empty()) std::cout << "  " << DIM << t.label << RESET;
    std::cout << "\n" << hline(TOTAL_WIDTH) << "\n\n";

    RunStatus s = result_status(r);
    std::cout << "  Status:   " << status_badge(s) << "\n";
    std::cout << "  Host:     " << r.host << "\n";
    std::cout << "  Run by:   " << r.run_by << "\n";

    // Format timestamps
    time_t started = (time_t)(r.started_at / 1000);
    char tbuf[32];
    struct tm* tm = localtime(&started);
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", tm);
    std::cout << "  Started:  " << tbuf << "\n";
    std::cout << "  Wall:     " << fs::format_duration_ms(r.wall_ms) << "\n";
    std::cout << "  Pass:     " << color(BGREEN, std::to_string(r.passed))
              << "   Fail: " << (r.failed > 0 ? color(BRED, std::to_string(r.failed)) : "0")
              << "\n";

    if (!r.timings.empty()) {
        std::cout << "\n  " << BOLD << "Timings:" << RESET << "\n";
        for (const auto& te : r.timings) {
            std::ostringstream dur;
            dur << std::fixed << std::setprecision(1) << te.elapsed_ms << "ms";
            std::cout << "    " << pad(te.label, 20) << " " << dur.str() << "\n";
        }
    }

    if (!r.metadata.empty()) {
        std::cout << "\n  " << BOLD << "Metadata:" << RESET << "\n";
        for (const auto& [k, v] : r.metadata)
            std::cout << "    " << k << ": " << v << "\n";
    }

    if (!t.node_profile.empty()) {
        std::cout << "\n  " << BOLD << "Node profile:" << RESET << " " << t.node_profile << "\n";
    }

    std::cout << "\n" << hline(TOTAL_WIDTH) << "\n";
    std::cout << DIM << "[b] back  [q] quit" << RESET << "\n";
    std::cout.flush();
}

// ── Main watch loop ───────────────────────────────────────────────────────

int run_watch(const std::string& trailhead_dir, const Registry& reg, int interval_ms) {
    std::string results_dir = trailhead_dir + "/results";
    fs::mkdir_p(results_dir);

    enter_raw_mode();
    std::cout << ansi::CURSOR_HIDE;

    int selected = 0;
    bool detail_mode = false;
    int total = (int)reg.tests.size();

    auto render_main = [&](const ResultIndex& idx) {
        using namespace ansi;
        std::cout << CLEAR;
        std::cout << BOLD << "TRAILHEAD" << RESET
                  << "  " << DIM << trailhead_dir << RESET
                  << "  " << now_str() << "\n";
        std::cout << hline(TOTAL_WIDTH) << "\n";
        std::cout << header_row() << "\n";
        std::cout << hline(TOTAL_WIDTH) << "\n";

        for (int i = 0; i < (int)reg.tests.size(); ++i) {
            const auto& t = reg.tests[i];
            const TestResult* r = latest_result(idx, t.name);
            std::cout << test_row(t, r, reg, i == selected) << "\n";
        }

        std::cout << hline(TOTAL_WIDTH) << "\n";
        std::cout << DIM
            << "[q] quit  [r] refresh  [↑/k] up  [↓/j] down  [enter] detail  [s] run selected"
            << RESET << "\n";
        std::cout.flush();
    };

    ResultIndex idx = load_all_results(results_dir);
    render_main(idx);

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
            if (key == '\r' || key == '\n') {
                if (total > 0) { detail_mode = true; redraw = true; }
            }
            // Auto-refresh
            ++ticks;
            if (ticks >= refresh_every) {
                idx = load_all_results(results_dir);
                ticks = 0;
                redraw = true;
            }
            if (redraw) render_main(idx);
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
                const auto& t = reg.tests[selected];
                const TestResult* r = latest_result(idx, t.name);
                if (r) render_detail(t, *r);
            }
        }
    }

    restore_terminal();
    std::cout << "\n";
    return 0;
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
