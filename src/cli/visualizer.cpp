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
    std::cout << ansi::CURSOR_HIDE;
    std::cout.flush();
    s_raw_mode = true;
}

static void restore_terminal() {
    if (s_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &s_orig_term);
        std::cout << ansi::CURSOR_SHOW;
        std::cout.flush();
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

    // If only ESC arrived, wait briefly for the rest of an arrow-key sequence.
    // Over SSH, the 3 bytes of \x1b[A can arrive in separate reads.
    if (r == 1 && buf[0] == 0x1b) {
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        tv.tv_sec = 0; tv.tv_usec = 50000; // 50 ms
        if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0)
            r += read(STDIN_FILENO, buf + 1, sizeof(buf) - 1);
    }

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
static const int COL_NODE   = 16;  // wider to fit build + [gpu/cpu] tag
static const int COL_STATUS =  8;
static const int COL_PASS   =  6;
static const int COL_FAIL   =  6;
static const int COL_TIME   = 14;
static const int COL_WALL   =  9;
static const int COL_WHEN   = 10;
static const int TOTAL_WIDTH = COL_NAME + COL_NODE + COL_STATUS + COL_PASS + COL_FAIL + COL_TIME + COL_WALL + COL_WHEN + 5;

// Is the given hardware compatible with a test's requires_hw?
// node_name == "local" is treated as GPU-capable iff has_local_gpu().
static bool hw_compatible(const std::string& node_name, const std::string& requires_hw,
                           const Registry& reg)
{
    if (requires_hw.empty() || requires_hw == "any") return true;
    if (node_name == "local") {
        bool local_gpu = has_local_gpu();
        if (requires_hw == "gpu") return local_gpu;
        if (requires_hw == "cpu") return !local_gpu;
        return true;
    }
    auto it = reg.nodes.find(node_name);
    if (it == reg.nodes.end()) return true;
    const auto& n = it->second;
    bool has_gpu = !n.gpu_type.empty() || !n.nodelist.empty();
    if (requires_hw == "gpu") return has_gpu;
    if (requires_hw == "cpu") return !has_gpu;
    return true;
}

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
      << pad("BUILD",  COL_NODE) << " "
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
    std::string build_tag = t.build_name;
    if (!t.requires_hw.empty() && t.requires_hw != "any")
        build_tag += " [" + t.requires_hw + "]";
    std::string node_col = pad(build_tag, COL_NODE);
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

// Forward declaration (defined after render_detail)
static int term_rows();

// ── Detail view ───────────────────────────────────────────────────────────

static void render_detail(const TestEntry& t, const TestResult* r,
                           int& detail_scroll, const std::string& live_status,
                           const std::vector<std::string>& live_output = {}) {
    using namespace ansi;
    std::ostringstream o;
    int rows_written = 0;

    // Helper: append a line and count it
    auto ln = [&](const std::string& s = "") {
        o << s << "\n";
        ++rows_written;
    };

    o << CURSOR_HOME;

    // ── Header ───────────────────────────────────────────────────────────────
    {
        std::ostringstream h;
        h << BOLD << "TRAILHEAD" << RESET << " — " << CYAN << t.name << RESET;
        if (!t.label.empty()) h << "  " << DIM << t.label << RESET;
        ln(h.str());
    }
    ln(hline(TOTAL_WIDTH));

    // ── Status / metadata ─────────────────────────────────────────────────
    if (!live_status.empty()) {
        ln("  Status:   " + status_badge(RunStatus::Running) + "  "
           + std::string(DIM) + live_status + RESET);
    } else if (r) {
        ln("  Status:   " + status_badge(result_status(*r)));
    } else {
        ln("  Status:   " + status_badge(RunStatus::Unknown));
    }

    if (r) {
        ln("  Host:     " + r->host);
        ln("  Run by:   " + r->run_by);

        if (r->started_at > 0) {
            time_t started = (time_t)(r->started_at / 1000);
            char tbuf[32];
            struct tm* tm_p = localtime(&started);
            strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", tm_p);
            ln("  Started:  " + std::string(tbuf));
        }
        ln("  Wall:     " + fs::format_duration_ms(r->wall_ms));
        {
            std::ostringstream ps;
            ps << "  Pass:     " << color(BGREEN, std::to_string(r->passed))
               << "   Fail: " << (r->failed > 0 ? color(BRED, std::to_string(r->failed)) : "0");
            ln(ps.str());
        }
    }

    // ── Pipeline ──────────────────────────────────────────────────────────
    {
        struct Phase {
            std::string label;
            double ms = 0;
            enum State { Done, Running, Pending } state = Pending;
            bool failed = false;
        };
        std::vector<Phase> phases;

        if (r) {
            for (const auto& te : r->timings) {
                Phase p; p.label = te.label; p.ms = te.elapsed_ms; p.state = Phase::Done;
                phases.push_back(p);
            }
            if (r->failed > 0 && !phases.empty())
                phases.back().failed = true;
        }
        bool is_running = !live_status.empty() && live_status != "QUEUED";
        if (is_running) {
            Phase p; p.label = "running"; p.state = Phase::Running;
            phases.push_back(p);
        }

        ln();
        ln("  " + std::string(BOLD) + "Pipeline" + RESET);

        if (phases.empty()) {
            ln("  " + std::string(DIM) + "(no timing phases — test has not run yet)" + RESET);
        } else {
            // Compute column width: enough for longest label + 2 padding, min 7
            int col_w = 7;
            for (const auto& p : phases)
                col_w = std::max(col_w, (int)p.label.size() + 2);
            // Cap so the pipeline fits in an 80-col terminal
            col_w = std::min(col_w, 80 / (int)phases.size());
            if (col_w < 7) col_w = 7;

            // Helper: repeat a UTF-8 string n times
            auto rep = [](const std::string& s, int n) {
                std::string r; for (int i = 0; i < n; ++i) r += s; return r;
            };

            // Dot + connector row  (● ──── ● ──── ○)
            {
                std::ostringstream dr; dr << "  ";
                for (int i = 0; i < (int)phases.size(); ++i) {
                    const auto& p = phases[i];
                    const char* dc;
                    // dot symbol: ● (3 bytes, 1 display col)
                    // ● = \xe2\x97\x8f  ○ = \xe2\x97\x8b
                    std::string sym;
                    if (p.state == Phase::Running)  { dc = BYELLOW; sym = "\xe2\x97\x8f"; }
                    else if (p.failed)              { dc = BRED;    sym = "\xe2\x97\x8f"; }
                    else if (p.state == Phase::Done){ dc = BGREEN;  sym = "\xe2\x97\x8f"; }
                    else                            { dc = GRAY;    sym = "\xe2\x97\x8b"; }
                    dr << color(dc, sym);

                    if (i + 1 < (int)phases.size()) {
                        // connector: space + (col_w-3) × ─ + space = col_w-1 display cols
                        // ─ = \xe2\x94\x80  (3 bytes, 1 display col each)
                        bool active = (phases[i+1].state != Phase::Pending);
                        const char* cc = active ? BGREEN : GRAY;
                        std::string conn = " " + rep("\xe2\x94\x80", col_w - 3) + " ";
                        dr << color(cc, conn);
                    }
                }
                ln(dr.str());
            }

            // Label row
            {
                std::ostringstream lr; lr << "  ";
                for (int i = 0; i < (int)phases.size(); ++i) {
                    const auto& p = phases[i];
                    std::string lbl = p.label;
                    if ((int)lbl.size() > col_w) lbl = lbl.substr(0, col_w);
                    bool last = (i + 1 == (int)phases.size());
                    lr << (last ? lbl : pad(lbl, col_w));
                }
                ln(lr.str());
            }

            // Duration row (only if any durations present)
            {
                bool any_dur = false;
                for (const auto& p : phases) if (p.state == Phase::Done) { any_dur = true; break; }
                if (any_dur) {
                    std::ostringstream dur; dur << "  ";
                    for (int i = 0; i < (int)phases.size(); ++i) {
                        const auto& p = phases[i];
                        std::string d;
                        if (p.state == Phase::Done) d = fmt_ms(p.ms);
                        bool last = (i + 1 == (int)phases.size());
                        dur << color(DIM, last ? d : pad(d, col_w));
                    }
                    ln(dur.str());
                }
            }
        }
    }

    // ── Metadata (user-visible keys) ──────────────────────────────────────
    {
        bool any = false;
        if (r) for (const auto& [k, v] : r->metadata)
            if (!k.empty() && k[0] != '_') { any = true; break; }
        if (any) {
            ln();
            ln("  " + std::string(BOLD) + "Metadata:" + RESET);
            for (const auto& [k, v] : r->metadata)
                if (!k.empty() && k[0] != '_')
                    ln("    " + k + ": " + v);
        }
    }

    // ── SLURM state ───────────────────────────────────────────────────────
    if (r) {
        auto it = r->metadata.find("_sacct");
        if (it != r->metadata.end() && !it->second.empty()) {
            std::string raw = it->second, state, rest;
            auto p = raw.find('|');
            if (p != std::string::npos) { state = raw.substr(0, p); rest = raw.substr(p+1); }
            else state = raw;
            bool bad = (state != "COMPLETED");
            const char* sc = bad ? BRED : BGREEN;
            std::ostringstream sl;
            sl << "  " << BOLD << "SLURM state:" << RESET << "  " << color(sc, state);
            if (!rest.empty()) sl << "  " << DIM << rest << RESET;
            ln();
            ln(sl.str());
        }
    }

    // ── Scrollable output ─────────────────────────────────────────────────
    // When the test is queued/running, show the live streaming output.
    // When done, show the saved output tail from the result.
    {
        bool is_live = !live_status.empty();
        std::vector<std::string> out_lines;

        if (is_live) {
            out_lines = live_output; // may be empty while still QUEUED
        } else if (r) {
            auto oit = r->metadata.find("_output_tail");
            if (oit != r->metadata.end() && !oit->second.empty()) {
                std::istringstream ss(oit->second);
                for (std::string l; std::getline(ss, l); )
                    out_lines.push_back(l);
            }
        }

        if (!out_lines.empty() || is_live) {
            int total_out = (int)out_lines.size();

            // Rows available: terminal height minus header rows, "Output" label (2),
            // footer (2), and up to 2 scroll indicator lines (↑/↓).
            static constexpr int FOOTER_ROWS = 2;
            int scroll_ind = 0;
            if (!is_live) {
                if (detail_scroll > 0) ++scroll_ind;  // will show ↑ line
                int tentative = std::max(3, term_rows() - rows_written - 2 - FOOTER_ROWS - scroll_ind);
                if (total_out > detail_scroll + tentative) ++scroll_ind;  // will show ↓ line
            }
            int avail = std::max(3, term_rows() - rows_written - 2 - FOOTER_ROWS - scroll_ind);

            // When showing live output, auto-scroll to the bottom so new lines are visible.
            // When showing saved output, respect the user's scroll position.
            if (is_live)
                detail_scroll = std::max(0, total_out - avail);
            else {
                int max_scroll = std::max(0, total_out - avail);
                if (detail_scroll > max_scroll) detail_scroll = max_scroll;
                if (detail_scroll < 0)          detail_scroll = 0;
            }

            const char* hdr_color = is_live ? BYELLOW
                : (r && r->failed > 0 ? BRED : GRAY);

            ln();
            {
                std::ostringstream oh;
                oh << "  " << color(hdr_color, std::string(BOLD) + "Output" + RESET);
                if (is_live)
                    oh << "  " << std::string(DIM) << "(live)" << RESET;
                else if (total_out > avail)
                    oh << std::string(DIM) << " [" << (detail_scroll + 1)
                       << "\xe2\x80\x93"  // en-dash
                       << std::min(detail_scroll + avail, total_out)
                       << "/" << total_out << "]" << RESET;
                ln(oh.str());
            }

            if (out_lines.empty()) {
                ln("    " + std::string(DIM) + "(waiting for output...)" + RESET);
            } else {
                if (!is_live && detail_scroll > 0)
                    ln("    " + std::string(DIM) + "\xe2\x86\x91 " // ↑
                       + std::to_string(detail_scroll) + " lines above" + RESET);

                int vis_end = std::min(detail_scroll + avail, total_out);
                for (int i = detail_scroll; i < vis_end; ++i)
                    ln("    " + std::string(DIM) + out_lines[i] + RESET);

                int below = total_out - vis_end;
                if (!is_live && below > 0)
                    ln("    " + std::string(DIM) + "\xe2\x86\x93 " // ↓
                       + std::to_string(below) + " lines below" + RESET);
            }
        }
    }

    // ── Footer ────────────────────────────────────────────────────────────
    o << hline(TOTAL_WIDTH) << "\n";
    o << DIM << "[b/ESC] back  [j/" << "\xe2\x86\x93" << "] scroll down  "
      << "[k/" << "\xe2\x86\x91" << "] scroll up  [o] full output  [q] quit" << RESET << "\n";
    o << ERASE_DOWN;

    // Clear to end of line on every line to erase artifacts from longer previous frames
    std::string frame = o.str();
    std::string eol = std::string(ERASE_EOL) + "\n";
    size_t pos = 0;
    while ((pos = frame.find('\n', pos)) != std::string::npos)
        frame.replace(pos, 1, eol), pos += eol.size();
    std::cout << frame;
    std::cout.flush();
}

// ── Full output log viewer ────────────────────────────────────────────────

static void render_output_log(const std::string& test_name,
                               const std::vector<std::string>& lines,
                               int& scroll) {
    using namespace ansi;
    std::ostringstream o;
    int rows_written = 0;
    auto ln = [&](const std::string& s = "") { o << s << "\n"; ++rows_written; };

    o << CURSOR_HOME;
    ln(std::string(BOLD) + "TRAILHEAD" + RESET + " \xe2\x80\x94 full output: "
       + CYAN + test_name + RESET);
    ln(hline(TOTAL_WIDTH));

    int total = (int)lines.size();
    int avail = std::max(3, term_rows() - rows_written - 2 /*footer*/);
    int max_scroll = std::max(0, total - avail);
    if (scroll > max_scroll) scroll = max_scroll;
    if (scroll < 0)          scroll = 0;

    if (total == 0) {
        ln("  " + std::string(DIM) + "(no output file found)" + RESET);
    } else {
        if (scroll > 0)
            ln("  " + std::string(DIM) + "\xe2\x86\x91 " + std::to_string(scroll) + " lines above" + RESET);

        int vis_end = std::min(scroll + avail, total);
        for (int i = scroll; i < vis_end; ++i)
            ln("  " + std::string(DIM) + lines[i] + RESET);

        int below = total - vis_end;
        if (below > 0)
            ln("  " + std::string(DIM) + "\xe2\x86\x93 " + std::to_string(below) + " lines below" + RESET);
    }

    o << hline(TOTAL_WIDTH) << "\n";
    o << DIM << "[b/ESC] back  [j/\xe2\x86\x93] scroll down  [k/\xe2\x86\x91] scroll up  [q] quit" << RESET << "\n";
    o << ERASE_DOWN;

    std::string frame = o.str();
    std::string eol = std::string(ERASE_EOL) + "\n";
    size_t pos = 0;
    while ((pos = frame.find('\n', pos)) != std::string::npos)
        frame.replace(pos, 1, eol), pos += eol.size();
    std::cout << frame;
    std::cout.flush();
}

// ── Script preview ────────────────────────────────────────────────────────

static void render_script_preview(const std::string& test_name,
                                    const std::vector<std::string>& lines,
                                    int& scroll) {
    using namespace ansi;
    std::ostringstream o;
    int rows_written = 0;
    auto ln = [&](const std::string& s = "") { o << s << "\n"; ++rows_written; };

    o << CURSOR_HOME;
    ln(std::string(BOLD) + "TRAILHEAD" + RESET + " \xe2\x80\x94 sbatch preview: "
       + CYAN + test_name + RESET);
    ln(hline(TOTAL_WIDTH));

    int total = (int)lines.size();
    int avail = std::max(3, term_rows() - rows_written - 2 /*footer*/);
    int max_scroll = std::max(0, total - avail);
    if (scroll > max_scroll) scroll = max_scroll;
    if (scroll < 0)          scroll = 0;

    if (scroll > 0)
        ln("  " + std::string(DIM) + "\xe2\x86\x91 " + std::to_string(scroll) + " lines above" + RESET);

    int vis_end = std::min(scroll + avail, total);
    for (int i = scroll; i < vis_end; ++i) {
        const std::string& l = lines[i];
        if (!l.empty() && l[0] == '#')
            ln("  " + std::string(DIM) + l + RESET);   // comments dimmed
        else if (l.rfind("#SBATCH", 0) == 0)
            ln("  " + std::string(CYAN) + l + RESET);  // never reached but kept for clarity
        else
            ln("  " + l);
    }

    int below = total - vis_end;
    if (below > 0)
        ln("  " + std::string(DIM) + "\xe2\x86\x93 " + std::to_string(below) + " lines below" + RESET);

    o << hline(TOTAL_WIDTH) << "\n";
    o << DIM << "[b/ESC] back  [j/\xe2\x86\x93] scroll down  [k/\xe2\x86\x91] scroll up  "
      << "[s] submit  [q] quit" << RESET << "\n";
    o << ERASE_DOWN;

    std::string frame = o.str();
    std::string eol = std::string(ERASE_EOL) + "\n";
    size_t pos = 0;
    while ((pos = frame.find('\n', pos)) != std::string::npos)
        frame.replace(pos, 1, eol), pos += eol.size();
    std::cout << frame;
    std::cout.flush();
}

// ── Terminal size ─────────────────────────────────────────────────────────

static int term_rows() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return (int)ws.ws_row;
    return 24;
}

static int term_cols() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return (int)ws.ws_col;
    return 80;
}

// Return a width-char window into `s` starting at `offset`, wrapping with a gap.
// If s fits in width it is returned unchanged; otherwise a marquee slice is returned.
static std::string scroll_name(const std::string& s, int offset, int width) {
    if ((int)s.size() <= width) return s;
    static const std::string GAP = "   ";
    std::string buf = s + GAP;
    int len = (int)buf.size();
    int start = offset % len;
    std::string out;
    out.reserve(width);
    for (int i = 0; i < width; ++i)
        out += buf[(start + i) % len];
    return out;
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

// ── Hardware add/edit wizards ─────────────────────────────────────────────

static void wizard_add_hardware(Registry& reg, const std::string& th_dir,
                                  const std::string& project_root)
{
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Add hardware\n\n";

    auto read_line = [](const std::string& prompt, const std::string& def = "") -> std::string {
        if (def.empty()) std::cout << "  " << prompt << ": ";
        else             std::cout << "  " << prompt << " [" << def << "]: ";
        std::cout.flush();
        std::string s;
        std::getline(std::cin, s);
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        while (!s.empty() && s.back()  == ' ') s.pop_back();
        return s.empty() ? def : s;
    };

    std::string name = read_line("Name");
    if (name.empty()) { enter_raw_mode(); return; }

    std::string partition = read_line("Partition");

    std::string type_str = read_line("Target (1=CPU-only, 2=GPU model, 3=specific node)", "2");
    int ttype = 2;
    if (type_str == "1") ttype = 1;
    else if (type_str == "3") ttype = 3;

    std::string gpu_type, nodelist;
    if (ttype == 2)      gpu_type = read_line("GPU model (e.g. h200, rtx6000)");
    else if (ttype == 3) nodelist = read_line("Node list (e.g. d4067)");

    std::string cpus_str = read_line("CPUs per task", "1");
    int cpus = 1;
    try { cpus = std::stoi(cpus_str); } catch (...) {}

    std::string time_str   = read_line("Time limit", "01:00:00");
    std::string rsync_dest = read_line("Remote rsync destination (user@host:/path, blank = local only)");
    std::string cuda_arch  = read_line("CUDA arch (e.g. 90 for H200, 86 for RTX 3090; used as {{arch}} in configure_cmd)");

    enter_raw_mode();

    NodeProfile np;
    np.name          = name;
    np.partition     = partition;
    np.gpu_type      = gpu_type;
    np.nodelist      = nodelist;
    np.cpus_per_task = cpus;
    np.time          = time_str;
    np.rsync_dest    = rsync_dest;
    np.cuda_arch     = cuda_arch;

    reg.nodes[name] = np;
    save_registry(th_dir, reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = project_root;
    write_sbatch(th_dir, reg, opts);
}

static void wizard_edit_hardware(Registry& reg, const std::string& node_name,
                                   const std::string& th_dir, const std::string& project_root)
{
    auto it = reg.nodes.find(node_name);
    if (it == reg.nodes.end()) return;
    NodeProfile np = it->second;

    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Edit hardware: " << node_name << "\n\n";

    auto read_line = [](const std::string& prompt, const std::string& def = "") -> std::string {
        if (def.empty()) std::cout << "  " << prompt << ": ";
        else             std::cout << "  " << prompt << " [" << def << "]: ";
        std::cout.flush();
        std::string s;
        std::getline(std::cin, s);
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        while (!s.empty() && s.back()  == ' ') s.pop_back();
        return s.empty() ? def : s;
    };

    std::string name = read_line("Name", np.name);
    std::string partition = read_line("Partition", np.partition);

    int cur_type = np.gpu_type.empty() ? (np.nodelist.empty() ? 1 : 3) : 2;
    std::string type_str = read_line("Target (1=CPU-only, 2=GPU model, 3=specific node)",
                                      std::to_string(cur_type));
    int ttype = cur_type;
    if (type_str == "1") ttype = 1;
    else if (type_str == "2") ttype = 2;
    else if (type_str == "3") ttype = 3;

    std::string gpu_type, nodelist;
    if (ttype == 2)      gpu_type = read_line("GPU model", np.gpu_type);
    else if (ttype == 3) nodelist = read_line("Node list", np.nodelist);

    std::string cpus_str  = read_line("CPUs per task", std::to_string(np.cpus_per_task));
    int cpus = np.cpus_per_task;
    try { cpus = std::stoi(cpus_str); } catch (...) {}

    std::string time_str   = read_line("Time limit", np.time);
    std::string rsync_dest = read_line("Remote rsync destination (user@host:/path)", np.rsync_dest);
    std::string cuda_arch  = read_line("CUDA arch (e.g. 90 for H200; used as {{arch}} in configure_cmd)", np.cuda_arch);

    enter_raw_mode();

    if (name != node_name) reg.nodes.erase(node_name);
    np.name          = name;
    np.partition     = partition;
    np.gpu_type      = (ttype == 2) ? gpu_type : "";
    np.nodelist      = (ttype == 3) ? nodelist : "";
    np.cpus_per_task = cpus;
    np.time          = time_str;
    np.rsync_dest    = rsync_dest;
    np.cuda_arch     = cuda_arch;
    reg.nodes[name]  = np;

    save_registry(th_dir, reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = project_root;
    write_sbatch(th_dir, reg, opts);
}

// ── Add-task wizard ───────────────────────────────────────────────────────

// Single-select hardware picker. Returns selected node name, or "" if cancelled.
// Supports [a] add and [e] edit hardware in-place.
static std::string wizard_select_hardware(Registry& reg,
                                           const std::string& th_dir,
                                           const std::string& project_root) {
    auto rebuild_names = [&]() {
        std::vector<std::string> n;
        if (has_local_gpu()) n.push_back("local");
        for (const auto& [k, v] : reg.nodes) n.push_back(k);
        int start = (!n.empty() && n[0] == "local") ? 1 : 0;
        std::sort(n.begin() + start, n.end());
        return n;
    };

    auto names = rebuild_names();
    int cursor = 0;
    int name_scroll = 0;
    int scroll_ticks = 0;
    while (true) {
        using namespace ansi;
        int name_w = std::max(10, term_cols() / 3);
        {
            std::ostringstream o;
            o << CLEAR;
            o << BOLD << "TRAILHEAD" << RESET << " — Select hardware\n";
            o << hline(50) << "\n\n";
            if (names.empty()) {
                o << DIM << "  No hardware configured. Press [a] to add." << RESET << "\n";
            }
            for (int i = 0; i < (int)names.size(); ++i) {
                if (i == cursor) o << CYAN << BOLD;
                std::string disp = (i == cursor) ? scroll_name(names[i], name_scroll, name_w)
                                                 : names[i].substr(0, name_w);
                o << (i == cursor ? " > " : "   ") << disp;
                if (names[i] == "local") {
                    o << DIM << "  (this machine, sequential)" << RESET;
                } else {
                    auto nit = reg.nodes.find(names[i]);
                    if (nit != reg.nodes.end()) {
                        const auto& n = nit->second;
                        o << DIM << "  (" << n.partition;
                        if      (!n.gpu_type.empty())  o << ", gpu:" << n.gpu_type;
                        else if (!n.nodelist.empty())   o << ", node:" << n.nodelist;
                        else                            o << ", CPU-only";
                        if (!n.rsync_dest.empty())      o << ", " << n.rsync_dest;
                        o << ")" << RESET;
                    }
                }
                if (i == cursor) o << RESET;
                o << "\n";
            }
            o << "\n" << hline(50) << "\n";
            o << DIM << "[↑/k] up  [↓/j] down  [enter] select  [a] add  [e] edit  [x] delete  [ESC] cancel"
              << RESET << "\n";
            std::cout << o.str();
            std::cout.flush();
        }

        int k = read_key(50);
        if (k == -1) {
            if (++scroll_ticks < 6) continue;
            ++name_scroll; scroll_ticks = 0;
            continue;
        }
        scroll_ticks = 0;
        if (k == 27 || k == 'q') return "";
        if ((k == '\r' || k == '\n') && !names.empty()) return names[cursor];
        if (k == 1000 || k == 'k') {
            if (!names.empty()) cursor = (cursor - 1 + (int)names.size()) % (int)names.size();
            name_scroll = 0;
        }
        if (k == 1001 || k == 'j') {
            if (!names.empty()) cursor = (cursor + 1) % (int)names.size();
            name_scroll = 0;
        }
        if (k == 'a' || k == 'A') {
            wizard_add_hardware(reg, th_dir, project_root);
            names = rebuild_names();
            if (cursor >= (int)names.size()) cursor = std::max(0, (int)names.size() - 1);
        }
        if ((k == 'e' || k == 'E') && !names.empty()
            && cursor < (int)names.size() && names[cursor] != "local") {
            wizard_edit_hardware(reg, names[cursor], th_dir, project_root);
            names = rebuild_names();
            if (cursor >= (int)names.size()) cursor = std::max(0, (int)names.size() - 1);
        }
        if ((k == 'x' || k == 'X') && !names.empty()
            && cursor < (int)names.size() && names[cursor] != "local") {
            std::string target = names[cursor];
            // Confirm deletion
            restore_terminal();
            std::cout << ansi::YELLOW << "Delete hardware profile '" << target << "'? [y/N] " << ansi::RESET;
            std::cout.flush();
            std::string ans;
            std::getline(std::cin, ans);
            enter_raw_mode();
            if (ans == "y" || ans == "Y") {
                reg.nodes.erase(target);
                save_registry(th_dir, reg);
                SbatchOptions opts;
                opts.split        = true;
                opts.project_root = project_root;
                write_sbatch(th_dir, reg, opts);
            }
            names = rebuild_names();
            if (cursor >= (int)names.size()) cursor = std::max(0, (int)names.size() - 1);
        }
    }
}

// Inline build config creation form.
// Saves the new config into reg and writes registry to th_dir.
// Returns the new config name, or "" if cancelled.
static std::string wizard_create_build(Registry& reg, const std::string& th_dir) {
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — New build config\n\n";

    auto read_line = [](const std::string& prompt, const std::string& def = "") {
        if (def.empty()) std::cout << "  " << prompt << ": ";
        else             std::cout << "  " << prompt << " [" << def << "]: ";
        std::cout.flush();
        std::string s;
        std::getline(std::cin, s);
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        while (!s.empty() && s.back()  == ' ') s.pop_back();
        return s.empty() ? def : s;
    };

    std::string name      = read_line("Config name", "release");
    if (name.empty()) { enter_raw_mode(); return ""; }

    std::string dir       = read_line("Build directory", "build");
    std::string def_cfg   = "cmake -B " + dir + " -DCMAKE_BUILD_TYPE=Release";
    std::string configure = read_line("Configure command", def_cfg);
    std::string def_bld   = "cmake --build " + dir + " -j$(nproc)";
    std::string build_cmd = read_line("Build command", def_bld);
    std::string rsync     = read_line("Remote destination (user@host:/path, blank = local only)");

    enter_raw_mode();

    BuildConfig bc;
    bc.name          = name;
    bc.dir           = dir;
    bc.configure_cmd = configure;
    bc.build_cmd     = build_cmd;
    bc.rsync_dest    = rsync;

    reg.builds[name] = bc;
    save_registry(th_dir, reg);
    return name;
}

// Single-select build config picker. Returns selected build name, or "" for none/cancelled.
// [n] opens an inline form to create a new config on the spot.
static std::string wizard_select_build(Registry& reg, const std::string& th_dir) {
    auto rebuild_names = [&]() {
        std::vector<std::string> n;
        n.push_back("(none)");
        for (const auto& [k, v] : reg.builds) n.push_back(k);
        std::sort(n.begin() + 1, n.end());
        return n;
    };

    auto names = rebuild_names();
    int cursor = 0;
    int name_scroll = 0;
    int scroll_ticks = 0;
    while (true) {
        using namespace ansi;
        int name_w = std::max(10, term_cols() / 3);
        {
            std::ostringstream o;
            o << CLEAR;
            o << BOLD << "TRAILHEAD" << RESET << " — Link build config\n";
            o << hline(60) << "\n\n";
            for (int i = 0; i < (int)names.size(); ++i) {
                if (i == cursor) o << CYAN << BOLD;
                std::string disp = (i == cursor) ? scroll_name(names[i], name_scroll, name_w)
                                                 : names[i].substr(0, name_w);
                o << (i == cursor ? " > " : "   ") << disp;
                if (i > 0) {
                    auto it = reg.builds.find(names[i]);
                    if (it != reg.builds.end()) {
                        const auto& bc = it->second;
                        if (!bc.configure_cmd.empty())
                            o << DIM << "  (" << bc.configure_cmd << ")" << RESET;
                        if (!bc.rsync_dest.empty())
                            o << DIM << "  \xe2\x86\x92 " << bc.rsync_dest << RESET;
                    }
                }
                if (i == cursor) o << RESET;
                o << "\n";
            }
            o << "\n" << hline(60) << "\n";
            o << DIM << "[↑/k] up  [↓/j] down  [enter] select  [n] new config  [ESC] cancel"
              << RESET << "\n";
            std::cout << o.str();
            std::cout.flush();
        }

        int k = read_key(50);
        if (k == -1) {
            if (++scroll_ticks < 6) continue;
            ++name_scroll; scroll_ticks = 0;
            continue;
        }
        scroll_ticks = 0;
        if (k == 27 || k == 'q') return "";
        if (k == '\r' || k == '\n') return (cursor == 0) ? "" : names[cursor];
        if (k == 1000 || k == 'k') { cursor = (cursor - 1 + (int)names.size()) % (int)names.size(); name_scroll = 0; }
        if (k == 1001 || k == 'j') { cursor = (cursor + 1) % (int)names.size(); name_scroll = 0; }
        if (k == 'n' || k == 'N') {
            std::string new_name = wizard_create_build(reg, th_dir);
            names = rebuild_names();
            if (!new_name.empty()) {
                // Position cursor on the newly created config
                for (int i = 0; i < (int)names.size(); ++i)
                    if (names[i] == new_name) { cursor = i; break; }
            }
        }
    }
}

// ── Sub-registry helpers ───────────────────────────────────────────────────

// Last path component of a sub_dir path (e.g. "a/b/c" → "c").
static std::string sub_name_from_dir(const std::string& sub_dir) {
    auto slash = sub_dir.rfind('/');
    return (slash != std::string::npos) ? sub_dir.substr(slash + 1) : sub_dir;
}

// Strip "subname/" prefix from a merged test name to recover the original name.
static std::string strip_sub_prefix(const std::string& name, const std::string& sub_dir) {
    std::string prefix = sub_name_from_dir(sub_dir) + "/";
    if (name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix)
        return name.substr(prefix.size());
    return name;
}

// Reload the parent registry and re-merge sub-registries into reg.
static void reload_merged(Registry& reg, const std::string& th_dir,
                           const std::string& project_root)
{
    auto fresh = load_registry(th_dir);
    if (fresh) {
        reg = *fresh;
        merge_sub_registries(reg, project_root);
    }
}

// ── Add-task wizard ───────────────────────────────────────────────────────

// Orchestrates the wizard: destination → name → editor → hw → build → target → workdir → save.
// Leaves the terminal in raw+cursor-hidden state when it returns.
static bool run_add_wizard(Registry& reg,
                            const std::string& th_dir,
                            const std::string& project_root)
{
    // ── Step 0: destination (only shown when sub-registries are declared) ─
    std::string dest_sub_dir;   // "" = parent, "gunrock" = sub-registry
    std::string eff_th_dir       = th_dir;
    std::string eff_project_root = project_root;
    Registry sub_reg_storage;
    Registry* dest_reg = &reg;  // points to the registry we'll add into

    if (!reg.sub_registries.empty()) {
        // Build list: "(this project)" first, then each sub-registry
        std::vector<std::string> choices;
        choices.push_back("");  // "" = parent
        for (const auto& s : reg.sub_registries) choices.push_back(s);

        int cursor = 0;
        int name_scroll = 0;
        int scroll_ticks = 0;
        bool cancelled = false;
        while (true) {
            using namespace ansi;
            int name_w = std::max(10, term_cols() / 2);
            {
                std::ostringstream o;
                o << CLEAR;
                o << BOLD << "TRAILHEAD" << RESET << " — Add to which registry?\n";
                o << hline(50) << "\n\n";
                for (int i = 0; i < (int)choices.size(); ++i) {
                    if (i == cursor) o << CYAN << BOLD;
                    o << (i == cursor ? " > " : "   ");
                    if (choices[i].empty()) {
                        o << "(this project)";
                    } else {
                        std::string disp = (i == cursor) ? scroll_name(choices[i], name_scroll, name_w)
                                                         : choices[i].substr(0, name_w);
                        o << disp;
                    }
                    if (i == cursor) o << RESET;
                    o << "\n";
                }
                o << "\n" << hline(50) << "\n";
                o << DIM << "[↑/k] up  [↓/j] down  [enter] select  [ESC] cancel"
                  << RESET << "\n";
                std::cout << o.str();
                std::cout.flush();
            }

            int k = read_key(50);
            if (k == -1) {
                if (++scroll_ticks < 6) continue;
                ++name_scroll; scroll_ticks = 0;
                continue;
            }
            scroll_ticks = 0;
            if (k == 27 || k == 'q') { cancelled = true; break; }
            if (k == '\r' || k == '\n') { dest_sub_dir = choices[cursor]; break; }
            if ((k == 1000 || k == 'k') && cursor > 0) { --cursor; name_scroll = 0; }
            if ((k == 1001 || k == 'j') && cursor + 1 < (int)choices.size()) { ++cursor; name_scroll = 0; }
        }
        if (cancelled) { enter_raw_mode(); return false; }

        if (!dest_sub_dir.empty()) {
            eff_th_dir       = project_root + "/" + dest_sub_dir + "/.trailhead";
            eff_project_root = project_root + "/" + dest_sub_dir;
            auto loaded = load_registry(eff_th_dir);
            if (!loaded) { enter_raw_mode(); return false; }
            sub_reg_storage = *loaded;
            dest_reg = &sub_reg_storage;
        }
    }

    // ── Step 1: name ─────────────────────────────────────────────────────
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Add task";
    if (!dest_sub_dir.empty())
        std::cout << ansi::DIM << "  → " << dest_sub_dir << ansi::RESET;
    std::cout << "\n\n";
    std::cout << "  Task name: ";
    std::cout.flush();

    std::string name;
    std::getline(std::cin, name);
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back()  == ' ') name.pop_back();
    for (auto& c : name) if (c == ' ') c = '_';

    if (name.empty()) {
        enter_raw_mode();
        return false;
    }

    // ── Step 2: editor ───────────────────────────────────────────────────
    std::string cmd = wizard_open_editor(name);

    enter_raw_mode();

    if (cmd.empty()) {
        std::cout << ansi::CLEAR;
        std::cout << ansi::color(ansi::BRED, "No command entered — cancelled.") << "\n\n";
        std::cout << ansi::DIM << "Press any key to return..." << ansi::RESET;
        std::cout.flush();
        read_key(10000);
        return false;
    }

    // ── Step 3: hardware requirement ─────────────────────────────────────
    std::string requires_hw;
    {
        using namespace ansi;
        std::cout << CLEAR;
        std::cout << BOLD << "TRAILHEAD" << RESET << " — Hardware requirement\n";
        std::cout << hline(50) << "\n\n";
        std::cout << "  [1] Any (default)\n";
        std::cout << "  [2] GPU required\n";
        std::cout << "  [3] CPU only\n\n";
        std::cout << hline(50) << "\n";
        std::cout << DIM << "[1/2/3] select  [ESC] any (default)" << RESET << "\n";
        std::cout.flush();
        int k = read_key(30000);
        if (k == '2') requires_hw = "gpu";
        else if (k == '3') requires_hw = "cpu";
    }

    // ── Step 4: build config (from destination registry, [n] creates new) ──
    std::string selected_build = wizard_select_build(*dest_reg, eff_th_dir);

    // ── Step 5: cmake target (only when a build is selected) ─────────────
    std::string target;
    if (!selected_build.empty()) {
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — CMake target\n\n";
        std::cout << "  cmake --build <dir> --target [" << name << "]: ";
        std::cout.flush();
        std::string t_in;
        std::getline(std::cin, t_in);
        while (!t_in.empty() && t_in.front() == ' ') t_in.erase(t_in.begin());
        while (!t_in.empty() && t_in.back()  == ' ') t_in.pop_back();
        target = t_in.empty() ? name : t_in;
        enter_raw_mode();
    }

    // ── Step 6: working directory ─────────────────────────────────────────
    std::string workdir = ".";
    {
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Working directory\n\n";
        std::cout << "  Run test from directory [.]: ";
        std::cout.flush();
        std::string wd_in;
        std::getline(std::cin, wd_in);
        while (!wd_in.empty() && wd_in.front() == ' ') wd_in.erase(wd_in.begin());
        while (!wd_in.empty() && wd_in.back()  == ' ') wd_in.pop_back();
        if (!wd_in.empty()) workdir = wd_in;
        enter_raw_mode();
    }

    // ── Step 7: create entry and save ─────────────────────────────────────
    dest_reg->tests.erase(std::remove_if(dest_reg->tests.begin(), dest_reg->tests.end(),
        [&](const TestEntry& t){ return t.name == name; }),
        dest_reg->tests.end());

    TestEntry t;
    t.name        = name;
    t.cmd         = cmd;
    t.build_name  = selected_build;
    t.requires_hw = requires_hw;
    t.target      = target;
    t.workdir     = workdir;
    dest_reg->tests.push_back(t);

    save_registry(eff_th_dir, *dest_reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = eff_project_root;
    write_sbatch(eff_th_dir, *dest_reg, opts);

    if (!dest_sub_dir.empty())
        reload_merged(reg, th_dir, project_root);

    return true;
}

// ── Edit-task wizard ──────────────────────────────────────────────────────

// Edit an existing test in-place. For sub-registry tests, edits are written
// to the sub-registry's own registry and the parent view is reloaded.
static bool run_edit_wizard(Registry& reg, int test_idx,
                             const std::string& th_dir,
                             const std::string& project_root)
{
    const std::string test_sub_dir = reg.tests[test_idx].sub_dir;

    // Resolve which registry to edit and what the un-prefixed test name is
    std::string eff_th_dir = th_dir;
    std::string eff_project_root = project_root;
    Registry sub_reg;
    Registry* work_reg = &reg;
    int work_idx = test_idx;

    if (!test_sub_dir.empty()) {
        eff_th_dir       = project_root + "/" + test_sub_dir + "/.trailhead";
        eff_project_root = project_root + "/" + test_sub_dir;
        auto loaded = load_registry(eff_th_dir);
        if (!loaded) { enter_raw_mode(); return false; }
        sub_reg  = *loaded;
        work_reg = &sub_reg;
        std::string actual_name = strip_sub_prefix(reg.tests[test_idx].name, test_sub_dir);
        work_idx = -1;
        for (int i = 0; i < (int)sub_reg.tests.size(); ++i) {
            if (sub_reg.tests[i].name == actual_name) { work_idx = i; break; }
        }
        if (work_idx < 0) { enter_raw_mode(); return false; }
    }

    TestEntry& test = work_reg->tests[work_idx];

    // ── Step 1: name (pre-filled, blank = keep current) ──────────────────
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Edit task";
    if (!test_sub_dir.empty())
        std::cout << ansi::DIM << "  (sub-registry: " << test_sub_dir << ")" << ansi::RESET;
    std::cout << "\n\n";
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

    if (cmd.empty()) {
        std::cout << ansi::CLEAR;
        std::cout << ansi::color(ansi::BRED, "No command entered — cancelled.") << "\n\n";
        std::cout << ansi::DIM << "Press any key to return..." << ansi::RESET;
        std::cout.flush();
        read_key(10000);
        return false;
    }

    // ── Step 3: hardware requirement (pre-filled) ────────────────────────
    std::string requires_hw = test.requires_hw;
    {
        using namespace ansi;
        std::cout << CLEAR;
        std::cout << BOLD << "TRAILHEAD" << RESET << " — Hardware requirement\n";
        std::cout << hline(50) << "\n\n";
        std::cout << "  [1] Any" << (requires_hw.empty() || requires_hw == "any" ? "  ← current" : "") << "\n";
        std::cout << "  [2] GPU required" << (requires_hw == "gpu" ? "  ← current" : "") << "\n";
        std::cout << "  [3] CPU only"     << (requires_hw == "cpu" ? "  ← current" : "") << "\n\n";
        std::cout << hline(50) << "\n";
        std::cout << DIM << "[1/2/3] change  [ESC/enter] keep current" << RESET << "\n";
        std::cout.flush();
        int k = read_key(30000);
        if      (k == '1') requires_hw = "";
        else if (k == '2') requires_hw = "gpu";
        else if (k == '3') requires_hw = "cpu";
        // else: keep existing value
    }

    // ── Step 4: build config (pre-filled, [n] creates new) ───────────────
    std::string selected_build = wizard_select_build(*work_reg, eff_th_dir);
    // Empty string from wizard means "(none)" was chosen, not cancelled.
    // Re-apply existing build_name if user just pressed ESC (q).
    // Since wizard returns "" for both cancel and (none), we keep whatever was chosen.

    // ── Step 5: cmake target (only when build selected, pre-filled) ──────
    std::string target = test.target;
    if (!selected_build.empty()) {
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — CMake target\n\n";
        std::string def_target = target.empty() ? name : target;
        std::cout << "  cmake --build <dir> --target [" << def_target << "]: ";
        std::cout.flush();
        std::string t_in;
        std::getline(std::cin, t_in);
        while (!t_in.empty() && t_in.front() == ' ') t_in.erase(t_in.begin());
        while (!t_in.empty() && t_in.back()  == ' ') t_in.pop_back();
        target = t_in.empty() ? def_target : t_in;
        enter_raw_mode();
    } else {
        target = "";
    }

    // ── Step 6: working directory (pre-filled) ────────────────────────────
    std::string workdir = test.workdir;
    {
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Working directory\n\n";
        std::cout << "  Run test from directory [" << (workdir.empty() ? "." : workdir) << "]: ";
        std::cout.flush();
        std::string wd_in;
        std::getline(std::cin, wd_in);
        while (!wd_in.empty() && wd_in.front() == ' ') wd_in.erase(wd_in.begin());
        while (!wd_in.empty() && wd_in.back()  == ' ') wd_in.pop_back();
        if (!wd_in.empty()) workdir = wd_in;
        enter_raw_mode();
    }

    // Remove old sbatch file if name changed
    if (name != test.name) {
        std::string old_sbatch = eff_th_dir + "/sbatch/" + test.name + ".sbatch";
        ::unlink(old_sbatch.c_str());
    }

    // ── Update entry and save ─────────────────────────────────────────────
    test.name        = name;
    test.cmd         = cmd;
    test.requires_hw = requires_hw;
    test.build_name  = selected_build;
    test.target      = target;
    test.workdir     = workdir;

    save_registry(eff_th_dir, *work_reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = eff_project_root;
    write_sbatch(eff_th_dir, *work_reg, opts);

    if (!test_sub_dir.empty())
        reload_merged(reg, th_dir, project_root);
    return true;
}

// ── Delete confirmation ────────────────────────────────────────────────────

static bool wizard_confirm_delete(Registry& reg, int test_idx,
                                   const std::string& th_dir,
                                   const std::string& project_root)
{
    const std::string display_name  = reg.tests[test_idx].name;
    const std::string test_sub_dir  = reg.tests[test_idx].sub_dir;

    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Delete task\n\n";
    std::cout << "  Delete " << ansi::color(ansi::BRED, display_name) << "?\n";
    if (!test_sub_dir.empty())
        std::cout << "  " << ansi::DIM << "(from sub-registry: " << test_sub_dir << ")"
                  << ansi::RESET << "\n";
    std::cout << "\n";
    std::cout << ansi::DIM << "  [y] Yes, delete    [n / ESC] Cancel" << ansi::RESET << "\n";
    std::cout.flush();

    while (true) {
        int k = read_key(30000);
        if (k == 'y' || k == 'Y') {
            if (!test_sub_dir.empty()) {
                std::string sub_th_dir   = project_root + "/" + test_sub_dir + "/.trailhead";
                std::string actual_name  = strip_sub_prefix(display_name, test_sub_dir);
                auto loaded = load_registry(sub_th_dir);
                if (loaded) {
                    auto& ts = loaded->tests;
                    ts.erase(std::remove_if(ts.begin(), ts.end(),
                        [&](const TestEntry& t){ return t.name == actual_name; }), ts.end());
                    save_registry(sub_th_dir, *loaded);
                }
                reload_merged(reg, th_dir, project_root);
            } else {
                std::string sbatch = th_dir + "/sbatch/" + display_name + ".sbatch";
                ::unlink(sbatch.c_str());
                reg.tests.erase(reg.tests.begin() + test_idx);
                save_registry(th_dir, reg);
                SbatchOptions opts;
                opts.split        = true;
                opts.project_root = project_root;
                write_sbatch(th_dir, reg, opts);
            }
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

    std::cout << ansi::ALT_SCREEN_ON;
    std::cout.flush();
    enter_raw_mode();

    // Select hardware at startup if nodes are available
    std::string selected_hw;
    if (auto_run) {
        selected_hw = "local"; // non-interactive: always run locally
    } else if (!reg.nodes.empty() && run_fn) {
        selected_hw = wizard_select_hardware(reg, trailhead_dir, project_root);
    }

    int selected      = 0;
    int scroll        = 0;
    bool detail_mode  = false;
    int detail_scroll = 0;
    int detail_ticks  = 0;
    bool preview_mode = false;
    int preview_scroll = 0;
    std::vector<std::string> preview_lines;
    bool output_mode = false;
    int output_scroll = 0;
    std::vector<std::string> output_lines;
    int total = 0;
    std::vector<int> filtered; // indices into reg.tests, filtered by selected_hw

    // Rebuild the filtered list and update total. Call after hardware change or test add/delete.
    auto rebuild_filtered = [&]() {
        filtered.clear();
        std::unordered_set<std::string> seen_targets; // for local dedup
        for (int i = 0; i < (int)reg.tests.size(); ++i) {
            const auto& t = reg.tests[i];
            if (selected_hw == "local") {
                // Local runs each unique cmake target once.
                if (!t.target.empty()) {
                    if (seen_targets.count(t.target)) continue;
                    seen_targets.insert(t.target);
                }
            }
            filtered.push_back(i);
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
          << "[q] quit  [↑/k/↓/j] nav  [enter] detail  [s] submit  [p] preview  [R] run all  [r] refresh  [a] add  [e] edit  [d] delete  [h] hw"
          << RESET << "\n";

        // Log panel — fixed at snapshot taken above
        if (!snap.empty()) {
            o << "\n";
            for (const auto& line : snap)
                o << DIM << "  " << line << RESET << "\n";
        }

        // Erase everything below the current frame (handles shrinking content)
        o << ERASE_DOWN;

        // Clear to end of line on every line to erase artifacts from wider previous frames
        std::string frame = o.str();
        std::string eol = std::string(ERASE_EOL) + "\n";
        size_t pos = 0;
        while ((pos = frame.find('\n', pos)) != std::string::npos)
            frame.replace(pos, 1, eol), pos += eol.size();
        std::cout << frame;
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

        if (!detail_mode && !preview_mode && !output_mode) {
            bool redraw = false;
            if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) break;
            if (key == 'j' || key == 1001 /*down*/) { selected = (selected + 1) % std::max(total, 1); redraw = true; }
            if (key == 'k' || key == 1000 /*up*/)   { selected = (selected - 1 + std::max(total, 1)) % std::max(total, 1); redraw = true; }
            if (key == 'r') { idx = load_all_results(results_dir); ticks = 0; redraw = true; }
            if ((key == 'R') && total > 0 && run_fn) {
                if (selected_hw.empty() && !reg.nodes.empty()) {
                    job_log->push("No hardware selected — press [h] to choose");
                } else {
                    int incompatible = 0;
                    for (int i = 0; i < total; ++i) {
                        const auto& t = reg.tests[filtered[i]];
                        if (!t.requires_hw.empty() && t.requires_hw != "any" &&
                            !selected_hw.empty() &&
                            !hw_compatible(selected_hw, t.requires_hw, reg))
                            ++incompatible;
                    }
                    if (incompatible > 0)
                        job_log->push("[warn] " + std::to_string(incompatible)
                                      + " test(s) may be incompatible with " + selected_hw);
                    job_log->push("Submitting all " + std::to_string(total) + " tests...");
                    for (int i = 0; i < total; ++i)
                        run_fn(reg.tests[filtered[i]].name, selected_hw);
                }
                redraw = true;
            }
            if ((key == 'h' || key == 'H') && run_fn) {
                std::string hw = wizard_select_hardware(reg, trailhead_dir, project_root);
                if (!hw.empty()) { selected_hw = hw; selected = 0; scroll = 0; }
                rebuild_filtered();
                redraw = true;
            }
            if ((key == 's' || key == 'S') && total > 0 && run_fn) {
                if (selected_hw.empty() && !reg.nodes.empty()) {
                    job_log->push("No hardware selected — press [h] to choose");
                } else {
                    const auto& t = reg.tests[filtered[selected]];
                    if (!t.requires_hw.empty() && t.requires_hw != "any" &&
                        !selected_hw.empty() &&
                        !hw_compatible(selected_hw, t.requires_hw, reg))
                        job_log->push("[warn] " + t.name + " requires " + t.requires_hw
                                      + " but " + selected_hw + " may not provide it");
                    run_fn(t.name, selected_hw);
                }
                redraw = true;
            }
            if ((key == 'p' || key == 'P') && total > 0) {
                const auto& t = reg.tests[filtered[selected]];
                SbatchOptions opts;
                opts.node_name    = (selected_hw == "local") ? "" : selected_hw;
                opts.project_root = project_root;

                std::string script;
                if (selected_hw.empty() || selected_hw == "local") {
                    std::ostringstream ss;
                    ss << "#!/bin/bash\n# Local run — no sbatch\n\n";
                    if (!t.build_name.empty() && !t.target.empty()) {
                        auto bit = reg.builds.find(t.build_name);
                        if (bit != reg.builds.end()) {
                            const auto& bc = bit->second;
                            std::string raw = bc.dir.empty() ? "build" : bc.dir;
                            std::string eff = bc.sub_dir.empty() ? raw : bc.sub_dir + "/" + raw;
                            if (!bc.configure_cmd.empty())
                                ss << "# configure (if CMakeCache.txt absent)\n"
                                   << bc.configure_cmd << "\n\n";
                            ss << "cmake --build " << eff << " --target " << t.target << "\n\n";
                        }
                    }
                    std::string wd = (t.workdir.empty() || t.workdir == ".") ? "" : t.workdir;
                    if (!wd.empty()) ss << "cd " << wd << "\n";
                    ss << t.cmd << "\n";
                    script = ss.str();
                } else {
                    script = generate_test_script(t, selected_hw, reg, opts);
                }

                preview_lines.clear();
                std::istringstream ss(script);
                for (std::string l; std::getline(ss, l); )
                    preview_lines.push_back(l);

                preview_mode = true;
                preview_scroll = 0;
                render_script_preview(t.name, preview_lines, preview_scroll);
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
                    detail_scroll = 0;
                    detail_ticks  = 0;
                    const auto& t = reg.tests[filtered[selected]];
                    const TestResult* r = latest_result(idx, t.name);
                    std::string live = job_log ? job_log->get_live(t.name) : "";
                    auto live_out = job_log ? job_log->get_live_output(t.name) : std::vector<std::string>{};
                    render_detail(t, r, detail_scroll, live, live_out);
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
        } else if (preview_mode) {
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                preview_mode = false;
                render_main(idx);
            } else if (key == 'q' || key == 'Q') {
                break;
            } else if ((key == 's' || key == 'S') && run_fn && total > 0) {
                preview_mode = false;
                const auto& t = reg.tests[filtered[selected]];
                run_fn(t.name, selected_hw);
                render_main(idx);
            } else if (key != -1) {
                if (key == 'j' || key == 1001 /*down*/) ++preview_scroll;
                else if (key == 'k' || key == 1000 /*up*/) preview_scroll = std::max(0, preview_scroll - 1);
                render_script_preview(reg.tests[filtered[selected]].name, preview_lines, preview_scroll);
            }
        } else if (output_mode) {
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                output_mode = false;
                const auto& t = reg.tests[filtered[selected]];
                const TestResult* r = latest_result(idx, t.name);
                std::string live = job_log ? job_log->get_live(t.name) : "";
                auto live_out = job_log ? job_log->get_live_output(t.name) : std::vector<std::string>{};
                render_detail(t, r, detail_scroll, live, live_out);
            } else if (key == 'q' || key == 'Q') {
                break;
            } else if (key != -1) {
                if (key == 'j' || key == 1001 /*down*/) ++output_scroll;
                else if (key == 'k' || key == 1000 /*up*/) output_scroll = std::max(0, output_scroll - 1);
                render_output_log(reg.tests[filtered[selected]].name, output_lines, output_scroll);
            }
        } else {
            // Detail mode
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                detail_mode = false;
                render_main(idx);
            } else if (key == 'q' || key == 'Q') {
                break;
            } else if ((key == 'o' || key == 'O') && total > 0) {
                const auto& t = reg.tests[filtered[selected]];
                const TestResult* r = latest_result(idx, t.name);
                output_lines.clear();
                if (r) {
                    std::string out_path = result_output_path(*r);
                    auto content = out_path.empty() ? std::nullopt : fs::read_file(out_path);
                    if (content) {
                        std::istringstream ss(*content);
                        for (std::string l; std::getline(ss, l); )
                            output_lines.push_back(l);
                    }
                }
                output_mode = true;
                output_scroll = (int)output_lines.size(); // start at bottom
                render_output_log(t.name, output_lines, output_scroll);
            } else {
                bool detail_redraw = false;
                if (key == 'j' || key == 1001 /*down*/) { ++detail_scroll; detail_redraw = true; }
                else if (key == 'k' || key == 1000 /*up*/) { detail_scroll = std::max(0, detail_scroll - 1); detail_redraw = true; }
                else if (key != -1) { detail_redraw = true; }

                ++detail_ticks;
                if (detail_ticks >= refresh_every) { detail_ticks = 0; detail_redraw = true; }

                if (detail_redraw) {
                    idx = load_all_results(results_dir);
                    const auto& t = reg.tests[filtered[selected]];
                    const TestResult* r = latest_result(idx, t.name);
                    std::string live = job_log ? job_log->get_live(t.name) : "";
                    auto live_out = job_log ? job_log->get_live_output(t.name) : std::vector<std::string>{};
                    render_detail(t, r, detail_scroll, live, live_out);
                }
            }
        }
    }

    restore_terminal();
    std::cout << ansi::ALT_SCREEN_OFF;
    std::cout.flush();
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
