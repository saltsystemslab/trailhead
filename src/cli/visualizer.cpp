#include "visualizer.hpp"
#include "sbatch_gen.hpp"
#include "local_run.hpp"
#include "remote_run.hpp"
#include "../core/registry.hpp"
#include "../util/ansi.hpp"
#include "../util/file_util.hpp"
#include "../util/process.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <chrono>
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
    raw.c_lflag &= ~(ICANON | ECHO | ISIG);
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
// Uses a static leftover buffer so that bytes unconsumed from one read are not
// discarded — preventing partial escape sequences from leaking as bare characters
// (e.g. 'A' from \x1b[A triggering the add-test wizard when holding the up arrow).
// Unread bytes from a previous read_key() call. File-scope (not function-local)
// so flush_input() can clear them when a modal prompt opens.
static char s_key_leftover[64] = {};
static int  s_key_leftover_len = 0;

// Discard buffered keystrokes — both read_key's leftover and the kernel's unread
// tty input — so a modal prompt (add/edit wizard, hardware picker) isn't driven
// by keys pressed before it opened. Without this, fast scrolling can leave a
// half-finished arrow sequence buffered that the wizard reads as a bare ESC and
// instantly cancels on.
static void flush_input() {
    s_key_leftover_len = 0;
    tcflush(STDIN_FILENO, TCIFLUSH);
}

static int read_key(int timeout_ms) {
    // Buffer comfortably larger than one escape sequence so a fast key-repeat
    // stream rarely has a read() boundary land in the middle of "\x1b[A".
    char buf[64] = {};
    int  r       = 0;

    if (s_key_leftover_len > 0) {
        r = s_key_leftover_len;
        memcpy(buf, s_key_leftover, s_key_leftover_len);
        s_key_leftover_len = 0;
    } else {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) return -1;

        r = (int)read(STDIN_FILENO, buf, sizeof(buf));
        if (r <= 0) return -1;
    }

    // If buf begins an escape sequence but doesn't yet hold a complete one, wait
    // briefly for the rest before parsing. Critical for fast scrolling: an 8-/64-
    // byte read can split "\x1b[A" after the '[', and without completion the
    // leftover "\x1b[" is read as ESC + '[' while the orphaned trailing 'A' then
    // arrives as a bare key — opening the add-test wizard. Over SSH the 3 bytes
    // can likewise arrive in separate reads.
    auto incomplete = [&]() {
        if (r < 1 || buf[0] != 0x1b) return false;       // not an escape sequence
        if (r == 1) return true;                          // just ESC
        if (buf[1] != '[') return false;                  // ESC + other (e.g. Alt-key)
        if (r == 2) return true;                          // ESC '[' — need final byte
        if (r == 3 && (buf[2] == '5' || buf[2] == '6')) return true; // need '~' (PgUp/PgDn)
        return false;
    };
    for (int tries = 0; tries < 4 && incomplete() && r < (int)sizeof(buf); ++tries) {
        fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = {0, 50000}; // 50 ms
        if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) break;
        int n = (int)read(STDIN_FILENO, buf + r, sizeof(buf) - r);
        if (n <= 0) break;
        r += n;
    }

    int key      = -1;
    int consumed = 1;

    if (r >= 3 && buf[0] == 0x1b && buf[1] == '[') {
        // Check for tilde-terminated sequences: \x1b[5~ (PgUp), \x1b[6~ (PgDn)
        if (r >= 4 && buf[3] == '~') {
            if (buf[2] == '5') { key = 1004; consumed = 4; }
            if (buf[2] == '6') { key = 1005; consumed = 4; }
        }
        if (key == -1) {
            switch (buf[2]) {
                case 'A': key = 1000; consumed = 3; break; // Up
                case 'B': key = 1001; consumed = 3; break; // Down
                case 'C': key = 1002; consumed = 3; break; // Right
                case 'D': key = 1003; consumed = 3; break; // Left
            }
        }
    }

    if (key == -1) {
        key      = (unsigned char)buf[0];
        consumed = 1;
    }

    // Save any bytes beyond what we consumed for the next call.
    if (r > consumed) {
        s_key_leftover_len = r - consumed;
        memcpy(s_key_leftover, buf + consumed, s_key_leftover_len);
    }

    return key;
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
        case RunStatus::Pass:      return ansi::color(ansi::BGREEN,  " PASS ");
        case RunStatus::Fail:      return ansi::color(ansi::BRED,    " FAIL ");
        case RunStatus::BuildFail: return ansi::color(ansi::MAGENTA, " BFAIL");
        case RunStatus::Running:   return ansi::color(ansi::BYELLOW, " RUN  ");
        default:                   return ansi::color(ansi::GRAY,    "  --- ");
    }
}

static std::string scroll_name(const std::string& s, int offset, int width);

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

static std::string header_row(int name_w = COL_NAME) {
    using namespace ansi;
    std::ostringstream o;
    o << BOLD
      << pad("NAME",   name_w) << " "
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
                              const std::string& live_status = "",
                              int name_offset = 0,
                              int name_w = COL_NAME,
                              int batch_mark = -1)  // -1 hide, 0 unchecked, 1 checked
{
    using namespace ansi;
    RunStatus status = r ? result_status(*r) : RunStatus::Unknown;

    std::string name_col = selected ? scroll_name(t.name, name_offset, name_w)
                                     : pad(t.name, name_w);
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
            case RunStatus::Pass:      stat_color = BGREEN;  break;
            case RunStatus::Fail:      stat_color = BRED;    break;
            case RunStatus::BuildFail: stat_color = MAGENTA; break;
            default:                   stat_color = GRAY;    break;
        }
    }

    std::ostringstream row;
    if (batch_mark >= 0)
        row << (batch_mark > 0 ? color(BGREEN, "[x] ") : dim("[ ] "));
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

    // ── Test output (TH_OUTPUT / TH_OUTPUT_START..STOP) ───────────────────
    // Folded into the scrollable output section below.

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
        int th_output_count = 0;  // lines from TH_OUTPUT — styled differently

        if (is_live) {
            out_lines = live_output; // may be empty while still QUEUED
        } else if (r) {
            // Include TH_OUTPUT / TH_OUTPUT_START..STOP lines first
            if (!r->output_lines.empty()) {
                out_lines.insert(out_lines.end(), r->output_lines.begin(), r->output_lines.end());
                th_output_count = (int)r->output_lines.size();
            }
            // Load full .out sidecar instead of truncated _output_tail
            std::string out_path = result_output_path(*r);
            auto content = out_path.empty() ? std::nullopt : fs::read_file(out_path);
            if (content) {
                bool in_stderr = false;
                std::istringstream ss(*content);
                for (std::string l; std::getline(ss, l); ) {
                    if (l == "--- stderr ---") { in_stderr = true; continue; }
                    if (in_stderr) continue;  // hide build stderr (compiler warnings)
                    out_lines.push_back(l);
                }
            } else {
                // Fallback to _output_tail if .out file missing
                auto oit = r->metadata.find("_output_tail");
                if (oit != r->metadata.end() && !oit->second.empty()) {
                    bool in_stderr = false;
                    std::istringstream ss(oit->second);
                    for (std::string l; std::getline(ss, l); ) {
                        if (l == "--- stderr ---") { in_stderr = true; continue; }
                        if (in_stderr) continue;
                        out_lines.push_back(l);
                    }
                }
            }
        }

        if (!out_lines.empty() || is_live) {
            int total_out = (int)out_lines.size();

            // Rows available for output lines.
            // Budget: term_rows()-1 total newlines (the Nth \n from CURSOR_HOME
            // scrolls the screen if N == term_rows(), clipping the header).
            // Subtract: rows_written so far, 2 for blank+Output-header,
            //           FOOTER_ROWS for the footer, scroll_ind for ↑/↓ indicators.
            static constexpr int FOOTER_ROWS = 2;
            int scroll_ind = 0;
            if (!is_live) {
                if (detail_scroll > 0) ++scroll_ind;
                int tentative = std::max(3, term_rows() - 1 - rows_written - 2 - FOOTER_ROWS - scroll_ind);
                if (total_out > detail_scroll + tentative) ++scroll_ind;
            }
            int avail = std::max(3, term_rows() - 1 - rows_written - 2 - FOOTER_ROWS - scroll_ind);

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
                for (int i = detail_scroll; i < vis_end; ++i) {
                    const char* style = (i < th_output_count) ? CYAN : DIM;
                    ln("    " + std::string(style) + out_lines[i] + RESET);
                }

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
    int scroll_ind = 0;
    if (scroll > 0) ++scroll_ind;
    {
        int tentative = std::max(3, term_rows() - 1 - rows_written - 2 - scroll_ind);
        if (total > scroll + tentative) ++scroll_ind;
    }
    int avail = std::max(3, term_rows() - 1 - rows_written - 2 - scroll_ind);
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
    o << DIM << "[b/ESC] back  [j/\xe2\x86\x93] down  [k/\xe2\x86\x91] up  [d/PgDn] page down  [u/PgUp] page up  [q] quit" << RESET << "\n";
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
    int scroll_ind = 0;
    if (scroll > 0) ++scroll_ind;
    {
        int tentative = std::max(3, term_rows() - 1 - rows_written - 2 - scroll_ind);
        if (total > scroll + tentative) ++scroll_ind;
    }
    int avail = std::max(3, term_rows() - 1 - rows_written - 2 - scroll_ind);
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
    if ((int)s.size() <= width) return ansi::pad(s, width);
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

// Truncate `s` to at most `max_width` *visible* columns, preserving ANSI escape
// sequences (which take no width) and dropping any printable overflow. Used for
// the log panel so a long message can't wrap onto a second terminal row, which
// would throw off the frame's row accounting and scroll the header off-screen.
static std::string truncate_display(const std::string& s, int max_width) {
    if (max_width < 0) max_width = 0;
    std::string out;
    int vis = 0;
    bool had_ansi = false;
    for (size_t i = 0; i < s.size(); ) {
        if (s[i] == '\x1b') {                       // ANSI escape: copy verbatim
            size_t j = i + 1;
            if (j < s.size() && s[j] == '[') {
                ++j;
                while (j < s.size() && !(s[j] >= 0x40 && s[j] <= 0x7e)) ++j;
                if (j < s.size()) ++j;              // include the final byte
            }
            out.append(s, i, j - i);
            had_ansi = true;
            i = j;
        } else {
            if (vis >= max_width) break;
            out += s[i++];
            ++vis;
        }
    }
    if (had_ansi) out += ansi::RESET;               // avoid colour bleed
    return out;
}

// ── Wizard helpers ────────────────────────────────────────────────────────

// Open $VISUAL/$EDITOR/vim to edit a list of shell lines (node preamble).
// Returns the lines as a vector; empty lines and '#' comments are stripped.
static std::vector<std::string> wizard_edit_preamble(const std::string& node_name,
                                                       const std::vector<std::string>& existing = {})
{
    std::string tmpfile = "/tmp/trailhead_preamble_" + std::to_string(getpid()) + ".sh";
    {
        std::ofstream f(tmpfile);
        f << "# Node preamble for: " << node_name << "\n"
          << "# Lines starting with '#' are ignored.\n"
          << "# Add module loads, exports, etc. that this cluster needs.\n"
          << "# Example:\n"
          << "#   module load cmake cuda/12.0\n"
          << "#   export CUDA_HOME=/path/to/cuda\n"
          << "#   export PATH=$CUDA_HOME/bin:$PATH\n"
          << "#\n";
        for (const auto& l : existing) f << l << "\n";
    }
    const char* ev = getenv("VISUAL");
    if (!ev) ev = getenv("EDITOR");
    std::cout << ansi::ALT_SCREEN_OFF;
    std::cout.flush();
    system((std::string(ev ? ev : "vim") + " " + tmpfile).c_str());
    std::cout << ansi::ALT_SCREEN_ON;
    std::cout.flush();
    std::vector<std::string> lines;
    {
        std::ifstream f(tmpfile);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] == '#') continue;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            lines.push_back(line);
        }
    }
    ::unlink(tmpfile.c_str());
    // Strip trailing blank lines
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    return lines;
}

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

    // Exit the alternate screen so the editor gets a clean normal terminal.
    // Some terminals don't support nested alt-screen buffers, causing the
    // editor to fight with the TUI for the same buffer → dropped inputs.
    std::cout << ansi::ALT_SCREEN_OFF;
    std::cout.flush();

    system((std::string(ev ? ev : "vim") + " " + tmpfile).c_str());

    // Restore alternate screen for the rest of the wizard.
    std::cout << ansi::ALT_SCREEN_ON;
    std::cout.flush();

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
    np.preamble      = wizard_edit_preamble(name);

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

    np.preamble = wizard_edit_preamble(name, np.preamble);

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
    flush_input();  // drop keys buffered before the picker opened (e.g. scroll)
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
                std::string disp = (int)names[i].size() > name_w
                    ? scroll_name(names[i], name_scroll, name_w)
                    : names[i];
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
        if (k == 27 || k == 'q' || k == 3 /*Ctrl-C*/) return "";
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
    std::string def_cfg   = "cmake -B " + dir + " -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={{arch}}";
    std::string configure = read_line("Configure command", def_cfg);
    std::string def_bld   = "cmake --build " + dir + " -j$(nproc)";
    std::string build_cmd = read_line("Build command", def_bld);

    enter_raw_mode();

    BuildConfig bc;
    bc.name          = name;
    bc.dir           = dir;
    bc.configure_cmd = configure;
    bc.build_cmd     = build_cmd;

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
    int cursor = (names.size() > 1) ? 1 : 0;
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
                std::string disp = (int)names[i].size() > name_w
                    ? scroll_name(names[i], name_scroll, name_w)
                    : names[i];
                o << (i == cursor ? " > " : "   ") << disp;
                if (i > 0) {
                    auto it = reg.builds.find(names[i]);
                    if (it != reg.builds.end()) {
                        const auto& bc = it->second;
                        if (!bc.configure_cmd.empty())
                            o << DIM << "  (" << bc.configure_cmd << ")" << RESET;
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
        if (k == 27 || k == 'q' || k == 3 /*Ctrl-C*/) return "";
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
    flush_input();  // drop keys buffered before the wizard opened (e.g. scroll)

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
                        std::string disp = (int)choices[i].size() > name_w
                            ? scroll_name(choices[i], name_scroll, name_w)
                            : choices[i];
                        o << disp;
                    }
                    if (i == cursor) o << RESET;
                    o << "\n";
                }
                o << "\n" << hline(50) << "\n";
                o << DIM << "[↑/k] up  [↓/j] down  [enter] select  [n] new sub-registry  [ESC] cancel"
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
            if (k == 27 || k == 'q' || k == 3 /*Ctrl-C*/) { cancelled = true; break; }
            if (k == '\r' || k == '\n') { dest_sub_dir = choices[cursor]; break; }
            if ((k == 1000 || k == 'k') && cursor > 0) { --cursor; name_scroll = 0; }
            if ((k == 1001 || k == 'j') && cursor + 1 < (int)choices.size()) { ++cursor; name_scroll = 0; }
            if (k == 'n' || k == 'N') {
                restore_terminal();
                std::cout << ansi::CLEAR;
                std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Add sub-registry\n\n";
                std::cout << "  Relative path to subdirectory (e.g. n_queens): ";
                std::cout.flush();
                std::string new_sub;
                std::getline(std::cin, new_sub);
                while (!new_sub.empty() && new_sub.front() == ' ') new_sub.erase(new_sub.begin());
                while (!new_sub.empty() && new_sub.back()  == ' ') new_sub.pop_back();
                enter_raw_mode();
                if (!new_sub.empty()) {
                    std::string sub_reg_path = project_root + "/" + new_sub + "/.trailhead/registry.json";
                    bool already = false;
                    for (const auto& s : reg.sub_registries) if (s == new_sub) { already = true; break; }
                    if (!already && fs::exists(sub_reg_path)) {
                        reg.sub_registries.push_back(new_sub);
                        save_registry(th_dir, reg);
                        choices.push_back(new_sub);
                        cursor = (int)choices.size() - 1;
                    }
                }
                continue;
            }
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

    // ── Step 7: tags ────────────────────────────────────────────────────────
    std::vector<std::string> tags;
    {
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Tags\n\n";
        std::cout << "  Comma-separated tags (e.g. perf,gpu) []: ";
        std::cout.flush();
        std::string tag_in;
        std::getline(std::cin, tag_in);
        std::istringstream tss(tag_in);
        std::string tok;
        while (std::getline(tss, tok, ',')) {
            while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
            while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
            if (!tok.empty()) tags.push_back(tok);
        }
        enter_raw_mode();
    }

    // ── Step 8: create entry and save ─────────────────────────────────────
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
    t.tags        = tags;
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
    flush_input();  // drop keys buffered before the wizard opened (e.g. scroll)
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

    // ── Step 7: tags (pre-filled) ────────────────────────────────────────
    std::vector<std::string> tags = test.tags;
    {
        std::string current;
        for (size_t i = 0; i < tags.size(); ++i) {
            if (i) current += ",";
            current += tags[i];
        }
        restore_terminal();
        std::cout << ansi::CLEAR;
        std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Tags\n\n";
        std::cout << "  Comma-separated tags [" << (current.empty() ? "" : current) << "]: ";
        std::cout.flush();
        std::string tag_in;
        std::getline(std::cin, tag_in);
        if (!tag_in.empty()) {
            tags.clear();
            std::istringstream tss(tag_in);
            std::string tok;
            while (std::getline(tss, tok, ',')) {
                while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
                while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
                if (!tok.empty()) tags.push_back(tok);
            }
        }
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
    test.tags        = tags;

    save_registry(eff_th_dir, *work_reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = eff_project_root;
    write_sbatch(eff_th_dir, *work_reg, opts);

    if (!test_sub_dir.empty())
        reload_merged(reg, th_dir, project_root);
    return true;
}

// ── Clone-task wizard ─────────────────────────────────────────────────────

// Clone an existing test: copies all fields, prompts for a new name, opens
// the editor with the existing command so the user can tweak args.
static bool run_clone_wizard(Registry& reg, int test_idx,
                              const std::string& th_dir,
                              const std::string& project_root)
{
    flush_input();  // drop keys buffered before the wizard opened (e.g. scroll)
    const TestEntry& src = reg.tests[test_idx];
    const std::string test_sub_dir = src.sub_dir;

    std::string eff_th_dir = th_dir;
    std::string eff_project_root = project_root;
    Registry sub_reg;
    Registry* dest_reg = &reg;

    if (!test_sub_dir.empty()) {
        eff_th_dir       = project_root + "/" + test_sub_dir + "/.trailhead";
        eff_project_root = project_root + "/" + test_sub_dir;
        auto loaded = load_registry(eff_th_dir);
        if (!loaded) { enter_raw_mode(); return false; }
        sub_reg  = *loaded;
        dest_reg = &sub_reg;
    }

    // ── Step 1: new name ─────────────────────────────────────────────────
    std::string base_name = test_sub_dir.empty() ? src.name : strip_sub_prefix(src.name, test_sub_dir);
    restore_terminal();
    std::cout << ansi::CLEAR;
    std::cout << ansi::BOLD << "TRAILHEAD" << ansi::RESET << " — Clone task\n\n";
    std::cout << "  New name [" << base_name << "_copy]: ";
    std::cout.flush();

    std::string name;
    std::getline(std::cin, name);
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back()  == ' ') name.pop_back();
    for (auto& c : name) if (c == ' ') c = '_';
    if (name.empty()) name = base_name + "_copy";

    // ── Step 2: editor (pre-filled with source cmd) ──────────────────────
    std::string cmd = wizard_open_editor(name, src.cmd);
    enter_raw_mode();

    if (cmd.empty()) return false;

    // ── Step 3: create entry (copy all fields, override name + cmd) ──────
    TestEntry t = src;
    t.name    = name;
    t.cmd     = cmd;
    t.sub_dir = "";  // not serialized; will be set by merge on reload
    dest_reg->tests.push_back(t);

    save_registry(eff_th_dir, *dest_reg);
    SbatchOptions opts;
    opts.split        = true;
    opts.project_root = eff_project_root;
    write_sbatch(eff_th_dir, *dest_reg, opts);

    if (!test_sub_dir.empty())
        reload_merged(reg, th_dir, project_root);
    return true;
}

// ── Delete confirmation ────────────────────────────────────────────────────

static bool wizard_confirm_delete(Registry& reg, int test_idx,
                                   const std::string& th_dir,
                                   const std::string& project_root)
{
    flush_input();  // drop keys buffered before the prompt opened (e.g. scroll)
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
        if (k == 'n' || k == 'N' || k == 27 || k == 'q' || k == 3 /*Ctrl-C*/ || k < 0) return false;
    }
}

// ── Main watch loop ───────────────────────────────────────────────────────

int run_watch(const std::string& trailhead_dir, Registry& reg, int interval_ms,
              std::shared_ptr<JobLog> job_log,
              std::function<void(const std::string&, const std::string&)> run_fn,
              std::function<void(const std::string&, const std::string&)> cancel_fn,
              std::function<void(const std::string&,
                                 const std::vector<std::string>&, int)> batch_fn,
              std::function<void(const std::string&,
                                 const std::vector<std::string>&)> clean_build_fn,
              std::string project_root, bool auto_run, int repeat) {
    std::string results_dir = trailhead_dir + "/results";
    fs::mkdir_p(results_dir);

    std::cout << ansi::ALT_SCREEN_ON;
    std::cout.flush();
    enter_raw_mode();
    tcflush(STDIN_FILENO, TCIFLUSH); // discard type-ahead from build/setup output

    // Select hardware at startup if nodes are available
    std::string selected_hw;
    if (auto_run) {
        selected_hw = "local"; // non-interactive: always run locally
    } else if (!reg.nodes.empty() && run_fn) {
        selected_hw = wizard_select_hardware(reg, trailhead_dir, project_root);
    }

    int selected      = 0;
    int scroll        = 0;
    int name_scroll   = 0;
    int name_scroll_ticks = 0;
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
    std::string tag_filter;    // empty = show all, else only tests with this tag
    bool batch_select_mode = false;       // [B]: pick tests for a batch-run
    std::set<std::string> batch_sel;      // test names marked for the batch
    int  batch_size = 50;                 // tests per chunk for the batch-run

    // Rebuild the filtered list and update total. Call after hardware change or test add/delete.
    auto rebuild_filtered = [&]() {
        filtered.clear();
        for (int i = 0; i < (int)reg.tests.size(); ++i) {
            const auto& t = reg.tests[i];
            if (!tag_filter.empty()) {
                bool has_tag = false;
                for (const auto& tg : t.tags)
                    if (tg == tag_filter) { has_tag = true; break; }
                if (!has_tag) continue;
            }
            filtered.push_back(i);
        }
        total = (int)filtered.size();
        if (selected >= total) selected = std::max(0, total - 1);
        if (scroll  >= total) scroll   = std::max(0, total - 1);
    };
    rebuild_filtered();

    // Fixed chrome: title(1) + hline(1) + header(1) + hline(1) + hline(1) + keys(variable)
    // keys row is 157 visible cols; wraps on terminals narrower than that.
    static constexpr int FIXED_ROWS_NO_KEYS = 5;
    static constexpr int FOOTER_VIS_LEN     = 185;  // visible width of the keys row
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
        int keys_rows = std::max(1, (FOOTER_VIS_LEN + term_cols() - 1) / term_cols());
        int vis = std::max(term_rows() - FIXED_ROWS_NO_KEYS - keys_rows - log_rows - 1, 3);
        bool need_up   = scroll > 0;
        bool need_down = (scroll + vis) < total;  // preliminary check; refined below
        // Account for indicator rows in the visible count before clamping
        int vis_tests = vis - (need_up ? 1 : 0) - (need_down ? 1 : 0);
        if (vis_tests < 1) vis_tests = 1;
        clamp_scroll(vis_tests);
        // Recompute after clamping — clamping can flip need_up (scroll was 0, now >0),
        // shrinking vis_tests by 1.  Re-clamp a second time with the corrected count so
        // the selected row is never pushed outside the visible window by the indicator.
        need_up   = scroll > 0;
        need_down = (scroll + vis_tests) < total;
        vis_tests = vis - (need_up ? 1 : 0) - (need_down ? 1 : 0);
        if (vis_tests < 1) vis_tests = 1;
        clamp_scroll(vis_tests);
        need_up   = scroll > 0;
        need_down = (scroll + vis_tests) < total;

        // Buffer the entire frame and write it in one shot to minimise
        // partial-frame flicker over high-latency connections (SSH).
        std::ostringstream o;

        // Move to top-left without clearing — old content is overwritten in place.
        // Build the title separately and truncate it to the terminal width so a
        // long project path can't wrap the header onto a second row.
        o << CURSOR_HOME;
        {
            std::ostringstream title;
            title << BOLD << "TRAILHEAD" << RESET
                  << "  " << DIM << trailhead_dir << RESET
                  << "  " << now_str();
            if (job_log && job_log->active > 0)
                title << "  " << color(BYELLOW, std::to_string(job_log->active.load())
                                       + " job(s) running");
            if (!selected_hw.empty())
                title << "  " << color(CYAN, "hw:" + selected_hw);
            o << truncate_display(title.str(), term_cols());
        }
        o << "\n";
        // Actual non-name visible width: column widths + 7 separator spaces.
        // TOTAL_WIDTH uses +5 (hline), so rows are TOTAL_WIDTH-COL_NAME+2 wide without the name.
        static const int FIXED_W = COL_NODE + COL_STATUS + COL_PASS + COL_FAIL + COL_TIME + COL_WALL + COL_WHEN + 7;
        int dyn_name_w = std::max(COL_NAME, term_cols() - FIXED_W - 1);
        int dyn_total_w = dyn_name_w + FIXED_W;
        o << hline(dyn_total_w) << "\n";
        o << header_row(dyn_name_w) << "\n";
        o << hline(dyn_total_w) << "\n";

        if (need_up)
            o << DIM << "  ↑ " << scroll << " more above" << RESET << "\n";

        int end = std::min(scroll + vis_tests, total);
        for (int i = scroll; i < end; ++i) {
            const auto& t = reg.tests[filtered[i]];
            const TestResult* r = latest_result(idx, t.name, selected_hw);
            std::string live = job_log ? job_log->get_live(t.name, selected_hw) : "";
            int mark = -1, nw = dyn_name_w;
            if (batch_select_mode) {            // reserve 4 cols for the "[x] " marker
                mark = batch_sel.count(t.name) ? 1 : 0;
                nw = std::max(COL_NAME, dyn_name_w - 4);
            }
            o << test_row(t, r, reg, i == selected, live,
                          i == selected ? name_scroll : 0,
                          nw, mark) << "\n";
        }

        if (need_down)
            o << DIM << "  ↓ " << (total - end) << " more below" << RESET << "\n";

        o << hline(dyn_total_w) << "\n";
        if (batch_select_mode) {
            o << color(BGREEN, "BATCH ")
              << DIM << std::to_string(batch_sel.size()) << " selected"
              << "  size " << batch_size << "  "
              << "[s/space] toggle  [A] all  [b] size  [enter] run selected  [esc] cancel"
              << RESET;
        } else {
            o << DIM
              << "[q] quit  [↑/k/↓/j] nav  [enter] detail  [s] submit  [x] cancel  [R] run all  [F] rerun fails  [B] batch  [a] add  [e] edit  [c] clone  [d] delete  [f] filter  [h] hw"
              << RESET;
            if (!tag_filter.empty())
                o << "  " << BOLD << "tag:" << tag_filter << RESET;
        }
        o << "\n";

        // Log panel — fixed at snapshot taken above. Truncate each line to the
        // terminal width (minus the 2-char indent) so a long message can't wrap
        // onto a second row; otherwise the extra rows push the whole frame past
        // term_rows() and the terminal scrolls the header off the top.
        if (!snap.empty()) {
            o << "\n";
            int log_w = std::max(0, term_cols() - 2);
            for (const auto& line : snap)
                o << DIM << "  " << truncate_display(line, log_w) << RESET << "\n";
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

    // Run a batch-run for the given test names on the current hardware: hand the
    // terminal to batch-run's plain output, then restore the TUI and reload results.
    auto run_batch = [&](const std::vector<std::string>& names) {
        if (names.empty() || !batch_fn) return;
        restore_terminal();
        std::cout << ansi::ALT_SCREEN_OFF;
        std::cout.flush();
        batch_fn(selected_hw, names, batch_size);
        std::cout << ansi::ALT_SCREEN_ON;
        std::cout.flush();
        enter_raw_mode();
        flush_input();
        idx = load_all_results(results_dir);
    };

    // auto_run: submit all visible tests immediately, then watch until done
    int rounds_submitted = 0;
    int64_t session_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (auto_run && run_fn && total > 0) {
        std::string msg = "auto-run: queuing " + std::to_string(total) + " test(s)";
        if (repeat > 1) msg += " x" + std::to_string(repeat) + " rounds";
        job_log->push(msg + "...");
        for (int i = 0; i < total; ++i)
            run_fn(reg.tests[filtered[i]].name, selected_hw);
        rounds_submitted = 1;
    }

    // Ticks elapsed since last full refresh
    int ticks = 0;
    int tick_ms = (selected_hw == "local") ? 500 : 100;
    int refresh_every = std::max(1, interval_ms / tick_ms);

    while (true) {
        int key = read_key(tick_ms);

        if (batch_select_mode) {
            bool redraw = false;
            if (key == 27 /*ESC*/ || key == 'q' || key == 3 /*Ctrl-C*/) {
                batch_select_mode = false; redraw = true;
            } else if (key == 'j' || key == 1001 /*down*/) {
                selected = (selected + 1) % std::max(total, 1); name_scroll = 0; redraw = true;
            } else if (key == 'k' || key == 1000 /*up*/) {
                selected = (selected - 1 + std::max(total, 1)) % std::max(total, 1); name_scroll = 0; redraw = true;
            } else if ((key == 's' || key == 'S' || key == ' ') && total > 0) {
                const std::string& nm = reg.tests[filtered[selected]].name;
                if (!batch_sel.erase(nm)) batch_sel.insert(nm);  // toggle
                redraw = true;
            } else if (key == 'b' || key == 'B') {
                // Prompt for tests-per-chunk (batch size).
                restore_terminal();
                std::cout << ansi::ALT_SCREEN_OFF << ansi::CLEAR
                          << "Batch size — tests per chunk [" << batch_size << "]: " << std::flush;
                std::string line; std::getline(std::cin, line);
                try { int v = std::stoi(line); if (v > 0) batch_size = v; } catch (...) {}
                std::cout << ansi::ALT_SCREEN_ON; std::cout.flush();
                enter_raw_mode(); flush_input();
                redraw = true;
            } else if (key == 'a' || key == 'A') {
                std::vector<std::string> names;            // [A]: run every visible test
                for (int i = 0; i < total; ++i) names.push_back(reg.tests[filtered[i]].name);
                batch_select_mode = false;
                run_batch(names);
                redraw = true;
            } else if (key == '\r' || key == '\n') {
                std::vector<std::string> names(batch_sel.begin(), batch_sel.end());
                batch_select_mode = false;
                if (!names.empty()) run_batch(names);
                else if (job_log) job_log->push("batch: nothing selected");
                redraw = true;
            }
            // Keep the board live (clock, marquee, incoming results) while choosing.
            ++ticks;
            if (++name_scroll_ticks >= 3) { ++name_scroll; name_scroll_ticks = 0; redraw = true; }
            if (ticks >= refresh_every) { refresh_results(results_dir, idx); ticks = 0; redraw = true; }
            if (redraw) render_main(idx);
        } else if (!detail_mode && !preview_mode && !output_mode) {
            bool redraw = false;
            if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) break;
            if (key == 'j' || key == 1001 /*down*/) { selected = (selected + 1) % std::max(total, 1); name_scroll = 0; name_scroll_ticks = 0; redraw = true; }
            if (key == 'k' || key == 1000 /*up*/)   { selected = (selected - 1 + std::max(total, 1)) % std::max(total, 1); name_scroll = 0; name_scroll_ticks = 0; redraw = true; }
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
            if (key == 'B' && batch_fn) {
                if (selected_hw.empty() || selected_hw == "local") {
                    job_log->push(selected_hw == "local"
                        ? "batch-run needs a remote node — press [h] to choose"
                        : "No hardware selected — press [h] to choose");
                } else if (total > 0) {
                    // Enter batch-select mode: toggle tests with [s], [A] runs
                    // all, [enter] runs the selected set.
                    batch_select_mode = true;
                    batch_sel.clear();
                }
                redraw = true;
            }
            if ((key == 'h' || key == 'H') && run_fn) {
                std::string hw = wizard_select_hardware(reg, trailhead_dir, project_root);
                if (!hw.empty()) { selected_hw = hw; selected = 0; scroll = 0; }
                rebuild_filtered();
                redraw = true;
            }
            if (key == 'F' && run_fn && total > 0) {
                // Re-run the failed tests on this hardware. Tests that BUILD-failed
                // get their build dir wiped first so the rebuild starts clean.
                if (selected_hw.empty() && !reg.nodes.empty()) {
                    job_log->push("No hardware selected — press [h] to choose");
                } else {
                    std::vector<std::string> failed, bfail;
                    for (int i = 0; i < total; ++i) {
                        const auto& t = reg.tests[filtered[i]];
                        const TestResult* r = latest_result(idx, t.name, selected_hw);
                        if (!r) continue;
                        RunStatus st = result_status(*r);
                        if (st == RunStatus::BuildFail) { failed.push_back(t.name); bfail.push_back(t.name); }
                        else if (st == RunStatus::Fail)  failed.push_back(t.name);
                    }
                    if (failed.empty()) {
                        job_log->push("No failed tests to re-run");
                    } else {
                        if (!bfail.empty())
                            job_log->push("Re-running " + std::to_string(failed.size())
                                + " failed (" + std::to_string(bfail.size())
                                + " build-fail → wiping build dirs)...");
                        else
                            job_log->push("Re-running " + std::to_string(failed.size()) + " failed test(s)...");
                        // Wipe (possibly slow ssh) + resubmit off the UI thread so
                        // the board stays responsive; the wipe completes before the
                        // resubmit so the rebuild sees a clean tree.
                        std::string hw = selected_hw;
                        std::thread([clean_build_fn, run_fn, hw, failed, bfail]() {
                            if (!bfail.empty() && clean_build_fn) clean_build_fn(hw, bfail);
                            for (const auto& n : failed) run_fn(n, hw);
                        }).detach();
                    }
                }
                redraw = true;
            }
            if (key == 'f') {
                // Collect all unique tags across tests
                std::vector<std::string> all_tags;
                for (const auto& t : reg.tests)
                    for (const auto& tg : t.tags) {
                        bool dup = false;
                        for (const auto& a : all_tags) if (a == tg) { dup = true; break; }
                        if (!dup) all_tags.push_back(tg);
                    }
                std::sort(all_tags.begin(), all_tags.end());
                if (all_tags.empty()) {
                    if (job_log) job_log->push("No tags defined on any test");
                } else {
                    // Cycle: "" → tag1 → tag2 → ... → ""
                    auto it = std::find(all_tags.begin(), all_tags.end(), tag_filter);
                    if (it == all_tags.end() || ++it == all_tags.end())
                        tag_filter.clear();
                    else
                        tag_filter = *it;
                    if (tag_filter.empty()) {
                        if (job_log) job_log->push("Filter cleared — showing all tests");
                    } else {
                        if (job_log) job_log->push("Filter: tag=" + tag_filter);
                    }
                    selected = 0; scroll = 0;
                    rebuild_filtered();
                }
                redraw = true;
            }
            if ((key == 'x' || key == 'X') && total > 0) {
                const auto& t = reg.tests[filtered[selected]];
                std::string live = job_log ? job_log->get_live(t.name, selected_hw) : "";
                if (live.empty()) {
                    if (job_log) job_log->push("[" + t.name + "] not running");
                } else {
                    std::string tname = t.name;
                    // Cancel only this test's job on the current page's hardware,
                    // leaving the same test running on other nodes untouched.
                    if (cancel_fn) cancel_fn(tname, selected_hw);
                    // Clear queued submission (pre-SLURM: QUEUED/RSYNC)
                    clear_queued_submission(trailhead_dir, tname, selected_hw);
                    job_log->set_live(tname, "", selected_hw);
                    // scancel SLURM job in background to avoid freezing the TUI
                    auto pending = load_pending_jobs(trailhead_dir);
                    for (auto& p : pending) {
                        if (p.name != tname || p.node_name != selected_hw) continue;
                        clear_pending_job(trailhead_dir, tname, selected_hw);
                        std::string remote = p.remote;
                        std::string job_id = p.job_id;
                        std::thread([job_log, tname, remote, job_id]() {
                            std::string cmd = "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                                + remote + " scancel " + job_id;
                            auto r = proc::run(cmd, {}, {}, 15, "", nullptr, true);
                            if (r.exit_code == 0)
                                job_log->push("[" + tname + "] cancelled job " + job_id);
                            else
                                job_log->push("[" + tname + "] scancel failed: " + r.stderr_str);
                        }).detach();
                        break;
                    }
                    if (job_log) job_log->push("[" + tname + "] cancelling...");
                }
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
                    std::string run_cmd = t.cmd;
                    std::string run_wd;
                    if (!t.build_name.empty()) {
                        auto bit = reg.builds.find(t.build_name);
                        if (bit != reg.builds.end()) {
                            const auto& bc = bit->second;
                            std::string raw = bc.dir.empty() ? "build" : bc.dir;
                            std::string eff = bc.sub_dir.empty() ? raw : bc.sub_dir + "/" + raw;
                            if (!t.target.empty() && !bc.configure_cmd.empty()) {
                                std::string cfg = bc.configure_cmd;
                                auto ap = cfg.find("{{arch}}");
                                if (ap != std::string::npos)
                                    cfg.replace(ap, 8, "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,once | head -1 | tr -d '.')");
                                bool from_build = cfg.size() >= 3 && cfg.substr(cfg.size()-3) == " ..";
                                ss << "# configure (if build system absent)\n";
                                std::string chk = "([ -f " + eff + "/Makefile ] || [ -f " + eff + "/build.ninja ])";
                                if (from_build)
                                    ss << chk << " || (mkdir -p " << eff << " && cd " << eff << " && " << cfg << ")\n\n";
                                else
                                    ss << chk << " || " << cfg << "\n\n";
                            }
                            if (!t.target.empty())
                                ss << "cmake --build " << eff << " --target " << t.target << "\n\n";
                            if (t.workdir.empty() || t.workdir == ".") {
                                run_wd = eff;
                                const std::string prefix = raw + "/";
                                if (run_cmd.rfind(prefix, 0) == 0)
                                    run_cmd = "./" + run_cmd.substr(prefix.size());
                            }
                        }
                    }
                    if (run_wd.empty() && !t.workdir.empty() && t.workdir != ".")
                        run_wd = t.workdir;
                    if (!run_wd.empty()) ss << "cd " << run_wd << "\n";
                    ss << run_cmd << "\n";
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
            if (key == 'c' && total > 0) {
                run_clone_wizard(reg, filtered[selected], trailhead_dir, project_root);
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
            if (key == 'C' && total > 0) {
                const std::string& tname = reg.tests[filtered[selected]].name;
                auto it = idx.find(tname);
                if (it != idx.end()) {
                    int n = 0;
                    for (const auto& r : it->second) {
                        if (!r.result_file.empty()) {
                            ::unlink(r.result_file.c_str());
                            std::string out = result_output_path(r);
                            if (!out.empty()) ::unlink(out.c_str());
                            ++n;
                        }
                    }
                    idx.erase(it);
                    job_log->push("Cleared " + std::to_string(n) + " result(s) for " + tname);
                }
                redraw = true;
            }
            if (key == 'w' || key == 'W') {
                auto removed = wipe_build_dirs(reg, project_root);
                // Also clear the setup sentinel so setup re-runs on next job
                std::string sentinel = trailhead_dir + "/setup_done";
                if (fs::exists(sentinel)) {
                    ::unlink(sentinel.c_str());
                    job_log->push("wipe: cleared setup_done sentinel");
                }
                if (removed.empty()) {
                    job_log->push("wipe: no build directories found");
                } else {
                    for (const auto& d : removed)
                        job_log->push("wipe: removed " + d);
                    job_log->push("Wiped " + std::to_string(removed.size()) + " build director"
                                  + (removed.size() == 1 ? "y" : "ies"));
                }
                redraw = true;
            }
            if (key == '\r' || key == '\n') {
                if (total > 0) {
                    detail_mode = true;
                    detail_scroll = 0;
                    detail_ticks  = 0;
                    const auto& t = reg.tests[filtered[selected]];
                    const TestResult* r = latest_result(idx, t.name, selected_hw);
                    std::string live = job_log ? job_log->get_live(t.name, selected_hw) : "";
                    auto live_out = job_log ? job_log->get_live_output(t.name) : std::vector<std::string>{};
                    render_detail(t, r, detail_scroll, live, live_out);
                    // don't set redraw — detail was just rendered
                }
            }
            // Auto-refresh
            ++ticks;
            // Advance name marquee every 3 ticks (~300ms) when name is long
            if (++name_scroll_ticks >= 3) { ++name_scroll; name_scroll_ticks = 0; redraw = true; }
            if (ticks >= refresh_every) {
                refresh_results(results_dir, idx);
                ticks = 0;
                redraw = true;
            }
            // In auto_run mode, keep the actively running/submitting test in view.
            if (auto_run && job_log && total > 0) {
                for (int i = 0; i < total; ++i) {
                    const std::string live = job_log->get_live(reg.tests[filtered[i]].name);
                    if (!live.empty() && live != "QUEUED") {
                        if (selected != i) { selected = i; redraw = true; }
                        break;
                    }
                }
            }
            if (redraw) render_main(idx);

            // auto_run: once all active jobs finish, either queue next round or exit
            if (auto_run && rounds_submitted > 0 && job_log && job_log->active.load() == 0) {
                if (rounds_submitted < repeat) {
                    ++rounds_submitted;
                    job_log->push("auto-run: round " + std::to_string(rounds_submitted)
                                  + "/" + std::to_string(repeat) + "...");
                    for (int i = 0; i < total; ++i)
                        run_fn(reg.tests[filtered[i]].name, selected_hw);
                } else {
                    break;
                }
            }
        } else if (preview_mode) {
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                preview_mode = false;
                render_main(idx);
            } else if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) {
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
                const TestResult* r = latest_result(idx, t.name, selected_hw);
                std::string live = job_log ? job_log->get_live(t.name, selected_hw) : "";
                auto live_out = job_log ? job_log->get_live_output(t.name) : std::vector<std::string>{};
                render_detail(t, r, detail_scroll, live, live_out);
            } else if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) {
                break;
            } else if (key != -1) {
                int page = std::max(1, term_rows() - 6);
                if (key == 'j' || key == 1001 /*down*/) ++output_scroll;
                else if (key == 'k' || key == 1000 /*up*/) output_scroll = std::max(0, output_scroll - 1);
                else if (key == 1005 /*PgDn*/ || key == 'd') output_scroll += page;
                else if (key == 1004 /*PgUp*/ || key == 'u') output_scroll = std::max(0, output_scroll - page);
                render_output_log(reg.tests[filtered[selected]].name, output_lines, output_scroll);
            }
        } else {
            // Detail mode
            if (key == 'b' || key == 'B' || key == 27 /*ESC*/) {
                detail_mode = false;
                render_main(idx);
            } else if (key == 'q' || key == 'Q' || key == 3 /*Ctrl-C*/) {
                break;
            } else if ((key == 'o' || key == 'O') && total > 0) {
                const auto& t = reg.tests[filtered[selected]];
                const TestResult* r = latest_result(idx, t.name, selected_hw);
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
                output_scroll = 0; // start at top
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
                    const TestResult* r = latest_result(idx, t.name, selected_hw);
                    std::string live = job_log ? job_log->get_live(t.name, selected_hw) : "";
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

    // Helper: collect passing session results for a test (started_at >= session_start_ms)
    auto session_results = [&](const std::string& name) -> std::vector<const TestResult*> {
        std::vector<const TestResult*> out;
        auto it = final_idx.find(name);
        if (it == final_idx.end()) return out;
        for (const auto& r : it->second) {
            if (r.started_at >= session_start_ms && r.failed == 0 && r.exit_code == 0)
                out.push_back(&r);
        }
        return out;
    };

    auto median_of = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2 == 1) ? v[n/2] : (v[n/2-1] + v[n/2]) / 2.0;
    };
    auto mean_of = [](const std::vector<double>& v) -> double {
        if (v.empty()) return 0.0;
        double s = 0; for (double x : v) s += x; return s / v.size();
    };

    bool multi = repeat > 1;

    // For each test, find the ordered set of timing labels it reports.
    // Tests with 2+ labels: show per-label columns, suppress reported_ms (sum is meaningless).
    // Tests with 1 label:   show reported_ms only (= that label's value).
    std::unordered_map<std::string, std::vector<std::string>> test_label_sets;
    std::vector<std::string> all_labels; // labels from multi-label tests only
    {
        for (int i : filtered) {
            const auto& tname = reg.tests[i].name;
            std::unordered_set<std::string> seen;
            std::vector<std::string> ordered;
            auto collect = [&](const TestResult* r) {
                if (!r) return;
                for (const auto& te : r->timings)
                    if (seen.insert(te.label).second) ordered.push_back(te.label);
            };
            if (multi)
                for (const auto* r : session_results(tname)) collect(r);
            else
                collect(latest_result(final_idx, tname));
            test_label_sets[tname] = std::move(ordered);
        }
        std::unordered_set<std::string> seen_labels;
        for (int i : filtered) {
            const auto& lbls = test_label_sets[reg.tests[i].name];
            if (lbls.size() < 2) continue;
            for (const auto& lbl : lbls)
                if (seen_labels.insert(lbl).second)
                    all_labels.push_back(lbl);
        }
    }

    auto timing_val = [](const TestResult* r, const std::string& lbl) -> double {
        for (const auto& te : r->timings) if (te.label == lbl) return te.elapsed_ms;
        return 0.0;
    };
    auto reported_ms_of = [](const TestResult* r) -> double {
        double s = 0; for (const auto& te : r->timings) s += te.elapsed_ms; return s;
    };
    // Does this test report 2+ timing labels?
    auto is_multi_label = [&](const std::string& name) {
        auto it = test_label_sets.find(name);
        return it != test_label_sets.end() && it->second.size() >= 2;
    };

    // ── CSV ───────────────────────────────────────────────────────────────────
    std::ofstream csv(csv_path);
    if (multi) {
        csv << "test_name,status,runs,reported_ms_median,reported_ms_mean";
        for (const auto& lbl : all_labels) csv << "," << lbl << "_median," << lbl << "_mean";
        csv << "\n";
    } else {
        csv << "test_name,status,reported_ms";
        for (const auto& lbl : all_labels) csv << "," << lbl;
        csv << "\n";
    }

    int any_failed = 0;
    for (int i : filtered) {
        const auto& t = reg.tests[i];
        const TestResult* latest = latest_result(final_idx, t.name);
        if (!latest) { csv << t.name << ",NO_RESULT\n"; any_failed = 1; continue; }
        bool mlabel = is_multi_label(t.name);
        csv << std::fixed << std::setprecision(3);
        if (multi) {
            auto rs = session_results(t.name);
            bool any_fail = (latest->failed > 0 || latest->exit_code != 0);
            std::string status = any_fail ? "FAIL" : (rs.empty() ? "NO_RESULT" : "PASS");
            if (any_fail || rs.empty()) any_failed = 1;
            // reported_ms: value for single-label tests, blank for multi-label
            csv << t.name << "," << status << "," << rs.size() << ",";
            if (!mlabel) {
                std::vector<double> v; for (const auto* r : rs) v.push_back(reported_ms_of(r));
                csv << median_of(v) << "," << mean_of(v);
            } else {
                csv << ","; // blank reported_ms for multi-label tests
            }
            for (const auto& lbl : all_labels) {
                std::vector<double> v; for (const auto* r : rs) v.push_back(timing_val(r, lbl));
                double med = median_of(v), mn = mean_of(v);
                if (med == 0.0 && mn == 0.0) csv << ",,";
                else csv << "," << med << "," << mn;
            }
            csv << "\n";
        } else {
            std::string status = (latest->failed > 0 || latest->exit_code != 0) ? "FAIL" : "PASS";
            if (latest->failed > 0 || latest->exit_code != 0) any_failed = 1;
            csv << t.name << "," << status << ",";
            if (!mlabel) csv << reported_ms_of(latest);
            for (const auto& lbl : all_labels) {
                double v = timing_val(latest, lbl);
                csv << "," << (v == 0.0 ? "" : std::to_string(v));
            }
            csv << "\n";
        }
    }
    csv.close();

    std::cout << "Results written to: " << csv_path << "\n\n";

    // ── Summary table ─────────────────────────────────────────────────────────
    // Dynamic name column width
    size_t name_w = 8;
    for (int i : filtered) name_w = std::max(name_w, reg.tests[i].name.size() + 2);

    int col_w = 16;
    int runs_w = multi ? 6 : 0;
    std::cout << std::left << std::setw((int)name_w) << "TEST" << std::setw(8) << "STATUS";
    if (multi) std::cout << std::setw(runs_w) << "RUNS";
    std::cout << std::setw(col_w) << (multi ? "reported_ms(med)" : "reported_ms");
    for (const auto& lbl : all_labels)
        std::cout << std::setw(col_w) << (multi ? lbl + "(med)" : lbl);
    int total_w = (int)name_w + 8 + runs_w + col_w * (1 + (int)all_labels.size());
    std::cout << "\n" << std::string(total_w, '-') << "\n";

    for (int i : filtered) {
        const auto& t = reg.tests[i];
        const TestResult* latest = latest_result(final_idx, t.name);
        if (!latest) {
            std::cout << std::left << std::setw((int)name_w) << t.name << "NO_RESULT\n";
            continue;
        }
        bool any_fail = (latest->failed > 0 || latest->exit_code != 0);
        bool mlabel   = is_multi_label(t.name);
        std::cout << std::left << std::setw((int)name_w) << t.name
                  << std::setw(8) << (any_fail ? "FAIL" : "PASS")
                  << std::fixed << std::setprecision(3);
        if (multi) {
            auto rs = session_results(t.name);
            std::cout << std::setw(runs_w) << rs.size();
            // reported_ms: value for single-label, dash for multi-label
            if (!mlabel) {
                std::vector<double> v; for (const auto* r : rs) v.push_back(reported_ms_of(r));
                std::cout << std::setw(col_w) << median_of(v);
            } else {
                std::cout << std::setw(col_w) << "—";
            }
            for (const auto& lbl : all_labels) {
                std::vector<double> v; for (const auto* r : rs) v.push_back(timing_val(r, lbl));
                double med = median_of(v);
                std::cout << std::setw(col_w) << (med == 0.0 ? "—" : std::to_string(med).substr(0,7));
            }
        } else {
            std::cout << std::setw(col_w) << (mlabel ? "—" : std::to_string(reported_ms_of(latest)).substr(0,9));
            for (const auto& lbl : all_labels) {
                double v = timing_val(latest, lbl);
                std::cout << std::setw(col_w) << (v == 0.0 ? "—" : std::to_string(v).substr(0,9));
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ── TH_OUTPUT blocks ──────────────────────────────────────────────────────
    for (int i : filtered) {
        const auto& t = reg.tests[i];
        const TestResult* latest = latest_result(final_idx, t.name);
        if (!latest || latest->output_lines.empty()) continue;
        std::cout << ansi::BOLD << t.name << ansi::RESET << ":\n";
        for (const auto& ol : latest->output_lines)
            std::cout << "  " << ol << "\n";
        std::cout << "\n";
    }

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
