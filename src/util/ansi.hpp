#pragma once
#include <string>
#include <sstream>

namespace trailhead::ansi {

// Reset / styles
constexpr const char* RESET  = "\033[0m";
constexpr const char* BOLD   = "\033[1m";
constexpr const char* DIM    = "\033[2m";
constexpr const char* ITALIC = "\033[3m";

// Foreground colors
constexpr const char* RED     = "\033[31m";
constexpr const char* GREEN   = "\033[32m";
constexpr const char* YELLOW  = "\033[33m";
constexpr const char* BLUE    = "\033[34m";
constexpr const char* MAGENTA = "\033[35m";
constexpr const char* CYAN    = "\033[36m";
constexpr const char* WHITE   = "\033[37m";
constexpr const char* GRAY    = "\033[90m";

// Bright variants
constexpr const char* BRED    = "\033[91m";
constexpr const char* BGREEN  = "\033[92m";
constexpr const char* BYELLOW = "\033[93m";
constexpr const char* BCYAN   = "\033[96m";
constexpr const char* BWHITE  = "\033[97m";

// Cursor / screen control
constexpr const char* CLEAR         = "\033[2J\033[H";
constexpr const char* ERASE_DOWN    = "\033[J";   // erase from cursor to end of screen
constexpr const char* CLEAR_LINE    = "\033[2K\r";
constexpr const char* CURSOR_HIDE   = "\033[?25l";
constexpr const char* CURSOR_SHOW   = "\033[?25h";
constexpr const char* CURSOR_HOME   = "\033[H";
constexpr const char* SAVE_CURSOR   = "\033[s";
constexpr const char* RESTORE_CURSOR= "\033[u";

inline std::string move_to(int row, int col) {
    std::ostringstream o;
    o << "\033[" << row << ";" << col << "H";
    return o.str();
}

inline std::string color(const char* col, const std::string& text) {
    return std::string(col) + text + RESET;
}

inline std::string bold(const std::string& text) { return std::string(BOLD) + text + RESET; }
inline std::string dim(const std::string& text)  { return std::string(DIM)  + text + RESET; }

// Pad/truncate string to exactly width chars (ASCII only)
inline std::string pad(const std::string& s, int width, char fill = ' ') {
    if ((int)s.size() >= width) return s.substr(0, width);
    return s + std::string(width - s.size(), fill);
}

inline std::string rpad(const std::string& s, int width, char fill = ' ') {
    if ((int)s.size() >= width) return s.substr(0, width);
    return std::string(width - s.size(), fill) + s;
}

// Horizontal rule
inline std::string hline(int width, char c = '-') {
    return std::string(width, c);
}

} // namespace trailhead::ansi
