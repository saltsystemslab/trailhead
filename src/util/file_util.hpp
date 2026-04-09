#pragma once
#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cerrno>

// POSIX
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>

namespace trailhead::fs {

inline bool exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

inline bool is_dir(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline time_t mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

// Create directory tree (like mkdir -p)
inline bool mkdir_p(const std::string& path) {
    std::string p = path;
    for (size_t i = 1; i < p.size(); ++i) {
        if (p[i] == '/') {
            p[i] = '\0';
            ::mkdir(p.data(), 0755);
            p[i] = '/';
        }
    }
    return ::mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
}

// Read entire file to string
inline std::optional<std::string> read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return std::nullopt;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Write string to file atomically (via tmp + rename)
inline bool write_file_atomic(const std::string& path, const std::string& content) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f) return false;
        f << content;
    }
    return ::rename(tmp.c_str(), path.c_str()) == 0;
}

// List files in a directory matching optional suffix filter
inline std::vector<std::string> list_dir(const std::string& path, const std::string& suffix = "") {
    std::vector<std::string> out;
    DIR* d = opendir(path.c_str());
    if (!d) return out;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name == "." || name == "..") continue;
        if (!suffix.empty()) {
            if (name.size() < suffix.size()) continue;
            if (name.substr(name.size() - suffix.size()) != suffix) continue;
        }
        out.push_back(path + "/" + name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// Walk up from cwd looking for a directory named ".trailhead"
// Returns the parent of .trailhead (i.e., the project root)
inline std::optional<std::string> find_trailhead_root(const std::string& start = "") {
    char buf[4096];
    std::string cwd = start.empty()
        ? (getcwd(buf, sizeof(buf)) ? std::string(buf) : std::string())
        : start;
    if (cwd.empty()) return std::nullopt;

    std::string path = cwd;
    for (int i = 0; i < 12; ++i) {
        if (is_dir(path + "/.trailhead"))
            return path;
        size_t slash = path.rfind('/');
        if (slash == std::string::npos || path == "/") break;
        path = (slash == 0) ? "/" : path.substr(0, slash);
    }
    return std::nullopt;
}

// Get the .trailhead/ directory, searching up from cwd
inline std::optional<std::string> find_trailhead_dir(const std::string& start = "") {
    auto root = find_trailhead_root(start);
    if (!root) return std::nullopt;
    return *root + "/.trailhead";
}

// Human-readable relative time ("3 min ago", "just now", etc.)
inline std::string relative_time(time_t then) {
    time_t now_t = time(nullptr);
    long diff = (long)(now_t - then);
    if (diff < 5)   return "just now";
    if (diff < 60)  return std::to_string(diff) + "s ago";
    if (diff < 3600) return std::to_string(diff / 60) + "m ago";
    if (diff < 86400) return std::to_string(diff / 3600) + "h ago";
    return std::to_string(diff / 86400) + "d ago";
}

// Human-readable duration from milliseconds
inline std::string format_duration_ms(int64_t ms) {
    if (ms < 0) return "?";
    if (ms < 1000) return std::to_string(ms) + "ms";
    double s = ms / 1000.0;
    if (s < 60.0) {
        std::ostringstream ss;
        ss.precision(2); ss << std::fixed << s << "s";
        return ss.str();
    }
    int mins = (int)s / 60;
    int secs = (int)s % 60;
    std::ostringstream ss;
    ss << mins << "m" << secs << "s";
    return ss.str();
}

} // namespace trailhead::fs
