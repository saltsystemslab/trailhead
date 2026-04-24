#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <cerrno>

// POSIX
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <poll.h>

namespace trailhead::proc {

struct RunResult {
    int    exit_code  = -1;
    bool   timed_out  = false;
    std::string stdout_str;
    std::string stderr_str;
};

// Split a shell-style command string into argv (very basic: no quoting support)
inline std::vector<std::string> split_cmd(const std::string& cmd) {
    std::vector<std::string> out;
    std::string tok;
    for (char c : cmd) {
        if (c == ' ' || c == '\t') {
            if (!tok.empty()) { out.push_back(tok); tok.clear(); }
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) out.push_back(tok);
    return out;
}

// Run a command, capturing stdout/stderr, with a timeout.
// use_shell=true: wraps cmd in sh -c (supports &&, pipes, multi-line, etc.)
// on_line: called for each line of stdout (for live progress); may be null.
inline RunResult run(
    const std::string& cmd,
    const std::vector<std::string>& extra_args = {},
    const std::unordered_map<std::string,std::string>& extra_env = {},
    int timeout_sec = 0,
    const std::string& workdir = "",
    std::function<void(const std::string&)> on_stdout_line = nullptr,
    bool use_shell = false)
{
    RunResult result;

    // Build argv
    std::vector<std::string> parts;
    if (use_shell) {
        // Combine cmd + extra_args into a single shell string
        std::string full = cmd;
        for (const auto& a : extra_args) { full += ' '; full += a; }
        parts = {"/bin/sh", "-c", full};
    } else {
        parts = split_cmd(cmd);
        for (const auto& a : extra_args) parts.push_back(a);
    }
    if (parts.empty()) { result.exit_code = -1; return result; }

    std::vector<const char*> argv;
    for (const auto& p : parts) argv.push_back(p.c_str());
    argv.push_back(nullptr);

    // Pipes for stdout and stderr
    int out_pipe[2], err_pipe[2];
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        result.exit_code = -1;
        return result;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        result.exit_code = -1;
        return result;
    }

    if (pid == 0) {
        // Child: redirect stdin to /dev/null so child processes don't compete
        // with the TUI for terminal input (would cause dropped keystrokes in editors).
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull != -1) { dup2(devnull, STDIN_FILENO); close(devnull); }

        close(out_pipe[0]); close(err_pipe[0]);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(out_pipe[1]); close(err_pipe[1]);

        if (!workdir.empty()) {
            if (chdir(workdir.c_str()) != 0) _exit(127);
        }

        // Set extra env vars
        for (const auto& [k, v] : extra_env) {
            setenv(k.c_str(), v.c_str(), 1);
        }

        execvp(argv[0], const_cast<char* const*>(argv.data()));
        // If exec fails, signal it via exit code 127 (command not found)
        _exit(127);
    }

    // Parent
    close(out_pipe[1]);
    close(err_pipe[1]);

    // Non-blocking reads
    fcntl(out_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(err_pipe[0], F_SETFL, O_NONBLOCK);

    auto deadline = timeout_sec > 0
        ? (time(nullptr) + timeout_sec)
        : (time_t)0;

    std::string out_buf, err_buf;
    // Separate buffer for line-splitting — keeps out_buf (result.stdout_str) intact
    std::string line_buf;

    auto drain = [&](int fd, std::string& buf, bool is_stdout) {
        char tmp[4096];
        while (true) {
            ssize_t n = read(fd, tmp, sizeof(tmp));
            if (n <= 0) break;
            buf.append(tmp, n);
            if (is_stdout && on_stdout_line) {
                line_buf.append(tmp, n);
                size_t pos;
                while ((pos = line_buf.find('\n')) != std::string::npos) {
                    on_stdout_line(line_buf.substr(0, pos));
                    line_buf.erase(0, pos + 1);
                }
            }
        }
    };

    while (true) {
        // Check if child is done
        int status;
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            // Drain remaining output
            drain(out_pipe[0], out_buf, true);
            drain(err_pipe[0], err_buf, false);
            if (WIFEXITED(status))   result.exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) result.exit_code = 128 + WTERMSIG(status);
            break;
        }

        // Poll for I/O
        struct pollfd pfds[2] = {
            {out_pipe[0], POLLIN, 0},
            {err_pipe[0], POLLIN, 0}
        };
        int timeout_ms = timeout_sec > 0 ? 200 : 200;
        poll(pfds, 2, timeout_ms);

        if (pfds[0].revents & POLLIN) drain(out_pipe[0], out_buf, true);
        if (pfds[1].revents & POLLIN) drain(err_pipe[0], err_buf, false);

        // Timeout check
        if (deadline > 0 && time(nullptr) >= deadline) {
            kill(pid, SIGTERM);
            usleep(200000);
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, 0);
            result.timed_out = true;
            result.exit_code = -1;
            break;
        }
    }

    close(out_pipe[0]);
    close(err_pipe[0]);

    result.stdout_str = std::move(out_buf);
    result.stderr_str = std::move(err_buf);
    return result;
}

} // namespace trailhead::proc
