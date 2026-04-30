/*
 * trailhead.h — lightweight C/C++ helper for reporting test results.
 *
 * Include this in any test (C or C++). No linking required.
 * Trailhead captures stdout and parses the TRAILHEAD: lines automatically.
 *
 * Markers are silent unless enabled by either:
 *   1. Compile-time: -DTRAILHEAD_ENABLED
 *   2. Runtime:      TRAILHEAD_ENABLED=1 environment variable
 *
 * Trailhead sets the env var automatically when running tests, so no
 * build changes are needed. User builds produce no extra output.
 */

#ifndef TRAILHEAD_H
#define TRAILHEAD_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ── Runtime enable check (cached after first call) ────────────────────── */

static inline int _th_enabled(void) {
#ifdef TRAILHEAD_ENABLED
    return 1;
#else
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("TRAILHEAD_ENABLED");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
#endif
}

/* ── Output protocol ─────────────────────────────────────────────────────
 *   TRAILHEAD:pass:<label>
 *   TRAILHEAD:fail:<label>
 *   TRAILHEAD:time:<label>:<elapsed_ms>
 *   TRAILHEAD:meta:<key>:<value>
 *   TRAILHEAD:output:<text>
 *   TRAILHEAD:output_start / TRAILHEAD:output_stop
 * ──────────────────────────────────────────────────────────────────────── */

#define TH_PASS(label) \
    do { if (_th_enabled()) { printf("TRAILHEAD:pass:%s\n", (label)); fflush(stdout); } } while(0)

#define TH_FAIL(label) \
    do { if (_th_enabled()) { printf("TRAILHEAD:fail:%s\n", (label)); fflush(stdout); } } while(0)

#define TH_CHECK(cond, label) \
    do { if (cond) TH_PASS(label); else TH_FAIL(label); } while(0)

#define TH_META(key, value) \
    do { if (_th_enabled()) { printf("TRAILHEAD:meta:%s:%s\n", (key), (value)); fflush(stdout); } } while(0)

#define TH_OUTPUT(text) \
    do { if (_th_enabled()) { printf("TRAILHEAD:output:%s\n", (text)); fflush(stdout); } } while(0)

#define TH_OUTPUT_START() \
    do { if (_th_enabled()) { printf("TRAILHEAD:output_start\n"); fflush(stdout); } } while(0)

#define TH_OUTPUT_STOP() \
    do { if (_th_enabled()) { printf("TRAILHEAD:output_stop\n"); fflush(stdout); } } while(0)

#define TH_TIME_EMIT(label, elapsed_ms) \
    do { if (_th_enabled()) { printf("TRAILHEAD:time:%s:%.6f\n", (label), (double)(elapsed_ms)); fflush(stdout); } } while(0)

/* Internal: declare a timer variable. Uses POSIX clock_gettime. */
#define _TH_TIMER_VAR(label) _th_t_##label
#define _TH_WALL_MS(start, end) \
    (((double)((end).tv_sec  - (start).tv_sec)  * 1e3) + \
     ((double)((end).tv_nsec - (start).tv_nsec) * 1e-6))

#define TH_TIME_BEGIN(label) \
    struct timespec _TH_TIMER_VAR(label); \
    clock_gettime(CLOCK_MONOTONIC, &_TH_TIMER_VAR(label))

#define TH_TIME_END(label) \
    do { \
        struct timespec _th_end_##label; \
        clock_gettime(CLOCK_MONOTONIC, &_th_end_##label); \
        TH_TIME_EMIT(#label, _TH_WALL_MS(_TH_TIMER_VAR(label), _th_end_##label)); \
    } while(0)

/* ── Node lock (flock-based) ─────────────────────────────────────────────
 * Serialize GPU-sensitive sections across parallel test processes on the
 * same node.  Uses POSIX flock() on a shared file.
 *
 *   TH_ACQUIRE_NODE_LOCK()   — blocks until exclusive lock is held
 *   TH_RELEASE_NODE_LOCK()   — releases the lock
 *
 * Lock file: /tmp/trailhead_<nodeid>.lock
 * Node ID sourced from: SLURMD_NODENAME > SLURM_NODELIST > hostname
 * Only active when TRAILHEAD_ENABLED — no-op in normal builds.
 * ──────────────────────────────────────────────────────────────────────── */
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

static int _th_lock_fd = -1;

static inline void _th_acquire_node_lock(void) {
    if (!_th_enabled()) return;
    if (_th_lock_fd >= 0) return;
    const char* node = getenv("SLURMD_NODENAME");
    if (!node) node = getenv("SLURM_NODELIST");
    if (!node) node = "local";
    char path[512];
    snprintf(path, sizeof(path), "/tmp/trailhead_%s.lock", node);
    _th_lock_fd = open(path, O_CREAT | O_RDWR, 0666);
    if (_th_lock_fd < 0) return;
    flock(_th_lock_fd, LOCK_EX);
}

static inline void _th_release_node_lock(void) {
    if (_th_lock_fd < 0) return;
    flock(_th_lock_fd, LOCK_UN);
    close(_th_lock_fd);
    _th_lock_fd = -1;
}

#define TH_ACQUIRE_NODE_LOCK() _th_acquire_node_lock()
#define TH_RELEASE_NODE_LOCK() _th_release_node_lock()

/* ── C++ extras ──────────────────────────────────────────────────────────
 * TH_SCOPE("label"): RAII guard that emits timing at end of scope.
 * ──────────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
#include <ctime>
#include <cstdio>
#include <cstring>
#include <string>

namespace trailhead {

/* Accept both const char* and std::string in macro arguments. */
inline const char* to_cstr(const char* s)        { return s; }
inline const char* to_cstr(const std::string& s)  { return s.c_str(); }

struct TimingScope {
    char label[128];
    struct timespec start;
    bool enabled;
    TimingScope(const char* lbl) : enabled(_th_enabled()) {
        if (!enabled) return;
        size_t n = strlen(lbl);
        if (n >= sizeof(label)) n = sizeof(label) - 1;
        memcpy(label, lbl, n);
        label[n] = '\0';
        clock_gettime(CLOCK_MONOTONIC, &start);
    }
    ~TimingScope() {
        if (!enabled) return;
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double ms = ((double)(end.tv_sec  - start.tv_sec)  * 1e3)
                  + ((double)(end.tv_nsec - start.tv_nsec) * 1e-6);
        printf("TRAILHEAD:time:%s:%.6f\n", label, ms);
        fflush(stdout);
    }
    TimingScope(const TimingScope&) = delete;
    TimingScope& operator=(const TimingScope&) = delete;
};

} // namespace trailhead

/* TH_SCOPE("label") — declare in a block, timing emitted at } */
#define TH_SCOPE(label) \
    trailhead::TimingScope _th_scope_##__LINE__(label)

/* ── Redefine macros for C++ to accept both const char* and std::string ── */
#undef TH_PASS
#undef TH_FAIL
#undef TH_CHECK
#undef TH_META
#undef TH_TIME_EMIT
#undef TH_OUTPUT
#undef TH_OUTPUT_START
#undef TH_OUTPUT_STOP

#define TH_PASS(label) \
    do { if (_th_enabled()) { printf("TRAILHEAD:pass:%s\n", trailhead::to_cstr(label)); fflush(stdout); } } while(0)

#define TH_FAIL(label) \
    do { if (_th_enabled()) { printf("TRAILHEAD:fail:%s\n", trailhead::to_cstr(label)); fflush(stdout); } } while(0)

#define TH_CHECK(cond, label) \
    do { if (cond) TH_PASS(label); else TH_FAIL(label); } while(0)

#define TH_META(key, value) \
    do { if (_th_enabled()) { printf("TRAILHEAD:meta:%s:%s\n", trailhead::to_cstr(key), trailhead::to_cstr(value)); fflush(stdout); } } while(0)

#define TH_TIME_EMIT(label, elapsed_ms) \
    do { if (_th_enabled()) { printf("TRAILHEAD:time:%s:%.6f\n", trailhead::to_cstr(label), (double)(elapsed_ms)); fflush(stdout); } } while(0)

#define TH_OUTPUT(text) \
    do { if (_th_enabled()) { printf("TRAILHEAD:output:%s\n", trailhead::to_cstr(text)); fflush(stdout); } } while(0)

#define TH_OUTPUT_START() \
    do { if (_th_enabled()) { printf("TRAILHEAD:output_start\n"); fflush(stdout); } } while(0)

#define TH_OUTPUT_STOP() \
    do { if (_th_enabled()) { printf("TRAILHEAD:output_stop\n"); fflush(stdout); } } while(0)

#endif /* __cplusplus */

#endif /* TRAILHEAD_H */
