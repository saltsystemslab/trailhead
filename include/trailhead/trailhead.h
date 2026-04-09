/*
 * trailhead.h — lightweight C/C++ helper for reporting test results.
 *
 * Include this in any test (C or C++). No linking required.
 * Trailhead captures stdout and parses the TRAILHEAD: lines automatically.
 *
 * C usage:
 *   #include "trailhead/trailhead.h"
 *   int main() {
 *       TH_META("gpu", "H200");
 *       TH_TIME_BEGIN("construction");
 *       // ... work ...
 *       TH_TIME_END("construction");
 *       TH_CHECK(result == expected, "correctness");
 *   }
 *
 * C++ usage — RAII timing scope:
 *   {
 *       TH_SCOPE("sort");   // timing_end called automatically at }
 *       std::sort(v.begin(), v.end());
 *   }
 *   TH_CHECK(v.front() == 0, "sorted");
 */

#ifndef TRAILHEAD_H
#define TRAILHEAD_H

#include <stdio.h>
#include <time.h>

/* ── Output protocol ─────────────────────────────────────────────────────
 * Each marker is one line on stdout. Trailhead strips these lines from
 * the captured output and builds a result JSON from them.
 *
 *   TRAILHEAD:pass:<label>
 *   TRAILHEAD:fail:<label>
 *   TRAILHEAD:time:<label>:<elapsed_ms>
 *   TRAILHEAD:meta:<key>:<value>
 * ──────────────────────────────────────────────────────────────────────── */

#define TH_PASS(label) \
    (printf("TRAILHEAD:pass:%s\n", (label)), (void)fflush(stdout))

#define TH_FAIL(label) \
    (printf("TRAILHEAD:fail:%s\n", (label)), (void)fflush(stdout))

#define TH_CHECK(cond, label) \
    ((cond) ? TH_PASS(label) : TH_FAIL(label))

#define TH_META(key, value) \
    (printf("TRAILHEAD:meta:%s:%s\n", (key), (value)), (void)fflush(stdout))

/* ── Timing (C, POSIX) ───────────────────────────────────────────────────
 * TH_TIME_BEGIN / TH_TIME_END pair:
 *   TH_TIME_BEGIN("phase1");
 *   // ... work ...
 *   TH_TIME_END("phase1");
 *
 * TH_TIME_EMIT: emit a pre-computed duration in milliseconds:
 *   TH_TIME_EMIT("phase1", 42.5);
 * ──────────────────────────────────────────────────────────────────────── */

#define TH_TIME_EMIT(label, elapsed_ms) \
    (printf("TRAILHEAD:time:%s:%.6f\n", (label), (double)(elapsed_ms)), (void)fflush(stdout))

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

/* ── C++ extras ──────────────────────────────────────────────────────────
 * TH_SCOPE("label"): RAII guard that emits timing at end of scope.
 * ──────────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
#include <ctime>
#include <cstdio>
#include <cstring>

namespace trailhead {

struct TimingScope {
    char label[128];
    struct timespec start;
    TimingScope(const char* lbl) {
        /* strncpy + null-terminate */
        size_t n = strlen(lbl);
        if (n >= sizeof(label)) n = sizeof(label) - 1;
        memcpy(label, lbl, n);
        label[n] = '\0';
        clock_gettime(CLOCK_MONOTONIC, &start);
    }
    ~TimingScope() {
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

#endif /* __cplusplus */

#endif /* TRAILHEAD_H */
