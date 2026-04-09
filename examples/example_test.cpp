// Example test using the trailhead reporter plugin.
// Build: g++ -std=c++17 -I../include example_test.cpp -o example_test
// Run:   ./example_test
// Then:  trailhead show example_test

#include "trailhead/reporter.hpp"
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>

// Simulate some work
static bool run_sort_test() {
    std::vector<int> v(100000);
    std::iota(v.begin(), v.end(), 0);
    std::reverse(v.begin(), v.end());
    std::sort(v.begin(), v.end());
    return v.front() == 0 && v.back() == 99999;
}

static bool run_sum_test() {
    long long sum = 0;
    for (int i = 1; i <= 1000000; ++i) sum += i;
    return sum == 500000500000LL;
}

int main() {
    trailhead::Reporter r("example_test");
    r.meta("description", "Example trailhead reporter usage");
    r.meta("data_size", "100000");

    // Timed region: sort test
    {
        auto scope = r.time_scope("sort");
        bool ok = run_sort_test();
        r.check(ok, "sort correctness");
    }

    // Timed region: sum test
    {
        auto scope = r.time_scope("sum");
        bool ok = run_sum_test();
        r.check(ok, "sum correctness");
    }

    // Manual timing
    r.timing_begin("sleep");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    r.timing_end("sleep");
    r.pass("sleep completed");

    // Deliberate failure example (comment out to see all pass)
    // r.check(false, "intentional failure");

    // Results written automatically by ~Reporter()
    return r.all_passed() ? 0 : 1;
}
