/*
 * Example using trailhead.h (not reporter.hpp) — results via stdout markers.
 * Useful when you don't want RAII or file I/O in the test binary itself.
 *
 * Build: g++ -std=c++17 -I../include example_header_test.cpp -o example_header_test
 */
#include "trailhead/trailhead.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

int main() {
    TH_META("version", "1.0");
    TH_META("description", "trailhead.h stdout-marker example");

    // RAII timing scope
    {
        TH_SCOPE("iota_sort");
        std::vector<int> v(50000);
        std::iota(v.begin(), v.end(), 0);
        std::reverse(v.begin(), v.end());
        std::sort(v.begin(), v.end());
        TH_CHECK(v.front() == 0 && v.back() == 49999, "sort correctness");
    }

    // Manual timing
    TH_TIME_BEGIN(sum);
    long long s = 0;
    for (int i = 0; i < 1000000; ++i) s += i;
    TH_TIME_END(sum);
    TH_CHECK(s == 499999500000LL, "sum correctness");

    // Emit a pre-computed timing
    TH_TIME_EMIT("constant", 0.001);

    TH_PASS("final check");
    return 0;
}
