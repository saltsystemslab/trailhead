/*
 * Example test using trailhead.h — pure C, no C++ required.
 * Build: cc -std=c11 -I../include example_c_test.c -o example_c_test
 * Run:   ./example_c_test
 */
#include "trailhead/trailhead.h"
#include <stdlib.h>
#include <string.h>

static int compare_int(const void* a, const void* b) {
    return *(const int*)a - *(const int*)b;
}

int main(void) {
    TH_META("language", "C");
    TH_META("description", "C example using trailhead.h macros");

    /* Timed sort */
    int arr[10000];
    for (int i = 0; i < 10000; i++) arr[i] = 10000 - i;

    TH_TIME_BEGIN(sort);
    qsort(arr, 10000, sizeof(int), compare_int);
    TH_TIME_END(sort);

    TH_CHECK(arr[0] == 1,     "sort: first element");
    TH_CHECK(arr[9999] == 10000, "sort: last element");

    /* String check */
    TH_TIME_BEGIN(strcmp);
    int eq = strcmp("trailhead", "trailhead") == 0;
    TH_TIME_END(strcmp);
    TH_CHECK(eq, "strcmp equal");

    /* Deliberate pass */
    TH_PASS("always passes");

    return 0;
}
