cmake_minimum_required(VERSION 3.16)
project(trailhead VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── trailhead CLI binary ───────────────────────────────────────────────────
add_executable(trailhead
    src/main.cpp
    src/core/registry.cpp
    src/cli/visualizer.cpp
    src/cli/sbatch_gen.cpp
)

target_include_directories(trailhead PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Reporter header-only interface library ─────────────────────────────────
# Other projects can link this to get the include path for reporter.hpp.
add_library(trailhead_reporter INTERFACE)
target_include_directories(trailhead_reporter INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Examples ───────────────────────────────────────────────────────────────
option(TRAILHEAD_BUILD_EXAMPLES "Build example tests" ON)

if(TRAILHEAD_BUILD_EXAMPLES)
    # C++ reporter.hpp example
    add_executable(example_test examples/example_test.cpp)
    target_link_libraries(example_test PRIVATE trailhead_reporter)

    # C++ trailhead.h stdout-marker example
    add_executable(example_header_test examples/example_header_test.cpp)
    target_link_libraries(example_header_test PRIVATE trailhead_reporter)

    # Pure C trailhead.h example
    enable_language(C)
    add_executable(example_c_test examples/example_c_test.c)
    target_include_directories(example_c_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ── Install ────────────────────────────────────────────────────────────────
install(TARGETS trailhead DESTINATION bin)
install(DIRECTORY include/trailhead DESTINATION include)
