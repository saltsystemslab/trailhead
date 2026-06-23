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
    src/cli/remote_run.cpp
    src/cli/local_run.cpp
    src/cli/batch_run.cpp
    src/cli/datasets_runtime.cpp
    src/cli/matrix.cpp
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


# ── Install ────────────────────────────────────────────────────────────────
install(TARGETS trailhead DESTINATION bin)
install(DIRECTORY include/trailhead DESTINATION include)
