cmake_minimum_required(VERSION 3.13)
project(simple-profiler)
find_package(Threads)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-std=c++17")

# add_link_options("-B/usr/local/libexec/mold")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wl,--no-as-needed -Wall -Wextra -Wundef")

include_directories(perfetto/sdk)
add_library(perfetto STATIC perfetto/sdk/perfetto.cc)

add_library(simple-profiler STATIC src/VisualProfiler.hpp)
target_link_libraries(simple-profiler perfetto pthread ${CMAKE_THREAD_LIBS_INIT})

add_executable(test-simple-profiler src/main.cc)
target_compile_definitions(test-simple-profiler PUBLIC PROFILING)
target_link_libraries(test-simple-profiler perfetto pthread ${CMAKE_THREAD_LIBS_INIT})