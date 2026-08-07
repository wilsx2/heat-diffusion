cmake_minimum_required(VERSION 3.18)
project(wacheat LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Parallelize
include(ProcessorCount)
ProcessorCount(N_CORES)
if(N_CORES GREATER 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N_CORES})
endif()

# Release optimizations
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release"
OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )
endif()

# Packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.0
)
FetchContent_MakeAvailable(spdlog)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
FetchContent_Declare(
    multi
    GIT_REPOSITORY https://gitlab.com/correaa/boost-multi.git
)
FetchContent_MakeAvailable(multi)

# Application
add_executable(wacheat
    src/main.cpp
)
target_include_directories(wacheat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(wacheat PUBLIC
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    argparse
    multi
    spdlog
)
