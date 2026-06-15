cmake_minimum_required(VERSION 3.16)

project(ScanExascaleReportAccelNTDu
    VERSION 1.0.0
    DESCRIPTION "HPC accelerator inventory tool for Linux (NPU/TPU/DPU)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ScanExascaleReportAccelNTDu
    ScanExascaleReportAccelNTDu.cpp
)

target_compile_options(ScanExascaleReportAccelNTDu PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(ScanExascaleReportAccelNTDu PRIVATE -O2)
endif()

install(TARGETS ScanExascaleReportAccelNTDu
    RUNTIME DESTINATION bin
)