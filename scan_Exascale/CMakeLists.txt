cmake_minimum_required(VERSION 3.14)
project(ScanExascale LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_VERBOSE "Enable default verbose build flags" ON)

if(ENABLE_VERBOSE)
    message(STATUS "Verbose build enabled")
endif()

# -----------------------
# hwloc
# -----------------------
find_package(PkgConfig QUIET)

set(HAVE_HWLOC FALSE)

# Try config-package mode first
find_package(hwloc QUIET CONFIG)
if(TARGET hwloc::hwloc)
    message(STATUS "Found hwloc via CMake config package")
    set(HAVE_HWLOC TRUE)
else()
    # Try pkg-config mode
    if(PkgConfig_FOUND)
        pkg_check_modules(HWLOC QUIET IMPORTED_TARGET hwloc)
        if(TARGET PkgConfig::HWLOC)
            message(STATUS "Found hwloc via pkg-config")
            set(HAVE_HWLOC TRUE)
        endif()
    endif()
endif()

if(NOT HAVE_HWLOC)
    message(FATAL_ERROR
        "hwloc not found. Install libhwloc-dev (or equivalent) and/or set hwloc_DIR/CMAKE_PREFIX_PATH.")
endif()

# -----------------------
# yaml-cpp
# -----------------------
find_package(yaml-cpp REQUIRED)

# -----------------------
# Target
# -----------------------
add_executable(ScanExascaleForLinux
    ScanExascaleForLinux.cpp
)

if(TARGET hwloc::hwloc)
    target_link_libraries(ScanExascaleForLinux PRIVATE hwloc::hwloc)
elseif(TARGET PkgConfig::HWLOC)
    target_link_libraries(ScanExascaleForLinux PRIVATE PkgConfig::HWLOC)
endif()

if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(ScanExascaleForLinux PRIVATE yaml-cpp::yaml-cpp)
else()
    target_link_libraries(ScanExascaleForLinux PRIVATE yaml-cpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ScanExascaleForLinux PRIVATE -Wall -Wextra -Wpedantic)
endif()