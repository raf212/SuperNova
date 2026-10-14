cmake_minimum_required(VERSION 3.20)

project(TestSA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    SUPERNOVA_ROOT
    ""
    CACHE PATH
    "Path to the LCIM-BitTheorium repository root"
)

set(
    SUPERNOVA_BUILD_DIR
    ""
    CACHE PATH
    "Path to the already-built SuperNova AdaptiveCpp build directory"
)

if(NOT SUPERNOVA_ROOT)
    message(FATAL_ERROR "Set -DSUPERNOVA_ROOT=<repository root>.")
endif()

if(NOT SUPERNOVA_BUILD_DIR)
    message(FATAL_ERROR
        "Set -DSUPERNOVA_BUILD_DIR=<repository>/build-acpp-windows."
    )
endif()

if(NOT EXISTS "${SUPERNOVA_ROOT}/core/headers")
    message(FATAL_ERROR
        "SuperNova headers not found at "
        "${SUPERNOVA_ROOT}/core/headers"
    )
endif()

# AdaptiveCpp's CMake integration consumes ACPP_TARGETS.
if(NOT DEFINED ACPP_TARGETS OR "${ACPP_TARGETS}" STREQUAL "")
    set(
        ACPP_TARGETS
        "generic"
        CACHE STRING
        "AdaptiveCpp compilation targets"
    )
endif()

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(Threads REQUIRED)

if(NOT COMMAND add_sycl_to_target)
    message(FATAL_ERROR
        "AdaptiveCpp was found, but add_sycl_to_target() is unavailable."
    )
endif()

# Locate the core that was already built by:
#   cmake --build build-acpp-windows --target SuperNova
#
# For Ninja + clang-cl on Windows this is normally atomiccim_core.lib.
find_library(
    SUPERNOVA_CORE_LIBRARY
    NAMES
        atomiccim_core
        libatomiccim_core
    PATHS
        "${SUPERNOVA_BUILD_DIR}"
        "${SUPERNOVA_BUILD_DIR}/lib"
        "${SUPERNOVA_BUILD_DIR}/Release"
    NO_DEFAULT_PATH
)

if(NOT SUPERNOVA_CORE_LIBRARY)
    message(FATAL_ERROR
        "Could not find the already-built atomiccim_core library under "
        "${SUPERNOVA_BUILD_DIR}. Build SuperNova first, or set "
        "-DSUPERNOVA_CORE_LIBRARY=<full path to atomiccim_core.lib>."
    )
endif()

message(STATUS "TestSA: SuperNova root = ${SUPERNOVA_ROOT}")
message(STATUS "TestSA: SuperNova core = ${SUPERNOVA_CORE_LIBRARY}")
message(STATUS "TestSA: ACPP_TARGETS = ${ACPP_TARGETS}")

add_library(SuperNovaCore STATIC IMPORTED GLOBAL)

set_target_properties(
    SuperNovaCore
    PROPERTIES
        IMPORTED_LOCATION "${SUPERNOVA_CORE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SUPERNOVA_ROOT}/core/headers"
)

add_executable(
    TestSA
    TestSA.cpp
)

target_link_libraries(
    TestSA
    PRIVATE
        SuperNovaCore
        Threads::Threads
)

target_compile_definitions(
    TestSA
    PRIVATE
        SUPERNOVA_ENABLE_ADAPTIVECPP=1
)

# This is the important part: TestSA.cpp itself is compiled through
# AdaptiveCpp's SYCL compilation flow.
add_sycl_to_target(
    TARGET TestSA
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/TestSA.cpp"
)
