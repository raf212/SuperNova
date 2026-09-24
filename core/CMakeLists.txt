cmake_minimum_required(VERSION 3.14)

project(SuperNova LANGUAGES CXX)

# =============================================================================
# Global language / generator-independent configuration
# =============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++20 is the minimum, not the maximum. A caller may configure with
# -DCMAKE_CXX_STANDARD=23 (or newer) and this file will preserve that choice.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ language standard (minimum: 20)")
if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "SuperNova requires C++20 or newer.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Unix Makefiles, Ninja, etc.) use CMAKE_BUILD_TYPE.
# Multi-config generators (Visual Studio, Xcode, Ninja Multi-Config) select the
# configuration at build time, e.g. --config Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type for single-config generators"
        FORCE
    )
endif()

message(STATUS "SuperNova: generator = ${CMAKE_GENERATOR}")
message(STATUS "SuperNova: compiler  = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "SuperNova: C++ standard = C++${CMAKE_CXX_STANDARD}")

# =============================================================================
# Build options
# =============================================================================

option(
    SUPERNOVA_BUILD_CLI
    "Build the SuperNova CLI executable when main.cpp exists"
    ON
)

option(
    SUPERNOVA_BUILD_PYTHON
    "Build the optional pybind11 Python extension"
    OFF
)

option(
    SUPERNOVA_FETCH_PYBIND11
    "Allow CMake to download pybind11 when Python bindings are enabled"
    OFF
)

option(
    SUPERNOVA_ENABLE_NUMA
    "Use libnuma when both its header and library are available"
    ON
)

option(
    SUPERNOVA_ENABLE_IPO
    "Enable interprocedural/link-time optimization in Release when supported"
    ON
)

option(
    SUPERNOVA_NATIVE_CPU
    "Use native CPU tuning where the selected compiler has a true native option"
    OFF
)

option(
    SUPERNOVA_MSVC_AVX2
    "Explicitly enable /arch:AVX2 for MSVC-style compiler frontends"
    OFF
)

option(
    SUPERNOVA_FAST_FP
    "Allow relaxed floating-point semantics for speed"
    OFF
)

option(
    SUPERNOVA_WARNINGS_AS_ERRORS
    "Treat compiler warnings as build errors"
    OFF
)

# Optional local pybind11 checkout. This path is consulted only when
# SUPERNOVA_BUILD_PYTHON=ON.
set(
    PYBIND11_DIR
    ""
    CACHE PATH
    "Path to a local pybind11 checkout (optional)"
)

# =============================================================================
# Paths
# =============================================================================

set(HEADERS_DIR  "${CMAKE_SOURCE_DIR}/headers")
set(SRC_DIR      "${CMAKE_SOURCE_DIR}/src")
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/bindings")

# =============================================================================
# Required standard-library / threading capabilities
# =============================================================================

find_package(Threads REQUIRED)

include(CheckCXXSourceCompiles)

# SuperNova uses C++20 atomic_ref and atomic wait/notify. Some toolchains may
# accept -std=c++20 while shipping an older standard library, so fail early with
# a clear configure-time message instead of a confusing source error later.
set(_SUPERNOVA_ATOMIC_TEST_SOURCE [[
    #include <atomic>
    #include <cstdint>

    #ifndef __cpp_lib_atomic_wait
    #error "C++20 atomic wait/notify is unavailable"
    #endif

    int main()
    {
        alignas(std::atomic_ref<std::uint64_t>::required_alignment)
        std::uint64_t word = 0;

        std::atomic_ref<std::uint64_t> ref(word);
        ref.store(1u, std::memory_order_release);
        const auto value = ref.load(std::memory_order_acquire);

        std::atomic<std::uint32_t> signal{0u};
        signal.notify_all();

        return value == 1u ? 0 : 1;
    }
]])

set(SUPERNOVA_ATOMIC_LIBRARY "")
check_cxx_source_compiles(
    "${_SUPERNOVA_ATOMIC_TEST_SOURCE}"
    SUPERNOVA_CXX20_ATOMICS_WORK_WITHOUT_LIBATOMIC
)

if(NOT SUPERNOVA_CXX20_ATOMICS_WORK_WITHOUT_LIBATOMIC)
    find_library(SUPERNOVA_ATOMIC_LIBRARY NAMES atomic)

    if(SUPERNOVA_ATOMIC_LIBRARY)
        set(_SUPERNOVA_SAVED_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
        set(CMAKE_REQUIRED_LIBRARIES "${SUPERNOVA_ATOMIC_LIBRARY}")

        check_cxx_source_compiles(
            "${_SUPERNOVA_ATOMIC_TEST_SOURCE}"
            SUPERNOVA_CXX20_ATOMICS_WORK_WITH_LIBATOMIC
        )

        set(CMAKE_REQUIRED_LIBRARIES "${_SUPERNOVA_SAVED_REQUIRED_LIBRARIES}")
        unset(_SUPERNOVA_SAVED_REQUIRED_LIBRARIES)
    endif()

    if(NOT SUPERNOVA_CXX20_ATOMICS_WORK_WITH_LIBATOMIC)
        message(FATAL_ERROR
            "SuperNova requires a C++20 standard library with std::atomic_ref "
            "and atomic wait/notify support. The selected compiler/standard "
            "library could not compile and link those features."
        )
    endif()
endif()

unset(_SUPERNOVA_ATOMIC_TEST_SOURCE)

# =============================================================================
# IPO / LTO support
# =============================================================================

include(CheckIPOSupported)
set(SUPERNOVA_IPO_SUPPORTED OFF)

if(SUPERNOVA_ENABLE_IPO)
    check_ipo_supported(
        RESULT SUPERNOVA_IPO_SUPPORTED
        OUTPUT SUPERNOVA_IPO_ERROR
        LANGUAGES CXX
    )

    if(SUPERNOVA_IPO_SUPPORTED)
        message(STATUS "SuperNova: IPO/LTO available; enabling it for Release targets")
    else()
        message(STATUS
            "SuperNova: IPO/LTO unavailable; continuing without it: "
            "${SUPERNOVA_IPO_ERROR}"
        )
    endif()
endif()

# =============================================================================
# Optional NUMA support
# =============================================================================

set(HAVE_LIBNUMA OFF)

if(SUPERNOVA_ENABLE_NUMA AND UNIX AND NOT APPLE)
    find_path(NUMA_INCLUDE_DIR NAMES numa.h)
    find_library(NUMALIB NAMES numa libnuma)

    if(NUMA_INCLUDE_DIR AND NUMALIB)
        set(HAVE_LIBNUMA ON)
        message(STATUS "SuperNova: libnuma enabled (${NUMALIB})")
    else()
        message(STATUS "SuperNova: libnuma not found; building without NUMA support")
    endif()
else()
    message(STATUS "SuperNova: NUMA support disabled or not applicable on this platform")
endif()

# =============================================================================
# Sources
# =============================================================================

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cc"
    "${SRC_DIR}/*.cxx"
)

file(GLOB_RECURSE CORE_HEADERS CONFIGURE_DEPENDS
    "${HEADERS_DIR}/*.h"
    "${HEADERS_DIR}/*.hpp"
    "${HEADERS_DIR}/*.hh"
    "${HEADERS_DIR}/*.hxx"
)

if(NOT CORE_SOURCES)
    message(FATAL_ERROR
        "No source files found in ${SRC_DIR} or its subdirectories. "
        "SuperNova expects at least one .cpp/.cc/.cxx source file."
    )
endif()

# =============================================================================
# Compiler-family detection
# =============================================================================

# Covers cl.exe and compiler frontends that intentionally emulate the MSVC
# command-line/ABI (for example clang-cl / some IntelLLVM configurations).
set(SUPERNOVA_MSVC_STYLE_FRONTEND OFF)
if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set(SUPERNOVA_MSVC_STYLE_FRONTEND ON)
endif()

set(SUPERNOVA_GNU_STYLE_FRONTEND OFF)
if(
    NOT SUPERNOVA_MSVC_STYLE_FRONTEND AND
    CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$"
)
    set(SUPERNOVA_GNU_STYLE_FRONTEND ON)
endif()

# =============================================================================
# Per-target optimization / warning policy
# =============================================================================

function(supernova_optimize_compile TARGET_NAME)

    if(SUPERNOVA_MSVC_STYLE_FRONTEND)

        target_compile_options(
            ${TARGET_NAME}
            PRIVATE
                /W4
                $<$<CONFIG:Release>:/O2>
                # Aggressive automatic inlining. SuperNova has many tiny
                # façade/packing/validation helpers in hot APC/Fabric paths.
                $<$<CONFIG:Release>:/Ob3>
                $<$<CONFIG:Release>:/Oi>
                $<$<CONFIG:Release>:/Ot>
                $<$<CONFIG:Release>:/Gy>
                $<$<CONFIG:Release>:/Gw>
        )

        if(SUPERNOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${TARGET_NAME} PRIVATE /WX)
        endif()

        # MSVC does not provide a true equivalent of -march=native. Keep the
        # semantic distinction explicit instead of silently calling AVX2
        # "native".
        if(SUPERNOVA_NATIVE_CPU)
            message(WARNING
                "SUPERNOVA_NATIVE_CPU has no exact MSVC-style equivalent. "
                "No ISA flag was added. Use SUPERNOVA_MSVC_AVX2=ON explicitly "
                "if an AVX2-only binary is acceptable."
            )
        endif()

        if(SUPERNOVA_MSVC_AVX2)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                    $<$<CONFIG:Release>:/arch:AVX2>
            )
        endif()

        if(SUPERNOVA_FAST_FP)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                    $<$<CONFIG:Release>:/fp:fast>
            )
        endif()

    elseif(SUPERNOVA_GNU_STYLE_FRONTEND)

        target_compile_options(
            ${TARGET_NAME}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                $<$<CONFIG:Release>:-O3>
        )

        if(SUPERNOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${TARGET_NAME} PRIVATE -Werror)
        endif()

        if(SUPERNOVA_NATIVE_CPU)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                    $<$<CONFIG:Release>:-march=native>
                    $<$<CONFIG:Release>:-mtune=native>
            )
        endif()

        if(SUPERNOVA_FAST_FP)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                    $<$<CONFIG:Release>:-ffast-math>
            )
        endif()

    else()

        # Unknown compiler: stay standards-only. CMake still supplies the
        # compiler's normal Release flags for the selected configuration.
        message(STATUS
            "SuperNova: no custom warning/optimization flags for compiler "
            "'${CMAKE_CXX_COMPILER_ID}'. Using standard CMake/toolchain flags."
        )

        if(SUPERNOVA_NATIVE_CPU)
            message(WARNING
                "SUPERNOVA_NATIVE_CPU requested, but no portable native-tuning "
                "flag is known for '${CMAKE_CXX_COMPILER_ID}'. Ignoring it."
            )
        endif()

        if(SUPERNOVA_FAST_FP)
            message(WARNING
                "SUPERNOVA_FAST_FP requested, but no compiler-specific fast-FP "
                "flag is configured for '${CMAKE_CXX_COMPILER_ID}'. Ignoring it."
            )
        endif()

        if(SUPERNOVA_WARNINGS_AS_ERRORS)
            message(WARNING
                "SUPERNOVA_WARNINGS_AS_ERRORS requested, but no compiler-specific "
                "warnings-as-errors flag is configured for '${CMAKE_CXX_COMPILER_ID}'."
            )
        endif()

    endif()

    # Explicit for consistency even though most Release toolchains already
    # define NDEBUG.
    target_compile_definitions(
        ${TARGET_NAME}
        PRIVATE
            $<$<CONFIG:Release>:NDEBUG>
    )

    if(SUPERNOVA_IPO_SUPPORTED)
        set_property(
            TARGET ${TARGET_NAME}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()

endfunction()

function(supernova_optimize_link TARGET_NAME)

    if(SUPERNOVA_MSVC_STYLE_FRONTEND)
        target_link_options(
            ${TARGET_NAME}
            PRIVATE
                $<$<CONFIG:Release>:/OPT:REF>
                $<$<CONFIG:Release>:/OPT:ICF>
                $<$<CONFIG:Release>:/INCREMENTAL:NO>
        )
    endif()

endfunction()

# =============================================================================
# Core library configuration
# =============================================================================

function(supernova_configure_core_target TARGET_NAME)

    # C++20 is the minimum. A caller may select C++23/26 globally.
    target_compile_features(
        ${TARGET_NAME}
        PUBLIC
            cxx_std_20
    )

    target_include_directories(
        ${TARGET_NAME}
        PUBLIC
            "${HEADERS_DIR}"
    )

    target_link_libraries(
        ${TARGET_NAME}
        PUBLIC
            Threads::Threads
    )

    if(SUPERNOVA_ATOMIC_LIBRARY)
        target_link_libraries(
            ${TARGET_NAME}
            PUBLIC
                "${SUPERNOVA_ATOMIC_LIBRARY}"
        )
    endif()

    if(HAVE_LIBNUMA)
        target_compile_definitions(
            ${TARGET_NAME}
            PRIVATE
                HAVE_LIBNUMA=1
        )

        target_include_directories(
            ${TARGET_NAME}
            PRIVATE
                "${NUMA_INCLUDE_DIR}"
        )

        target_link_libraries(
            ${TARGET_NAME}
            PRIVATE
                "${NUMALIB}"
        )
    endif()

    supernova_optimize_compile(${TARGET_NAME})

endfunction()

# =============================================================================
# Native core static library
# =============================================================================
#
# This target is intentionally NON-PIC unless the toolchain itself chooses PIC.
# It is the core linked into the native SuperNova executable and benchmarks.
# Python bindings must not force this target to become position-independent,
# because even a small code-generation difference is undesirable in ns-scale
# microbenchmarks.
# =============================================================================

add_library(
    atomiccim_core
    STATIC
    ${CORE_SOURCES}
    ${CORE_HEADERS}
)

supernova_configure_core_target(atomiccim_core)

# Better IDE organization; harmless for generators that ignore source groups.
source_group(
    TREE "${SRC_DIR}"
    PREFIX "Source Files"
    FILES ${CORE_SOURCES}
)

source_group(
    TREE "${HEADERS_DIR}"
    PREFIX "Header Files"
    FILES ${CORE_HEADERS}
)

# =============================================================================
# Optional PIC core for Python
# =============================================================================
#
# ELF/Mach-O Python extensions normally require position-independent code.
# Build the same core sources a second time only when Python is requested.
# This costs additional BUILD TIME but keeps the native CLI/benchmark core
# free from -fPIC/PIC code generation.
# =============================================================================

if(SUPERNOVA_BUILD_PYTHON)
    add_library(
        atomiccim_core_pic
        STATIC
        ${CORE_SOURCES}
        ${CORE_HEADERS}
    )

    set_property(
        TARGET atomiccim_core_pic
        PROPERTY POSITION_INDEPENDENT_CODE ON
    )

    supernova_configure_core_target(atomiccim_core_pic)
endif()

# =============================================================================
# Optional CLI executable
# =============================================================================

set(MAIN_SRC "${CMAKE_SOURCE_DIR}/main.cpp")

if(SUPERNOVA_BUILD_CLI)
    if(EXISTS "${MAIN_SRC}")
        add_executable(
            SuperNova
            "${MAIN_SRC}"
        )

        target_link_libraries(
            SuperNova
            PRIVATE
                atomiccim_core
        )

        target_include_directories(
            SuperNova
            PRIVATE
                "${HEADERS_DIR}"
        )

        supernova_optimize_compile(SuperNova)
        supernova_optimize_link(SuperNova)
    else()
        message(STATUS "SuperNova: main.cpp not found; CLI target will not be built")
    endif()
endif()

# =============================================================================
# Optional Python bindings
# =============================================================================

if(SUPERNOVA_BUILD_PYTHON)

    file(GLOB_RECURSE BINDING_SRCS CONFIGURE_DEPENDS
        "${BINDINGS_DIR}/*.cpp"
        "${BINDINGS_DIR}/*.cc"
        "${BINDINGS_DIR}/*.cxx"
    )

    if(NOT BINDING_SRCS)
        message(FATAL_ERROR
            "SUPERNOVA_BUILD_PYTHON=ON, but no binding source files were found "
            "under ${BINDINGS_DIR}."
        )
    endif()

    # Prefer a user-provided local checkout, then an already-installed package.
    # Network fetching is opt-in so the C++ core can always configure offline.
    if(PYBIND11_DIR)
        if(EXISTS "${PYBIND11_DIR}/CMakeLists.txt")
            message(STATUS "SuperNova: using local pybind11 at ${PYBIND11_DIR}")
            add_subdirectory("${PYBIND11_DIR}" EXCLUDE_FROM_ALL)
        else()
            message(FATAL_ERROR
                "PYBIND11_DIR was set, but no CMakeLists.txt exists at: "
                "${PYBIND11_DIR}"
            )
        endif()
    else()
        set(PYBIND11_FINDPYTHON ON)
        find_package(pybind11 CONFIG QUIET)

        if(NOT pybind11_FOUND)
            if(SUPERNOVA_FETCH_PYBIND11)
                include(FetchContent)

                FetchContent_Declare(
                    pybind11
                    GIT_REPOSITORY https://github.com/pybind/pybind11.git
                    GIT_TAG        v3.0.1
                    GIT_SHALLOW    TRUE
                )

                FetchContent_MakeAvailable(pybind11)
            else()
                message(FATAL_ERROR
                    "Python bindings were requested, but pybind11 was not found. "
                    "Either install pybind11, set PYBIND11_DIR, or configure with "
                    "-DSUPERNOVA_FETCH_PYBIND11=ON to allow downloading it."
                )
            endif()
        endif()
    endif()

    set(
        PREFERRED_BINDING
        "${BINDINGS_DIR}/PackedCellCoreBindings.cpp"
    )

    if(EXISTS "${PREFERRED_BINDING}")
        set(MODULE_SRC "${PREFERRED_BINDING}")
    else()
        list(GET BINDING_SRCS 0 MODULE_SRC)
        message(STATUS "SuperNova: using binding source ${MODULE_SRC}")
    endif()

    pybind11_add_module(
        SuperNovaBind
        ${MODULE_SRC}
    )

    target_link_libraries(
        SuperNovaBind
        PRIVATE
            atomiccim_core_pic
    )

    target_include_directories(
        SuperNovaBind
        PRIVATE
            "${HEADERS_DIR}"
            "${BINDINGS_DIR}"
    )

    supernova_optimize_compile(SuperNovaBind)
    supernova_optimize_link(SuperNovaBind)

    source_group(
        TREE "${BINDINGS_DIR}"
        PREFIX "Binding Files"
        FILES ${BINDING_SRCS}
    )

    # Preserve the original behavior: place the extension beside the repository
    # Python entry point rather than inside the build tree.
    get_filename_component(
        SUPERNOVA_REPOSITORY_DIR
        "${CMAKE_SOURCE_DIR}"
        DIRECTORY
    )

    set_target_properties(
        SuperNovaBind
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${SUPERNOVA_REPOSITORY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${SUPERNOVA_REPOSITORY_DIR}"
    )

    foreach(CONFIG_TYPE DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(
            SuperNovaBind
            PROPERTIES
                "LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE}" "${SUPERNOVA_REPOSITORY_DIR}"
                "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE}" "${SUPERNOVA_REPOSITORY_DIR}"
        )
    endforeach()

endif()

# =============================================================================
# Convenience targets
# =============================================================================

# These are dependency aliases, not recursive `cmake --build` invocations.
# Therefore they work with Ninja, Makefiles, Visual Studio, Xcode, etc.
add_custom_target(build_core DEPENDS atomiccim_core)

if(TARGET SuperNovaBind)
    add_custom_target(build_python_module DEPENDS SuperNovaBind)
endif()

# =============================================================================
# Configuration summary
# =============================================================================

message(STATUS "")
message(STATUS "SuperNova configuration summary")
message(STATUS "  C++ standard          : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build CLI             : ${SUPERNOVA_BUILD_CLI}")
message(STATUS "  Build Python bindings : ${SUPERNOVA_BUILD_PYTHON}")
message(STATUS "  Native core uses PIC   : NO (Python uses atomiccim_core_pic)")
message(STATUS "  Fetch pybind11        : ${SUPERNOVA_FETCH_PYBIND11}")
message(STATUS "  NUMA                   : ${HAVE_LIBNUMA}")
message(STATUS "  IPO/LTO                : ${SUPERNOVA_IPO_SUPPORTED}")
message(STATUS "  Native CPU tuning      : ${SUPERNOVA_NATIVE_CPU}")
message(STATUS "  MSVC AVX2              : ${SUPERNOVA_MSVC_AVX2}")
message(STATUS "  Fast FP                : ${SUPERNOVA_FAST_FP}")
message(STATUS "  Warnings as errors     : ${SUPERNOVA_WARNINGS_AS_ERRORS}")
message(STATUS "")
