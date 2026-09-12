cmake_minimum_required(VERSION 3.14)

project(SuperNova LANGUAGES CXX)

# ============================================================
# Global configuration
# ============================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For single-config generators such as Ninja/Makefiles:
# default to Release, NOT Debug.
#
# NOTE:
# Visual Studio is a multi-config generator, so CMAKE_BUILD_TYPE
# does not choose Debug/Release there. Build with:
#
#   cmake --build build --config Release
#
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Choose the build type."
        FORCE
    )
endif()

# ============================================================
# Performance options
# ============================================================

option(
    SUPERNOVA_ENABLE_IPO
    "Enable interprocedural/link-time optimization in Release"
    ON
)

option(
    SUPERNOVA_NATIVE_CPU
    "Generate CPU-specific instructions for this machine"
    OFF
)

option(
    SUPERNOVA_FAST_FP
    "Allow relaxed floating-point semantics for speed"
    OFF
)

# ============================================================
# IPO / LTO support
# ============================================================

include(CheckIPOSupported)

set(SUPERNOVA_IPO_SUPPORTED OFF)

if(SUPERNOVA_ENABLE_IPO)
    check_ipo_supported(
        RESULT SUPERNOVA_IPO_SUPPORTED
        OUTPUT SUPERNOVA_IPO_ERROR
        LANGUAGES CXX
    )

    if(SUPERNOVA_IPO_SUPPORTED)
        message(STATUS "SuperNova: IPO/LTO supported and enabled for Release")
    else()
        message(WARNING
            "SuperNova: IPO/LTO requested but unavailable:\n"
            "${SUPERNOVA_IPO_ERROR}"
        )
    endif()
endif()

# ============================================================
# Paths
# ============================================================

set(HEADERS_DIR  "${CMAKE_SOURCE_DIR}/headers")
set(SRC_DIR      "${CMAKE_SOURCE_DIR}/src")
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/bindings")

# This is a PATH, not a boolean option.
set(
    PYBIND11_DIR
    ""
    CACHE PATH
    "Path to a local pybind11 checkout (optional)"
)

# ============================================================
# pybind11
# ============================================================

if(PYBIND11_DIR)
    if(EXISTS "${PYBIND11_DIR}/CMakeLists.txt")
        message(STATUS "Using local pybind11 at: ${PYBIND11_DIR}")
        add_subdirectory("${PYBIND11_DIR}" EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "PYBIND11_DIR set but not found: ${PYBIND11_DIR}"
        )
    endif()
else()
    include(FetchContent)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.1
    )

    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================
# Optional libnuma
# ============================================================

find_library(NUMALIB NAMES numa libnuma)

if(NUMALIB)
    message(STATUS "Found libnuma: ${NUMALIB}")
    set(HAVE_LIBNUMA 1)
else()
    message(STATUS
        "libnuma not found; building without NUMA support (optional)."
    )
endif()

# ============================================================
# Sources
# ============================================================

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
        "Expecting at least one .cpp file."
    )
endif()

# ============================================================
# Optimization helpers
# ============================================================

function(atomiccim_optimize_compile TARGET_NAME)

    # --------------------------------------------------------
    # MSVC
    # --------------------------------------------------------

    if(MSVC)

        target_compile_options(
            ${TARGET_NAME}
            PRIVATE

            # Warnings in every configuration
            /W4
            /WX

            # ---------------- RELEASE ONLY ----------------

            # Maximum optimization for speed.
            $<$<CONFIG:Release>:/O2>

            # More aggressive automatic inlining than /Ob2.
            # Important for APC's many tiny helper functions.
            $<$<CONFIG:Release>:/Ob3>

            # Replace suitable CRT/function calls with intrinsics.
            $<$<CONFIG:Release>:/Oi>

            # Favor speed rather than size.
            $<$<CONFIG:Release>:/Ot>

            # Package functions separately so linker can
            # discard/fold/reorganize them.
            $<$<CONFIG:Release>:/Gy>

            # Whole-program global-data optimization support.
            $<$<CONFIG:Release>:/Gw>
        )

        # CPU-specific build.
        #
        # Only turn this ON if the machine that will execute
        # SuperNova supports AVX2.
        if(SUPERNOVA_NATIVE_CPU)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                $<$<CONFIG:Release>:/arch:AVX2>
            )
        endif()

        # Deliberately NOT default.
        #
        # /fp:fast allows transformations that can alter
        # floating-point numerical semantics.
        if(SUPERNOVA_FAST_FP)
            target_compile_options(
                ${TARGET_NAME}
                PRIVATE
                $<$<CONFIG:Release>:/fp:fast>
            )
        endif()

    # --------------------------------------------------------
    # GCC / Clang
    # --------------------------------------------------------

    else()

        target_compile_options(
            ${TARGET_NAME}
            PRIVATE

            -Wall
            -Wextra
            -Wpedantic

            $<$<CONFIG:Release>:-O3>
        )

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

    endif()

    # Explicitly define NDEBUG in Release.
    target_compile_definitions(
        ${TARGET_NAME}
        PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
    )

    # Enable cross-translation-unit optimization.
    if(SUPERNOVA_IPO_SUPPORTED)
        set_property(
            TARGET ${TARGET_NAME}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()

endfunction()


function(atomiccim_optimize_link TARGET_NAME)

    if(MSVC)

        target_link_options(
            ${TARGET_NAME}
            PRIVATE

            # Remove unreferenced functions/data.
            $<$<CONFIG:Release>:/OPT:REF>

            # Fold identical COMDATs/functions.
            $<$<CONFIG:Release>:/OPT:ICF>

            # LTCG and incremental linking should not be mixed.
            $<$<CONFIG:Release>:/INCREMENTAL:NO>
        )

    endif()

endfunction()

# ============================================================
# Core static library
# ============================================================

add_library(
    atomiccim_core
    STATIC
    ${CORE_SOURCES}
    ${CORE_HEADERS}
)

target_include_directories(
    atomiccim_core
    PUBLIC
        ${HEADERS_DIR}
)

target_compile_features(
    atomiccim_core
    PUBLIC
        cxx_std_20
)

atomiccim_optimize_compile(atomiccim_core)

# Better IDE organization.
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

if(NUMALIB)
    target_compile_definitions(
        atomiccim_core
        PRIVATE
        HAVE_LIBNUMA=1
    )

    target_link_libraries(
        atomiccim_core
        PRIVATE
        ${NUMALIB}
    )
endif()

# ============================================================
# CLI executable
# ============================================================

set(MAIN_SRC "${CMAKE_SOURCE_DIR}/main.cpp")

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
        ${HEADERS_DIR}
    )

    atomiccim_optimize_compile(SuperNova)
    atomiccim_optimize_link(SuperNova)

endif()

# ============================================================
# Python bindings
# ============================================================

get_filename_component(
    SUPERNOVA_REPOSITORY_DIR
    "${CMAKE_SOURCE_DIR}"
    DIRECTORY
)

file(GLOB_RECURSE BINDING_SRCS CONFIGURE_DEPENDS
    "${BINDINGS_DIR}/*.cpp"
    "${BINDINGS_DIR}/*.cc"
    "${BINDINGS_DIR}/*.cxx"
)

if(BINDING_SRCS)

    set(
        PREFERRED_BINDING
        "${BINDINGS_DIR}/PackedCellCoreBindings.cpp"
    )

    if(EXISTS "${PREFERRED_BINDING}")
        set(MODULE_SRC "${PREFERRED_BINDING}")
    else()
        list(GET BINDING_SRCS 0 MODULE_SRC)
        message(STATUS "Using binding source: ${MODULE_SRC}")
    endif()

    pybind11_add_module(
        SuperNovaBind
        ${MODULE_SRC}
    )

    target_link_libraries(
        SuperNovaBind
        PRIVATE
        atomiccim_core
    )

    target_include_directories(
        SuperNovaBind
        PRIVATE
        ${HEADERS_DIR}
        ${BINDINGS_DIR}
    )

    atomiccim_optimize_compile(SuperNovaBind)
    atomiccim_optimize_link(SuperNovaBind)

    source_group(
        TREE "${BINDINGS_DIR}"
        PREFIX "Binding Files"
        FILES ${BINDING_SRCS}
    )

    # Put the Python extension beside apc.py.
    set_target_properties(
        SuperNovaBind
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                "${SUPERNOVA_REPOSITORY_DIR}"

            RUNTIME_OUTPUT_DIRECTORY
                "${SUPERNOVA_REPOSITORY_DIR}"
    )

    # Multi-config generators such as Visual Studio.
    foreach(CONFIG_TYPE DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)

        set_target_properties(
            SuperNovaBind
            PROPERTIES
                "LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE}"
                    "${SUPERNOVA_REPOSITORY_DIR}"

                "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE}"
                    "${SUPERNOVA_REPOSITORY_DIR}"
        )

    endforeach()

else()

    message(STATUS
        "No binding sources found in bindings/ or its "
        "subdirectories — Python module will not be built."
    )

endif()

# ============================================================
# Convenience targets
# ============================================================

if(TARGET SuperNovaBind)

    add_custom_target(
        build_python_module

        COMMAND
            ${CMAKE_COMMAND}
            --build ${CMAKE_BINARY_DIR}
            --target SuperNovaBind
            --config $<CONFIG>

        COMMENT
            "Build Python extension module (SuperNovaBind)"
    )

endif()

add_custom_target(
    build_core

    COMMAND
        ${CMAKE_COMMAND}
        --build ${CMAKE_BINARY_DIR}
        --target atomiccim_core
        --config $<CONFIG>

    COMMENT
        "Build atomiccim_core"
)