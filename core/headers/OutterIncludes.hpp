#pragma once

// ============================================================
// C++ STANDARD LIBRARY (core runtime, storage, numerical code)
// ============================================================
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ostream>
#include <random>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

// ============================================================
// WINDOWS (only when compiling for Windows)
// ============================================================
#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif

// ============================================================
// POSIX / LINUX / MACOS (only on supported Unix-like targets)
// ============================================================
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// ============================================================
// ADAPTIVECPP / SYCL (feature-gated, never required in CPU-only builds)
// Keep the existing SuperNova feature-macro compatibility.
// ============================================================
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP) && !defined(SN_ACPP_ENABLED)
    #define SN_ACPP_ENABLED
#endif

#if defined(SN_ACPP_ENABLED)
    #include <sycl/sycl.hpp>
#endif