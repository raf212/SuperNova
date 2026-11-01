#pragma once

#include <type_traits>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <optional>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <cstddef>
#include <array>
#include <thread>
#include <stdexcept>
#include <memory>
#include <bit>
#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace BidirectionalInMemGraph 
{
    #define DEFAULT_MAX_TRIES 128
    static constexpr uint8_t LEN_OF_BYTE_IN_BITS = CHAR_BIT;
    static constexpr size_t BIT_COUNT_OF_UINT64_T = LEN_OF_BYTE_IN_BITS * sizeof(uint64_t);
    static constexpr uint8_t BIT_COUNT_OF_UINT32_T = LEN_OF_BYTE_IN_BITS * sizeof(uint32_t);
    static constexpr unsigned UNSIGNED_ZERO = 0u;
    static constexpr uint16_t MINIMUM_APC_CELL_COUNT = 128u;
    static constexpr uint64_t FABRIC_CELL_SENTINAL = UINT64_MAX;

    static constexpr uint8_t EDGE_COUNT = 2u;

    static constexpr uint64_t LIMIT_OF_CELL_COUNT = INT64_MAX / sizeof(uint64_t); //8 exabytes-limited by os

    enum class MacroColumnOfAPC : uint8_t
    {
        BOTTOM_UP_SLOT  = 0,
        TOP_DOWN_SLOT = 1,
        LATERAL_MESAGE = 2,
        STATE_SLOT = 3,
        ERROR_SLOT = 4,
        WEIGHTLESS_LOOKUP = 5,
        WEIGHT_SLOT = 6,
        AUX_SLOT = 7,
        EXTRA_SLOT = 8,
        FREE_SLOT     = 9
    };

    enum class FabricSegments : uint8_t
    {
        SLAB_RECORD_MAP = 0,
        MATRIX_VIEW_TABLE = 1,
        VALUE_PARENT_EDGE_TABLE_H = 2,
        VOLATILE_PARENT_EDGE_TABLE_V = 3,
        APC_HANDLE_TABLE = 4,
        WORK_QUEUE = 5,
        MODEL_DESCRIPTION = 6,
        SEGMENT_POOL = 7
    };

    enum class StateOfAPC : uint8_t
    {
        FREE = 0,
        RESERVED = 1,
        LIVE = 2,
        RETIRED = 3,
        HAULTED = 4
    };

    enum class PreBuildModels : uint8_t
    {
        NONE = 0,
        TRANSFORMER = 1,
        STRUCTRURAL_ADAPTATION = 2,
        GHGF = 3
    };
    static constexpr PreBuildModels LAST_DEFINED_MODEL_IDX = PreBuildModels::GHGF;
    
    static  constexpr uint64_t MaskLowBitsForU64(unsigned n) noexcept
    {
        if (n == UNSIGNED_ZERO) return uint64_t(0);
        if (n >= BIT_COUNT_OF_UINT64_T) return ~uint64_t(0);
        // produce low-n ones without shifting by >= width
        return ((uint64_t(1) << n) - 1u);                  
    }

    static  constexpr uint32_t MaskLowBitsForU32(unsigned n) noexcept
    {
        if (n == UNSIGNED_ZERO) return uint32_t(0);
        if (n >= BIT_COUNT_OF_UINT32_T) return ~uint32_t(0);
        // produce low-n ones without shifting by >= width
        return ((uint32_t(1) << n) - 1u);                  
    }
}
