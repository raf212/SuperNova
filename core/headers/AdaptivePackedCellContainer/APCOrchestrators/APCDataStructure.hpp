
#pragma once 
#include <array>
#include <utility>
#include "SharedConf.hpp"

namespace BidirectionalInMemGraph
{

    class APCFinilizer;
    class AdaptivePackedCellContainer;

    struct APCDataStructure 
    {
        enum class HeaderIdentifierOfAPC : uint8_t
        {
            // Identity
            MAGIC_ID = 0,
            APC_SLOT_IDX = 1,
            GHGF_ROLE_CELL = 2,
            EOF_APC_HEADER                      = 7
        };

        static constexpr uint8_t META_CELL_COUNT = static_cast<uint8_t>(HeaderIdentifierOfAPC::EOF_APC_HEADER) + 1u;

        static constexpr uint32_t BRANCH_MAGIC = 0x41504342u;//big-endian
        static constexpr uint32_t EOF_HEADER = 0x72616600;//big-endian
        static constexpr uint32_t APC_INDEX_BOUND_SENTINAL = UINT32_MAX;
        static constexpr size_t APC_CACHELINE_SIZE = 64u;
        static constexpr uint8_t DEFAULT_DIRECTED_PARENT_PER_AXIS = 8u;
        //Later transfer it to GHGF side.
        static constexpr uint8_t GHGF_MAX_DIRECTED_PARENT_PER_AXIS  = 64u;        

        struct RangeOfAPC
        {
            size_t BeginIndex = UNSIGNED_ZERO;
            size_t EndIndex = UNSIGNED_ZERO;
            bool IsValid = false;
        };

        static constexpr uint8_t CountOfMacroColumn() noexcept
        {
            return static_cast<uint8_t>(MacroColumnOfAPC::FREE_SLOT) - static_cast<uint8_t>(MacroColumnOfAPC::BOTTOM_UP_SLOT) + 1;
        }

        static constexpr uint16_t RegionBit(MacroColumnOfAPC column) noexcept
        {
            return static_cast<uint16_t>(uint16_t{1u} << static_cast<uint16_t>(column));
        }

        static constexpr uint16_t ValidRegionMask() noexcept
        {
            return static_cast<uint16_t>((uint16_t{1u} << CountOfMacroColumn()) - 1u);
        }

        static constexpr std::optional<uint8_t> CompactRegionIndex(
            uint16_t active_mask,
            MacroColumnOfAPC column
        ) noexcept
        {
            if (
                (active_mask & RegionBit(column)) == UNSIGNED_ZERO ||
                (active_mask & static_cast<uint16_t>(~ValidRegionMask())) != UNSIGNED_ZERO
            )
            {
                return std::nullopt;
            }

            const uint16_t lower = static_cast<uint16_t>(active_mask & static_cast<uint16_t>(RegionBit(column) - 1u));
            return static_cast<uint8_t>(std::popcount(lower));
        }

        static constexpr bool IsValid32BitAPCUnit(uint64_t index) noexcept
        {
            return index < APC_INDEX_BOUND_SENTINAL;
        }

        static constexpr bool IsValidFabricUnit(uint64_t index) noexcept
        {
            return index < FABRIC_CELL_SENTINAL;
        }

        static constexpr bool IsCapacityOfAPCValid(uint64_t capacity) noexcept
        {
            return capacity >= MINIMUM_APC_CELL_COUNT &&
                IsValid32BitAPCUnit(capacity);
        }

        static constexpr bool IsValidEven64(uint64_t value) noexcept
        {
            return 
                (value & 1u) == UNSIGNED_ZERO;
        }

        struct CacheOfAPC
        {
            APCFinilizer* FabricOwnerPtr_{nullptr};
            std::byte* RawAPCBasePtr_{nullptr};
            uint32_t APCSlotIdx_{APCDataStructure::APC_INDEX_BOUND_SENTINAL};
            uint64_t* GenerationCellPtr_{nullptr};
            uint32_t CurrentGeneration_{UNSIGNED_ZERO};
        };
    };

    using ADS = APCDataStructure;


    class APCUseScope final
    {
        friend class FabricToAPCLinker;
    private:
        uint64_t* ControlCell_{nullptr};
        explicit APCUseScope(uint64_t* control_cell) noexcept
            : ControlCell_(control_cell)
        {}
    
    public:
        constexpr APCUseScope() noexcept = default;

        APCUseScope(const APCUseScope&) = delete;
        APCUseScope& operator = (const APCUseScope&) = delete;

        APCUseScope(APCUseScope&& other) noexcept
            :ControlCell_(std::exchange(other.ControlCell_, nullptr))
        {}

        APCUseScope& operator = (APCUseScope&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            Release();
            ControlCell_ = std::exchange(other.ControlCell_, nullptr);
            return *this;
        }

        ~APCUseScope() noexcept
        {
            Release();
        }

        explicit constexpr operator bool() const noexcept
        {
            return ControlCell_ != nullptr;
        }

        void Release() noexcept
        {
            if (!ControlCell_)
            {
                return;
            }
            std::atomic_ref<uint64_t>(*ControlCell_).fetch_sub(1u, std::memory_order_release);
            ControlCell_ = nullptr;
        }
    };


    struct TwinU32ToU64
    {
        static constexpr uint64_t PackDoubleUnsigned32In64(uint32_t low_32, uint32_t high_32) noexcept
        {
            return (
                (uint64_t{low_32} << UNSIGNED_ZERO) |
                (uint64_t{high_32} << BIT_COUNT_OF_UINT32_T)
            );
        }

        static constexpr uint32_t ExtractLow32Of64(uint64_t packed_value) noexcept
        {
            return static_cast<uint32_t>((packed_value >> UNSIGNED_ZERO) & MaskLowBitsForU64(BIT_COUNT_OF_UINT32_T));
        }

        static constexpr uint32_t ExtractHigh32Of64(uint64_t packed_value) noexcept
        {
            return static_cast<uint32_t>((packed_value >> BIT_COUNT_OF_UINT32_T) & MaskLowBitsForU64(BIT_COUNT_OF_UINT32_T));
        }
    };
    
}