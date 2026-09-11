
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
            APC_LIFE_CYCLE                      = 6,
            EOF_APC_HEADER                      = 7
        };

        static constexpr uint8_t META_CELL_COUNT = static_cast<uint8_t>(HeaderIdentifierOfAPC::EOF_APC_HEADER) + 1u;

        static constexpr uint32_t BRANCH_MAGIC = 0x41504342u;//big-endian
        static constexpr uint32_t EOF_HEADER = 0x72616600;//big-endian
        static constexpr uint32_t APC_INDEX_BOUND_SENTINAL = UINT32_MAX;
        static constexpr size_t APC_CACHELINE_SIZE = 64u;
        static constexpr uint8_t DEFAULT_DIRECTED_PARENT_PER_AXIS = 8u;
        static constexpr uint8_t COMPILED_MAX_DIRECT_PARENTS_PER_AXIS = 64u;

        struct RangeOfAPC
        {
            size_t BeginIndex = UNSIGNED_ZERO;
            size_t EndIndex = UNSIGNED_ZERO;
            bool IsValid = false;
        };

        static constexpr uint8_t CountOfMacroColumn() noexcept
        {
            return static_cast<uint8_t>(MacroColumnOfAPC::FREE_SLOT) - static_cast<uint8_t>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE) + 1;
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

        static constexpr bool InLimitOfUint8(uint32_t version) noexcept
        {
            return version < UINT8_MAX &&
                version > UNSIGNED_ZERO;
        }

        static constexpr bool IsCapacityOfAPCValid(uint64_t capacity) noexcept
        {
            return capacity >= MINIMUM_APC_CELL_COUNT &&
                IsValid32BitAPCUnit(capacity);
        }

        static constexpr bool IsPowerOfTwoValue(uint64_t value) noexcept
        {
            return value != UNSIGNED_ZERO && (value & (value - 1u)) == UNSIGNED_ZERO;
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
            uint64_t* APCGenerationCellPtr_{nullptr};
            uint32_t ExpectedGeneration_{UNSIGNED_ZERO};
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


    struct Twin28Plus8
    {
        static constexpr uint8_t LEN_OF_28_BIT = 28u;
        static constexpr uint32_t UINT28_MAX = UINT32_MAX & MaskLowBitsForU32(LEN_OF_28_BIT);

        struct CarrierTwin28
        {
            uint32_t Lowest28bit = UINT32_MAX;
            uint32_t Mid28Bit = UINT32_MAX;
            uint8_t High8Bit = UINT8_MAX;
            bool IsValid = false;
        };
        
        static constexpr bool IsCarrierValid(CarrierTwin28& carrier) noexcept
        {
            carrier.IsValid = 
                carrier.Lowest28bit <= UINT28_MAX &&
                carrier.Mid28Bit <= UINT28_MAX &&
                carrier.High8Bit <= UINT8_MAX;
            return carrier.IsValid;
        }

        static constexpr std::optional<uint64_t> PackValues(
            CarrierTwin28& carrier
        ) noexcept
        {
            if (!IsCarrierValid(carrier))
            {
                return std::nullopt;
            }
            
            return(
                (uint64_t(carrier.Lowest28bit) << UNSIGNED_ZERO) |
                (uint64_t(carrier.Mid28Bit) << LEN_OF_28_BIT) |
                (uint64_t(carrier.High8Bit) << (LEN_OF_28_BIT * 2))
            );
        }

        static constexpr CarrierTwin28 UnpackUnitToCarrier(uint64_t value) noexcept
        {
            CarrierTwin28 carrier{};
            carrier.Lowest28bit = static_cast<uint32_t>((value >> UNSIGNED_ZERO) & MaskLowBitsForU64(LEN_OF_28_BIT));
            carrier.Mid28Bit = static_cast<uint32_t>((value >> LEN_OF_28_BIT) & MaskLowBitsForU64(LEN_OF_28_BIT));
            carrier.High8Bit = static_cast<uint8_t>((value >> (LEN_OF_28_BIT * 2)) & MaskLowBitsForU64(8u));
            IsCarrierValid(carrier);
            return carrier;
        }

    };
    
}