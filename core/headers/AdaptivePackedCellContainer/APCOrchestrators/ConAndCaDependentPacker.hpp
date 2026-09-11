#pragma once 
#include "APCDataStructure.hpp"

namespace BidirectionalInMemGraph 
{
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