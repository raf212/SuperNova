#pragma once
#include <functional>
#include "SchemaOrchestratorForRegion.hpp"

namespace BidirectionalInMemGraph
{

    struct DescriptionOfAPC 
    {

        
        struct SeqLockAndStateStruct
        {
            uint32_t SeqLock = UINT32_MAX;
            StateOfAPC StateOfTheAPC = StateOfAPC::RETIRED;
            bool IsValid = false;
        };

        static_assert(sizeof(SeqLockAndStateStruct) <= sizeof(uint64_t));

        static constexpr uint64_t ComposeSeqLockAndState(SeqLockAndStateStruct& files) noexcept
        {
            if (
                !ADS::IsValid32BitAPCUnit(files.SeqLock) ||
                !ValidateStateAgainstSeqLock(files)
            )
            {
                return FABRIC_CELL_SENTINAL;
            }
            return TwinU32ToU64::PackDoubleUnsigned32In64(files.SeqLock, static_cast<uint32_t>(files.StateOfTheAPC));
        }

        static constexpr bool GetSeqLockAndLifeCycle(
            uint64_t desc_id_state,
            SeqLockAndStateStruct& values
        ) noexcept
        {
            values = SeqLockAndStateStruct{};
            values.SeqLock = TwinU32ToU64::ExtractLow32Of64(desc_id_state);
            values.StateOfTheAPC = static_cast<StateOfAPC>(TwinU32ToU64::ExtractHigh32Of64(desc_id_state));

            if (!ADS::IsValidFabricUnit(values.SeqLock))
            {
                return false;
            }
            return ValidateStateAgainstSeqLock(values);
        }

        static constexpr bool ValidateStateAgainstSeqLock(SeqLockAndStateStruct& files) noexcept
        {
            if (!ADS::IsValidFabricUnit(files.SeqLock))
            {
                files.IsValid = false;
                return false;
            }
            if (
                files.StateOfTheAPC == StateOfAPC::RESERVED &&
                ADS::IsValidEven64(files.SeqLock)
            )
            {
                files.IsValid = false;
                return false;
            }
            if (
                files.StateOfTheAPC != StateOfAPC::RESERVED &&
                !ADS::IsValidEven64(files.SeqLock)
            )
            {
                files.IsValid = false;
                return false;
            }

            files.IsValid = true;
            return true;
        }

        static constexpr bool IsTransitionStateLeagal(StateOfAPC current_state, StateOfAPC desired_state) noexcept
        {
            return (current_state == StateOfAPC::FREE && desired_state == StateOfAPC::RESERVED) ||
                (current_state == StateOfAPC::RESERVED && desired_state == StateOfAPC::FREE) ||
                (current_state == StateOfAPC::RESERVED && desired_state == StateOfAPC::LIVE) ||
                (current_state == StateOfAPC::LIVE && desired_state == StateOfAPC::RESERVED) ||
                (current_state == StateOfAPC::RESERVED && desired_state == StateOfAPC::RETIRED) ||
                (current_state == StateOfAPC::RETIRED && desired_state == StateOfAPC::RESERVED) ||
                (current_state == StateOfAPC::LIVE && desired_state == StateOfAPC::HAULTED) ||
                (current_state == StateOfAPC::HAULTED && desired_state == StateOfAPC::LIVE);
                
        }

    };

    struct HeaderOrchestrator : DescriptionOfAPC
    {
        static constexpr uint8_t LEN_OF_APC_META_BUFFER_OR_COUNT = ADS::META_CELL_COUNT;

        using APCMetaBuffer = std::array<uint64_t, LEN_OF_APC_META_BUFFER_OR_COUNT>;

        static constexpr bool InitializeDefaultHeaderBuffer(
            APCMetaBuffer& header,
            uint32_t apc_slot_idx,
            uint32_t capacity_of_apc
        ) noexcept
        {
            for (uint64_t& word : header)
            {
                word = UNSIGNED_ZERO;
            }
            
            if (
                !ADS::IsCapacityOfAPCValid(capacity_of_apc) ||
                !ADS::IsValid32BitAPCUnit(apc_slot_idx)
            )
            {
                return false;
            }

            header[static_cast<std::size_t>(ADS::HeaderIdentifierOfAPC::MAGIC_ID)] = ADS::BRANCH_MAGIC;
            header[static_cast<std::size_t>(ADS::HeaderIdentifierOfAPC::APC_SLOT_IDX)] = apc_slot_idx;
            header[static_cast<std::size_t>(ADS::HeaderIdentifierOfAPC::EOF_APC_HEADER)] = ADS::EOF_HEADER;

            return true;
        }


    };
    

}