#pragma once
#include <functional>
#include <utility>
#include "APCOrchestrators/SNTOComputerBridgeAndView.hpp"

namespace BidirectionalInMemGraph
{

    class AdaptivePackedCellContainer;

    class FabricToAPCLinker 
    {
        friend class APCFinilizer;
    public:
        enum class SeqLockedOperation : uint8_t
        {
            FOUND = 0,
            NONE = 1,
            RETRY = 2
        };

        uint32_t GetThisSlotIdx() noexcept
        {
            return
                IsActiveAPC() ? APCCache_.APCSlotIdx_ : ADS::APC_INDEX_BOUND_SENTINAL;
        }

        bool IsActiveAPC() noexcept;

    protected:

        ADS::CacheOfAPC APCCache_{};

        struct RelationOparation 
        {
            APCUseScope Use_{};
            uint32_t RelationLocator_ = UINT32_MAX;
            SeqLockedOperation MutationOP_ = SeqLockedOperation::NONE;
        };

        bool IsFabricBound_() const noexcept;

        APCUseScope AcquireAPCUse_() noexcept;

        void ReleseFabricBindingOnly_() noexcept;

        bool BindExternalRawFabricBacking_(
            uint64_t* raw_cells_ptr,
            APCFinilizer* fabric_owner,
            uint64_t fabric_slot_idx,
            uint64_t* generation_cell,
            uint32_t expected_generation
        ) noexcept;

        bool InitiateAPCMetaHeader() noexcept;

        bool ReadAPCMetaUnit(
            ADS::HeaderIdentifierOfAPC meta_idx,
            uint64_t& return_value
        ) noexcept;

    };
        
    
}