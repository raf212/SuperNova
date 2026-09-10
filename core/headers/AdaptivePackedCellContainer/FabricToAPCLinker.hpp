#pragma once
#include <functional>
#include <utility>
#include "APCOrchestrators/ViewOrchestrator.hpp"

namespace BidirectionalInMemGraph
{
    
    class VagueTemoraryPremativeFabric;
    class AdaptivePackedCellContainer;

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

    struct CacheOfAPC
    {

    };

    class FabricToAPCLinker 
    {
        friend class VagueTemoraryPremativeFabric;
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
                IsActiveAPC() ? APCSlotIdx_ : APCDataStructure::APC_INDEX_BOUND_SENTINAL;
        }

        bool IsActiveAPC() noexcept;

    protected:
        VagueTemoraryPremativeFabric* FabricOwnerPtr_{nullptr};
        std::byte* RawAPCBasePtr_{nullptr};
        uint32_t APCSlotIdx_{APCDataStructure::APC_INDEX_BOUND_SENTINAL};
        uint64_t* APCGenerationCellPtr_{nullptr};
        uint32_t ExpectedGeneration_{UNSIGNED_ZERO};

        struct RelationOparation 
        {
            AdaptivePackedCellContainer* APCPtr_ = nullptr;
            uint32_t RelationLocator_ = UINT32_MAX;
            SeqLockedOperation MutationOP_ = SeqLockedOperation::NONE;
        };

        bool IsFabricBound_() const noexcept;

        APCUseScope AcquireAPCUse_() noexcept;

        void ReleseFabricBindingOnly_() noexcept;

        bool BindExternalRawFabricBacking_(
            uint64_t* raw_cells_ptr,
            VagueTemoraryPremativeFabric* fabric_owner,
            uint64_t fabric_slot_idx,
            uint64_t* generation_cell,
            uint32_t expected_generation
        ) noexcept;

        bool InitiateAPCMetaHeader() noexcept;

        bool ReadAPCMetaUnit(
            APCDataStructure::HeaderIdentifierOfAPC meta_idx,
            uint64_t& return_value
        ) noexcept;

    };
        
    
}