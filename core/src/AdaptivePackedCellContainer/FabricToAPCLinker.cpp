#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph
{
    using HAS = HandleOfAPCStatic;

    bool FabricToAPCLinker::BindExternalRawFabricBacking_(
        uint64_t* raw_cells_ptr,
        APCFinilizer* fabric_owner,
        uint64_t fabric_slot_idx,
        uint64_t* generation_cell,
        uint32_t expected_generation
    ) noexcept
    {
        if (
            !raw_cells_ptr ||
            !fabric_owner ||
            !ADS::IsCapacityOfAPCValid(fabric_owner->FabCache_->PerAPCRuntimeCellCount_) ||
            !ADS::ValidateSlotIdx(fabric_slot_idx) ||
            IsFabricBound_()
        )
        {
            return false;
        }
        const ADS::RangeOfAPC range_of_this_apc = fabric_owner->GetSegmentPoolRange(fabric_slot_idx);
        if (
            !range_of_this_apc.IsValid ||
            range_of_this_apc.EndIndex - range_of_this_apc.BeginIndex != fabric_owner->FabCache_->PerAPCRuntimeCellCount_
        )
        {
            return false;
        }

        SchemaOrchestrator::ConstRSSpan region_row = fabric_owner->MetrixViewRow_(
            static_cast<uint32_t>(fabric_slot_idx)
        );

        if (region_row.size() != fabric_owner->FabCache_->ActiveRegionCount_)
        {
            return false;
        }

        APCCache_.APCSlotIdx_ = static_cast<uint32_t>(fabric_slot_idx);
        APCCache_.RawAPCBasePtr_ = reinterpret_cast<std::byte*>(raw_cells_ptr);
        APCCache_.FabricOwnerPtr_ = fabric_owner;
        APCCache_.GenerationCellPtr_ = generation_cell;
        APCCache_.CurrentGeneration_ = expected_generation;
        return true;
    }


    void FabricToAPCLinker::ReleseFabricBindingOnly_() noexcept
    {
        APCCache_ = ADS::CacheOfAPC{};
    }

    APCUseScope FabricToAPCLinker::AcquireAPCUse_() noexcept
    {
        if (!IsFabricBound_())
        {
            return APCUseScope{};
        }

        auto phase = APCCache_.FabricOwnerPtr_->ExecutionGate().TryRead(true);
        if (!phase) return {};

        std::atomic_ref<uint64_t> control(*APCCache_.GenerationCellPtr_);
        uint64_t observed = control.load(std::memory_order_acquire);

        for (;;)
        {
            const HandleOfAPCStatic::ControlValues current =
                HandleOfAPCStatic::ReadControlCell(observed);

            if (
                current.Closed ||
                current.Generation != APCCache_.CurrentGeneration_ ||
                !HandleOfAPCStatic::IsGenerationValid(current.Generation) ||
                current.ActiveAccess == UINT32_MAX
            )
            {
                return APCUseScope{};
            }

            HandleOfAPCStatic::ControlValues desired = current;
            ++desired.ActiveAccess;

            if (control.compare_exchange_weak(
                observed,
                HandleOfAPCStatic::MakeControlCell(desired),
                std::memory_order_acquire,
                std::memory_order_relaxed
            ))
            {
                return APCUseScope(APCCache_.GenerationCellPtr_, std::move(phase));
            }
        }
    }

    bool FabricToAPCLinker::IsActiveAPC() noexcept
    {
        if (!IsFabricBound_())
        {
            return false;
        }

        auto phase = APCCache_.FabricOwnerPtr_->ExecutionGate().TryTransientRead(true);
        if (!phase) return false;

        const uint64_t raw = std::atomic_ref<const uint64_t>(*APCCache_.GenerationCellPtr_).load(std::memory_order_acquire);

        return HandleOfAPCStatic::IsOpenGeneration(raw, APCCache_.CurrentGeneration_);
    }


}