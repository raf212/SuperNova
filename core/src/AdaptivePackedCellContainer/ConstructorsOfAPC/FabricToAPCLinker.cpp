#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph
{
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
            !ADS::IsCapacityOfAPCValid(fabric_owner->FabCache_.PerAPCRuntimeCellCount_) ||
            !ADS::IsValid32BitAPCUnit(fabric_slot_idx) ||
            IsFabricBound_()
        )
        {
            return false;
        }
        const ADS::RangeOfAPC range_of_this_apc = fabric_owner->GetSegmentPoolRange(fabric_slot_idx);
        if (
            !range_of_this_apc.IsValid ||
            range_of_this_apc.EndIndex - range_of_this_apc.BeginIndex != fabric_owner->FabCache_.PerAPCRuntimeCellCount_
        )
        {
            return false;
        }

        std::span<SchemaDefinition::RegionSchemaRecord> region_row = fabric_owner->MetrixViewRow_(
            static_cast<uint32_t>(fabric_slot_idx)
        );

        if (region_row.size() != fabric_owner->FabCache_.ActiveRegionCount_)
        {
            return false;
        }

        APCCache_.APCSlotIdx_ = static_cast<uint32_t>(fabric_slot_idx);
        APCCache_.RawAPCBasePtr_ = reinterpret_cast<std::byte*>(raw_cells_ptr);
        APCCache_.FabricOwnerPtr_ = fabric_owner;
        APCCache_.APCGenerationCellPtr_ = generation_cell;
        APCCache_.ExpectedGeneration_ = expected_generation;
        return true;
    }


    void FabricToAPCLinker::ReleseFabricBindingOnly_() noexcept
    {
        APCCache_ = ADS::CacheOfAPC{};
    }

    bool FabricToAPCLinker::InitiateAPCMetaHeader() noexcept
    {
        using DSA = DescriptionOfAPC;

        HeaderOrchestrator::APCMetaBuffer header_meta_buffer{};

        if (!IsFabricBound_())
        {
            return false;
        }

        DSA::SeqLockAndStateStruct current_state =
            APCCache_.FabricOwnerPtr_->ReadAPCStateAtomically_(APCCache_.APCSlotIdx_);

        if (
            !current_state.IsValid ||
            current_state.StateOfTheAPC != StateOfAPC::RESERVED ||
            !HeaderOrchestrator::InitializeDefaultHeaderBuffer(
                header_meta_buffer,
                APCCache_.APCSlotIdx_,
                APCCache_.FabricOwnerPtr_->FabCache_.PerAPCRuntimeCellCount_
            )
        )
        {
            return false;
        }

        const uint64_t raw_new_state_seq = DSA::ComposeSeqLockAndState(current_state);

        header_meta_buffer[static_cast<uint8_t>(ADS::HeaderIdentifierOfAPC::APC_LIFE_CYCLE)] = raw_new_state_seq;

        const ADS::RangeOfAPC range_of_this_apc = APCCache_.FabricOwnerPtr_->GetSegmentPoolRange(APCCache_.APCSlotIdx_);


        return
            range_of_this_apc.IsValid &&
            ADS::IsValidFabricUnit(raw_new_state_seq) &&
            APCCache_.FabricOwnerPtr_->ForceNxLenMemCopy(
                range_of_this_apc.BeginIndex,
                ADS::META_CELL_COUNT,
                header_meta_buffer.data()
            );
    }

    bool FabricToAPCLinker::ReadAPCMetaUnit(
        ADS::HeaderIdentifierOfAPC meta_idx,
        uint64_t& return_value
    ) noexcept
    {
        if (!IsActiveAPC())
        {
            return false;
        }
        const uint8_t idx_u = static_cast<uint8_t>(meta_idx);
        const ADS::RangeOfAPC range_of_this_apc = APCCache_.FabricOwnerPtr_->GetSegmentPoolRange(APCCache_.APCSlotIdx_);
        const size_t slab_idx = static_cast<uint64_t>(range_of_this_apc.BeginIndex + idx_u);
        return range_of_this_apc.IsValid && APCCache_.FabricOwnerPtr_->AtomicallyLoadReadAUnit(slab_idx, return_value);
    }

    bool FabricToAPCLinker::IsFabricBound_() const noexcept
    {
        return APCCache_.FabricOwnerPtr_ != nullptr &&
            APCCache_.RawAPCBasePtr_ != nullptr &&
            APCCache_.APCGenerationCellPtr_ != nullptr &&
            ADS::IsValid32BitAPCUnit(APCCache_.APCSlotIdx_) &&
            HandleOfAPCStatic::IsGenerationValid(APCCache_.ExpectedGeneration_);
    }


    APCUseScope FabricToAPCLinker::AcquireAPCUse_() noexcept
    {
        if (!IsFabricBound_())
        {
            return APCUseScope{};
        }

        std::atomic_ref<uint64_t> control(*APCCache_.APCGenerationCellPtr_);
        uint64_t observed = control.load(std::memory_order_acquire);

        for (;;)
        {
            const HandleOfAPCStatic::ControlValues current =
                HandleOfAPCStatic::ReadControlCell(observed);

            if (
                current.Closed ||
                current.Generation != APCCache_.ExpectedGeneration_ ||
                !HandleOfAPCStatic::IsGenerationValid(current.Generation) ||
                current.ActiveAccess == UINT32_MAX
            )
            {
                return APCUseScope{};
            }

            HandleOfAPCStatic::ControlValues desired = current;
            ++desired.ActiveAccess;
            const uint64_t desired_raw = HandleOfAPCStatic::MakeControlCell(desired);

            if (control.compare_exchange_weak(
                observed,
                desired_raw,
                std::memory_order_acq_rel,
                std::memory_order_acquire
            ))
            {
                return APCUseScope(APCCache_.APCGenerationCellPtr_);
            }
        }
    }

    bool FabricToAPCLinker::IsActiveAPC() noexcept
    {
        if (!IsFabricBound_())
        {
            return false;
        }

        const uint64_t raw = std::atomic_ref<const uint64_t>(*APCCache_.APCGenerationCellPtr_).load(std::memory_order_acquire);

        return HandleOfAPCStatic::IsOpenGeneration(raw, APCCache_.ExpectedGeneration_);
    }


}