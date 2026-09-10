#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph
{
    bool FabricToAPCLinker::BindExternalRawFabricBacking_(
        uint64_t* raw_cells_ptr,
        VagueTemoraryPremativeFabric* fabric_owner,
        uint64_t fabric_slot_idx,
        uint64_t* generation_cell,
        uint32_t expected_generation
    ) noexcept
    {
        if (
            !raw_cells_ptr ||
            !fabric_owner ||
            !APCDataStructure::IsCapacityOfAPCValid(fabric_owner->PerAPCRuntimeCellCount_) ||
            !APCDataStructure::IsValid32BitAPCUnit(fabric_slot_idx) ||
            IsFabricBound_()
        )
        {
            return false;
        }
        const APCDataStructure::RangeOfAPC range_of_this_apc = fabric_owner->GetSegmentPoolRange(fabric_slot_idx);
        if (
            !range_of_this_apc.IsValid ||
            range_of_this_apc.EndIndex - range_of_this_apc.BeginIndex != fabric_owner->PerAPCRuntimeCellCount_
        )
        {
            return false;
        }

        std::span<SchemaDefinition::RegionSchemaRecord> region_row = fabric_owner->MetrixViewRow_(
            static_cast<uint32_t>(fabric_slot_idx)
        );

        if (region_row.size() != fabric_owner->ActiveRegionCount_)
        {
            return false;
        }

        APCSlotIdx_ = static_cast<uint32_t>(fabric_slot_idx);
        RawAPCBasePtr_ = reinterpret_cast<std::byte*>(raw_cells_ptr);
        FabricOwnerPtr_ = fabric_owner;
        APCGenerationCellPtr_ = generation_cell;
        ExpectedGeneration_ = expected_generation;
        return true;
    }


    void FabricToAPCLinker::ReleseFabricBindingOnly_() noexcept
    {
        FabricOwnerPtr_ = nullptr;
        RawAPCBasePtr_ = nullptr;
        APCSlotIdx_ = APCDataStructure::APC_INDEX_BOUND_SENTINAL;
        APCGenerationCellPtr_ = nullptr;
        ExpectedGeneration_ = UNSIGNED_ZERO;
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
            FabricOwnerPtr_->ReadAPCStateAtomically_(APCSlotIdx_);

        if (
            !current_state.IsValid ||
            current_state.StateOfTheAPC != StateOfAPC::RESERVED ||
            !HeaderOrchestrator::InitializeDefaultHeaderBuffer(
                header_meta_buffer,
                APCSlotIdx_,
                FabricOwnerPtr_->PerAPCRuntimeCellCount_
            )
        )
        {
            return false;
        }

        const uint64_t raw_new_state_seq = DSA::ComposeSeqLockAndState(current_state);

        header_meta_buffer[static_cast<uint8_t>(APCDataStructure::HeaderIdentifierOfAPC::APC_LIFE_CYCLE)] = raw_new_state_seq;

        const APCDataStructure::RangeOfAPC range_of_this_apc = FabricOwnerPtr_->GetSegmentPoolRange(APCSlotIdx_);


        return
            range_of_this_apc.IsValid &&
            APCDataStructure::IsValidFabricUnit(raw_new_state_seq) &&
            FabricOwnerPtr_->ForceNxLenMemCopy(
                range_of_this_apc.BeginIndex,
                APCDataStructure::META_CELL_COUNT,
                header_meta_buffer.data()
            );
    }

    bool FabricToAPCLinker::ReadAPCMetaUnit(
        APCDataStructure::HeaderIdentifierOfAPC meta_idx,
        uint64_t& return_value
    ) noexcept
    {
        if (!IsActiveAPC())
        {
            return false;
        }
        const uint8_t idx_u = static_cast<uint8_t>(meta_idx);
        const APCDataStructure::RangeOfAPC range_of_this_apc = FabricOwnerPtr_->GetSegmentPoolRange(APCSlotIdx_);
        const size_t slab_idx = static_cast<uint64_t>(range_of_this_apc.BeginIndex + idx_u);
        return range_of_this_apc.IsValid && FabricOwnerPtr_->AtomicallyLoadReadAUnit(slab_idx, return_value);
    }

    bool FabricToAPCLinker::IsFabricBound_() const noexcept
    {
        return FabricOwnerPtr_ != nullptr &&
            RawAPCBasePtr_ != nullptr &&
            APCGenerationCellPtr_ != nullptr &&
            APCDataStructure::IsValid32BitAPCUnit(APCSlotIdx_) &&
            HandleOfAPCStatic::IsGenerationValid(ExpectedGeneration_);
    }


    APCUseScope FabricToAPCLinker::AcquireAPCUse_() noexcept
    {
        if (!IsFabricBound_())
        {
            return APCUseScope{};
        }

        std::atomic_ref<uint64_t> control(*APCGenerationCellPtr_);

        const uint64_t before = control.fetch_add(1u, std::memory_order_acquire);

        const HandleOfAPCStatic::ControlValues values = HandleOfAPCStatic::ReadControlCell(before);

        if (
            values.Closed ||
            values.Generation != ExpectedGeneration_ ||
            values.ActiveAccess == APCDataStructure::APC_INDEX_BOUND_SENTINAL
        )
        {
            control.fetch_sub(1u, std::memory_order_release);
            return APCUseScope{};
        }
        
        return APCUseScope(APCGenerationCellPtr_);
    }

    bool FabricToAPCLinker::IsActiveAPC() noexcept
    {
        if (!IsFabricBound_())
        {
            return false;
        }

        const uint64_t raw = std::atomic_ref<const uint64_t>(*APCGenerationCellPtr_).load(std::memory_order_acquire);

        return HandleOfAPCStatic::IsOpenGeneration(raw, ExpectedGeneration_);
    }


}