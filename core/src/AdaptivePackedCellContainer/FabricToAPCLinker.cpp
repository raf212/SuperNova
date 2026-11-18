#include <SuperNova>

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

        SchemaOrchestrator::ConstRSSpan region_row = fabric_owner->MatrixViewRow_(
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

    bool FabricToAPCLinker::StampModelAPCGeneration_() noexcept
    {
        using SD = SchemaDefinition;
        if (!IsFabricBound_()) return false;
        HAS::LifeCycleControl current_state{};
        if (
            APCCache_.FabricOwnerPtr_->ReadAPCStateAtomically_(
                APCCache_.APCSlotIdx_, current_state
            ) != SeqLockedOperation::FOUND ||
            current_state.State != StateOfAPC::RESERVED
        ) 
        {
            return false;
        }

        if (GMI::ModelAPC* assignmen = APCCache_.FabricOwnerPtr_->ModelAPC_(APCCache_.APCSlotIdx_)) 
        {
            APCFinilizer* fabric = APCCache_.FabricOwnerPtr_;
            const GMI::ModelIR ir = GMI::GetModelIR(
                *fabric->FabCache_, 
                fabric->DisposableFCache___.SlabBasePtr_
            );
            const GMI::IndividualModelDescription* owner = ir.ModelForSLot(APCCache_.APCSlotIdx_);
            if (!owner) return false;
            for (const SD::RegionSchemaRecord& schema : fabric->MatrixViewRow_(APCCache_.APCSlotIdx_)) 
            {
                if (
                    !SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::REGION_DISABLED) &&
                    schema.ModelInstance != owner->Instance
                ) 
                {
                    return false;
                }
            }
            // Preserve roles supplied by imported Model IR on first creation.
            // Only a reused slot with a different generation loses its stale role.
            std::atomic_ref<uint32_t> generation(assignmen->Generation);
            const uint32_t previous = generation.load(std::memory_order_acquire);
            if (
                previous != UNSIGNED_ZERO &&
                previous != APCCache_.CurrentGeneration_
            ) 
            {
                std::atomic_ref<uint32_t>(assignmen->Role).store(
                    UNSIGNED_ZERO, std::memory_order_release
                );
            }
            generation.store(
                APCCache_.CurrentGeneration_,
                std::memory_order_release
            );
        }
        else 
        {
            for (const SD::RegionSchemaRecord& schema : APCCache_.FabricOwnerPtr_->MatrixViewRow_(
                APCCache_.APCSlotIdx_
            )) 
            {
                if (
                    !SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::REGION_DISABLED) &&
                    schema.ModelInstance != UNSIGNED_ZERO
                ) 
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool FabricToAPCLinker::IsFabricBound_() const noexcept
    {
        return APCCache_.FabricOwnerPtr_ != nullptr &&
            APCCache_.RawAPCBasePtr_ != nullptr && 
            APCCache_.GenerationCellPtr_ != nullptr &&
            ADS::ValidateSlotIdx(APCCache_.APCSlotIdx_) &&
            HandleOfAPCStatic::IsGenerationValid(APCCache_.CurrentGeneration_);
    }



}