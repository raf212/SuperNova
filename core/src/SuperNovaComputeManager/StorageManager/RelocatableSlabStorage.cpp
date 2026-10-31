#include <SuperNova>

namespace BidirectionalInMemGraph
{
    CoreOfFabricCoordinator::DetachFabric RelocatableSlabStorage::DetachFabric() noexcept
    {
        if (ExecutionGate().IsPoisoned()) return {};

        auto phase = ExecutionGate().TryMaintenance();
        if (!phase) return {};

        CFC::DetachFabric detached{};
        if (
            !IsFabricActive() ||
            !FabCache_ ||
            !QuiesceFabric_()
        )
        {
            return detached;
        }

        detached.Slab_ = SlabBasePtr_;
        detached.CellCount_ = FabCache_->SlabCellCount_;
        detached.Ownership_ = BackingOwnership_;

        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        FabricInitialized_.store(false, std::memory_order_release);
        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        return detached;
    }

    bool RelocatableSlabStorage::SaveFabric(std::span<uint64_t> destination) noexcept
    {
        if (ExecutionGate().IsPoisoned()) return false;

        auto phase = ExecutionGate().TryMaintenance();
        if (!phase) return false;

        if (
            !IsFabricActive() ||
            !FabCache_ ||
            destination.data() == nullptr ||
            destination.size() < FabCache_->SlabCellCount_ ||
            IsInternalBuffer(destination.data(), destination.size())
        )
        {
            return false;
        }
        
        if (!QuiesceFabric_())
        {
            return false;
        }

        std::memcpy(
            destination.data(),
            SlabBasePtr_,
            FabCache_->SlabCellCount_ * sizeof(uint64_t)
        );

        bool reopened = ReopenLiveAPCGenerations_();

        FabricInitialized_.store(reopened, std::memory_order_release);

        return reopened;
    }

    bool RelocatableSlabStorage::FlushNumericalBacking() noexcept
    {
        ExclusiveLease phase = ExecutionGate().TryMaintenance();
        return phase &&
            !ExecutionGate_.IsPoisoned() &&
            NumericalBackingPtr_->Flush();
    }

    bool RelocatableSlabStorage::PrefetchNumerical(
        uint32_t slot,
        uint32_t cell_offset,
        uint32_t cells
    ) noexcept
    {
        ReadLease read_phase = ExecutionGate().TryRead();
        if (
            !read_phase ||
            !IsFabricActive() ||
            slot >= FabCache_->CountOfAPC_ ||
            cell_offset > FabCache_->PerAPCRuntimeCellCount_ ||
            cells > (FabCache_->PerAPCRuntimeCellCount_  - cell_offset)
        ) 
        {
            return false;
        }

        return !IsDiskPaged() ||
            NumericalBackingPtr_->Prefetch(
                static_cast<uint64_t>(slot) * FabCache_->PerAPCRuntimeCellCount_ + cell_offset, 
                cells
            );
    }



    bool RelocatableSlabStorage::AttachFabric(
        uint64_t* raw_cells,
        uint64_t cell_count,
        CFC::FabricBackigOwnership ownership,
        const CFC::FabricInitConf& storage
    ) noexcept
    {
        auto phase = ExecutionGate().TryMaintenance();
        if (!phase) return false;

        if (
            !raw_cells || cell_count < CFC::FABRIC_UNIT_COUNT || 
            ownership == CFC::FabricBackigOwnership::NONE ||
            (reinterpret_cast<uintptr_t>(raw_cells) % alignof(FabricCache) != UNSIGNED_ZERO)  ||
            raw_cells == SlabBasePtr_
        )
        {
            return false;
        }

        FabricCache cache{};
        std::memcpy(
            &cache,
            raw_cells,
            sizeof(cache)
        );

        if (!StorageAllocationDef::ValidateFabricCache(cache, cell_count))
        {
            return false;
        }

        ShutDownFabric();

        SlabBasePtr_ = raw_cells;
        FabCache_ = reinterpret_cast<FabricCache*>(std::memcpy(
            raw_cells,
            &cache,
            sizeof(cache)
        ));

        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        DefaultRegionTable_ = SD::RegionSchemaTable{};

        if (!ValidateAttachedFabricLayout_())
        {
            ResetScalarsofTheFabric_();
            return false;
        }
        
        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            uint64_t* cell = GetAPCGenerationPtr_(i);
            if (!cell)
            {
                ResetScalarsofTheFabric_();
                return false;
            }

            const HandleOfAPCStatic::ControlValues control = HandleOfAPCStatic::ReadControlCell(std::atomic_ref<const uint64_t>(*cell).load(std::memory_order_acquire));

            if (!control.Closed || control.ActiveAccess != UNSIGNED_ZERO || !HandleOfAPCStatic::IsGenerationValid(control.Generation))
            {
                ResetScalarsofTheFabric_();
                return false;
            }
        }
        if (!ReopenLiveAPCGenerations_())
        {
            QuiesceFabric_();
            ResetScalarsofTheFabric_();
            return false;
        }
        
        BackingOwnership_ = ownership;
        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        FabricInitialized_.store(true, std::memory_order_release);
        (void)ExecutionGate().ConfirmedRestored();

        return true;
        
    }

}