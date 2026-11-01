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
            IsDiskPaged() ||
            !QuiesceFabric_()
        )
        {
            return detached;
        }
        
        FabricInitialized_.store(false, std::memory_order_release);
        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);

        detached.Slab_ = SlabBasePtr_;
        detached.CellCount_ = FabCache_->SlabCellCount_;
        detached.Ownership_ = BackingOwnership_;

        __NumericalBacking__.Close();
        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        HotCellCount_ = UNSIGNED_ZERO;
        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        return detached;
    }

    bool RelocatableSlabStorage::SaveHotSubFabric(std::span<uint64_t> destination) noexcept
    {
        if (ExecutionGate().IsPoisoned()) return false;

        ExclusiveLease ex_phase = ExecutionGate().TryMaintenance();
        if (
            !ex_phase ||
            !IsFabricActive() ||
            !destination.data() ||
            destination.size() < FabCache_->SegmentPoolBegin_ ||
            IsInternalBuffer(destination.data(), destination.size())
        ) 
        {
            return false;
        }

        if (!QuiesceFabric_()) return false;

        std::memcpy(
            destination.data(), 
            SlabBasePtr_, 
            FabCache_->SegmentPoolBegin_ * sizeof(uint64_t)
        );

        const bool reopened = ReopenLiveAPCGenerations_();
        FabricInitialized_.store(reopened, std::memory_order_release);
        return reopened;
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
        std::memcpy(
            destination.data() + FabCache_->SegmentPoolBegin_,
            NumericalBase_(),
            (FabCache_->SlabCellCount_ - FabCache_->SegmentPoolBegin_) * sizeof(uint64_t)
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
            __NumericalBacking__.Flush();
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
            __NumericalBacking__.Prefetch(
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

        if (
            storage.HostAlloc &&
            !storage.HostAlloc->IsValid()
        )
        {
            return false;
        }

        FabricCache cache{};
        std::memcpy(&cache, raw_cells, sizeof(cache));

        const bool hot_only = !storage.SegmentPoolPath.empty() && cell_count == cache.SegmentPoolBegin_;
        if (
            (hot_only && cell_count != cache.SlabCellCount_) ||
            !StorageAllocationDef::ValidateFabricCache(cache)
        ) 
        {
            return false;
        }

        Numerics::Replacement replacement{*this};
        AllocatorOfFabric_ = storage.HostAlloc ? *storage.HostAlloc : RawPackedCellAllocator{};
        if (
            !storage.SegmentPoolPath.empty() &&
            !__NumericalBacking__.Create(storage.SegmentPoolPath, (cache.SlabCellCount_ - cache.SegmentPoolBegin_))
        ) 
        {
            return false;
        }

        SlabBasePtr_ = raw_cells;
        HotCellCount_ = cell_count;
        FabCache_ = cache.Construct(raw_cells);

        RefreshNumericalBase_();
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
        
        if (IsDiskPaged() && !hot_only) 
        {
            std::copy_n(
                raw_cells + cache.SegmentPoolBegin_,
                cache.SlabCellCount_ - cache.SegmentPoolBegin_,
                NumericalBase_()
            );
        }

        if (IsDiskPaged() && storage.Cache == CFC::NumericalCachePolicy::PRE_TOUCHED) 
        {
            __NumericalBacking__.WarmAll();
        }
        __NumericalBacking__.CommitCreation();
        BackingOwnership_ = ownership;
        SealedDAGRevision_.fetch_add(replacement.Revision() + 1, std::memory_order_release);
        FabricInitialized_.store(true, std::memory_order_release);
        (void)ExecutionGate().ConfirmedRestored();

        replacement.Commit();
        return true;
        
    }

}

namespace BidirectionalInMemGraph::Numerics 
{
    Replacement::Replacement(SlabToFabricConverterAndCordinator& self) noexcept :
        Self_(self),
        Previous_(
            self.SlabBasePtr_, self.HotCellCount_, self.AllocatorOfFabric_,
            self.BackingOwnership_ == CFC::FabricBackigOwnership::OWNED
        ),
        Numbers_(std::move(self.__NumericalBacking__)),
        Cache_(self.FabCache_),
        Ownership_(self.BackingOwnership_),
        Allocator_(self.AllocatorOfFabric_),
        Schemas_(self.DefaultRegionTable_),
        Active_(self.FabricInitialized_.load(std::memory_order_acquire)),
        Track_(self.TrackDAGRevision_.load(std::memory_order_acquire)),
        Revision_(self.SealedDAGRevision_.load(std::memory_order_acquire))
    {
        self.SlabBasePtr_ = nullptr;
        self.StorageBasePtr_ = nullptr;
        self.HotCellCount_ = UNSIGNED_ZERO;
        self.FabCache_ = nullptr;
        self.BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        self.FabricInitialized_.store(false, std::memory_order_release);
        self.TrackDAGRevision_.store(false, std::memory_order_release);
    }

    Replacement::~Replacement()
    {
        if (Complete_) return;
        Self_.ShutDownFabric();
        Self_.SlabBasePtr_ = Previous_.Relese();
        Self_.HotCellCount_ = Previous_.Count();
        Self_.FabCache_ = Cache_;
        Self_.__NumericalBacking__ = std::move(Numbers_);
        Self_.RefreshNumericalBase_();
        Self_.BackingOwnership_ = Ownership_;
        Self_.AllocatorOfFabric_ = Allocator_;
        Self_.DefaultRegionTable_ = Schemas_;
        Self_.TrackDAGRevision_.store(Track_, std::memory_order_release);
        Self_.SealedDAGRevision_.store(Revision_, std::memory_order_release);
        Self_.FabricInitialized_.store(Active_, std::memory_order_release);
    }
}