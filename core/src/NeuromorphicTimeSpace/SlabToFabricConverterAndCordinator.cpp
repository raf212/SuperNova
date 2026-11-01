#include <SuperNova>
namespace BidirectionalInMemGraph
{

    uint64_t* SlabToFabricConverterAndCordinator::AllocatePackedCellRaw_(size_t count_of_cells) noexcept
    {
        return AllocatorOfFabric_.Allocate(count_of_cells);
    }

    void SlabToFabricConverterAndCordinator::FreeRawPackedCells_(uint64_t* data, size_t cell_count) noexcept
    {
        AllocatorOfFabric_.Free(data, cell_count);
    }

    void SlabToFabricConverterAndCordinator::ResetScalarsofTheFabric_() noexcept
    {
        TrackDAGRevision_.store(false, std::memory_order_release);
        FabricInitialized_.store(false, std::memory_order_release);
        InitializationInProgress_.store(false, std::memory_order_release);

        __NumericalBacking__.Close();
        BackingOwnership_ = CoreOfFabricCoordinator::FabricBackigOwnership::NONE;
        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        StorageBasePtr_ = nullptr;
        HotCellCount_ = UNSIGNED_ZERO;
    }

    bool SlabToFabricConverterAndCordinator::QuiesceFabric_() noexcept
    {
        if (!SlabBasePtr_ || !FabCache_)
        {
            return false;
        }

        FabricInitialized_.store(false, std::memory_order_release);
        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            uint64_t* control_cell = GetAPCGenerationPtr_(i);
            if (!control_cell)
            {
                return false;
            }
            
            std::atomic_ref<uint64_t>(*control_cell).fetch_or(HandleOfAPCStatic::CLOSED_MASK, std::memory_order_acq_rel);
        }

        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            uint64_t* control_cell = GetAPCGenerationPtr_(i);
            if (!control_cell)
            {
                return false;
            }
            std::atomic_ref<uint64_t> control(*control_cell);
            for (;;)
            {
                HandleOfAPCStatic::ControlValues values = HandleOfAPCStatic::ReadControlCell(
                    control.load(std::memory_order_acquire)
                );
                if (values.ActiveAccess == UNSIGNED_ZERO)
                {
                    break;
                }
                std::this_thread::yield();
            }
        }
        
        return true;
    }


    bool SlabToFabricConverterAndCordinator::ReopenLiveAPCGenerations_() noexcept
    {
        if (!SlabBasePtr_ || !FabCache_)
        {
            return false;
        }

        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            uint64_t* control_cell = GetAPCGenerationPtr_(i);
            if (!control_cell)
            {
                return false;
            }

            const uint64_t raw = std::atomic_ref<uint64_t>(*control_cell).load(std::memory_order_acquire);
            const HandleOfAPCStatic::ControlValues values = HandleOfAPCStatic::ReadControlCell(raw);
            if (
                values.ActiveAccess != UNSIGNED_ZERO ||
                !values.Closed ||
                !HandleOfAPCStatic::IsGenerationValid(values.Generation)
            )
            {
                return false;
            }

            HAS::LifeCycleControl state{};

            if (ReadAPCStateAtomically_(i, state) != SeqLockedOperation::FOUND)
            {
                return false;
            }
            
            if (state.State == StateOfAPC::RESERVED)
            {
                return false;
            }
            
            if (state.State == StateOfAPC::LIVE)
            {
                if (!OpenAPCGeneration_(i, values.Generation))
                {
                    return false;
                }   
            }
        }
        return true;
    }

    struct SlabToFabricConverterAndCordinator::InitFlagReset
    {
        std::atomic<bool>& Flag;
        ~InitFlagReset()
        {
            Flag.store(false, std::memory_order_release);
        }
    };

    bool SlabToFabricConverterAndCordinator::__InitSlabAndCache__(const CFC::FabricInitConf& storage, CFC::FabricCache& cache) noexcept
    {
        const size_t  physical_cells = storage.SegmentPoolPath.empty() ?
            cache.SlabCellCount_ : cache.SegmentPoolBegin_;

        SlabBasePtr_ = AllocatePackedCellRaw_(physical_cells);
        if (!SlabBasePtr_) return false;
        
        std::fill_n(SlabBasePtr_, physical_cells, uint64_t{UNSIGNED_ZERO});

        FabCache_->Construct(SlabBasePtr_);
        BackingOwnership_ = CFC::FabricBackigOwnership::OWNED;
        if (
            !storage.SegmentPoolPath.empty() &&
            !__NumericalBacking__.Create(
                storage.SegmentPoolPath,
                (cache.SlabCellCount_ - cache.SegmentPoolBegin_)
            )
        ) 
        {
            return false;
        }
        RefreshNumericalBase_();
        FabCache_->ModelDiaphragm_.DynamicsRevision_ = storage.DynamicRevision;
        return true;
    }

    bool SlabToFabricConverterAndCordinator::InitializeFabric(
        uint32_t slot_count,
        uint32_t slot_cell_count,
        const SchemaDefinition::FabricRegionConfig& region_conf,
        uint32_t max_direct_parent_per_axis,
        const CFC::FabricInitConf& storage
    ) noexcept
    {
        ExclusiveLease exc_phase = ExecutionGate_.TryMaintenance();
        if (!exc_phase) return false;
        bool expected = false;
        if (
            !InitializationInProgress_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)
        ) 
        {
            return false;
        }

        InitFlagReset flag{InitializationInProgress_};
        const RawPackedCellAllocator allocator =  storage.HostAlloc ? *storage.HostAlloc : RawPackedCellAllocator{};
        const GMI::ModelCatalog catalog{
            storage.Models,
            storage.Dependencies,
            storage.Dynamics                                                                                                                          
        };

        if (
            !allocator.IsValid() ||
            !storage.DynamicRevision ||
            !catalog.ValidCatalog()
        ) 
        {
            return false;
        }

        //CODE:: INITCACHE_FIX_PRIO_01
        CFC::FabricCache cache{};
        {
            cache.FormateVersion_ = CFC::FORMAT_VERSION;
            cache.CountOfAPC_ = slot_count;
            cache.PerAPCRuntimeCellCount_ = slot_cell_count;
            cache.MaxDirectParentsPerAxis_ = max_direct_parent_per_axis;
            cache.ActiveRegionMask_ = region_conf.ActiveRegionMask;
            cache.MatrixBatchCapacity_ = region_conf.BatchCapacity;
            cache.HasDefaultRegionTable_ = region_conf.IsDefault;
            cache.ModelDiaphragm_.ModelCount_ = static_cast<uint16_t>(storage.Models.size());
            cache.ModelDiaphragm_.ModelDependencyCount_ = static_cast<uint32_t>(storage.Dependencies.size());
            cache.ModelDiaphragm_.ModelDynamicsByteCount_ = storage.Dynamics.size();
        }

        StorageAllocationDef layout;
        if (!layout.BuildStorage(cache)) return false;
        Numerics::Replacement replacement{*this};
        AllocatorOfFabric_ = allocator;

        if (!__InitSlabAndCache__(storage, cache)) return false;

    }

    void SlabToFabricConverterAndCordinator::ShutDownFabric() noexcept
    {
        // Preserve the original blocking/draining shutdown contract.
        // Disallow fresh Fabric access before waiting for existing read pins.
        if (ExecutionGate().EpochOwnedByCaller()) return;
        const bool was_initialized = FabricInitialized_.exchange(false, std::memory_order_acq_rel);
        auto phase = ExecutionGate().TryMaintenance();
        while (!phase)
        {
            std::this_thread::yield();
            phase = ExecutionGate().TryMaintenance();
        }

        uint64_t* old_ptr = SlabBasePtr_;
        const size_t old_count = FabCache_ ? FabCache_->SlabCellCount_ : UNSIGNED_ZERO;
        const CFC::FabricBackigOwnership old_ownership = BackingOwnership_;
        if (
            old_ptr &&
            FabCache_ &&
            was_initialized
        )
        {
            QuiesceFabric_();
        }
        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        FabricInitialized_.store(false, std::memory_order_release);

        if (old_ptr && old_ownership == CFC::FabricBackigOwnership::OWNED)
        {
            FreeRawPackedCells_(old_ptr, old_count);
        }

        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        ResetScalarsofTheFabric_();
    }

}
