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

        FabCache_ = cache.Construct(SlabBasePtr_);
        HotCellCount_ = physical_cells;
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

    void SlabToFabricConverterAndCordinator::__InitModelStorage__(
        const GMI::ModelCatalog& catalog, 
        std::byte* directory,
        uint64_t count_of_apc
    ) noexcept
    {
        if (!catalog.Models.empty()) 
        {
            std::memcpy(directory, catalog.Models.data(), catalog.Models.size_bytes());
        }
        if (!catalog.Dependencies.empty()) 
        {
            std::memcpy(
                directory + catalog.Models.size_bytes(), 
                catalog.Dependencies.data(), 
                catalog.Dependencies.size_bytes()
            );
        }
        if (!catalog.Dynamics.empty()) 
        {
            std::memcpy(
                directory + catalog.Models.size_bytes() + catalog.Dependencies.size_bytes(),
                catalog.Dynamics.data(),
                catalog.Dynamics.size()
            );
        }

        for (uint32_t s = 0; s < count_of_apc; s++) 
        {
            std::construct_at(NodeModel_(s));
        }
    }


    bool SlabToFabricConverterAndCordinator::__InitMatrixRegionTable__(const StorageAllocationDef& layout) noexcept
    {
        for (const CFC::FabricSegmentBounds& range : layout.DefaultSegments__) 
        {
            WriteARecordBookOfTSCEntry_(
                range.OwnerTableOfTheBounds, 
                range.BeginIndex, 
                range.EndIndex
            );
        }

        const CFC::FabricSegmentBounds& matrix_table_bounds = layout.DefaultSegments__[
            static_cast<uint8_t>(FabricSegments::MATRIX_VIEW_TABLE)
        ];

        if (!ConstructMatrixViewRecords_(matrix_table_bounds.BeginIndex, matrix_table_bounds.EndIndex)) 
        {
            return false;
        }

        if (FabCache_->HasDefaultRegionTable_) 
        {
            for (uint32_t s = 0; s < FabCache_->CountOfAPC_; s++) 
            {
                if (
                    !PrepareMatrixViewRow_(s, DefaultRegionTable_) ||
                    (
                        !IsDiskPaged() &&
                        !InitializeRegionProtocolStorage_(s)
                    )
                ) 
                {
                    return false;
                }
            }
        }
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
        using SAD = StorageAllocationDef;

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

        SAD layout;
        if (!layout.BuildStorage(cache)) return false;
        Numerics::Replacement replacement{*this};
        AllocatorOfFabric_ = allocator;

        if (!__InitSlabAndCache__(storage, cache)) return false;

        __InitModelStorage__(
            catalog,
            reinterpret_cast<std::byte*>(SlabBasePtr_ + cache.ModelDiaphragm_.LocalDirectoryBeginIdx_),
            cache.CountOfAPC_
        );

        if (!__InitMatrixRegionTable__(layout)) return false;
        
        //IDELS
        IdleAFabricTableClassRangesMemory_(FabricSegments::WORK_QUEUE);

        if (
            !InitializeStructHotRowApcHandleTable_() ||
            !InitializeEdgeTable_(FabricSegments::VALUE_PARENT_EDGE_TABLE_H) ||
            !InitializeEdgeTable_(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V)
        ) 
        {
            return false;
        }

        if (
            IsDiskPaged() &&
            storage.Cache == CFC::NumericalCachePolicy::PRE_TOUCHED
        ) 
        {
            __NumericalBacking__.WarmAll();
        }

        __NumericalBacking__.CommitCreation();
        SealedDAGRevision_.store(replacement.Revision() + 1, std::memory_order_release);
        FabricInitialized_.store(true, std::memory_order_release);
        (void) ExecutionGate().ConfirmedRestored();
        replacement.Commit();
        return true;
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
        const size_t old_count = HotCellCount_;
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
        HotCellCount_ = UNSIGNED_ZERO;
        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        FabricInitialized_.store(false, std::memory_order_release);

        if (old_ptr && old_ownership == CFC::FabricBackigOwnership::OWNED)
        {
            FreeRawPackedCells_(old_ptr, old_count);
        }

        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        ResetScalarsofTheFabric_();
    }

    bool SlabToFabricConverterAndCordinator::ValidateAttachedFabricLayout_() noexcept
    {
        if (
            !FabCache_ || !SlabBasePtr_ ||
            !StorageAllocationDef::ValidateFabricCache(*FabCache_)
        ) 
        {
            return false;
        }

        const GMI::ModelCatalog catalog = GMI::GetModelCatalog(*FabCache_, SlabBasePtr_);
        if (!catalog.ValidCatalog()) return false;

        StorageAllocationDef layout{};
        CFC::FabricCache expected = *FabCache_;
        if (!layout.BuildStorage(expected)) return false;
        for (const CFC::FabricSegmentBounds& range : layout.DefaultSegments__) 
        {
            if (
                range.BeginIndex != range.EndIndex &&
                !CheckRecordBookRange_(range.OwnerTableOfTheBounds, range.BeginIndex, range.EndIndex)
            ) 
            {
                return false;
            }
        }

        for (uint32_t s = 0; s < FabCache_->CountOfAPC_; s++) 
        {
            const SD::ConstRSSpan row = MetrixViewRow_(s);
            if (row.size() != FabCache_->ActiveRegionCount_) 
            {
                return false;
            }
            uint16_t mask = UNSIGNED_ZERO;
            uint64_t previous_end = UNSIGNED_ZERO;
            for (const SD::RegionSchemaRecord& schema : row) 
            {
                if (SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::REGION_DISABLED)) 
                {
                    continue;
                }

                if (
                    static_cast<uint8_t>(schema.Region) >= ADS::CountOfMacroColumn() ||
                    (mask & ADS::RegionBit(schema.Region)) ||
                    schema.CellOffset < previous_end ||
                    !SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::SEALED) ||
                    !SD::ValidateStortedRegionSchema(
                        schema, 
                        FabCache_->PerAPCRuntimeCellCount_, 
                        FabCache_->MatrixBatchCapacity_
                    ) ||
                    !catalog.Accept(schema)
                ) 
                {
                    return false;
                }

                mask = static_cast<uint16_t>(mask | ADS::RegionBit(schema.Region));
                previous_end = uint64_t{schema.CellOffset} + schema.CellCount;
            }

            if ((mask != UNSIGNED_ZERO && mask != FabCache_->ActiveRegionMask_))
            {
                return false;
            }

            const GMI::NodeModelDescription* description = NodeModel_(s);
            if (
                !description ||
                (description->Instance) >  catalog.Models.size()
            ) 
            {
                return false;
            }
        }
        return true;
    }

}
