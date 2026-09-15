#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    uint64_t* SlabToFabricConverterAndCordinator::AllocatePackedCellRaw_(size_t count_of_cells) noexcept
    {
        auto allocation_function = AllocatorOfFabric_.AllocatePackedCellStorage ? 
            AllocatorOfFabric_.AllocatePackedCellStorage : &RawPackedCellAllocator::DefaultAllocateAtomicCells;
        
        size_t alignment = AllocatorOfFabric_.Alignment ? AllocatorOfFabric_.Alignment : BIT_COUNT_OF_UINT64_T;
        alignment = std::max<size_t>(alignment, alignof(uint64_t));
        alignment = std::max<size_t>(alignment, BIT_COUNT_OF_UINT64_T);

        return allocation_function(count_of_cells, alignment, AllocatorOfFabric_.User);

    }

    void SlabToFabricConverterAndCordinator::FreeRawPackedCells_(uint64_t* packed_cell_memory_ptr, size_t packed_cell_count) noexcept
    {
        RawPackedCellAllocator::FreeFunction free_function = AllocatorOfFabric_.FreePackedCellStorage ?
                            AllocatorOfFabric_.FreePackedCellStorage : &RawPackedCellAllocator::DefaultFreeAtomicCells;
        size_t alignment = AllocatorOfFabric_.Alignment ? AllocatorOfFabric_.Alignment : BIT_COUNT_OF_UINT64_T;
        alignment = std::max<size_t>(alignment, alignof(uint64_t));
        alignment = std::max<size_t>(alignment, BIT_COUNT_OF_UINT64_T);

        free_function(packed_cell_memory_ptr, packed_cell_count, alignment, AllocatorOfFabric_.User);
    }

    void SlabToFabricConverterAndCordinator::ResetScalarsofTheFabric_() noexcept
    {
        BackingOwnership_ = CoreOfFabricCoordinator::FabricBackigOwnership::NONE;
        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        FabricInitialized_.store(false, std::memory_order_release);
        InitializationInProgress_.store(false, std::memory_order_release);
    }

    bool SlabToFabricConverterAndCordinator::ValidateAttachedFabricLayout_() noexcept
    {
        if (!FabCache_ || !SlabBasePtr_)
        {
            return false;
        }

        const uint64_t count = FabCache_->CountOfAPC_;
        const uint64_t matrix_end = FabCache_->MatrixViewTableBeginIndex_ + count * FabCache_->MatrixViewRowCellCount_;
        const uint64_t value_edge_end = FabCache_->HorizontalEdgeBeginIdx_ + count * FabCache_->EdgeTableRecordWidth_;
        const uint64_t volatile_edge_end = FabCache_->VerticalEdgeBeginIdx_ + count * FabCache_->EdgeTableRecordWidth_;
        const uint64_t handle_end = FabCache_->HandleTableBeginIndex_ + count * HandleOfAPCStatic::HANDLE_TABLE_WIDTH;
        const uint64_t dag_end = FabCache_->CompiledDAGTableBeginIdx_ + count * CoreOfFabricCoordinator::COMPILED_DAG_LEN;
        const uint64_t segment_end = FabCache_->SegmentPoolBegin_ + count * FabCache_->PerAPCRuntimeCellCount_;

        if (
            value_edge_end > FabCache_->SlabCellCount_ ||
            matrix_end > FabCache_->SlabCellCount_ ||
            volatile_edge_end > FabCache_->SlabCellCount_ ||
            handle_end > FabCache_->SlabCellCount_ ||
            dag_end > FabCache_->SlabCellCount_ ||
            segment_end > FabCache_->SlabCellCount_
        )
        {
            return false;
        }
        
        return
            CheckRecordBookRange_(FabricSegments::SLAB_RECORD_MAP, FabCache_->RecordBookBeginIndex_, FabCache_->RecordBookEndIndex_) &&
            CheckRecordBookRange_(FabricSegments::MATRIX_VIEW_TABLE, FabCache_->MatrixViewTableBeginIndex_, matrix_end) &&
            CheckRecordBookRange_(FabricSegments::VALUE_PARENT_EDGE_TABLE_H, FabCache_->HorizontalEdgeBeginIdx_, value_edge_end) &&
            CheckRecordBookRange_(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V, FabCache_->VerticalEdgeBeginIdx_, volatile_edge_end) &&
            CheckRecordBookRange_(FabricSegments::COMPILED_DAG_TABLE, FabCache_->CompiledDAGTableBeginIdx_, dag_end) &&
            CheckRecordBookRange_(FabricSegments::APC_HANDLE_TABLE, FabCache_->HandleTableBeginIndex_, handle_end) &&
            CheckRecordBookRange_(FabricSegments::SEGMENT_POOL, FabCache_->SegmentPoolBegin_, segment_end);
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
                if (values.ActiveAccess != UNSIGNED_ZERO)
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

            const DSA::SeqLockAndStateStruct state = ReadAPCStateAtomically_(i);
            if (!state.IsValid)
            {
                return false;
            }
            
            if (state.StateOfTheAPC == StateOfAPC::RESERVED)
            {
                return false;
            }
            
            if (state.StateOfTheAPC == StateOfAPC::LIVE)
            {
                if (!!OpenAPCGeneration_(i, values.Generation))
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
        uint8_t max_direct_parent_per_axis
    ) noexcept
    {
        bool expected = false;
        if (!InitializationInProgress_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return false;
        }

        struct InitGuardSTFC
        {
            SlabToFabricConverterAndCordinator* SelfPtr{};
            bool SuccesInit{false};


            ~InitGuardSTFC()
            {
                if (!SuccesInit && SelfPtr)
                {
                    SelfPtr->FabricInitialized_.store(false, std::memory_order_release);
                }
                
                if (SelfPtr)
                {
                    SelfPtr->InitializationInProgress_.store(false, std::memory_order_release);
                }
            }
        } internal_init_guard{this, false};
        
        ShutDownFabric();

        InitializationInProgress_.store(true, std::memory_order_release);

        if (slot_count == UNSIGNED_ZERO || !ADS::IsValid32BitAPCUnit(slot_count))
        {
            return false;
        }
        
        if (!ADS::IsCapacityOfAPCValid(slot_cell_count))
        {
            return false;
        }

        const uint16_t active_mask = region_conf.ActiveRegionMask;
        const uint8_t active_count = static_cast<uint8_t>(std::popcount(active_mask));

        if (
            active_mask == UNSIGNED_ZERO ||
            (active_mask & static_cast<uint16_t>(~ADS::ValidRegionMask())) != UNSIGNED_ZERO ||
            active_count == UNSIGNED_ZERO ||
            region_conf.BatchCapacity == UNSIGNED_ZERO ||
            slot_cell_count % SD::REGION_ALIGNMENT_CELLS != UNSIGNED_ZERO
        )
        {
            return false;
        }

        if (
            !EdgeBuilder::IsValidConfigurableParentCapacity(max_direct_parent_per_axis) ||
            slot_count > (uint32_t{1u} << EdgeBuilder::RELATION_SLOT_BITS)
        )
        {
            return false;
        }

        FabricCache cache{};

        cache.FormateVersion_ = CoreOfFabricCoordinator::FORMAT_VERSION;
        cache.CountOfAPC_ = slot_count;
        cache.PerAPCRuntimeCellCount_ = slot_cell_count;
        cache.MaxDirectParentsPerAxis_ = max_direct_parent_per_axis;
        cache.EdgeTableRecordWidth_ = EdgeBuilder::EdgeTableRecordWidth(max_direct_parent_per_axis);
        cache.ActiveRegionMask_ = active_mask;
        cache.ActiveRegionMask_ = active_mask;
        cache.MatrixBatchCapacity_ = region_conf.BatchCapacity;
        cache.MatrixViewRowCellCount_ = static_cast<uint16_t>(
            static_cast<uint16_t>(FabCache_->ActiveRegionCount_) *
            SD::RegionSchemaCellCount()
        );
        cache.RegionAlignmentCellCount_ = SD::REGION_ALIGNMENT_CELLS;
        cache.FirstFreeIdx_ = UNSIGNED_ZERO;

        size_t cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(CoreOfFabricCoordinator::FABRIC_UNIT_COUNT);
        const size_t record_book_begin = cursor;
        const size_t record_book_end = record_book_begin + static_cast<size_t>(RecordBookConf::RECORD_BOOK_INTERNAL_SEGMENT_COUNT) * CoreOfFabricCoordinator::RECORD_BOOK_WIDTH;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(record_book_end);
        const size_t horizontal_edge_begin = cursor;
        const size_t horizontal_edge_end = horizontal_edge_begin + static_cast<size_t>(cache.CountOfAPC_) * cache.EdgeTableRecordWidth_;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(horizontal_edge_end);
        const size_t matrix_view_table_begin = cursor;
        const size_t matrix_view_table_end = matrix_view_table_begin + static_cast<size_t>(cache.CountOfAPC_ * cache.MatrixViewRowCellCount_);
        
        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(matrix_view_table_end);
        const size_t vertical_edge_begin = cursor;
        const size_t vertical_edge_end = vertical_edge_begin + 
                static_cast<size_t>(cache.CountOfAPC_) * cache.EdgeTableRecordWidth_;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(vertical_edge_end);
        const size_t apc_handle_table_begin = cursor;
        const size_t apc_handle_table_end = apc_handle_table_begin + static_cast<size_t>(cache.CountOfAPC_ * HandleOfAPCStatic::HANDLE_TABLE_WIDTH);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(apc_handle_table_end);
        const size_t compiled_dag_begin = cursor;
        const size_t compiled_dag_end = compiled_dag_begin + static_cast<size_t>(cache.CountOfAPC_ * CoreOfFabricCoordinator::COMPILED_DAG_LEN);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(compiled_dag_end);
        const size_t device_planner_begain = cursor;
        const size_t device_planner_end = device_planner_begain + static_cast<size_t>(cache.CountOfAPC_ * CoreOfFabricCoordinator::DEVICE_PLANNER_RECORD_LEN);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(device_planner_end);
        const size_t work_queue_begin = cursor;
        const size_t work_queue_end = work_queue_begin + static_cast<size_t>(cache.CountOfAPC_ * CoreOfFabricCoordinator::WORK_RECORD_WIDTH_OF_FABRIC);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(work_queue_end);
        cache.SegmentPoolBegin_ = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(std::max<size_t>(cursor, CoreOfFabricCoordinator::DEFAULT_FABRIC_CONTROLIO_LENGTH));
        cache.SlabCellCount_ = cache.SegmentPoolBegin_ + static_cast<size_t>(cache.CountOfAPC_ * cache.PerAPCRuntimeCellCount_);
        cache.HorizontalEdgeBeginIdx_ = horizontal_edge_begin;
        cache.VerticalEdgeBeginIdx_ = vertical_edge_begin;
        cache.HandleTableBeginIndex_ = apc_handle_table_begin;
        cache.MatrixViewTableBeginIndex_ = matrix_view_table_begin;
        cache.HasDefaultRegionTable_ = region_conf.IsDefault;
        if (cache.SlabCellCount_ == UNSIGNED_ZERO || cache.SlabCellCount_ >= FABRIC_CELL_SENTINAL)
        {
            return false;
        }
        SlabBasePtr_ = AllocatePackedCellRaw_(cache.SlabCellCount_);
        if (!SlabBasePtr_)
        {
            return false;
        }

        for (size_t idx = 0; idx < FabCache_->SlabCellCount_; idx++)
        {
            DirectlyStoreFabricUnit64(idx, UNSIGNED_ZERO);
        }

        FabCache_ = std::construct_at(reinterpret_cast<FabricCache*>(SlabBasePtr_), cache);

        //RECORD_BOOK_OF_TABLE_SEGMENT_CLASS - ENTRIES
        WriteARecordBookOfTSCEntry_(FabricSegments::SLAB_RECORD_MAP, record_book_begin, record_book_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::VALUE_PARENT_EDGE_TABLE_H, horizontal_edge_begin, horizontal_edge_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V, vertical_edge_begin, vertical_edge_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::APC_HANDLE_TABLE, apc_handle_table_begin, apc_handle_table_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::COMPILED_DAG_TABLE, compiled_dag_begin, compiled_dag_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::DEVICE_PLANNER_TABLE, device_planner_begain, device_planner_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::WORK_QUEUE, work_queue_begin, work_queue_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::MATRIX_VIEW_TABLE, matrix_view_table_begin, matrix_view_table_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::SEGMENT_POOL, FabCache_->SegmentPoolBegin_, FabCache_->SlabCellCount_);



        if (!ConstructMatrixViewRecords_(matrix_view_table_begin, matrix_view_table_end))
        {
            return false;
        }

        if (FabCache_->HasDefaultRegionTable_)
        {
            for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
            {
                if (
                    !PrepareMatrixViewRow_(i, DefaultRegionTable_) ||
                    !InitializeRegionProtocolStorage_(i)
                )
                {
                    return false;
                }
            }
        }
        
        

        if (!InitializeAPCGenerationTable_())
        {
            return false;
        }
        
        //IDLE UNUSED FabricSegments
        IdleAFabricTableClassRangesMemory_(FabricSegments::DEVICE_PLANNER_TABLE);
        IdleAFabricTableClassRangesMemory_(FabricSegments::WORK_QUEUE);
        //END:: IDELING

        //INIT: EDGE TABLES
        if (
            !InitializeEdgeTable_(FabricSegments::VALUE_PARENT_EDGE_TABLE_H) ||
            !InitializeEdgeTable_(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V) ||
            !InitializeCompiledDAGTAble_()
        )
        {
            return false;
        }
        //END::: 
        //INIT:Life Cycle
        InitAllAPCLifeCycleState();

        //CONFERMATION
        FabricInitialized_.store(true, std::memory_order_release);
        internal_init_guard.SuccesInit = true;
        return true;
        
    }
    
    void SlabToFabricConverterAndCordinator::ShutDownFabric() noexcept
    {
        const bool was_active = FabricInitialized_.exchange(false, std::memory_order_acq_rel);

        if (was_active && SlabBasePtr_)
        {
            for (uint32_t slot = 0u; slot < FabCache_->CountOfAPC_; ++slot)
            {
                std::atomic_ref<uint64_t>(*GetAPCGenerationPtr_(slot)).fetch_or(
                    HandleOfAPCStatic::CLOSED_MASK,
                    std::memory_order_acq_rel
                );
            }

            for (uint32_t slot = 0u; slot < FabCache_->CountOfAPC_; ++slot)
            {
                std::atomic_ref<uint64_t> control(*GetAPCGenerationPtr_(slot));
                while (
                    HandleOfAPCStatic::ReadControlCell(control.load(std::memory_order_acquire)).ActiveAccess != UNSIGNED_ZERO
                )
                {
                    std::this_thread::yield();
                }
            }
        }

        uint64_t* old_ptr = SlabBasePtr_;
        const size_t old_count = FabCache_->SlabCellCount_;
        SlabBasePtr_ = nullptr;
        FabCache_->SlabCellCount_ = UNSIGNED_ZERO;

        if (old_ptr)
        {
            FreeRawPackedCells_(old_ptr, old_count);
        }

        FabCache_->CompiledDAGTableBeginIdx_ = UNSIGNED_ZERO;
        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        ResetScalarsofTheFabric_();
    }


    bool SlabToFabricConverterAndCordinator::SaveFabric(std::span<uint64_t> destination) noexcept
    {
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

    bool SlabToFabricConverterAndCordinator::AttachFabric(
        uint64_t* raw_cells,
        uint64_t cell_count,
        CFC::FabricBackigOwnership ownership 
    ) noexcept
    {
        if (
            !raw_cells || cell_count < CFC::FABRIC_UNIT_COUNT || 
            ownership == CFC::FabricBackigOwnership::NONE ||
            (reinterpret_cast<uintptr_t>(raw_cells) % alignof(FabricCache) != UNSIGNED_ZERO)
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

        if (!APCRelocationDef::ValidateFabricCache(cache, cell_count))
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

            if (!control.Closed || control.ActiveAccess != UNSIGNED_ZERO || HandleOfAPCStatic::IsGenerationValid(control.Generation))
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

        return true;
        
    }
}
