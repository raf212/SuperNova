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
        SlabBasePtr_ = nullptr;
        SlabCellCount_ = UNSIGNED_ZERO;
        PerAPCRuntimeCellCount_ = UNSIGNED_ZERO;
        CountOfAPC_ = UNSIGNED_ZERO;
        MaxDirectParentsPerAxis_ = UNSIGNED_ZERO;
        EdgeTableRecordWidth_ = UNSIGNED_ZERO;
        SegmentPoolBegin_ = CoreOfFabricCoordinator::FABRIC_UNIT_COUNT;
        FabricInitialized_.store(false, std::memory_order_release);
        InitializationInProgress_.store(false, std::memory_order_release);
    }



    void SlabToFabricConverterAndCordinator::InitializeCompleateFabricMetaIndices_(size_t record_book_begin, size_t record_book_end) noexcept
    {
        using FMI = CoreOfFabricCoordinator::FabricMetaIndicies;

        for (size_t i = 0; i < CoreOfFabricCoordinator::FABRIC_UNIT_COUNT; i++)
        {
            DirectlyStoreFabricUnit64(i, UNSIGNED_ZERO);
        }

        SlabBasePtr_[static_cast<size_t>(FMI::MAGIC)] = CoreOfFabricCoordinator::FABRIC_MAGIC;
        SlabBasePtr_[static_cast<size_t>(FMI::TOTAL_CELLS)] = SlabCellCount_;
        SlabBasePtr_[static_cast<size_t>(FMI::SEGMENT_POOL_BEGIN_IDX)] = SegmentPoolBegin_;
        SlabBasePtr_[static_cast<size_t>(FMI::PER_APC_RUNTIME_CELL_COUNT)] = PerAPCRuntimeCellCount_;
        SlabBasePtr_[static_cast<size_t>(FMI::RECORD_BOOK_OF_TSC_BEGIN)] = record_book_begin;
        SlabBasePtr_[static_cast<size_t>(FMI::RECORD_BOOK_OF_TSC_END)] = record_book_end;
        SlabBasePtr_[static_cast<size_t>(FMI::FIRST_FREE_IDX)] = UNSIGNED_ZERO;
        SlabBasePtr_[static_cast<size_t>(FMI::EDGE_TABLE_RECORD_WIDTH)] = EdgeTableRecordWidth_;
        SlabBasePtr_[static_cast<size_t>(FMI::MAX_DIRECT_PARENTS_PER_AXIS)] = MaxDirectParentsPerAxis_;
        SlabBasePtr_[static_cast<size_t>(FMI::ACTIVE_REGION_MASK)] = ActiveRegionMask_;
        SlabBasePtr_[static_cast<size_t>(FMI::ACTIVE_REGION_COUNT)] = ActiveRegionCount_;
        SlabBasePtr_[static_cast<size_t>(FMI::REGION_SCHEMA_RECORD_CELL_COUNT)] = SD::RegionSchemaCellCount();
        SlabBasePtr_[static_cast<size_t>(FMI::DEVICE_VIEW_ROW_CELL_COUNT)] = MatrixViewRowCellCount_;
        SlabBasePtr_[static_cast<size_t>(FMI::MATRIC_BATCH_CAPACITY)] = MatrixBatchCapacity_;
        SlabBasePtr_[static_cast<size_t>(FMI::REGION_ALLIGNMENT_CELL_COUNT)] = SD::REGION_ALIGNMENT_CELLS;
        SlabBasePtr_[static_cast<size_t>(FMI::EOF_FABRIC_HEADER)] = CoreOfFabricCoordinator::FABRIC_META_EOF;
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

        MaxDirectParentsPerAxis_ = max_direct_parent_per_axis;
        EdgeTableRecordWidth_ = static_cast<uint16_t>(EdgeBuilder::EdgeTableRecordWidth(MaxDirectParentsPerAxis_));
        CountOfAPC_ = static_cast<uint64_t>(slot_count);
        PerAPCRuntimeCellCount_ = static_cast<uint32_t>(slot_cell_count);

        ActiveRegionMask_ = active_mask;
        ActiveRegionCount_ = active_count;
        MatrixBatchCapacity_ = region_conf.BatchCapacity;
        MatrixViewRowCellCount_ = static_cast<uint16_t>(
            static_cast<uint16_t>(ActiveRegionCount_) *
            SD::RegionSchemaCellCount()
        );

        size_t cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(CoreOfFabricCoordinator::FABRIC_UNIT_COUNT);
        const size_t record_book_begin = cursor;
        const size_t record_book_end = record_book_begin + static_cast<size_t>(RecordBookConf::RECORD_BOOK_INTERNAL_SEGMENT_COUNT) * CoreOfFabricCoordinator::RECORD_BOOK_WIDTH;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(record_book_end);
        const size_t horizontal_edge_begin = cursor;
        const size_t horizontal_edge_end = horizontal_edge_begin + static_cast<size_t>(CountOfAPC_) * EdgeTableRecordWidth_;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(horizontal_edge_end);
        const size_t matrix_view_table_begin = cursor;
        const size_t matrix_view_table_end = matrix_view_table_begin + static_cast<size_t>(CountOfAPC_ * MatrixViewRowCellCount_);
        
        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(matrix_view_table_end);
        const size_t vertical_edge_begin = cursor;
        const size_t vertical_edge_end = vertical_edge_begin + 
                static_cast<size_t>(CountOfAPC_) * EdgeTableRecordWidth_;

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(vertical_edge_end);
        const size_t apc_handle_table_begin = cursor;
        const size_t apc_handle_table_end = apc_handle_table_begin + static_cast<size_t>(CountOfAPC_ * HandleOfAPCStatic::HANDLE_TABLE_WIDTH);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(apc_handle_table_end);
        const size_t compiled_dag_begin = cursor;
        const size_t compiled_dag_end = compiled_dag_begin + static_cast<size_t>(CountOfAPC_ * CoreOfFabricCoordinator::COMPILED_DAG_LEN);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(compiled_dag_end);
        const size_t device_planner_begain = cursor;
        const size_t device_planner_end = device_planner_begain + static_cast<size_t>(CountOfAPC_ * CoreOfFabricCoordinator::DEVICE_PLANNER_RECORD_LEN);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(device_planner_end);
        const size_t work_queue_begin = cursor;
        const size_t work_queue_end = work_queue_begin + static_cast<size_t>(CountOfAPC_ * CoreOfFabricCoordinator::WORK_RECORD_WIDTH_OF_FABRIC);

        cursor = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(work_queue_end);
        SegmentPoolBegin_ = CoreOfFabricCoordinator::DefaultFabricAlignment16Cell_(std::max<size_t>(cursor, CoreOfFabricCoordinator::DEFAULT_FABRIC_CONTROLIO_LENGTH));
        SlabCellCount_ = SegmentPoolBegin_ + static_cast<size_t>(CountOfAPC_ * PerAPCRuntimeCellCount_);

        if (SlabCellCount_ == UNSIGNED_ZERO || SlabCellCount_ >= FABRIC_CELL_SENTINAL)
        {
            return false;
        }
        SlabBasePtr_ = AllocatePackedCellRaw_(SlabCellCount_);
        if (!SlabBasePtr_)
        {
            return false;
        }

        for (size_t idx = 0; idx < SlabCellCount_; idx++)
        {
            DirectlyStoreFabricUnit64(idx, UNSIGNED_ZERO);
        }

        InitializeCompleateFabricMetaIndices_(record_book_begin, record_book_end);

        //RECORD_BOOK_OF_TABLE_SEGMENT_CLASS - ENTRIES
        WriteARecordBookOfTSCEntry_(FabricSegments::SLAB_RECORD_MAP, record_book_begin, record_book_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::VALUE_PARENT_EDGE_TABLE_H, horizontal_edge_begin, horizontal_edge_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V, vertical_edge_begin, vertical_edge_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::APC_HANDLE_TABLE, apc_handle_table_begin, apc_handle_table_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::COMPILED_DAG_TABLE, compiled_dag_begin, compiled_dag_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::DEVICE_PLANNER_TABLE, device_planner_begain, device_planner_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::WORK_QUEUE, work_queue_begin, work_queue_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::MATRIX_VIEW_TABLE, matrix_view_table_begin, matrix_view_table_end);
        WriteARecordBookOfTSCEntry_(FabricSegments::SEGMENT_POOL, SegmentPoolBegin_, SlabCellCount_);

        HorizontalEdgeBeginIdx_ = horizontal_edge_begin;
        VerticalEdgeBeginIdx_ = vertical_edge_begin;
        HandleTableBeginIndex_ = apc_handle_table_begin;
        MatrixViewTableBeginIndex_ = matrix_view_table_begin;

        if (!ConstructMatrixViewRecords_(matrix_view_table_begin, matrix_view_table_end))
        {
            return false;
        }

        if (HasDefaultRegionTable_)
        {
            for (uint32_t i = 0; i < CountOfAPC_; i++)
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

        FabricInitialized_.store(false, std::memory_order_release);
        uint64_t* old_ptr = SlabBasePtr_;
        const size_t old_count = SlabCellCount_;
        SlabBasePtr_ = nullptr;
        SlabCellCount_ = UNSIGNED_ZERO;
        if (old_ptr)
        {
            FreeRawPackedCells_(old_ptr, old_count);
        }
        ResetScalarsofTheFabric_();
    }


}
