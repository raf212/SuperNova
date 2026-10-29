#pragma once 
#include "GlobalStorage.hpp"

namespace BidirectionalInMemGraph
{

struct StorageAllocationDef
{
    using CFC = CoreOfFabricCoordinator;
    using SD = SchemaDefinition;
    using EB = EdgeBuilder;

    using SegmentBounds = std::array<CFC::FabricSegmentBounds, CFC::RECORD_BOOK_INTERNAL_SEGMENT_COUNT>;
    SegmentBounds DefaultSegments__{};

    static constexpr bool FabCacheValid(CFC::FabricCache& cache) noexcept
    {
        if (
            cache.CountOfAPC_ == UNSIGNED_ZERO ||
            !ADS::ValidateSlotIdx(cache.CountOfAPC_) ||
            !ADS::IsCapacityOfAPCValid(cache.PerAPCRuntimeCellCount_) ||
            cache.PerAPCRuntimeCellCount_ % SD::REGION_ALIGNMENT_CELLS != UNSIGNED_ZERO ||
            !EB::IsValidConfigurableParentCapacity(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_) ||
            !EB::IsBoundedRelationLocatorSize(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_) ||
            cache.ActiveRegionMask_ == UNSIGNED_ZERO ||
            (cache.ActiveRegionMask_ & ~ADS::ValidRegionMask()) ||
            cache.MatrixBatchCapacity_ == UNSIGNED_ZERO ||
            cache.ModelDiaphragm_.ModelCount_ == GMI::ModelDependency::INVALID_MODEL_IDX ||
            cache.ModelDiaphragm_.DynamicsRevision_ == UNSIGNED_ZERO
        ) 
        {
            return false;
        }
        return true;
    }

private :

    static constexpr std::optional<uint64_t> __LocalRecords__(CFC::FabricCache& cache) noexcept
    {
        uint64_t local_records = cache.ModelDiaphragm_.ModelCount_ * (sizeof(GMI::ModelInfo) / sizeof(uint64_t)) +
            cache.ModelDiaphragm_.ModelDependencyCount_ * (sizeof(GMI::ModelDependency) / sizeof(uint64_t));
        return local_records > LIMIT_OF_CELL_COUNT ? std::nullopt : std::optional<uint64_t>{local_records};
    }

    static constexpr std::optional<uint64_t> __DynamicRecords__(CFC::FabricCache& cache) noexcept
    {
        uint64_t dynamics = (cache.ModelDiaphragm_.ModelDynamicsByteCount_ / sizeof(uint64_t)) +
            (cache.ModelDiaphragm_.ModelDynamicsByteCount_ % sizeof(uint64_t) != UNSIGNED_ZERO);
        return dynamics > LIMIT_OF_CELL_COUNT ? std::nullopt : std::optional<uint64_t>{dynamics};
    }

    static constexpr uint64_t __SegmentBegin__(const SegmentBounds& segments, FabricSegments segment) noexcept
    {
        return segments[static_cast<size_t>(segment)].BeginIndex;
    }

    static constexpr bool __InitModelDescription__(CFC::FabricCache& cache, SegmentBounds& segments, uint64_t& cursor) noexcept
    {
        using FS = FabricSegments;
        std::optional<uint64_t> lrm = __LocalRecords__(cache);
        std::optional<uint64_t> drm = __DynamicRecords__(cache);

        if (
            !lrm.has_value() || !drm.has_value() || 
            lrm.value() > LIMIT_OF_CELL_COUNT - drm.value()
        ) 
        {
            return false;
        }

        cache.ModelDiaphragm_.DirectoryCellCount_ = lrm.value() + drm.value();

        const uint64_t per_slot_cells = sizeof(GMI::NodeModelDescription) / sizeof(uint64_t);
        const uint64_t node_cells = static_cast<uint64_t>(cache.CountOfAPC_) * per_slot_cells;
        if (node_cells > LIMIT_OF_CELL_COUNT - cursor) return false;
        const uint64_t directory_begin = CFC::DefaultFabricAlignment16Cell(cursor + node_cells);
        if (
            directory_begin > LIMIT_OF_CELL_COUNT ||
            cache.ModelDiaphragm_.DirectoryCellCount_ > LIMIT_OF_CELL_COUNT - directory_begin
        )
        {
            return false;
        }
        CFC::FabricSegmentBounds& model_bounds = segments[static_cast<size_t>(FS::MODEL_DESCRIPTION)];
        model_bounds = {
            cursor,
            directory_begin + cache.ModelDiaphragm_.DirectoryCellCount_,
            FS::MODEL_DESCRIPTION,
            true
        };
        cache.ModelDiaphragm_.GlobalTableBeginIdx_ = __SegmentBegin__(segments, FS::MODEL_DESCRIPTION);
        cache.ModelDiaphragm_.LocalDirectoryBeginIdx_ = directory_begin;
        cursor = CFC::DefaultFabricAlignment16Cell(model_bounds.EndIndex);
        return cursor > LIMIT_OF_CELL_COUNT ? false : true;
    }

    static constexpr void __InitHotCache__(CFC::FabricCache& cache, const SegmentBounds& segments) noexcept
    {
        using FS = FabricSegments;
        cache.RecordBookBeginIndex_ = __SegmentBegin__(segments, FS::SLAB_RECORD_MAP);
        cache.RecordBookEndIndex_=segments[static_cast<size_t>(FabricSegments::SLAB_RECORD_MAP)].EndIndex;
        cache.MatrixViewTableBeginIndex_ = __SegmentBegin__(segments, FS::MATRIX_VIEW_TABLE);
        cache.HorizontalEdgeBeginIdx_ = __SegmentBegin__(segments, FS::VALUE_PARENT_EDGE_TABLE_H);
        cache.VerticalEdgeBeginIdx_ = __SegmentBegin__(segments, FS::VOLATILE_PARENT_EDGE_TABLE_V);
        cache.HandleTableBeginIndex_ = __SegmentBegin__(segments, FS::APC_HANDLE_TABLE);
        cache.SegmentPoolBegin_ = __SegmentBegin__(segments, FS::SEGMENT_POOL);
        cache.SlabCellCount_= segments[static_cast<size_t>(FabricSegments::SEGMENT_POOL)].EndIndex;

    }

public :

    constexpr bool BuildStorage(CFC::FabricCache& cache) noexcept
    {
        using FS = FabricSegments;

        if (!FabCacheValid(cache)) return false;

        cache.ActiveRegionCount_ = static_cast<uint8_t>(std::popcount(cache.ActiveRegionMask_));
        if (SD::RegionSchemaCellCount() * cache.ActiveRegionCount_ > ADS::APC_INDEX_BOUND_SENTINAL) 
        {
            return false;
        }
        cache.MatrixViewRowCellCount_ = static_cast<uint32_t>(cache.ActiveRegionCount_ * SD::RegionSchemaCellCount());
        cache.EdgeTableRecordWidth_ = EB::EdgeTableRecordWidth(cache.MaxDirectParentsPerAxis_);
        cache.RegionAlignmentCellCount_ = SD::REGION_ALIGNMENT_CELLS;

        auto& segments = DefaultSegments__;
        segments = {};
        uint64_t cursor = CFC::DefaultFabricAlignment16Cell(CFC::FABRIC_UNIT_COUNT);

        const auto Place___ = [&](FS segment, uint64_t count, uint64_t width) noexcept -> bool
        {
            if (
                width > UNSIGNED_ZERO &&
                ((LIMIT_OF_CELL_COUNT - cursor) / width) < count
            ) 
            {
                return false;
            }
            CFC::FabricSegmentBounds& bounds = segments[static_cast<size_t>(segment)];
            bounds = {
                cursor, 
                cursor + (count * width),
                segment,
                true
            };
            cursor = CFC::DefaultFabricAlignment16Cell(bounds.EndIndex);
            return cursor <= LIMIT_OF_CELL_COUNT;
        };

        if (!Place___(FS::SLAB_RECORD_MAP, CFC::RECORD_BOOK_INTERNAL_SEGMENT_COUNT, CFC::RECORD_BOOK_WIDTH)) return false;
        // Preserve established structural cache-line placement after the record-book shrink.
        cursor = std::max<uint64_t>(cursor, 512u / sizeof(uint64_t));
        if (
            !Place___(FS::VALUE_PARENT_EDGE_TABLE_H, cache.CountOfAPC_, cache.EdgeTableRecordWidth_) ||
            !Place___(FS::MATRIX_VIEW_TABLE, cache.CountOfAPC_, cache.MatrixViewRowCellCount_) ||
            !Place___(FS::VOLATILE_PARENT_EDGE_TABLE_V, cache.CountOfAPC_, cache.EdgeTableRecordWidth_) ||
            !Place___(FS::APC_HANDLE_TABLE, cache.CountOfAPC_, HandleOfAPCStatic::HANDLE_TABLE_WIDTH) ||
            !Place___(FS::WORK_QUEUE, cache.CountOfAPC_, CFC::WORK_RECORD_WIDTH_OF_FABRIC) ||
            !__InitModelDescription__(cache, segments, cursor)
        ) 
        {
            return false;
        }
        cursor = std::max<uint64_t>(cursor, CFC::DEFAULT_FABRIC_CONTROLIO_LENGTH);
        if (!Place___(FS::SEGMENT_POOL, cache.CountOfAPC_, cache.PerAPCRuntimeCellCount_)) return false;
        __InitHotCache__(cache, segments);
        return true;
    }

    static constexpr bool MatchesLayout(const CFC::FabricCache& actual, const CFC::FabricCache& expected) noexcept
    {
        return
            actual.RecordBookBeginIndex_ == expected.RecordBookBeginIndex_ &&
            actual.RecordBookEndIndex_ == expected.RecordBookEndIndex_ &&
            actual.MatrixViewTableBeginIndex_ == expected.MatrixViewTableBeginIndex_ &&
            actual.HorizontalEdgeBeginIdx_ == expected.HorizontalEdgeBeginIdx_ &&
            actual.VerticalEdgeBeginIdx_ == expected.VerticalEdgeBeginIdx_ &&
            actual.HandleTableBeginIndex_ == expected.HandleTableBeginIndex_ &&
            actual.ModelDiaphragm_.GlobalTableBeginIdx_ == expected.ModelDiaphragm_.GlobalTableBeginIdx_ &&
            actual.ModelDiaphragm_.LocalDirectoryBeginIdx_ == expected.ModelDiaphragm_.LocalDirectoryBeginIdx_ &&
            actual.ModelDiaphragm_.DirectoryCellCount_ == expected.ModelDiaphragm_.DirectoryCellCount_ &&
            actual.SegmentPoolBegin_ == expected.SegmentPoolBegin_ &&
            actual.SlabCellCount_ == expected.SlabCellCount_ &&
            actual.EdgeTableRecordWidth_ == expected.EdgeTableRecordWidth_ &&
            actual.MatrixViewRowCellCount_ == expected.MatrixViewRowCellCount_ &&
            actual.ActiveRegionCount_ == expected.ActiveRegionCount_ &&
            actual.RegionAlignmentCellCount_ == expected.RegionAlignmentCellCount_;
    }

    static constexpr bool ValidateFabricCache(const CFC::FabricCache& cache, uint64_t supplied_cell_count) noexcept
    {
        if (
            cache.FormateVersion_ != CFC::FORMAT_VERSION ||
            cache.SegmentPoolDiskByteOffset_ != UNSIGNED_ZERO ||
            cache.ModelDiaphragm_.DynamicsRevision_ == UNSIGNED_ZERO ||
            cache.SlabCellCount_ != supplied_cell_count
        ) 
        {
            return false;
        }
        auto expected = cache;
        StorageAllocationDef planner;
        return planner.BuildStorage(expected) && MatchesLayout(cache, expected);
    }

};
}
