#pragma once 
#include "GlobalStorage.hpp"

namespace BidirectionalInMemGraph
{

struct StorageAllocationDef
{
    using CFC = CoreOfFabricCoordinator;
    using SD = SchemaDefinition;
    using EB = EdgeBuilder;

    inline static std::array<CFC::FabricSegmentBounds, CFC::RECORD_BOOK_INTERNAL_SEGMENT_COUNT> __DefaultSegments__{};

    static constexpr bool FabCacheValid(CFC::FabricCache& cache) noexcept
    {
        if (
            cache.CountOfAPC_ > UNSIGNED_ZERO ||
            !ADS::IsCapacityOfAPCValid(cache.PerAPCRuntimeCellCount_) ||
            cache.PerAPCRuntimeCellCount_ % SD::RegionSchemaCellCount() != UNSIGNED_ZERO ||
            !EB::IsValidConfigurableParentCapacity(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_) ||
            !EB::IsBoundedRelationLocatorSize(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_);
            cache.ActiveRegionMask_ == UNSIGNED_ZERO ||
            (cache.ActiveRegionMask_ & ~ADS::ValidRegionMask()) ||
            cache.MatrixBatchCapacity_ == UNSIGNED_ZERO ||
            cache.ModelDiaphragm_.ModelCount_ == GMI::ModelDependency::INVALID_MODEL_IDX ||
            cache.ModelDiaphragm_.DirectoryCellCount_ > GMI::ModelDependency::INVALID_MODEL_IDX
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
            cache.ModelDiaphragm_.DirectoryCellCount_ * (sizeof(GMI::ModelDependency) / sizeof(uint64_t));
        return local_records > LIMIT_OF_CELL_COUNT ? std::nullopt : std::optional<uint64_t>{local_records};
    }

    static constexpr std::optional<uint64_t> __DynamicRecords__(CFC::FabricCache& cache) noexcept
    {
        uint64_t dynamics = (cache.ModelDiaphragm_.ModelDynamicsByteCount_ / sizeof(uint64_t)) +
            (cache.ModelDiaphragm_.ModelDynamicsByteCount_ % sizeof(uint64_t) != UNSIGNED_ZERO);
        return dynamics > LIMIT_OF_CELL_COUNT ? std::nullopt : std::optional<uint64_t>{dynamics};
    }

    static constexpr uint64_t __SegmentBegin__(FabricSegments segment) noexcept
    {
        return __DefaultSegments__[static_cast<size_t>(segment)].BeginIndex;
    }

    static constexpr bool __InitModelDescription__(CFC::FabricCache& cache, uint64_t& cursor) noexcept
    {
        using FS = FabricSegments;
        std::optional<uint64_t> lrm = __LocalRecords__(cache);
        std::optional<uint64_t> drm = __DynamicRecords__(cache);

        if (
            !lrm.has_value() || !drm.has_value() || 
            lrm.value() + drm.value() > LIMIT_OF_CELL_COUNT
        ) 
        {
            return false;
        }

        cache.ModelDiaphragm_.DirectoryCellCount_ = lrm.value() + drm.value();

        CFC::FabricSegmentBounds& model_bounds = __DefaultSegments__[static_cast<size_t>(FS::MODEL_DESCRIPTION)];
        model_bounds = {
            cursor, 
            cursor + (cursor * sizeof(GMI::NodeModelDescription)) + cache.ModelDiaphragm_.DirectoryCellCount_,
            FS::MODEL_DESCRIPTION,
            true
        };
        cache.ModelDiaphragm_.GlobalTableBeginIdx_ = __SegmentBegin__(FS::MODEL_DESCRIPTION);
        cache.ModelDiaphragm_.LocalDirectoryBeginIdx_ = cursor;
        cursor = CFC::DefaultFabricAlignment16Cell(model_bounds.EndIndex);
        return cursor > LIMIT_OF_CELL_COUNT ? false : true;
    }

    static constexpr void __InitHotCache__(CFC::FabricCache& cache) noexcept
    {
        using FS = FabricSegments;
        cache.RecordBookBeginIndex_ = __SegmentBegin__(FS::SLAB_RECORD_MAP);
        cache.RecordBookEndIndex_=__DefaultSegments__[static_cast<size_t>(FabricSegments::SLAB_RECORD_MAP)].EndIndex;
        cache.MatrixViewTableBeginIndex_ = __SegmentBegin__(FS::MATRIX_VIEW_TABLE);
        cache.HorizontalEdgeBeginIdx_ = __SegmentBegin__(FS::VALUE_PARENT_EDGE_TABLE_H);
        cache.VerticalEdgeBeginIdx_ = __SegmentBegin__(FS::VOLATILE_PARENT_EDGE_TABLE_V);
        cache.HandleTableBeginIndex_ = __SegmentBegin__(FS::APC_HANDLE_TABLE);
        cache.SegmentPoolBegin_ = __SegmentBegin__(FS::SEGMENT_POOL);
        cache.SlabCellCount_=__DefaultSegments__[static_cast<size_t>(FabricSegments::SEGMENT_POOL)].EndIndex;

    }

public :

    static constexpr bool BuildStorage(CFC::FabricCache& cache)
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
            CFC::FabricSegmentBounds& bounds = __DefaultSegments__[static_cast<size_t>(segment)];
            bounds = {
                cursor, 
                cursor + (cursor * width),
                segment,
                true
            };
            cursor = CFC::DefaultFabricAlignment16Cell(bounds.EndIndex);
            return cursor < LIMIT_OF_CELL_COUNT;
        };

        if (
            !Place___(FS::SLAB_RECORD_MAP, CFC::RECORD_BOOK_INTERNAL_SEGMENT_COUNT, CFC::RECORD_BOOK_WIDTH) ||
            !Place___(FS::MATRIX_VIEW_TABLE, cache.CountOfAPC_, cache.MatrixViewRowCellCount_) ||
            !Place___(FS::VALUE_PARENT_EDGE_TABLE_H, cache.CountOfAPC_, cache.EdgeTableRecordWidth_) ||
            !Place___(FS::VOLATILE_PARENT_EDGE_TABLE_V, cache.CountOfAPC_, cache.EdgeTableRecordWidth_) ||
            !Place___(FS::APC_HANDLE_TABLE, cache.CountOfAPC_, HandleOfAPCStatic::HANDLE_TABLE_WIDTH) ||
            !Place___(FS::APC_HANDLE_TABLE, cache.CountOfAPC_, CFC::WORK_RECORD_WIDTH_OF_FABRIC) ||
            !__InitModelDescription__(cache, cursor) ||
            !Place___(FS::SEGMENT_POOL, cache.CountOfAPC_, cache.PerAPCRuntimeCellCount_)
        ) 
        {
            return false;
        }
        __InitHotCache__(cache);
        return true;
    }

    static constexpr bool ValidateFabricCache(const CFC::FabricCache& cache, uint64_t supplied_cell_count) noexcept
    {
        if (
            cache.FormateVersion_ != CFC::FORMAT_VERSION ||
            cache.SlabCellCount_ != supplied_cell_count ||
            cache.SlabCellCount_ <  CFC::FABRIC_UNIT_COUNT ||
            cache.SlabCellCount_ >= FABRIC_CELL_SENTINAL ||
            cache.CountOfAPC_ == UNSIGNED_ZERO ||
            cache.CountOfAPC_ > UINT32_MAX ||
            cache.PerAPCRuntimeCellCount_ > UINT32_MAX ||
            !ADS::IsCapacityOfAPCValid(cache.PerAPCRuntimeCellCount_) ||
            !EdgeBuilder::IsValidConfigurableParentCapacity(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_)||
            !EdgeBuilder::IsBoundedRelationLocatorSize(cache.MaxDirectParentsPerAxis_, cache.CountOfAPC_) ||
            cache.EdgeTableRecordWidth_ != EdgeBuilder::EdgeTableRecordWidth(cache.MaxDirectParentsPerAxis_) ||
            cache.ActiveRegionMask_ == UNSIGNED_ZERO ||
            (cache.ActiveRegionMask_ & ~ADS::ValidRegionMask()) != UNSIGNED_ZERO ||
            cache.ActiveRegionCount_ != std::popcount(cache.ActiveRegionMask_) ||
            cache.ActiveRegionCount_ == UNSIGNED_ZERO ||
            cache.MatrixBatchCapacity_ == UNSIGNED_ZERO ||
            cache.MatrixViewRowCellCount_ != cache.ActiveRegionCount_ * SchemaDefinition::RegionSchemaCellCount() ||
            cache.RecordBookBeginIndex_ >=  cache.RecordBookEndIndex_ ||
            cache.RecordBookEndIndex_ > cache.SlabCellCount_ ||
            cache.SegmentPoolBegin_ >= cache.SlabCellCount_ ||
            cache.CountOfAPC_ > ((UINT64_MAX - cache.SegmentPoolBegin_) / cache.PerAPCRuntimeCellCount_) 
        )
        {
            return false;
        }
        return 
            (cache.SegmentPoolBegin_ + (cache.CountOfAPC_ * cache.PerAPCRuntimeCellCount_)) == cache.SlabCellCount_;
    }

};
}