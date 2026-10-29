#pragma once 
#include "GlobalStorage.hpp"

namespace BidirectionalInMemGraph::StorageAllocationDef
{
    using CFC = CoreOfFabricCoordinator;
    using SD = SchemaDefinition;
    using EB = EdgeBuilder;

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
            cache.ModelDiaphragm_.ModelDependencyCount_ > GMI::ModelDependency::INVALID_MODEL_IDX
        ) 
        {
            return false;
        }
        return true;
    }

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

    static constexpr bool BuildStorage(CFC::FabricCache& cache)
    {
        if (!FabCacheValid(cache)) return false;

        cache.ActiveRegionCount_ = static_cast<uint8_t>(std::popcount(cache.ActiveRegionMask_));
        if (SD::RegionSchemaCellCount() * cache.ActiveRegionCount_ > ADS::APC_INDEX_BOUND_SENTINAL) 
        {
            return false;
        }
        cache.MatrixViewRowCellCount_ = static_cast<uint32_t>(cache.ActiveRegionCount_ * SD::RegionSchemaCellCount());
        cache.EdgeTableRecordWidth_ = EB::EdgeTableRecordWidth(cache.MaxDirectParentsPerAxis_);
        cache.RegionAlignmentCellCount_ = SD::REGION_ALIGNMENT_CELLS;

        std::optional<uint64_t> lrm = __LocalRecords__(cache);
        std::optional<uint64_t> drm = __DynamicRecords__(cache);

        if (
            !lrm.has_value() || !drm.has_value() || 
            lrm.value() + drm.value() > LIMIT_OF_CELL_COUNT
        ) 
        {
            return false;
        }

        cache.ModelDiaphragm_.ModelDependencyCount_ = lrm.value() + drm.value();
        uint64_t cursor = CFC::DefaultFabricAlignment16Cell(CFC::FABRIC_UNIT_COUNT);

        const auto Place___ = [&](FabricSegments segment, uint64_t count, uint64_t width) noexcept -> bool
        {
            if (
                width > UNSIGNED_ZERO &&
                ((LIMIT_OF_CELL_COUNT - cursor) / width) < count
            ) 
            {
                return false;
            }

        };

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
}