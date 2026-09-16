#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    bool RecordBookConstructor::CheckRecordBookRange_(FabricSegments segment, uint64_t expected_begin, uint64_t expected_end) noexcept
    {
        RecordBookConf::FabricSegmentBounds bounds{};
        return
            GetRecordMapCarrierRanges_(segment, bounds) &&
            bounds.IsValid &&
            bounds.BeginIndex == expected_begin &&
            bounds.EndIndex == expected_end;
    }

    void RecordBookConstructor::IdleAFabricTableClassRangesMemory_(FabricSegments table_class) noexcept
    {

        RecordBookConf::FabricSegmentBounds return_bounds{};
        if (!GetRecordMapCarrierRanges_(table_class, return_bounds))
        {
            return;
        }

        for (size_t idx = return_bounds.BeginIndex; idx < return_bounds.EndIndex; idx++)
        {
            DirectlyStoreFabricUnit64(idx, UNSIGNED_ZERO);
        }
    }

    bool RecordBookConstructor::GetRecordMapCarrierRanges_(
        const FabricSegments table_class,
        RecordBookConf::FabricSegmentBounds& return_bounds
    ) noexcept
    {
        return_bounds = {};
        const uint64_t entry_idx = CoreOfFabricCoordinator::GetStartingOfAnyFabricTable_(table_class);
        if (
            entry_idx + CoreOfFabricCoordinator::RECORD_BOOK_WIDTH > FabCache_->SlabCellCount_ ||
            !ReadAFabricU64Directly(
                entry_idx + static_cast<uint8_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::BEGIN64),
                return_bounds.BeginIndex
            ) ||
            !ReadAFabricU64Directly(
                entry_idx + static_cast<uint8_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::END64),
                return_bounds.EndIndex
            ) ||
            return_bounds.BeginIndex >= return_bounds.EndIndex ||
            return_bounds.EndIndex > FabCache_->SlabCellCount_
        )
        {
            return_bounds.IsValid = false;
            return false;
        }
        return_bounds.IsValid = true;
        return return_bounds.IsValid;
    }


    void RecordBookConstructor::WriteARecordBookOfTSCEntry_(
        FabricSegments table_class, 
        size_t begin, 
        size_t end
    ) noexcept
    {
        const size_t base_idx = CoreOfFabricCoordinator::GetStartingOfAnyFabricTable_(table_class);
        if (
            !ADS::IsValidFabricUnit(base_idx) || 
            (base_idx + CoreOfFabricCoordinator::RECORD_BOOK_WIDTH > FabCache_->SlabCellCount_) ||
            begin >= end || end > FabCache_->SlabCellCount_
        )
        {
            return;
        }

        DirectlyStoreFabricUnit64(
            base_idx + static_cast<size_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::BEGIN64), 
            begin
        );
        
        DirectlyStoreFabricUnit64(
            base_idx + static_cast<size_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::END64), 
            end
        );                
        
    }

        
}
