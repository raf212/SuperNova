#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

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
        const uint64_t entry_idx = GetStartingOfAnyFabricTable_(table_class);
        if (
            entry_idx + CoreOfFabricCoordinator::RECORD_BOOK_WIDTH > FVolatileCache_.SlabCellCount_ ||
            !ReadAFabricU64Directly(
                entry_idx + static_cast<uint8_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::BEGIN64),
                return_bounds.BeginIndex
            ) ||
            !ReadAFabricU64Directly(
                entry_idx + static_cast<uint8_t>(CoreOfFabricCoordinator::RecordBookInternalIndexing::END64),
                return_bounds.EndIndex
            ) ||
            return_bounds.BeginIndex >= return_bounds.EndIndex ||
            return_bounds.EndIndex > FVolatileCache_.SlabCellCount_
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
        const size_t base_idx = GetStartingOfAnyFabricTable_(table_class);
        if (
            !ADS::IsValidFabricUnit(base_idx) || 
            (base_idx + CoreOfFabricCoordinator::RECORD_BOOK_WIDTH > FVolatileCache_.SlabCellCount_) ||
            begin >= end || end > FVolatileCache_.SlabCellCount_
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


    uint64_t RecordBookConstructor::GetStartingOfAnyFabricTable_(
        FabricSegments table_class
    ) noexcept
    {   uint64_t record_map_begin = UNSIGNED_ZERO;
        const bool read_ok = ReadAFabricU64Directly(
            static_cast<size_t>(CoreOfFabricCoordinator::FabricMetaIndicies::RECORD_BOOK_OF_TSC_BEGIN),
            record_map_begin
        );
        if (
            !read_ok ||
            !ADS::IsValidFabricUnit(record_map_begin)
        )
        {
            return FABRIC_CELL_SENTINAL;
        }
        return static_cast<uint64_t>(record_map_begin + (static_cast<uint8_t>(table_class) * CoreOfFabricCoordinator::RECORD_BOOK_WIDTH));        
    }

        
}
