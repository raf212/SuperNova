#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    bool RegionViewConstructor::ResolveRegionView_(
        MacroColumnOfAPC column,
        uint32_t record_ordinal,
        ResolveRegionBiteView& out
    ) noexcept
    {
        using SD = SchemaDefinition;
        using ASG = APCStorageGeometry;

        std::span<SchemaDefinition::RegionSchemaRecord> region_row = APCCache_.FabricOwnerPtr_->MetrixViewRow_(
            static_cast<uint32_t>(APCCache_.APCSlotIdx_)
        );

        out = ResolveRegionBiteView{};
        if (
            !IsActiveAPC() ||
            !APCCache_.RawAPCBasePtr_  ||
            region_row.size() != APCCache_.FabricOwnerPtr_->FabCache_.ActiveRegionCount_
        )
        {
            return false;
        }

        const std::optional<uint8_t> compact_index = ADS::CompactRegionIndex(APCCache_.FabricOwnerPtr_->FabCache_.ActiveRegionMask_, column);
        if (!compact_index.has_value())
        {
            return false;
        }
        

        const SD::RegionSchemaRecord& stored = region_row[compact_index.value()];

        if (
            stored.Region != column ||
            !SD::ValidateStortedRegionSchema(
                stored,
                APCCache_.FabricOwnerPtr_->FabCache_.PerAPCRuntimeCellCount_,
                APCCache_.FabricOwnerPtr_->FabCache_.MatrixBatchCapacity_
            )
        )
        {
            return false;
        }

        const std::optional<uint64_t> matrix_bytes = SD::MatrixByteCount(stored);
        const std::optional<uint32_t> matrix_cells = SD::MatrixCellCount(stored);
        const std::optional<uint32_t> stride_cells = SD::RecordStrideCells(stored);
        const std::optional<uint32_t> record_count = SD::LogicalRecordCount(stored);

        if (
            !matrix_bytes.has_value() ||
            !matrix_cells.has_value() ||
            !stride_cells.has_value() ||
            !record_count.has_value() ||
            record_ordinal >= record_count.value()
        )
        {
            return false;
        }

        const uint64_t local_data_cell = static_cast<uint64_t>(stored.CellOffset) + (static_cast<std::uint64_t>(record_ordinal) * stride_cells.value());

        if (
            local_data_cell >= APCCache_.FabricOwnerPtr_->FabCache_.PerAPCRuntimeCellCount_ ||
            matrix_cells.value() > APCCache_.FabricOwnerPtr_->FabCache_.PerAPCRuntimeCellCount_ - local_data_cell
        )
        {
            return false;
        }
        
        out.Bytes = std::span<std::byte>(
            APCCache_.RawAPCBasePtr_ + (static_cast<size_t>(local_data_cell) * sizeof(uint64_t)),
            static_cast<size_t>(matrix_bytes.value())
        );
        out.Schema = &stored;
        out.RegionOrdinal = record_ordinal;
        return true;
    }

}