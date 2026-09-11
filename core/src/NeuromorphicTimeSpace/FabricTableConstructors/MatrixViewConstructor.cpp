#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"


namespace BidirectionalInMemGraph
{

    std::span<SchemaDefinition::RegionSchemaRecord> MatrixViewConstructor::MetrixViewRow_(uint32_t apc_slot) noexcept
    {
        if (
            !SlabBasePtr_ ||
            apc_slot >= FabCache_.CountOfAPC_ ||
            FabCache_.ActiveRegionCount_ == UNSIGNED_ZERO ||
            FabCache_.MatrixViewRowCellCount_ != FabCache_.ActiveRegionCount_ * SD::RegionSchemaCellCount()
        )
        {
            return {};
        }

        std::byte* table_bytes = reinterpret_cast<std::byte*>(SlabBasePtr_ + FabCache_.MatrixViewTableBeginIndex_);

        const size_t row_byte_offset = static_cast<size_t>(apc_slot) * FabCache_.MatrixViewRowCellCount_ * sizeof(uint64_t);

        SD::RegionSchemaRecord* row = std::launder(reinterpret_cast<SD::RegionSchemaRecord*>(table_bytes + row_byte_offset));

        return std::span<SD::RegionSchemaRecord>(row, FabCache_.ActiveRegionCount_);
    }

    bool MatrixViewConstructor::ConstructMatrixViewRecords_(size_t table_begin, size_t table_end) noexcept
    {
        const size_t total_records = static_cast<size_t>(FabCache_.CountOfAPC_) * FabCache_.ActiveRegionCount_;
        const size_t required_cells = total_records * SD::RegionSchemaCellCount();

        if (
            !SlabBasePtr_ ||
            table_begin >= table_end ||
            table_end - table_begin != required_cells ||
            table_end > FabCache_.SlabCellCount_
        )
        {
            return false;
        }
        
        std::byte* bytes = reinterpret_cast<std::byte*>(SlabBasePtr_ + table_begin);
        for (size_t i = 0; i < total_records; i++)
        {
            void* place = static_cast<void*>(bytes + (i * sizeof(SD::RegionSchemaRecord)));
            std::construct_at(reinterpret_cast<SD::RegionSchemaRecord*>(place), SD::RegionSchemaRecord{});
        }
        return true;
    }

    bool MatrixViewConstructor::PrepareMatrixViewRow_(
        uint32_t apc_slot,
        const SD::RegionSchemaTable& requested
    ) noexcept
    {
        std::span<SD::RegionSchemaRecord> destination = MetrixViewRow_(apc_slot);
        if (destination.size() != FabCache_.ActiveRegionCount_)
        {
            return false;
        }
        
        SD::RegionSchemaTable prepared{};
        uint16_t observed_mask = UNSIGNED_ZERO;
        uint8_t prepared_count = UNSIGNED_ZERO;
        uint32_t cursor = SD::AlignRegionCells(ADS::META_CELL_COUNT);

        for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
        {
            const SD::RegionSchemaRecord& source = requested[i];
            const MacroColumnOfAPC expected_region = static_cast<MacroColumnOfAPC>(i);
            if (
                source.Region != expected_region ||
                prepared_count >= prepared.size()
            )
            {
                return false;
            }

            if (SD::HasSchemaFlag(source.Flags, SD::SchemaFlags::REGION_DISABLED))
            {
                continue;
            }

            SD::RegionSchemaRecord record = source;
            record.CellOffset = SD::AlignRegionCells(cursor);

            if (
                SD::HasSchemaFlag(record.Flags, SD::SchemaFlags::BATCHED_LAST_DIM) &&
                record.MatrixWidth != FabCache_.MatrixBatchCapacity_
            )
            {
                return false;
            }
            
            if (!SD::FreshProtocolState(record))
            {
                return false;
            }

            if (!SD::ValidateStortedRegionSchema(
                record,
                FabCache_.PerAPCRuntimeCellCount_,
                FabCache_.MatrixBatchCapacity_
            ))
            {
                return false;
            }
            
            cursor = record.CellOffset + record.CellCount;
            observed_mask = static_cast<uint16_t>(observed_mask | ADS::RegionBit(expected_region));
            prepared[prepared_count++] = record;
        }
        
        if (
            observed_mask != FabCache_.ActiveRegionMask_ ||
            prepared_count != FabCache_.ActiveRegionCount_
        )
        {
            return false;
        }
        
        for (uint8_t i = 0; i < prepared_count; i++)
        {
            destination[i] = prepared[i];
        }
        return true;
    }

    void MatrixViewConstructor::ClearMatrixViewRow_(uint32_t apc_slot) noexcept
    {
        for (SD::RegionSchemaRecord& record : MetrixViewRow_(apc_slot))
        {
            record = SD::RegionSchemaRecord{};
        };
    }

    bool MatrixViewConstructor::InitializeRegionProtocolStorage_(uint32_t apc_slot) noexcept
    {
        if (apc_slot >= FabCache_.CountOfAPC_)
        {
            return false;
        }

        const size_t apc_begin = FabCache_.SegmentPoolBegin_ +  static_cast<size_t>(apc_slot) * FabCache_.PerAPCRuntimeCellCount_;
        std::span<SD::RegionSchemaRecord> row = MetrixViewRow_(apc_slot);
        if (row.size() != FabCache_.ActiveRegionCount_)
        {
            return false;
        }

        for (SD::RegionSchemaRecord& schema : row)
        {
            if (schema.Protocol != SD::SchemaProtocols::MPMC_FIXED_RECORD_QUEUE)
            {
                continue;
            }
            
            const std::optional<uint32_t> matrix_cells = SD::MatrixCellCount(schema);
            const std::optional<uint32_t> stride_cells = SD::RecordStrideCells(schema);
            const std::optional<uint32_t> record_count = SD::LogicalRecordCount(schema);

            if (
                !matrix_cells.has_value() ||
                !stride_cells.has_value() ||
                !record_count.has_value()
            )
            {
                return false;
            }
            
            schema.EnqueuePosition = UNSIGNED_ZERO;
            schema.DequeuePosition = UNSIGNED_ZERO;

            for (uint32_t i = 0; i < record_count.value(); i++)
            {
                const std::size_t sequense_cell = apc_begin + schema.CellOffset + 
                    (static_cast<size_t>(i) * stride_cells.value()) + matrix_cells.value();
                if (sequense_cell >= FabCache_.SlabCellCount_)
                {
                    return false;
                }
                SlabBasePtr_[sequense_cell] = i;
            }
        }
        
        return true;
    }




}