#pragma once
#include "APCDataStructure.hpp"

namespace BidirectionalInMemGraph
{

    struct SchemaOrchestrator 
    {

        static constexpr uint32_t REGION_ALIGNMENT_CELLS = static_cast<uint32_t>(ADS::APC_CACHELINE_SIZE / sizeof(std::uint64_t));
        static constexpr uint64_t NO_POSITION = FABRIC_CELL_SENTINAL;

        enum class SchemaProtocols : uint8_t
        {
            PRIVATE_REGION = 0,
            IMMUTABLE_SNAPSHOT = 1,
            ATOMIC_WORD_ARRAY = 2,
            MPMC_FIXED_RECORD_QUEUE = 3,
            DOUBLE_BUFFERED = 4
        };

        enum class DataTypeOfMacroColumn : uint8_t
        {
            UINT8_T = 0,
            UINT16_T = 1,
            UINT32_T = 2,
            UINT64_T = 3,
            INT8_T = 4,
            INT16_T = 5,
            INT32_T = 6,
            INT64_T = 7,
            FLOAT16_T = 8,
            FLOAT32_T = 9,
            FLOAT64_T = 10,
            CHAR = 11
        };

        enum class SchemaFlags : uint8_t
        {
            NONE = 0u,
            REQUIRED_POW_OF_TWO = 1u << 0u,
            ALLOW_QUICENT_SCHEMA_MUTATION = 1u << 1u ,
            ALLOW_TRAILING_PADDING = 1u << 2u,
            HAS_PER_SLOT_SEQUENSE = 1u << 3u,
            REGION_DISABLED = 1u << 4u,
            BATCHED_LAST_DIM = 1u << 5u,
            UNASSIGNED_UNUSED_NANNULL = UINT8_MAX
        };

        struct alignas(uint64_t) RegionSchemaRecord final
        {
            uint32_t CellOffset = ADS::APC_INDEX_BOUND_SENTINAL;
            uint32_t CellCount = UNSIGNED_ZERO;
            uint32_t MatrixHeight = UNSIGNED_ZERO;
            uint32_t MatrixWidth = UNSIGNED_ZERO;

            uint64_t EnqueuePosition = FABRIC_CELL_SENTINAL;
            uint64_t DequeuePosition = FABRIC_CELL_SENTINAL;
            
            MacroColumnOfAPC Region = MacroColumnOfAPC::FREE_SLOT;
            SchemaProtocols Protocol = SchemaProtocols::PRIVATE_REGION;
            DataTypeOfMacroColumn Dtype = DataTypeOfMacroColumn::UINT64_T;
            SchemaFlags Flags = SchemaFlags::REGION_DISABLED;   
            uint32_t SeqLockCounter = UNSIGNED_ZERO;
        };
        
        static_assert(sizeof(RegionSchemaRecord) == 5u * sizeof(std::uint64_t));
        static_assert(alignof(RegionSchemaRecord) == alignof(std::uint64_t));
        static_assert(std::is_trivially_copyable_v<RegionSchemaRecord>);
        static_assert(std::is_trivially_destructible_v<RegionSchemaRecord>);
        
        struct FabricRegionConfig final
        {
            uint16_t ActiveRegionMask = UNSIGNED_ZERO;
            uint16_t Reserved = UNSIGNED_ZERO;
            uint32_t BatchCapacity = UNSIGNED_ZERO;
        };

        using RegionSchemaTable = std::array<RegionSchemaRecord, ADS::CountOfMacroColumn()>;

    };
    
    struct SchemaValidator : public SchemaOrchestrator
    {

        static constexpr size_t RegionSchemaCellCount() noexcept
        {
            return sizeof(RegionSchemaRecord) / sizeof(uint64_t);
        }

        static constexpr bool IsMPMCQueue(const RegionSchemaRecord& provided_schema) noexcept
        {
            return provided_schema.Protocol == SchemaProtocols::MPMC_FIXED_RECORD_QUEUE;
        }

        static constexpr bool HasSchemaFlag(SchemaFlags current_flag, SchemaFlags desired_bit) noexcept
        {
            return (
                static_cast<uint8_t>(current_flag) & static_cast<uint8_t>(desired_bit)
            ) != UNSIGNED_ZERO;
        }


        static constexpr bool IsKnownSchemaFlags(SchemaFlags flags) noexcept
        {
            constexpr uint8_t KNOWN_FLAGS = 
                static_cast<uint8_t>(SchemaFlags::REQUIRED_POW_OF_TWO) |
                static_cast<uint8_t>(SchemaFlags::ALLOW_QUICENT_SCHEMA_MUTATION) |
                static_cast<uint8_t>(SchemaFlags::ALLOW_TRAILING_PADDING) |
                static_cast<uint8_t>(SchemaFlags::HAS_PER_SLOT_SEQUENSE) |
                static_cast<uint8_t>(SchemaFlags::BATCHED_LAST_DIM) |
                static_cast<uint8_t>(SchemaFlags::REGION_DISABLED);
            
            const uint8_t raw = static_cast<uint8_t>(flags);

            return flags != SchemaFlags::UNASSIGNED_UNUSED_NANNULL &&
                (raw & static_cast<uint8_t>(~KNOWN_FLAGS)) == UNSIGNED_ZERO;
                
        }

        static constexpr bool IsValuePowOfTwoU32(uint32_t value) noexcept
        {
            return value >= 2u && ((value & (value - 1u)) == UNSIGNED_ZERO);
        }


        template<class DType>
        static constexpr std::optional<DataTypeOfMacroColumn> CppTypeToRegionDType() noexcept
        {
            using U = std::remove_cv_t<DType>;

            if constexpr (std::is_same_v<U, uint8_t>)
                return DataTypeOfMacroColumn::UINT8_T;

            else if constexpr (std::is_same_v<U, uint16_t>)
                return DataTypeOfMacroColumn::UINT16_T;

            else if constexpr (std::is_same_v<U, uint32_t>)
                return DataTypeOfMacroColumn::UINT32_T;

            else if constexpr (std::is_same_v<U, uint64_t>)
                return DataTypeOfMacroColumn::UINT64_T;

            else if constexpr (std::is_same_v<U, int8_t>)
                return DataTypeOfMacroColumn::INT8_T;

            else if constexpr (std::is_same_v<U, int16_t>)
                return DataTypeOfMacroColumn::INT16_T;

            else if constexpr (std::is_same_v<U, int32_t>)
                return DataTypeOfMacroColumn::INT32_T;

            else if constexpr (std::is_same_v<U, int64_t>)
                return DataTypeOfMacroColumn::INT64_T;

            else if constexpr (std::is_same_v<U, float>)
                return DataTypeOfMacroColumn::FLOAT32_T;

            else if constexpr (std::is_same_v<U, double>)
                return DataTypeOfMacroColumn::FLOAT64_T;

            else if constexpr (std::is_same_v<U, char>)
                return DataTypeOfMacroColumn::CHAR;

            else
                return std::nullopt;
        }

        static constexpr std::optional<uint8_t> DTypeByteCount(DataTypeOfMacroColumn data_type) noexcept
        {
            switch (data_type)
            {
            case DataTypeOfMacroColumn::UINT8_T:
            case DataTypeOfMacroColumn::INT8_T:
            case DataTypeOfMacroColumn::CHAR:
                return static_cast<uint8_t>(sizeof(uint8_t));

            case DataTypeOfMacroColumn::UINT16_T:
            case DataTypeOfMacroColumn::INT16_T:
            case DataTypeOfMacroColumn::FLOAT16_T:
                return static_cast<uint8_t>(sizeof(uint16_t));

            case DataTypeOfMacroColumn::UINT32_T:
            case DataTypeOfMacroColumn::INT32_T:
            case DataTypeOfMacroColumn::FLOAT32_T:
                return static_cast<uint8_t>(sizeof(uint32_t));

            case DataTypeOfMacroColumn::UINT64_T:
            case DataTypeOfMacroColumn::INT64_T:
            case DataTypeOfMacroColumn::FLOAT64_T:
                return static_cast<uint8_t>(sizeof(uint64_t));

            default:
                return std::nullopt;
            }
        }

        static constexpr std::optional<uint64_t> MatrixByteCount(const RegionSchemaRecord& schema) noexcept
        {
            const std::optional<uint8_t> dtype_bytes = DTypeByteCount(schema.Dtype);
            if (
                !dtype_bytes.has_value() ||
                schema.MatrixHeight == UNSIGNED_ZERO ||
                schema.MatrixWidth == UNSIGNED_ZERO ||
                schema.MatrixHeight > UINT64_MAX / schema.MatrixWidth
            )
            {
                return std::nullopt;
            }

            const uint64_t elements = static_cast<uint64_t>(schema.MatrixHeight) * schema.MatrixWidth;

            if (elements > UINT64_MAX / dtype_bytes.value())
            {
                return std::nullopt;
            }
            
            return elements * dtype_bytes.value();
        }

        static constexpr std::optional<uint32_t> MatrixCellCount(const RegionSchemaRecord& schema) noexcept
        {
            const std::optional<uint64_t> bytes = MatrixByteCount(schema);
            if (!bytes.has_value())
            {
                return std::nullopt;
            }

            const uint64_t cells = bytes.value() / sizeof(uint64_t) + 
                static_cast<uint64_t>(bytes.value() % sizeof(uint64_t) != UNSIGNED_ZERO);

            return cells <= UINT32_MAX ?
                std::optional<uint32_t>(static_cast<uint32_t>(cells)) : std::nullopt;
            
        }

        static constexpr std::optional<std::uint32_t> RecordStrideCells(const RegionSchemaRecord& schema) noexcept
        {
            const std::optional<uint64_t> matrix_cell = MatrixCellCount(schema);
            if (!matrix_cell.has_value())
            {
                return std::nullopt;
            }

            const uint64_t raw = static_cast<uint64_t>(matrix_cell.value()) + 
                static_cast<uint64_t>(schema.Protocol == SchemaProtocols::MPMC_FIXED_RECORD_QUEUE);
            
            const uint64_t aligned = ((raw + REGION_ALIGNMENT_CELLS - 1u) / REGION_ALIGNMENT_CELLS) * REGION_ALIGNMENT_CELLS;

            return aligned <= UINT32_MAX ? 
                std::optional<uint32_t>(static_cast<uint32_t>(aligned)) : std::nullopt;
            
        }


        static constexpr std::optional<uint32_t> LogicalRecordCount(const RegionSchemaRecord& schema) noexcept
        {
            const std::optional<uint32_t> stride = RecordStrideCells(schema);
            if (
                !stride.has_value() ||
                stride.value() == UNSIGNED_ZERO ||
                schema.CellCount == UNSIGNED_ZERO ||
                schema.CellCount % stride.value() != UNSIGNED_ZERO
            )
            {
                return std::nullopt;
            }

            return schema.CellCount / stride.value();
        }

        static constexpr uint32_t AlignRegionCells(uint32_t cell) noexcept
        {
            return static_cast<uint32_t>(
                ((static_cast<uint64_t>(cell) + REGION_ALIGNMENT_CELLS - 1u) / REGION_ALIGNMENT_CELLS) * REGION_ALIGNMENT_CELLS
            );
        }

        static constexpr bool ValidateStortedRegionSchema(
            const RegionSchemaRecord& schema,
            uint32_t apc_cell_count,
            uint32_t  fabric_batch_capacity
        ) noexcept
        {
            if (
                !IsKnownSchemaFlags(schema.Flags) ||
                HasSchemaFlag(schema.Flags, SchemaFlags::REGION_DISABLED) ||
                schema.CellOffset < ADS::META_CELL_COUNT ||
                schema.CellOffset % REGION_ALIGNMENT_CELLS != UNSIGNED_ZERO ||
                schema.CellCount == UNSIGNED_ZERO ||
                schema.CellOffset > apc_cell_count ||
                schema.CellCount > apc_cell_count - schema.CellOffset
            )
            {
                return false;
            }
            
            if (
                HasSchemaFlag(schema.Flags, SchemaFlags::BATCHED_LAST_DIM) &&
                schema.MatrixWidth != fabric_batch_capacity
            )
            {
                return false;
            }

            const std::optional<uint32_t> record_count = LogicalRecordCount(schema);
            if (!record_count.has_value())
            {
                return false;
            }
            

            switch (schema.Protocol)
            {
            case SchemaProtocols::PRIVATE_REGION:
            case SchemaProtocols::IMMUTABLE_SNAPSHOT:
            case SchemaProtocols::ATOMIC_WORD_ARRAY:
                return 
                    record_count.value() == 1u &&
                    schema.EnqueuePosition == NO_POSITION &&
                    schema.DequeuePosition == NO_POSITION;
            
            case SchemaProtocols::DOUBLE_BUFFERED:
                return 
                    record_count.value() == 2u &&
                    schema.EnqueuePosition < 2u &&
                    schema.DequeuePosition < 2u;

            case SchemaProtocols::MPMC_FIXED_RECORD_QUEUE:
                return
                    record_count.value() >= 2u &&
                    (record_count.value() & (record_count.value() - 1u)) == UNSIGNED_ZERO &&
                    HasSchemaFlag(schema.Flags, SchemaFlags::REQUIRED_POW_OF_TWO) &&
                    HasSchemaFlag(schema.Flags, SchemaFlags::HAS_PER_SLOT_SEQUENSE);
            
            default:
                return false;
            }
        }


        static constexpr bool FreshProtocolState(const RegionSchemaRecord& record) noexcept
        {
            return 
                (record.Protocol == SchemaProtocols::DOUBLE_BUFFERED) ? 
                    (record.EnqueuePosition == 1u && record.DequeuePosition == 0u) : (record.Protocol == SchemaProtocols::MPMC_FIXED_RECORD_QUEUE) ? 
                        (record.EnqueuePosition == 0u && record.DequeuePosition == 0u) : (record.EnqueuePosition == NO_POSITION && record.DequeuePosition == NO_POSITION);
        }
    };

    static constexpr SchemaOrchestrator::SchemaFlags operator|(SchemaOrchestrator::SchemaFlags lhs, SchemaOrchestrator::SchemaFlags rhs) noexcept
    {
        return static_cast<SchemaOrchestrator::SchemaFlags>(
            static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs)
        );
    }

    struct SchemaDefinition : public SchemaValidator
    {
        friend class GHGFLayerModel;
    private:
        static constexpr bool AttachPrivateFloat32ToTable_(
            RegionSchemaTable& table,
            MacroColumnOfAPC region,
            uint32_t height,
            uint32_t width,
            SchemaFlags flags
        ) noexcept
        {
            RegionSchemaRecord& schema =
                table[static_cast<std::size_t>(region)];

            schema.Region = region;
            schema.Dtype = DataTypeOfMacroColumn::FLOAT32_T;
            schema.Protocol = SchemaProtocols::PRIVATE_REGION;
            schema.MatrixHeight = height;
            schema.MatrixWidth = width;
            schema.Flags = flags;

            return SealDesiredSchema(schema, UNSIGNED_ZERO);
        }

        static constexpr uint16_t GetActiveMaskOfRegionTable_(const RegionSchemaTable& table) noexcept
        {
            uint16_t mask = UNSIGNED_ZERO;
            for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
            {
                const RegionSchemaRecord& schema = table[i];
                if (!HasSchemaFlag(schema.Flags, SchemaFlags::REGION_DISABLED))
                {
                    mask = static_cast<uint16_t>(mask | ADS::RegionBit(static_cast<MacroColumnOfAPC>(i)));
                }
            }
            return mask;
        }

        static constexpr uint32_t RequiredCellsForSchemaTable_(const RegionSchemaTable& table) noexcept
        {
            uint32_t cursor = AlignRegionCells(ADS::META_CELL_COUNT);
            for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
            {
                const RegionSchemaRecord& schema = table[i];
                if (HasSchemaFlag(schema.Flags, SchemaFlags::REGION_DISABLED))
                {
                    continue;
                }

                cursor = AlignRegionCells(cursor);

                if (
                    schema.CellCount == UNSIGNED_ZERO ||
                    cursor > UINT32_MAX - schema.CellCount
                )
                {
                    return UNSIGNED_ZERO;
                }
                
                cursor += schema.CellCount;
            }

            cursor = AlignRegionCells(cursor);
            if (cursor < MINIMUM_APC_CELL_COUNT)
            {
                cursor = AlignRegionCells(MINIMUM_APC_CELL_COUNT);
            }
            
            return ADS::IsCapacityOfAPCValid(cursor) ? cursor : UNSIGNED_ZERO;
            
        }


        static constexpr std::optional<uint32_t> SetRecords_(
            RegionSchemaRecord& schema,
            uint32_t protocol_record_count
        ) noexcept
        {
            uint32_t record_count = 1u;
            switch (schema.Protocol)
            {
            case SchemaProtocols::PRIVATE_REGION:
            case SchemaProtocols::IMMUTABLE_SNAPSHOT:
            case SchemaProtocols::ATOMIC_WORD_ARRAY:
                if (protocol_record_count != 0u && protocol_record_count != 1u)
                {
                    return std::nullopt;
                }
                schema.EnqueuePosition = NO_POSITION;
                schema.DequeuePosition = NO_POSITION;
                break;

            case SchemaProtocols::DOUBLE_BUFFERED:
                if (protocol_record_count != 0u && protocol_record_count != 2u)
                {
                    return std::nullopt;
                }
                record_count = 2u;
                schema.EnqueuePosition = 1u; // write bank ordinal
                schema.DequeuePosition = 0u; // published/read bank ordinal
                break;

            case SchemaProtocols::MPMC_FIXED_RECORD_QUEUE:
                if (
                    protocol_record_count < 2u ||
                    (protocol_record_count & (protocol_record_count - 1u)) != 0u
                )
                {
                    return std::nullopt;
                }
                record_count = protocol_record_count;
                schema.Flags = schema.Flags |
                    SchemaFlags::REQUIRED_POW_OF_TWO |
                    SchemaFlags::HAS_PER_SLOT_SEQUENSE;
                schema.EnqueuePosition = 0u;
                schema.DequeuePosition = 0u;
                break;

            default:
                return std::nullopt;
            }

            return record_count;
        }

        
    public:

        static constexpr bool SealDesiredSchema(
            RegionSchemaRecord& schema,
            uint32_t protocol_record_count = UNSIGNED_ZERO
        ) noexcept
        {   
            const std::optional<uint32_t> stride = RecordStrideCells(schema);
            if (!stride.has_value())
            {
                schema = RegionSchemaRecord{};
                return false;
            }

            std::optional<uint32_t> record_count = SetRecords_(schema, protocol_record_count);

            if (!record_count.has_value())
            {
                schema = RegionSchemaRecord{};
                return false;
            }
            
            const uint64_t total_cells = static_cast<uint64_t>(stride.value()) * record_count.value();

            if (total_cells == UNSIGNED_ZERO || total_cells > UINT32_MAX)
            {
                schema = RegionSchemaRecord{};
                return false;
            }

            schema.CellCount = static_cast<uint32_t>(total_cells);
            return true;
        }


        static constexpr void MakeDisabledSchemaTable(RegionSchemaTable& schema_table) noexcept
        {
            schema_table = RegionSchemaTable{};
            for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
            {
                RegionSchemaRecord& schema = schema_table[i];
                schema.Region = static_cast<MacroColumnOfAPC>(i);
                schema.Flags = SchemaFlags::REGION_DISABLED;
            }
        }

    };
    
}