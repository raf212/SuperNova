#pragma once
#include <functional>
#include "../../AdaptivePackedCellContainer/APCOrchestrators/SchemaOrchestratorForRegion.hpp"

namespace BidirectionalInMemGraph
{
    class GHGFLayerModel final
    {
        friend class GHGFModelConstructor;
        friend class GHGFNode;
    public:
        using SD = SchemaDefinition;
        static constexpr uint32_t DEFAULT_BATCH_CAPACITY = 32u;

        enum class GHGFNodeRole : uint8_t
        {
            OBSERVATION = 1,
            VALUE = 2,
            VOLATILE = 3
        };

        enum class GHGFStateRow : uint8_t
        {
            MEAN = 0,
            EXPECTED_MEAN = 1,
            PRECISION = 2,
            EXPECTED_PRECISION = 3,
            CONDITIONAL_EXPECTED_PRECISION = 4,
            OBSERVED = 5,
            CURRENT_VARIANCE = 6,
            EFFECTIVE_PRECISION = 7
        };
        static constexpr uint8_t STATE_ROW_COUNT_HEIGHT = static_cast<uint8_t>(GHGFStateRow::EFFECTIVE_PRECISION) + 1;

        enum class GHGFErrorRow : uint8_t
        {
            VALUE_PREDICTION_ERROR = 0,
            VOLATILE_PREDICTION_ERROR = 1
        };
        static constexpr uint8_t ERROR_ROW_COUNT_HEIGHT = static_cast<uint8_t>(GHGFErrorRow::VOLATILE_PREDICTION_ERROR) + 1;

        enum class GHGFErrorValueIndexing : uint8_t
        {
            TONIC_VOLATILE = 0,
            TONIC_DRIFT = 1,
            AUTO_CONNECTION = 2
        };
        static constexpr uint8_t FIRST_COUPLING_INDEX = static_cast<uint8_t>(GHGFErrorValueIndexing::AUTO_CONNECTION) + 1;

        static constexpr uint8_t WEIGHT_ROW_HEIGHT = 1u;

        struct GHGFStorageProfile final
        {
            uint32_t BatchCapacity = UNSIGNED_ZERO;
            uint32_t RequiredAPCCells = UNSIGNED_ZERO;
            uint32_t ParameterCount = UNSIGNED_ZERO;
            uint16_t ActiveRegionMask = UNSIGNED_ZERO;
            uint8_t MaxDirectParentPerAxis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS;

            SD::FabricRegionConfig FabricConfig{};
            SD::RegionSchemaTable DefaultSchemaTable{};
            bool IsValid = false;
        };

        struct GHGFConnection
        {
            uint32_t Parent;
            uint32_t Child;
            FabricSegments Edge;
            float Coupling = 1.0f;
        };

        struct GHGFParameterRange
        {
            uint32_t Slot;
            uint32_t Index;
            float Lower;
            float Upper;
        };

        enum class GHGFPhase : uint8_t
        {
            NEEDS_RESET = 0,
            READY = 1,
            PREDICTED = 2
        };


        static constexpr uint32_t CouplingIndex(
            FabricSegments edge_table,
            uint8_t relation_ordinal,
            uint8_t max_direct_parent_per_axis
        ) noexcept
        {
            if (
                relation_ordinal >= max_direct_parent_per_axis ||
                (edge_table != FabricSegments::VALUE_PARENT_EDGE_TABLE_H &&
                 edge_table != FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V)
            )
            {
                return ADS::APC_INDEX_BOUND_SENTINAL;
            }

            const uint32_t axis_offset =
                edge_table == FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
                    ? static_cast<uint32_t>(max_direct_parent_per_axis)
                    : UNSIGNED_ZERO;

            return FIRST_COUPLING_INDEX + axis_offset + relation_ordinal;
        }

        static constexpr bool MakeDefaultGHGFStorageProfile(
            GHGFStorageProfile& profile,
            uint32_t batch_capacity = DEFAULT_BATCH_CAPACITY,
            uint8_t max_direct_parent_per_axis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS
        ) noexcept
        {
            profile = GHGFStorageProfile{};
            SD::MakeDisabledSchemaTable(profile.DefaultSchemaTable);

            if (
                batch_capacity == UNSIGNED_ZERO ||
                max_direct_parent_per_axis == UNSIGNED_ZERO ||
                max_direct_parent_per_axis > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS
            )
            {
                return false;
            }

            const uint32_t parameter_count = FIRST_COUPLING_INDEX + (EDGE_COUNT * static_cast<uint32_t>(max_direct_parent_per_axis));

            if (!SD::AttachPrivateFloat32ToTable_(
                profile.DefaultSchemaTable,
                MacroColumnOfAPC::STATE_SLOT,
                STATE_ROW_COUNT_HEIGHT,
                batch_capacity,
                SD::SchemaFlags::BATCHED_LAST_DIM
            ))
            {
                return false;
            }
            
            if (!SD::AttachPrivateFloat32ToTable_(
                profile.DefaultSchemaTable,
                MacroColumnOfAPC::ERROR_SLOT,
                ERROR_ROW_COUNT_HEIGHT,
                batch_capacity,
                SD::SchemaFlags::BATCHED_LAST_DIM
            ))
            {
                return false;
            }

            if (!SD::AttachPrivateFloat32ToTable_(
                profile.DefaultSchemaTable,
                MacroColumnOfAPC::WEIGHT_SLOT,
                WEIGHT_ROW_HEIGHT,
                parameter_count,
                SD::SchemaFlags::NONE
            ))
            {
                return false;
            }

            profile.BatchCapacity = batch_capacity;
            profile.MaxDirectParentPerAxis = max_direct_parent_per_axis;
            profile.ParameterCount = parameter_count;
            profile.ActiveRegionMask = SD::GetActiveMaskOfRegionTable_(profile.DefaultSchemaTable);
            profile.RequiredAPCCells = SD::RequiredCellsForSchemaTable_(profile.DefaultSchemaTable);

            if (
                profile.ActiveRegionMask == UNSIGNED_ZERO ||
                profile.RequiredAPCCells == UNSIGNED_ZERO
            )
            {
                return false;
            }
            
            profile.FabricConfig.ActiveRegionMask = profile.ActiveRegionMask;
            profile.FabricConfig.Reserved = UNSIGNED_ZERO;
            profile.FabricConfig.BatchCapacity = profile.BatchCapacity;

            profile.IsValid = true;

            return true;
        }

        static constexpr bool IsValidStoregeProfile(const GHGFStorageProfile& profile) noexcept
        {
            const uint32_t expected_paremeter_count = FIRST_COUPLING_INDEX + (EDGE_COUNT * static_cast<uint32_t>(profile.MaxDirectParentPerAxis));

            const uint16_t expected_mask = static_cast<uint16_t>(
                ADS::RegionBit(MacroColumnOfAPC::STATE_SLOT) |
                ADS::RegionBit(MacroColumnOfAPC::ERROR_SLOT) |
                ADS::RegionBit(MacroColumnOfAPC::WEIGHT_SLOT)
            );        

            const SD::RegionSchemaRecord& state = profile.DefaultSchemaTable[static_cast<uint8_t>(MacroColumnOfAPC::STATE_SLOT)];
            const SD::RegionSchemaRecord& error = profile.DefaultSchemaTable[static_cast<uint8_t>(MacroColumnOfAPC::ERROR_SLOT)];
            const SD::RegionSchemaRecord& weight = profile.DefaultSchemaTable[static_cast<uint8_t>(MacroColumnOfAPC::WEIGHT_SLOT)];

            return
                profile.IsValid &&
                profile.BatchCapacity != UNSIGNED_ZERO &&
                profile.MaxDirectParentPerAxis != UNSIGNED_ZERO &&
                profile.MaxDirectParentPerAxis <= ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS &&
                profile.ParameterCount == expected_paremeter_count &&
                profile.ActiveRegionMask == expected_mask &&
                profile.FabricConfig.ActiveRegionMask == expected_mask &&
                profile.FabricConfig.Reserved == UNSIGNED_ZERO &&
                profile.FabricConfig.BatchCapacity == profile.BatchCapacity &&
                ADS::IsCapacityOfAPCValid(profile.RequiredAPCCells) &&
                SD::GetActiveMaskOfRegionTable_(profile.DefaultSchemaTable) == expected_mask &&
                SD::RequiredCellsForSchemaTable_(profile.DefaultSchemaTable) == profile.RequiredAPCCells &&
                state.Region == MacroColumnOfAPC::STATE_SLOT &&
                state.Dtype == SD::DataTypeOfMacroColumn::FLOAT32_T &&
                state.Protocol == SD::SchemaProtocols::PRIVATE_REGION &&
                state.MatrixHeight == STATE_ROW_COUNT_HEIGHT &&
                state.MatrixWidth == profile.BatchCapacity &&
                state.Flags == SD::SchemaFlags::BATCHED_LAST_DIM &&
                error.Region == MacroColumnOfAPC::ERROR_SLOT &&
                error.Dtype == SD::DataTypeOfMacroColumn::FLOAT32_T &&
                error.Protocol == SD::SchemaProtocols::PRIVATE_REGION &&
                error.MatrixHeight == ERROR_ROW_COUNT_HEIGHT &&
                error.MatrixWidth == profile.BatchCapacity &&
                error.Flags == SD::SchemaFlags::BATCHED_LAST_DIM &&
                weight.Region == MacroColumnOfAPC::WEIGHT_SLOT &&
                weight.Dtype == SD::DataTypeOfMacroColumn::FLOAT32_T &&
                weight.Protocol == SD::SchemaProtocols::PRIVATE_REGION &&
                weight.MatrixHeight == WEIGHT_ROW_HEIGHT &&
                weight.MatrixWidth == profile.ParameterCount &&
                weight.Flags == SD::SchemaFlags::NONE;
        }


    private:
        
        struct GHGFCache final
        {
            uint32_t StateCellOffset_ = UNSIGNED_ZERO;
            uint32_t ErrorCellOffset_ = UNSIGNED_ZERO;
            uint32_t WeightCellOffset_ = UNSIGNED_ZERO;
            uint32_t NodeCount_ = UNSIGNED_ZERO;
            uint32_t ObservationCount_ = UNSIGNED_ZERO;
            uint32_t ActiveBatch_ = UNSIGNED_ZERO;
            uint64_t PreparedRevision_ = UNSIGNED_ZERO;
            bool ModelPrepared_ = false;
            GHGFLayerModel::GHGFPhase Phase_ = GHGFLayerModel::GHGFPhase::NEEDS_RESET;
        };

        struct StorageConst
        {
            // Initial storage contents.
            static constexpr float INITIAL_STORAGE_VALUE = 0.0f;

            // Initial beliefs and prediction state.
            static constexpr float INITIAL_BINARY_PROBABILITY = 0.5f;
            static constexpr float INITIAL_PRECISION = 1.0f;
            // Baseline dynamics; tonic volatility is stored on a logarithmic scale.
            static constexpr float OBSERVATION_TONIC_LOG_VOLATILITY = 0.0f;
            static constexpr float INITIAL_TONIC_LOG_VOLATILITY = -4.0f;
            static constexpr float INITIAL_AUTO_CONNECTION = 1.0f;
            // Initial strengths for connected H and V parents.
            static constexpr float INITIAL_VALUE_COUPLING = 1.0f;
            static constexpr float INITIAL_VOLATILITY_COUPLING = 1.0f;

            static constexpr float ZERO = 0.0f;
            static constexpr float ONE = 1.0f;
            static constexpr float HALF = 0.5f;
            static constexpr float BINARY_CLIP = 1.0e-6f;
            static constexpr float MIN_PRECISION = 1.0e-8f;
            static constexpr float MAX_PRECISION = 1.0e10f;
            static constexpr float INITIAL_SEARCH_STEP = 1.0f;
            static constexpr float MIN_SEARCH_STEP = 1.0e-3f;
            static constexpr uint32_t INVALID_SLOT = ADS::APC_INDEX_BOUND_SENTINAL;
        };

    };
    
}