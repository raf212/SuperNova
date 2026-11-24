#pragma once 

#include "ModelIR.hpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    inline constexpr uint32_t NO_TENSOR = ADS::APC_INDEX_BOUND_SENTINAL;

    enum class TensorRole : uint8_t 
    {
        Parameter = 1,
        Buffer = 2
    };

    struct TensorBindings final
    {
        uint32_t APCIndex{};
        MacroColumnOfAPC Column{};
        uint64_t TensorOffset{};
        uint32_t APCOffset{};
        uint32_t ElementCount{};
    };

    struct TensorDescription final
    {
        using DataType = SchemaDefinition::DataTypeOfMacroColumn;
        std::string Name{};
        TensorRole Role{};
        DataType DType{};
        uint32_t LayerIndex{};
        std::vector<uint32_t> Shape{};
        std::vector<TensorBindings> Bindings{};
    };

    enum class OperatorKind : uint8_t
    {
        Linear = 1,
        ReLU = 2
    };

    struct Operator final
    {
        OperatorKind Kind{};
        uint32_t Input{};
        uint32_t Weight{NO_TENSOR};
        uint32_t Bias{NO_TENSOR};
    };

    struct OperatorGraph final
    {
        std::vector<Operator> Operators{};
        uint32_t Output{};
    };
    
    struct DenseModelDynamics final
    {
        std::vector<TensorDescription> Tensors{};
        std::string TopologyFingerPrint{};
        std::optional<OperatorGraph> Graph{};

        // Configuration bytes in MODEL_DESCRIPTION, versioned little-endian.
        std::vector<std::byte> Encode() const;

        static DenseModelDynamics Decode(std::span<const std::byte> bytes);

        // Throws std::invalid_argument for invalid metadata or GMI assignments.
        void Validate(std::span<const LayerDescription> layers) const;
    };

    struct ModelSnapshot final
    {
        DenseModelDynamics DenseIR{};
        uint64_t StructureRevision{};
        std::vector<std::vector<ModelAPC>> Assignments{};
    };

    inline constexpr uint32_t INACTIVE_LAYER = ADS::APC_INDEX_BOUND_SENTINAL;

    // Element offsets and strides, relative to a private Fabric region.
    struct APCMatrixTile final
    {
        uint32_t Slot{};
        MacroColumnOfAPC Region{};
        uint32_t Offset{};
        uint32_t FirstRow{};
        uint32_t FirstColumn{};
        uint32_t Rows{};
        uint32_t Columns{};
        uint32_t LeadingDimension{};
    };

    struct BlockPlacement final
    {
        uint32_t Layer{};
        bool Initialized{}; // provisioning certifies all selectable connections and bias

        std::vector<uint64_t> InputColumns; // stable IDs in physical column order
        std::vector<APCMatrixTile> Weight;
        std::vector<APCMatrixTile> Bias;     // one column, if the operator has a bias
    };

    struct DenseBlock final
    {
        uint64_t Id{}; // stable IDs in physical row order
        std::vector<uint64_t> Nodes;
        std::vector<ModelAPC> APCs;
        std::vector<BlockPlacement> Placements;
    };

}