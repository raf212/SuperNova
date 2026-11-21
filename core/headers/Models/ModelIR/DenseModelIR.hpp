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
}