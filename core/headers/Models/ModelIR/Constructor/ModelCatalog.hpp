#pragma once 
#include "../../../AdaptivePackedCellContainer/APCOrchestrators/SchemaOrchestratorForRegion.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    enum class ModelRepresentation : uint8_t
    {
        NONE = 0,
        GHGF_GRAPH = 2,
        DENSE_TENSOR_STATE = 3
    };
    static constexpr ModelRepresentation LAST_DEFINED_MODEL_IDX = ModelRepresentation::DENSE_TENSOR_STATE;

    enum class LearningSystemKind
    {
        NONE = 0,
        BACKPROPAGATION = 1,
        GHGF_LOCAL = 2
    };
    
    struct alignas(uint64_t) ModelDependency final
    {
        static constexpr uint16_t INVALID_MODEL_IDX = UINT16_MAX;

        uint16_t Producer{INVALID_MODEL_IDX};
        uint16_t Consumer{INVALID_MODEL_IDX};
        MacroColumnOfAPC ProducerRegion{};
        MacroColumnOfAPC ConsumerRegion{};
        uint32_t Flags{};
    };

    // LayerAPCs[local_index] is authoritative construction/reconstruction order.
    // Slot is the physical APC slot; it need not equal local_index.
    struct alignas(uint64_t) ModelAPC final
    {
        uint32_t Slot{ADS::APC_INDEX_BOUND_SENTINAL};
        uint32_t Generation{};  // Zero until the APC is constructed.
        uint32_t Role{}; // Interpreted by the owning model family.
        uint32_t Reserved{};
        bool operator==(const ModelAPC& other) const noexcept
        {
            return Slot == other.Slot &&
                Generation == other.Generation &&
                Role == other.Role &&
                Reserved == other.Reserved;
        }
    };

    struct alignas(uint64_t) LayerDescription final
    {
        uint32_t OperatorId{};       // Model-specific operation descriptor ID.
        uint32_t FirstAPCRecord{};  // Index into ModelIR::LayerAPCs.
        uint32_t APCCount{};        // Ordered active assignments.
        uint32_t APCCapacity{};     // Includes reserved, unassigned records.
    };

    struct alignas(64) IndividualModelDescription
    {
        uint32_t Instance{};
        ModelRepresentation Model{ModelRepresentation::NONE};
        uint8_t StructuralADaptationAllowed{static_cast<uint8_t>(false)};
        uint16_t Reserved{};

        uint64_t ConfByteOffset{};
        uint64_t ConfByteCount{};

        uint32_t FirstLayer{}; // Index into ModelIR::Layers.
        uint32_t LayerCount{};
        uint32_t FirstDependency{};
        uint32_t DependencyCount{};

        uint64_t StructureRevision{1};

        uint64_t Unused_[2]{};
    };

    struct alignas(uint64_t) ModelStorageDiaphragm final
    {
        uint64_t LocalDirectoryBeginIdx_{UNSIGNED_ZERO};
        uint64_t DirectoryCellCount_{UNSIGNED_ZERO};
        uint16_t ModelCount_{UNSIGNED_ZERO};
        uint32_t ModelDependencyCount_{UNSIGNED_ZERO};
        uint32_t LayerCount_{UNSIGNED_ZERO};
        uint32_t APCRecordCount_{UNSIGNED_ZERO};
        uint64_t ModelDynamicsByteCount_{UNSIGNED_ZERO};
        uint64_t DynamicsRevision_{1};
    };

    // Each layer owns a different, capacity-bounded subarray of the serialized
    // ModelAPC directory. These are views, not persistent pointers.
    struct LayerAPCArray final
    {
        std::span<const ModelAPC> Active{};
        std::span<const ModelAPC> Storage{};
        bool Present{false};

        const ModelAPC* At(size_t position) const noexcept
        {
            return position < Active.size() ? &Active[position] : nullptr;
        }
    };

    struct LayerBuildDescription final
    {
        uint32_t OperatorId{};
        uint32_t MaxAllowedAPCs{};
        std::vector<ModelAPC> APCs{};
    };

    struct BuildModelDescriptionIndivisually final
    {
        uint32_t Instance{};
        ModelRepresentation Model{ModelRepresentation::NONE};
        bool StructuralADaptationAllowed{false};
        uint64_t StructureRevision{1};
        std::vector<std::byte> Configuration{};
        std::vector<ModelDependency> Dependencies{}; // Incoming dependencies, Consumer==Instance.
        std::vector<LayerBuildDescription> Layers{};
    };

}