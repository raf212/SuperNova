#pragma once 
#include "../AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    struct alignas(uint64_t) ModelInfo final
    {
        uint32_t Instance{};
        PreBuildModels Model{PreBuildModels::NONE};
        uint64_t ConfByteOffset{};
        uint64_t ConfByteCount{};
    private:
        uint8_t Reserved[sizeof(uint64_t) - sizeof(PreBuildModels)] {};
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

    struct alignas(uint64_t) NodeModelDescription final
    {
        uint32_t Instance{};
        uint32_t Role{};
    };

    struct alignas(uint64_t) ModelStorageDiaphragm final
    {
        uint64_t GlobalTableBeginIdx_{UNSIGNED_ZERO};
        uint64_t LocalDirectoryBeginIdx_{UNSIGNED_ZERO};
        uint64_t DirectoryCellCount_{UNSIGNED_ZERO};
        uint16_t ModelCount_{ModelDependency::INVALID_MODEL_IDX};
        uint32_t ModelDependencyCount_{UNSIGNED_ZERO};
        uint64_t ModelDynamicsByteCount_{UNSIGNED_ZERO};
    };

    struct ModelCatalog final
    {
        std::span<const ModelInfo> Models{};
        std::span<const ModelDependency> Dependencies{};
        std::span<const std::byte> Dynamics{};

        constexpr bool ValidInfo(const ModelInfo& mi) const noexcept;

        constexpr bool ValidateDependency(const ModelDependency& md) const noexcept;

        constexpr bool Validate() const noexcept;

        constexpr bool Accept(const SchemaDefinition::RegionSchemaRecord& region) noexcept;
    };
}