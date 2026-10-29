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
        uint16_t ModelCount_{UNSIGNED_ZERO};
        uint32_t ModelDependencyCount_{UNSIGNED_ZERO};
        uint64_t ModelDynamicsByteCount_{UNSIGNED_ZERO};
        uint64_t DynamicRevision_{UNSIGNED_ZERO};
    };

    struct ModelCatalog final
    {
        std::span<const ModelInfo> Models{};
        std::span<const ModelDependency> Dependencies{};
        std::span<const std::byte> Dynamics{};

        constexpr bool ValidInfo(const ModelInfo& mi) const noexcept
        {
            if (
                mi.Model == PreBuildModels::NONE ||
                mi.Model > LAST_DEFINED_MODEL_IDX ||
                mi.ConfByteOffset > Dynamics.size() ||
                mi.ConfByteCount > (Dynamics.size() - mi.ConfByteOffset)
            ) 
            {
                return false;
            }
            return true;
        }

        constexpr bool ValidateDependency(const ModelDependency& md) const noexcept
        {
            if (
                !ADS::ValidateSlotIdx(md.Producer) ||
                !ADS::ValidateSlotIdx(md.Consumer) ||
                md.Producer == Models.size() ||
                md.Consumer == Models.size() ||
                static_cast<uint8_t>(md.ProducerRegion) >= ADS::CountOfMacroColumn() ||
                static_cast<uint8_t>(md.ConsumerRegion) >= ADS::CountOfMacroColumn()
            ) 
            {
                return false;
            }
            return true;
        }

        constexpr bool Validate() const noexcept
        {
            if (Models.size() > ADS::APC_INDEX_BOUND_SENTINAL) 
            {
                return false;
            }

            for (size_t i = 0; i < Models.size(); i++) 
            {
                const ModelInfo& mi = Models[i];
                if (mi.Instance != i + 1 || !ValidInfo(mi)) 
                {
                    return false;
                }
            }

            for (const ModelDependency& md : Dependencies) 
            {
                if (!ValidateDependency(md)) return false;
            }
            return true;
        }
        
        constexpr bool Accept(const SchemaDefinition::RegionSchemaRecord& region) const noexcept
        {
            if (
                region.ModelInstance > Models.size() ||
                region.DependencyBegin > Dependencies.size() ||
                region.DependencyCount > Dependencies.size() - region.DependencyBegin ||
                (!region.ModelInstance  && region.DependencyCount > UNSIGNED_ZERO)
            ) 
            {
                return false;
            }

            for (uint32_t i = 0; i < region.DependencyCount; i++) 
            {
                const ModelDependency& dp = Dependencies[region.DependencyBegin + i];
                if (
                    dp.Consumer != region.ModelInstance ||
                    dp.ConsumerRegion != region.Region
                ) 
                {
                    return false;
                }
            }
            return true;
        }
    };
}