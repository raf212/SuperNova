#include "ModelCatalog.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    // A single model's layers and APC arrays. Local layer index is independent
    // of both the global layer directory and the physical APC slot number.
    struct ViewModelIndividually final
    {
        const IndividualModelDescription* Description{};
        std::span<const LayerDescription> Layers{};
        std::span<const ModelAPC> AssignmentDirectory{};

        explicit operator bool () const noexcept
        {
            return Description != nullptr;
        }

        LayerAPCArray LayerAPCs(uint32_t local_layer) const noexcept;
    };


    // A non-owning canonical IR view. Constructed from FabricInitConf on write,
    // reconstructed directly from MODEL_DESCRIPTION on read/attach.
    struct ModelIR final
    {
        std::span<const IndividualModelDescription>  Description{};
        std::span<const ModelDependency> Dependencies{};
        std::span<const LayerDescription> Layers{};
        std::span<const ModelAPC> LayerAPCs{};
        std::span<const std::byte> Dynamics{};

        ViewModelIndividually ModelView(uint32_t instance) const noexcept;

        std::span<const ModelDependency> IncomingDependencies(uint32_t instance) const noexcept;

        std::span<const std::byte> Configuration(uint32_t instance) const noexcept;

        bool valid(uint32_t slot_count) const noexcept;

        const ModelAPC* APCAt(uint32_t layer_index, uint32_t position) const noexcept;

        const IndividualModelDescription* ModelForRecord(size_t record_index) const noexcept;

        constexpr bool Accept(const SchemaDefinition::RegionSchemaRecord& region) const noexcept
        {
            if (
                region.ModelInstance > Dependencies.size() ||
                region.DependencyBegin > Dependencies.size() ||
                region.DependencyCount > (Dependencies.size() - region.DependencyBegin) ||
                (
                    region.ModelInstance == UNSIGNED_ZERO &&
                    region.DependencyCount != UNSIGNED_ZERO
                )
            ) 
            {
                return false;
            }

            if (region.DependencyCount == UNSIGNED_ZERO) return true;

            const IndividualModelDescription& dsc = Description[region.ModelInstance - 1];
            if (
                region.DependencyBegin < dsc.FirstDependency ||
                (uint64_t{region.DependencyBegin} + region.DependencyCount) >
                    (uint64_t{dsc.FirstDependency} + dsc.DependencyCount)
            ) 
            {
                return false;
            }

            for (uint32_t i = 0; i < region.DependencyCount; i++) 
            {
                const ModelDependency& dep = Dependencies[region.DependencyBegin + i];
                if (
                    dep.Consumer != region.ModelInstance ||
                    dep.ConsumerRegion != region.Region
                ) 
                {
                    return false;
                }
            }
            return true;
        }

    };
}