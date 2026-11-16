#include "ModelCatalog.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    // A single model's layers and APC arrays. Local layer index is independent
    // of both the global layer directory and the physical APC slot number.
    struct IndividualModelView final
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

        // // Model instance IDs are the existing 1-based ModelCatalog IDs.
        // // The returned view exposes one separate ordered APC array per layer.
        // IndividualModelView

    };
}