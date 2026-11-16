#include "ModelCatalog.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
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

}