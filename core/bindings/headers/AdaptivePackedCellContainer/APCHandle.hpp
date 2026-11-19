#pragma once

#include <SuperNova>

namespace SuperNovaBindings
{
    class APCHandle final : public BidirectionalInMemGraph::AdaptivePackedCellContainer
    {
        // Native APC handles borrow their fabric's slab.
        std::shared_ptr<BidirectionalInMemGraph::APCFinilizer> Fabric_;

    public:
        explicit APCHandle(std::shared_ptr<BidirectionalInMemGraph::APCFinilizer> fabric,
                           BidirectionalInMemGraph::AdaptivePackedCellContainer apc = {})
            : BidirectionalInMemGraph::AdaptivePackedCellContainer(std::move(apc)), Fabric_(std::move(fabric))
        {
        }

        const std::shared_ptr<BidirectionalInMemGraph::APCFinilizer>& Fabric() const noexcept { return Fabric_; }
    };
}
