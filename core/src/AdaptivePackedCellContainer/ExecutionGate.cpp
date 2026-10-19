#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

namespace BidirectionalInMemGraph
{
    void ReadLeas::Reset(FabricExecutionGate* owner) noexcept
    {
        if (Gate_ && Counted_) 
        {
            if (owner != Gate_) 
            {
                Gate_->HotStamp_.fetch_add(1, std::memory_order_release);
            }
            Gate_->State_.fetch_add(1, std::memory_order_release);
        }
        Gate_ = nullptr;
    }

    void ExclusiveLease::Reset(FabricExecutionGate* owner) noexcept
    {
        if (Gate_ && Counted_) 
        {
            if (owner != Gate_) 
            {
                std::terminate();
            }
            owner->Owner_ = nullptr;
            owner->Maintenance_ = false;
            Gate_->State_.fetch_add(~FabricExecutionGate::WRITER, std::memory_order_release);
        }
        Gate_ = nullptr;
    }
}