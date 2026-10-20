#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

namespace BidirectionalInMemGraph
{
    void ReadLease::Reset() noexcept
    {
        if (Gate_ && Counted_) 
        {
            if (FabricExecutionGate::Owner_ != Gate_)
            {
                Gate_->HotStamp_.fetch_add(1, std::memory_order_release);
            }
            Gate_->State_.fetch_sub(1, std::memory_order_release);
        }
        Gate_ = nullptr;
    }
    
    void ExclusiveLease::Reset() noexcept
    {
        if (Gate_ && Counted_) 
        {
            if (FabricExecutionGate::Owner_ != Gate_)
            {
                std::terminate();
            }
            FabricExecutionGate::Maintenance_ = false;
            FabricExecutionGate::Owner_ = nullptr;
            Gate_->State_.fetch_and(~FabricExecutionGate::WRITER, std::memory_order_release);
        }
        Gate_ = nullptr;
    }

    bool FabricExecutionGate::ConfirmedRestored() noexcept
    {
        if (Owner_ != this || !Maintenance_) 
        {
            return false;
        }
        Poisoned_.store(false, std::memory_order_release);
        return true;
    }

    ReadLease FabricExecutionGate::TryRead(bool allow_epoch_pin) noexcept
    {
        if (IsPoisoned()) 
        {
            return {};
        }
        if (Owner_ == this) 
        {
            if (!Maintenance_ && !allow_epoch_pin) 
            {
                return {};
            }
            uint64_t previous = State_.fetch_add(1, std::memory_order_relaxed);
            if ((previous & ~WRITER) == WRITER - 1) 
            {
                std::terminate();
            }
            return ReadLease(this, true);
        }
        uint64_t state = State_.load(std::memory_order_acquire);
        while (state < WRITER - 1) 
        {
            if (State_.compare_exchange_strong(state, state + 1, std::memory_order_acq_rel)) 
            {
                return ReadLease(this, true);
            }
        }
        return {}; // Gate is writer-held or saturated; never fall through.
    }

    ExclusiveLease FabricExecutionGate::TryEpoch() noexcept
    {
        if (Owner_ || IsPoisoned()) 
        {
            return {};    
        }
        uint64_t empty = 0u;
        if (!State_.compare_exchange_strong(empty, WRITER, std::memory_order_acq_rel)) 
        {
            return {};
        }
        Owner_ = this;
        Maintenance_ = false;
        return ExclusiveLease(this, true);
    }

    ExclusiveLease FabricExecutionGate::TryMaintenance() noexcept
    {
        if (Owner_ == this && Maintenance_) 
        {
            return ExclusiveLease(this, false);
        }
        if (Owner_) 
        {
            return {};
        }
        uint64_t empty = 0u;
        if (!State_.compare_exchange_strong(empty, WRITER, std::memory_order_acq_rel)) 
        {
            return {};
        }
        Owner_ = this;
        Maintenance_ = true;
        return ExclusiveLease(this, true);
    }
}
