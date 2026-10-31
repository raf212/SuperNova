#include <SuperNova>

namespace BidirectionalInMemGraph
{
    CoreOfFabricCoordinator::DetachFabric RelocatableSlabStorage::DetachFabric() noexcept
    {
        if (ExecutionGate().IsPoisoned()) return {};

        auto phase = ExecutionGate().TryMaintenance();
        if (!phase) return {};

        CFC::DetachFabric detached{};
        if (
            !IsFabricActive() ||
            !FabCache_ ||
            !QuiesceFabric_()
        )
        {
            return detached;
        }

        detached.Slab_ = SlabBasePtr_;
        detached.CellCount_ = FabCache_->SlabCellCount_;
        detached.Ownership_ = BackingOwnership_;

        SlabBasePtr_ = nullptr;
        FabCache_ = nullptr;
        BackingOwnership_ = CFC::FabricBackigOwnership::NONE;
        FabricInitialized_.store(false, std::memory_order_release);
        SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        return detached;
    }

    bool RelocatableSlabStorage::SaveFabric(std::span<uint64_t> destination) noexcept
    {
        if (ExecutionGate().IsPoisoned()) return false;

        auto phase = ExecutionGate().TryMaintenance();
        if (!phase) return false;

        if (
            !IsFabricActive() ||
            !FabCache_ ||
            destination.data() == nullptr ||
            destination.size() < FabCache_->SlabCellCount_ ||
            IsInternalBuffer(destination.data(), destination.size())
        )
        {
            return false;
        }
        
        if (!QuiesceFabric_())
        {
            return false;
        }

        std::memcpy(
            destination.data(),
            SlabBasePtr_,
            FabCache_->SlabCellCount_ * sizeof(uint64_t)
        );

        bool reopened = ReopenLiveAPCGenerations_();

        FabricInitialized_.store(reopened, std::memory_order_release);

        return reopened;
    }
}