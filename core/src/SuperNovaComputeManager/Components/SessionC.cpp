#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{
    using CFC = CoreOfFabricCoordinator;

    FabricExecutionGate& SessionCompute::NumericGate_() noexcept
    {
        return Binding_.Gate ? *Binding_.Gate : UnboundGate_;
    }

    bool SessionCompute::OwnsEpoch_() noexcept
    {
        return NumericGate_().EpochOwnedByCaller();
    }

    const uint64_t* SessionCompute::NumericalBase_() noexcept
    {
        return Binding_.NumericalCells().data();
    }

    void SessionCompute::SetError_(ErrorC error) noexcept
    {
        LastError_.store(error, std::memory_order_relaxed);
    }

    ReadLease SessionCompute::AdmitNumeric_() noexcept
    {
        if (!Binding_ || !EnsureConfigured_()) return {};
        if (Session_ && !Residency_.Active()) 
        {
            SetError_(ErrorC::INVALID_CONF);
            return {};
        }

        ReadLease admission = NumericGate_().TryTransientRead(true);
        if (!admission) SetError_(ErrorC::INVALID_CONF);
        return admission;
    }

    bool SessionCompute::FallbackAllowed_(const ExecutionPolicyC& policy) const noexcept
    {
        return Scheduler::FallbackAllowed(Config_, policy);
    }

    bool SessionCompute::EnsureConfigured_() noexcept
    {
        if (Configured_) return true;
        return Configure(Config_);
    }

    BackendC SessionCompute::Resolve_(size_t work_units, const ExecutionPolicyC& policy) noexcept
    {
        if (Residency_.Active()) return Residency_.Backend();
        return Scheduler::Resolve(work_units, Config_, policy, Device_);
    }

    bool SessionCompute::ValidateResolvedBackend_(BackendC backend, const ExecutionPolicyC& policy) noexcept
    {
        if (backend == BackendC::CPU) return true;

        if (backend == BackendC::GPU && Device_.NumericReady()) 
        {
            return true;
        }

        SetError_(ErrorC::ACCELERATOR_UNAVAILAVLE);
        (void) policy;
        return false;
    }

    bool SessionCompute::BindFabric(APCFinilizer& owner, FabricId id) noexcept
    {
        ReadLease admission = owner.ExecutionGate().TryRead();
        if (!admission || !owner.IsFabricActive()) 
        {
            return false;
        }
        std::span<uint64_t> hot_babe = owner.HotStorageDramUsm();
        std::span<uint64_t> probable_segmentpool = owner.NumericalStorage();

        if (!hot_babe.empty() || probable_segmentpool.empty()) return false;

        try 
        {
            if (!id) 
            {
                id = Binding_.Owner == &owner ?
                    Binding_.Id : FabricId::New();
            }
        } catch (...) 
        {
            return false;
        }

        return BindFabric({
            &owner, hot_babe, &owner.ExecutionGate(),
            id, probable_segmentpool, hot_babe.size()
        });
    }

    bool SessionCompute::BindFabric(FabricRuntimeBinding binding) noexcept
    {
        if (!binding || Residency_.Active() || Session_) 
        {
            return false;
        }

        ReadLease admission = binding.Gate->TryRead(true);
        if (!admission) return false;
        if (binding.Owner) 
        {
            APCFinilizer& owner = *binding.Owner;
            if (
                binding.Gate != &owner.ExecutionGate() ||
                !owner.IsFabricActive()
            ) 
            {
                return false;
            }

            const std::span<uint64_t> hot_fabric = owner.HotStorageDramUsm();
            const std::span<uint64_t> probable_segmentpool = owner.NumericalStorage();
            const std::span<uint64_t> contigious = owner.ContiguousStorage();

            if (
                binding.Cells.data() != hot_fabric.data() ||
                (
                    binding.Cells.size() != hot_fabric.size() &&
                    binding.Cells.size() != contigious.size()
                )
            ) 
            {
                return false;
            }

            CFC::FabricCache fab_cache{};
            if (binding.Cells.size_bytes() < sizeof(fab_cache)) return false;
            std::memcpy(&fab_cache, binding.Cells.data(), sizeof(fab_cache));
            if (
                !StorageAllocationDef::ValidateFabricCache(fab_cache) ||
                binding.Cells.size() < fab_cache.SegmentPoolBegin_
            )
            {
                return false;
            }

            if (binding.Numerics.empty()) 
            {
                if (binding.Cells.size() != fab_cache.SlabCellCount_) 
                {
                    return false;
                }
                binding.Numerics = binding.Cells.subspan(fab_cache.SegmentPoolBegin_);
                binding.NumericalBegainCell = fab_cache.SegmentPoolBegin_;
            }

            if (
                binding.Numerics.data() != probable_segmentpool.data() ||
                binding.Numerics.size() != probable_segmentpool.size() ||
                binding.NumericalBegainCell != fab_cache.SegmentPoolBegin_ ||
                binding.Numerics.size() != (fab_cache.SlabCellCount_ - fab_cache.SegmentPoolBegin_)
            ) 
            {
                return false;
            }
        }

        if (
            Binding_ != binding &&
            !Residency_.Evict(Device_)
        ) 
        {
            return false;
        }

        Binding_ = binding;
        return true;
    }

    bool SessionCompute::BindNumericStorage(
        std::span<uint64_t> cells, 
        FabricExecutionGate& gate, 
        FabricId id
    ) noexcept
    {
        try 
        {
            if (!id)
            {
                if (Binding_.Gate == &gate && Binding_.Cells.data()) 
                {
                    id = Binding_.Id;
                }
                else
                {
                    id = FabricId::New();
                }
            }
        }
        catch (...)
        {
            return false;
        }
        return BindFabric({
            nullptr, cells, &gate, id
        });
    }

}