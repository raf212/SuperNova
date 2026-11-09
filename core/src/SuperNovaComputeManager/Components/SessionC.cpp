#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{
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
}