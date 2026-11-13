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
        if (!Binding_ || !EnsureConfigured()) return {};
        if (Session_ && !Residency_.Active()) 
        {
            SetError_(ErrorC::INVALID_CONF);
            return {};
        }

        ReadLease admission = NumericGate_().TryTransientRead(true);
        if (!admission) SetError_(ErrorC::INVALID_CONF);
        return admission;
    }

    bool SessionCompute::CpuFallbackAllowed_(const ExecutionPolicyC& policy) const noexcept
    {
        return Scheduler::FallbackAllowed(Config_, policy);
    }

    bool SessionCompute::EnsureConfigured() noexcept
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

    bool SessionCompute::ReleseNumericalStorage() noexcept
    {
        if (!Close()) return false;
        Binding_ = {};
        return true;
    }

    bool SessionCompute::SetRegisteredRegionBudget(uint64_t bytes) noexcept
    {
        ExclusiveLease admission = NumericGate_().TryMaintenance();
        return admission && Residency_.SetCacheBudget(bytes);
    }

    bool SessionCompute::ConfirmRecovery() noexcept
    {
        if (Residency_.Active() || Session_) return false;
        ExclusiveLease restore = NumericGate_().TryMaintenance();
        return restore && NumericGate_().ConfirmedRestored();
    }

    bool SessionCompute::Synchronize() noexcept
    {
        if (!Device_.Synchronize()) 
        {
            SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
            return false;
        }
        return true;
    }

    bool SessionCompute::BeginSession(size_t work, ExecutionPolicyC policy, ExclusiveLease lease) noexcept
    {
        if (Session_ || Residency_.Active()) return false;
        if (!lease) lease = NumericGate_().TryEpoch();
        if (
            !lease || !lease.Owns(NumericGate_()) ||
            !Binding_ || !EnsureConfigured()
        ) 
        {
            return false;
        }
        SessionBackend_ = Resolve_(work, policy);
        if (!ValidateResolvedBackend_(SessionBackend_, policy)) return false;
        EpochLease_ = std::move(lease);
        Session_ = true;
        return true;
    }

    bool SessionCompute::EndSession() noexcept
    {
        if (!Session_) return EndEpoch();
        if (!OwnsEpoch_() || EpochActive()) return false;
        if (Residency_.UnpublishedWrites()) 
        {
            NumericGate_().NotifyHostChange();
        }
        if (!Residency_.PublishResidency(Device_, NumericGate_().HostStamp())) 
        {
            NumericGate_().Poison();
            return false;
        }
        Session_ = false;
        EpochLease_.Reset();
        return true;
    }

    bool SessionCompute::BeginEpoch(size_t expected_work, ExecutionPolicyC policy, ExclusiveLease lease) noexcept
    {
        if (!lease || !Session_) 
        {
            lease = NumericGate_().TryEpoch();
        }
        
        if (lease && !lease.Owns(NumericGate_())) return false;
        if (
            (!lease && !Session_) ||
            (Session_ && !OwnsEpoch_()) ||
            Residency_.Active() || !NumericalBase_() || !EnsureConfigured()
        ) 
        {
            return false;
        }

        const BackendC backend = Session_ ? SessionBackend_ : Resolve_(expected_work, policy);
        if (!ValidateResolvedBackend_(backend, policy)) return false;
        if (!Session_) EpochLease_ = std::move(lease);
        return true;
    }

    bool SessionCompute::EndEpoch(uint64_t* published_stamp) noexcept
    {
        if (
            (Residency_.Active() || Session_) &&
            OwnsEpoch_()
        ) 
        {   
            return false;
        }

        if (!Session_ && Residency_.Active() && Residency_.UnpublishedWrites()) 
        {
            NumericGate_().NotifyHostChange();
        }

        if (!Residency_.Finish(Device_, !Session_, NumericGate_().HostStamp())) 
        {
            NumericGate_().Poison();
            SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
            return false;
        }

        if (published_stamp) *published_stamp = NumericGate_().HostStamp();
        if (!Session_) EpochLease_.Reset();
        ClearError_();
        return true;
    }

    bool SessionCompute::DiscardEpoch() noexcept
    {
        if ((Residency_.Active() || Session_) && !OwnsEpoch_()) return false;
        if (
            (Residency_.Active() || Session_) &&
            (Residency_.Dirty() || Residency_.Faild())
        ) 
        {
            NumericGate_().Poison();
        }
        if (!Residency_.Discard(Device_)) return false;
        Session_ = false;
        EpochLease_.Reset();
        return true;
    }

    bool SessionCompute::Configure(const ConfigC& config) noexcept
    {
        if (!Residency_.Active() || Session_ || !Residency_.Empty()) 
        {
            SetError_(ErrorC::CONFIGIRED_AFTER_FABRIC_INIT);
            return false;
        }

        if (
            !CDV::ValidBackend(config.DefaultBackend) ||
            config.GPUWorkThreshold == UNSIGNED_ZERO ||
            config.GPUWorkThreshold > SIZE_MAX
        ) 
        {
            SetError_(ErrorC::INVALID_CONF);
            return false;
        }

        Config_ = config;
        Configured_ = false;
        (void)Device_.ConfigureGPU();
        if (
            !config.AllowCPUFallback &&
            config.DefaultBackend == BackendC::GPU &&
            !Device_.NumericReady()
        ) 
        {
            SetError_(ErrorC::ACCELERATOR_UNAVAILAVLE);
            return false;
        }

        Configured_ = true;
        ClearError_();
        return true;
    }

    RuntimeInfoC SessionCompute::GetRuntimeInfo() const noexcept
    {
        RuntimeInfoC info{};
        info.DefaultBackend = Config_.DefaultBackend;
    #if defined(SN_ACPP_ENABLED)
        info.AdaptiveCppCompiled = true;
    #else
        info.AdaptiveCppCompiled = false;
    #endif
        info.AcceleratorAvailable = Device_.Available();
        info.DeviceUSMAvailable = Device_.NumericReady();
        info.DualQueueAvailable = Device_.Available();
        info.Configured = Configured_;
        info.DeviceName = Device_.Name();
        return info;
    }

    

}