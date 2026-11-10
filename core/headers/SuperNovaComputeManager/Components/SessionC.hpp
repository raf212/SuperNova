#pragma once 
#include "../Runtime/NumericalResedency.hpp"
#include "../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    class SessionCompute 
    {
    private:
        using FabricId = CheckpointFormat::FabricId;

        friend class RegionCompute;
        friend class ArithmeticCompute;
        friend class CheckpointCompute;

        ConfigC Config_{};
        OptimizedPrimitivesC OptimizedPrm_{};
        Device Device_{};
        NumericResidency Residency_{};
        FabricRuntimeBinding Binding_{};
        FabricExecutionGate UnboundGate_{};
        ExclusiveLease EpochLease_{};
        bool Session_{false};
        BackendC SessionBackend_{BackendC::CPU};
        std::atomic<ErrorC> LastError_{ErrorC::NONE};
        bool Configured_{false};

    //Functions - Basic
        FabricExecutionGate& NumericGate_() noexcept;

        bool OwnsEpoch_() noexcept;

        const uint64_t* NumericalBase_() noexcept;

        void SetError_(ErrorC error) noexcept;

        void ClearError_() noexcept {SetError_(ErrorC::NONE);}

        ReadLease AdmitNumeric_() noexcept;

        bool FallbackAllowed_(const ExecutionPolicyC& policy) const noexcept;

        uint64_t SlabCellCount_() noexcept {return Binding_.NumericalCells().size();}

        BackendC Resolve_(size_t work_units, const ExecutionPolicyC& policy) noexcept; 

        bool ValidateResolvedBackend_(BackendC backend, const ExecutionPolicyC& policy) noexcept;

    //Functions - Template
        template<class... Ts>
        bool RowRangesPreparation_(size_t count, NumericRange<Ts>... ranges) noexcept;

        template<class... Ts>
        bool ArithmeticPolicyPreparation_(ExecutionPolicyC& policy) noexcept;

        template<class T>
        bool IsTypedRangeInSlab_(const T* data, size_t count) noexcept;

        template<class OP>
        bool Dispatch_(size_t element_count, size_t work_units, OP operation, const ExecutionPolicyC& policy) noexcept;

        template<class FN, class... Ts>
        bool WithNumericRanges_(size_t work_units, ExecutionPolicyC policy, FN function, NumericRange<Ts>... ranges) noexcept;

        template<class OP> 
        bool For(size_t element_count, OP operation, ExecutionPolicyC policy = {}) noexcept;

    public:
        using SD = SchemaDefinition;
        // template<class T>
        // using RegionT = SNTOComputerBridge<T>;

        SessionCompute() noexcept = default;
        SessionCompute(const SessionCompute&) = delete;
        SessionCompute& operator = (const SessionCompute&) = delete;
        ~SessionCompute() noexcept
        {
            if (!Close() || !DiscardEpoch()) 
            {
                std::terminate();
            }
        }

        ///FUNCTIONS
        bool BindFabric(APCFinilizer& owner, FabricId id = {}) noexcept;

        bool BindFabric(FabricRuntimeBinding binding) noexcept;

        bool BindNumericStorage(
            std::span<uint64_t> cells, 
            FabricExecutionGate& gate, 
            FabricId id ={}
        ) noexcept;

        bool ReleseNumericalStorage() noexcept;

        void NotifyExternalWrite() noexcept {NumericGate_().NotifyHostChange();}

        FabricId Identity() const noexcept {return Binding_.Id;}

        bool SetRegisteredRegionBudget(uint64_t bytes) noexcept;

        TransferStatics TransferStats() const noexcept {return Residency_.Statistics();}

        std::optional<ResidencyInfo> InspectResidency(const CheckpointFormat::RegionRef& ref) const noexcept
        {
            return Residency_.InspectResidency(ref);
        }

        bool SessionActive() const noexcept {return Session_;}

        bool BeginSession(size_t work, ExecutionPolicyC policy = {}, ExclusiveLease lease = {}) noexcept;

        bool EndSession() noexcept;

        bool Close() noexcept;

        bool EpochActive() noexcept {return Residency_.Active();}

        bool BeginEpoch(size_t expected_work, ExecutionPolicyC policy = {}, ExclusiveLease lease = {}) noexcept;

        bool EndEpoch(uint64_t* published_stamp = nullptr) noexcept;

        bool DiscardEpoch() noexcept;

        bool RecoveryRequired() noexcept {return NumericGate_().IsPoisoned();}

        bool ConfirmRecovery() noexcept;

        bool EnsureConfigured() noexcept;

        bool Configure(const ConfigC& config) noexcept;

        void SetOptimizedPremetives(const OptimizedPrimitivesC& premitive) noexcept
        {
            OptimizedPrm_ = premitive;
        }

        OptimizedPrimitivesC GetOptimizedPremetives() const noexcept {return OptimizedPrm_;}

        bool Synchronize() noexcept;

        ConfigC GetConfig() const noexcept {return Config_;}

        ErrorC LastError() const noexcept;

        bool SyncCompute() noexcept {return Device_.SyncComputeQ();}

        bool SyncControl() noexcept {return Device_.SyncControlQ();}

        RuntimeInfoC GetRuntimeInfo() const noexcept;

    #if defined (SN_ACPP_ENABLED)

        template<class T>
        std::unique_ptr<ResidentDeviceRegion<T>> CtrateResidentRegion(size_t count) noexcept;

        template<class Record>
        std::unique_ptr<DevicePlanDoubleBuffer<Record>> CreateDevicePlan(size_t capacity) noexcept;
    #endif

    };
}