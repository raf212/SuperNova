#pragma once
namespace BidirectionalInMemGraph::ComputerDetail
{
    template<class... Ts>
    bool SessionCompute::RowRangesPreparation_(size_t count, NumericRange<Ts>... ranges) noexcept
    {
        if (
            !EnsureConfigured() || 
            !(
                (
                    ranges.Count >= count &&
                    IsTypedRangeInSlab_(ranges.Host, ranges.Count)
                ) 
                && ...
            )
        )
        {
            SetError_(ErrorC::INVALID_REGION);
            return false;
        }
        return true;
    }


    template<class... Ts>
    bool SessionCompute::ArithmeticPolicyPreparation_(ExecutionPolicyC& policy) noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        const BackendC req_backend = policy.Backend == BackendC::AUTOMATIC ?
            Config_.DefaultBackend : policy.Backend;

        if (
            req_backend == BackendC::CPU || 
            !Device_.Available() ||
            (
                policy.Deterministic &&
                policy.WorkType == WorkTypeC::REDUCTION
            )
        ) 
        {
            return true;
        }

        const bool supported = (Device_.template SupportsArithmeticType<Ts>() && ...);
        if (supported) return true;

        if (!Residency_.Active() && CpuFallbackAllowed_(policy)) 
        {
            policy.Backend = BackendC::CPU;
            return true;
        }

        SetError_(ErrorC::INVALIDE_DTYPE);
        return false;
    #else
        (void) policy;
        return true;
    #endif
    }

    template<class T>
    bool SessionCompute::IsTypedRangeInSlab_(const T* data, size_t count) noexcept
    {
        if (Residency_.Active()) return Residency_.Contains(data, count);
        if (Binding_.Owner) return false;
        return Memory::IsTypedRangeInside(
            NumericalBase_(), SlabCellCount_(), data, count
        );
    }

    template<class OP>
    bool SessionCompute::Dispatch_(size_t element_count, size_t work_units, OP operation, const ExecutionPolicyC& policy) noexcept
    {
        if (element_count == UNSIGNED_ZERO) return false;

        const BackendC backend = Resolve_(work_units, policy);
        if (!ValidateResolvedBackend_(backend, policy)) return false;

        const bool ok = Executor::Run(Device_, backend, element_count, operation);
        if (!ok) 
        {
            SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
            return false;
        }
        ClearError_();
        return true;
    }

    template<class FN, class... Ts>
    bool SessionCompute::__ResolveActiveResidency__(ExecutionPolicyC policy, FN function, NumericRange<Ts>... ranges) noexcept
    {
        if (
            Residency_.Faild() ||
            (
                policy.Backend != BackendC::AUTOMATIC &&
                policy.Backend != Residency_.Backend()
            ) ||
            (
                policy.Deterministic &&
                policy.WorkType == WorkTypeC::REDUCTION &&
                Residency_.Backend() == BackendC::GPU
            )
        ) 
        {
            SetError_(ErrorC::INVALID_CONF);
        }

        policy.Backend = Residency_.Backend();

        auto GetAccess___ = [](const auto& range) noexcept -> OpenAccess
        {
            return range.Access == CDV::NumericalAccess::READ ? OpenAccess::READ :
                range.Access == CDV::NumericalAccess::WRITE ? OpenAccess::WRITE :
                OpenAccess::READ_WRITE;
        };

        std::tuple mapped = std::tuple{Residency_.Map(ranges.Host, ranges.Count, GetAccess___())...};
        const bool applied = std::apply([](auto... pointers)
            {
                return ((pointers != nullptr) && ...);
            },
            mapped
        );
        if (!applied) 
        {
            Residency_.CancelMapping();
            SetError_(ErrorC::INVALID_REGION);
            return false;
        }

        const bool ok = std::apply([&](auto... pointers)
            {
                return function(pointers..., policy);
            },
            mapped
        );
        if (!ok) 
        {
            Residency_.CancelMapping();
            const bool writable = ((ranges.Access != CDV::NumericalAccess::READ) || ...);
            if (!writable || LastError_ == ErrorC::DEVICE_EXECUTION_FAILED) 
            {
                Residency_.SetFail();
                NumericGate_().Poison();
            }
        }
        else 
        {
            Residency_.CommitMappings();
        }
        return ok;
    }

    template<class FN, class... Ts>
    bool SessionCompute::__ExecuteGpuNumerics__(ExecutionPolicyC policy, FN function, NumericRange<Ts>... ranges) noexcept
    {
        std::tuple scratch = std::tuple{StagedRange<Ts>(Device_, ranges)...};
        bool publishing = false;
        try 
        {
            const auto ApplyPrepare___ = std::apply([](auto&... staged_range) noexcept -> bool
                {
                    return (staged_range.PrepareStage() && ...);
                },
                scratch
            );

            if (
                !Device_.SyncComputeQ() || !ApplyPrepare___ ||
                !Device_.SyncComputeQ()
            ) 
            {
                SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
                return false;
            }

            const bool function_applied = std::apply([&](auto&... staged_range)
                {
                    return function(staged_range.data()..., policy);
                },
                scratch
            );

            if (!function_applied) 
            {
                return false;
            }

            if (!Device_.SyncComputeQ()) 
            {
                SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
                return false;
            }

            publishing = true;

            const bool downloaded = std::apply([](auto&... staged_range)
                {
                    return (staged_range.Download() && ...);
                },
                scratch
            );
            if (!downloaded) 
            {
                NumericGate_().Poison();
                SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
                return false;
            }

            ClearError_();
            return true;
        }
        catch (...)
        {
            (void) Device_.SyncComputeQ();
            if (publishing) NumericGate_().Poison();
            SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
            return false;
        }
    }



    template<class FN, class... Ts>
    bool SessionCompute::WithNumericRanges_(size_t work_units, ExecutionPolicyC policy, FN function, NumericRange<Ts>... ranges) noexcept
    {
        if (Residency_.Active()) 
        {
            if (__ResolveActiveResidency__(policy, function, ranges...)) return true;
            else 
                return false;
        }

        const BackendC backend = Resolve_(work_units, policy);
        if (!ValidateResolvedBackend_(backend, policy)) return false;
        policy.Backend = backend;
    
    #if defined (SN_ACPP_ENABLED)
        if (backend == BackendC::GPU) 
        {
            return __ExecuteGpuNumerics__(policy, function, ranges...);
        }
    #endif
        const bool ok = function(ranges.Host..., policy);
        if (!ok || ((ranges.Access != CDV::NumericalAccess::READ) || ...)) 
        {
            NumericGate_().Poison();
            SetError_(ErrorC::DEVICE_EXECUTION_FAILED);
        }

        return ok;
    }


}