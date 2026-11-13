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
    bool IsTypedRangeInSlab_(const T* data, size_t count) noexcept;


}