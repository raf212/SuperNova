#pragma once

#if defined (SN_ACPP_ENABLED)
namespace BidirectionalInMemGraph::ComputerDetail
{
    template<class DT>
    bool StagedRange<DT>::PrepareStage()
    {
        const std::optional<uint64_t> bytes = Memory::ByteCount<U>(Range_.Count);
        if (!bytes.has_value() ||!Range_.Host || Range_.Count == UNSIGNED_ZERO) 
        {
            return false;
        }
        
        uint64_t alignment = std::max(alignof(U), uint64_t{ADS::CACELINE_SIZE_IN_BYTES});

        if constexpr (AtomicTypeSupport<U>()) 
        {
            using Atomic = sycl::atomic_ref<U, sycl::memory_order::relaxed, 
                sycl::memory_scope::device, sycl::access::address_space::global_space>;

            alignment = std::max(alignment, uint64_t{Atomic::required_alignment});
        }

        Local_ = sycl::aligned_alloc_device<U>(
            alignment,
            Range_.Count,
            *Device_.ComputeQueuePtr()
        );
        if (!Local_) return false;

        if (Range_.Access != CDV::NumericalAccess::WRITE) {
            Device_.ComputeQueuePtr()->memcpy(
                Local_, 
                Range_.Host, 
                *bytes
            );
        }
        return true;
    }

    template<class DT>
    bool StagedRange<DT>::Download() 
    {
        if constexpr (std::is_const_v<DT>) return true;
        else
        {
            if (Range_.Access == CDV::NumericalAccess::READ) return true;
            
            const std::optional<uint64_t> bytes = Memory::ByteCount<U>(Range_.Count);
            if (!bytes.has_value() ||!Range_.Host || !Local_) 
            {
                return false;
            }
            Device_.ComputeQueuePtr()->memcpy(
                Range_.Host,
                Local_,
                *bytes
            );
            return true;
        }
    }
}
#endif