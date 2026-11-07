#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail
{
#if defined (SN_ACPP_ENABLED)
    RuntimeAcpp::RuntimeAcpp(sycl::device gpu) :
        Gpu(gpu), 
        Contex(Gpu),
        ComputeQueue(
            Contex, Gpu, [error = Failed](SEL list) noexcept{
                if (list.size()) 
                {
                    error->store(true, std::memory_order_relaxed);
                }
            },
            sycl::property_list{sycl::property::queue::in_order{}}
        ),
        ControlQueue(
            Contex, Gpu, [error = Failed](SEL list) noexcept{
                if (list.size()) 
                {
                    error->store(true, std::memory_order_relaxed);
                }
            },
            sycl::property_list{sycl::property::queue::in_order{}}
        )
    {}

    bool RuntimeAcpp::Wait(sycl::queue& queue) noexcept
    {
        try 
        {
            queue.wait_and_throw();
            return Healthy();
        }
        catch(...)
        {
            MarkFaild();
            try 
            {
                queue.wait();
            }
            catch(...)
            {
            }
            return false;
        }
    }

    bool RuntimeAcpp::Wait(sycl::event& event) noexcept
    {
        try 
        {
            event.wait_and_throw();
            return Healthy();
        }
        catch(...)
        {
            MarkFaild();
            try 
            {
                event.wait();
            }
            catch(...)
            {
            }
            return false;
        }
    }

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

#endif


    void Device::Reset()
    {
        (void)Sync();
    #if defined (SN_ACPP_ENABLED)
        Runtime_.reset();
    #endif
        FP16_ = FP64_ = false;
        Name_.clear();
    }

    bool Device::SyncComputeQ() noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        return !Runtime_ || Runtime_->Wait(Runtime_->ComputeQueue);
    #else
        return true;
    #endif
    }

    bool Device::SyncControlQ() noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        return !Runtime_ || Runtime_->Wait(Runtime_->ControlQueue);
    #else
        return true;
    #endif
    }

    bool Device::Available() const noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        return static_cast<bool>(Runtime_);
    #else
        return false;
    #endif
    }

    void Device::MarkField() noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        if (Runtime_) 
        {
            Runtime_->MarkFaild();
        }
    #endif
    }

    bool Device::NumericReady() const noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        return StagingAvailable() && Runtime_->Healthy();
    #else
        return false;
    #endif
    }

    bool Device::Drain() noexcept
    {
    #if defined(SN_ACPP_ENABLED)
        return !Runtime_ || Runtime_->Drain();
    #else
        return true;
    #endif
    }

    bool Device::Sync() noexcept
    {
        const bool compute = SyncComputeQ();
        const bool control = SyncControlQ();
        return compute && control;
    }

}