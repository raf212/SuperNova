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

#endif


    void Device::Reset()
    {
        (void)Synchronize();
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

    bool Device::Synchronize() noexcept
    {
        const bool compute = SyncComputeQ();
        const bool control = SyncControlQ();
        return compute && control;
    }

}