#include "SuperNovaComputeManager/ComputerDetail.hpp"

namespace ComputerDetail
{
    void DeviceSN::Reset() noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        Queue_.reset();
    #endif
        Availavle_ = false;
        SharedUSM_ = false;
        ConcurrentSharedUSM_ = false;
        Atomic64_ = false;
        FP16_ = false;
        FP64_ = false;
        Name_.clear();
    }

    bool DeviceSN::ConfigureGpu() noexcept
    {
        Reset();
    #if defined (SN_ACPP_ENABLED)
        try 
        {
            sycl::device device{sycl::gpu_selector_v};
            Queue_.emplace(
                device,
                sycl::property_list{sycl::property::queue::in_order{}}
            );
            Availavle_ = true;
            SharedUSM_ = device.has(sycl::aspect::usm_shared_allocations);
            ConcurrentSharedUSM_ = device.has(sycl::aspect::usm_shared_allocations);
            Atomic64_ = device.has(sycl::aspect::atomic64);
            FP16_ = device.has(sycl::aspect::fp16);
            FP64_ = device.has(sycl::aspect::fp64);
            Name_ = device.get_info<sycl::info::device::name>();
            return true;
        } catch (...) 
        {
            return false;
        }
    #endif  
        return false;
    }

    bool DeviceSN::Syncronize() noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        if (!Queue_.has_value()) 
        {
            return true;
        }

        try 
        {
            Queue_->wait_and_throw();
            return true;
        } catch (...) 
        {
            return false;
        }
    #endif
        return true;
    }

    

}