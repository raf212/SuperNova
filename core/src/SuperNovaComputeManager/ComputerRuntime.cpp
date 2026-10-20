#include "SuperNovaComputeManager/ComputerRuntime.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    bool Device::ConfigureGPU() noexcept
    {
        Reset();
    #if defined (SN_ACPP_ENABLED)
        try 
        {
            sycl::device gpu = sycl::device{sycl::gpu_selector_v};
            std::shared_ptr<Runtime> runtime = std::make_shared<Runtime>(gpu);
            FP16_ = gpu.has(sycl::aspect::fp16);
            FP64_ = gpu.has(sycl::aspect::fp64);
            Name_ = gpu.get_info<sycl::info::device::name>();
            Runtime_ = std::move(runtime);
            return true;
        }
        catch(...)
        {
            Reset();
        }
    #endif
        return false;
    }

    bool Device::StagingAvailable() const noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        try 
        {
            return Runtime_ && Runtime_->Gpu.has(sycl::aspect::usm_device_allocations);
        }
        catch(...)
        {
            return false;
        }
    #else
        return false;
    #endif
    }


}