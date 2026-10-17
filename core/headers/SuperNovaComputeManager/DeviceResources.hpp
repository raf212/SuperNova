#pragma once
#include "CpRuntimeManager.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{

#if defined (SN_ACPP_ENABLED)

    template<class DT>
    struct DevicePlanView final
    {
        const DT* Data;
        uint64_t count;
        uint64_t Revision;
    };

    enum class PlanUpdateResult : uint8_t
    {
        ACCEPTED = 0,
        BUSY = 1,
        INVALID = 2,
        FAILED = 3
    };

    template<class DT>
    class ResidentDeviceRegion final
    {
        template<class> friend class DevicePlanDoubleBuffer;
        static_assert(std::is_trivially_copyable_v<DT> && !std::is_const_v<DT>);
    private:
        std::shared_ptr<Runtime> Runtime_;
        DT* Data_;
        uint64_t Count_{0};
        std::optional<sycl::event> Tail_{};

    public:
        explicit ResidentDeviceRegion(std::shared_ptr<Runtime> runtime) noexcept :
            Runtime_(std::move(runtime))
        {}
        ResidentDeviceRegion(const ResidentDeviceRegion&) = delete;
        ResidentDeviceRegion& operator = (const ResidentDeviceRegion&) = delete;
        ~ResidentDeviceRegion() noexcept
        {
            if (!Data_) return;
            if (Tail_) (void)Runtime_->Wait(*Tail_);
            try 
            {
                sycl::free(Data_, Runtime_->Contex);
            }
            catch (...)
            {}
        }
    };  

#endif

}