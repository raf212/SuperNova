#pragma once
#include "ComputerDefAndVect.hpp"

namespace ComputerDetail 
{

    class DeviceSN final
    {
    private:
        bool Availavle_{false};
        bool SharedUSM_{false};
        bool ConcurrentSharedUSM_{false};
        bool Atomic64_{false};
        bool FP16_{false};
        bool FP64_{false};
        std::string Name_{};

    #if defined (SN_ACPP_ENABLED)
        std::optional<sycl::queue> Queue_{};
    #endif
    
    public:
        DeviceSN() noexcept = default;
        DeviceSN(const DeviceSN&) = delete;
        DeviceSN& operator=(const DeviceSN&) = delete;

        void Reset() noexcept;

        bool ConfigureGpu() noexcept;

        bool Syncronize() noexcept;
    
    };
}