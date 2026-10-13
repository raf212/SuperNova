#pragma once
#include "DeviceSchedularMemory.hpp"

namespace ComputerDetail 
{

    struct ExecutorSN final
    {
        template<class OP> 
        static constexpr bool CPU(
            uint64_t begin,
            uint64_t end,
            OP operation
        ) noexcept
        {
            try 
            {
                for (uint64_t i = 0; i < end; i++) 
                {
                    operation(i);
                }
                return true;

            } catch (...) 
            {
                return false;
            }
        }


    #if defined (SN_ACPP_ENABLED)
        
        template<class OP>
        static constexpr bool GPU(
            DeviceSN& device,
            uint64_t begin,
            uint64_t end,
            OP operation
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>, "DEVICE:: Operation must be trivally copyable.");

            if (end <= begin) 
            {
                return false;
            }

            sycl::queue* queue = device.Queue();
            if (!queue) 
            {
                return false;
            }

            try 
            {
                const size_t count = end - begin;
                queue->parallel_for(
                    sycl::range<1>{count},
                    [=](sycl::id<1> id)
                        {
                            operation(begin + static_cast<uint64_t>(id[0]));
                        }
                ).wait_and_throw();
                return true;

            } catch (...) 
            {
                return false;
            }
        }

        template<class OP>
        static bool CoOp(
            DeviceSN& device,
            size_t count,
            float cpu_fraction,
            OP operation
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>, "DEVICE:: Operation must be trivally copyable.");

            if (count == UNSIGNED_ZERO) 
            {
                return true;
            }

            if (!device.Queue() || !device.ConcurrentSharedUSM()) 
            {
                return false;
            }

            cpu_fraction = std::clamp(cpu_fraction, 0.0f, 1.0f);

            uint64_t cpu_count = static_cast<uint64_t>(static_cast<double>(count) * cpu_fraction);
            cpu_count = std::min(cpu_count, count);
            if (cpu_count == count) 
            {
                return CPU(UNSIGNED_ZERO, count, operation);
            }

            try 
            {
                const uint64_t gpu_count = count - cpu_count;
                sycl::event event = device.Queue()->parallel_for(
                    sycl::range<1>{gpu_count},
                    [=](sycl::id<1> id)
                    {
                        operation(cpu_count + static_cast<uint64_t>(id[0]));
                    }
                );
                const bool cpu_ok = CPU(UNSIGNED_ZERO, cpu_count, operation);
                event.wait_and_throw();
                return cpu_ok;
            } catch (...) 
            {
                return false;
            }
        }

    #endif
    };

}