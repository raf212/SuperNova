#pragma once
#include "ComputerDefAndVect.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    using CDV = ComputerDefAndVect;
    static constexpr uint64_t MAX_SLAB_CELLS = ADS::MAX_SLAB_CELLS;

#if defined (SN_ACPP_ENABLED)
    struct Runtime 
    {
        using SEL = sycl::exception_list;

        std::shared_ptr<std::atomic_bool> Failed = std::make_shared<std::atomic_bool>(false);
        sycl::device Gpu;
        sycl::context Contex;
        sycl::queue ComputeQueue;
        sycl::queue ControlQueue;

        explicit Runtime(sycl::device gpu) :
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

        constexpr bool Healthy() const noexcept
        {
            return !Failed->load(std::memory_order_relaxed);
        }

        constexpr void MarkFaild() noexcept
        {
            Failed->store(true, std::memory_order_relaxed);
        }

        constexpr bool Wait(sycl::queue& queue) noexcept
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

        constexpr bool Wait(sycl::event& event) noexcept
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

        ~Runtime() noexcept 
        {
            (void) Wait(ComputeQueue);
            (void) Wait(ControlQueue);
        }

    };
#endif


    class Device
    {
    private:

    #if defined (SN_ACPP_ENABLED)
        std::shared_ptr<Runtime> Runtime_{};
    #endif

        bool FP16_{false};
        bool FP64_{false};
        std::string Name_{};

    public:
        Device() noexcept = default;
        Device(const Device&) = delete;
        Device& operator = (const Device&) = delete;

        void Reset()
        {
            (void)Sync();
        #if defined (SN_ACPP_ENABLED)
            Runtime_.reset();
        #endif
            FP16_ = FP64_ = false;
            Name_.clear();
        }

        bool ConfigureGPU() noexcept;

        bool SyncComputeQ() noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return !Runtime_ || Runtime_->Wait(Runtime_->ComputeQueue);
        #else
            return false;
        #endif
        }

        bool SyncControlQ() noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return !Runtime_ || Runtime_->Wait(Runtime_->ControlQueue);
        #else
            return false;
        #endif
        }

        bool Sync() noexcept
        {
            const bool compute = SyncComputeQ();
            const bool control = SyncControlQ();
            return compute && control;
        }



    };

}