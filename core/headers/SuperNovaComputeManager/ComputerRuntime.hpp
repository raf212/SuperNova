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

        bool Availavle() noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return static_cast<bool>(Runtime_);
        #else
            return false;
        #endif
        }

        void MarkField() noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            if (Runtime_) 
            {
                Runtime_->MarkFaild();
            }
        #endif
        }

        bool StagingAvailable() const noexcept;

        bool NumericReady() const noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return StagingAvailable() && Runtime_->Healthy();
        #else
            return false;
        #endif
        }

        template<class DT>
        static constexpr bool AtomicTypeSupport() noexcept
        {
            using U = std::remove_cv_t<DT>;
            return (sizeof(U)== sizeof(uint32_t) || sizeof(U)== sizeof(uint64_t)) &&
                (std::is_same_v<U,int> || std::is_same_v<U,unsigned int> ||
                std::is_same_v<U,long> || std::is_same_v<U,unsigned long> ||
                std::is_same_v<U,long long> || std::is_same_v<U,unsigned long long> ||
                std::is_same_v<U,float> || std::is_same_v<U,double>);
        }

        template<class DT>
        bool SupportArithmeticType() const noexcept
        {
            using U = std::remove_cv_t<DT>;
        #if defined (SN_ACPP_ENABLED)
            if constexpr (std::is_same_v<U, sycl::half>) 
            {
                return FP16_;
            }
        #endif
            if constexpr (std::is_same_v<U, double>) 
            {
                return FP64_;
            }
            return true;
        }


        template<class DT> 
        bool SupportsDeviceAtomicRMW() const  noexcept
        {
            using U = std::remove_cv_t<DT>;
            if constexpr (!AtomicTypeSupport<DT>()) return false;
        #if defined (SN_ACPP_ENABLED)
            using Atomic = sycl::atomic_ref<U, sycl::memory_order_relaxed, sycl::memory_scope::device, 
                            sycl::access::address_space::global_space>;
            try 
            {
                return NumericReady() && 
                    SupportArithmeticType<DT>() &&
                    sizeof(U) % Atomic::required_alignment == UNSIGNED_ZERO &&
                    (
                        sizeof(U) != sizeof(uint64_t) ||
                        Runtime_->Gpu.has(sycl::aspect::atomic64)
                    );
            }
            catch(...)
            {
                return false;
            }
        #else
            return false;
        #endif
        }

        const std::string& Name() const noexcept {return Name_;}

    #if defined (SN_ACPP_ENABLED)
        std::shared_ptr<Runtime> RuntimeSharedPtr() const noexcept {return Runtime_;}

        sycl::queue* ControlQueuePtr() noexcept {return &Runtime_ ? &Runtime_->ControlQueue : nullptr;}

        sycl::queue* ComputeQueuePtr() noexcept {return &Runtime_ ? &Runtime_->ControlQueue : nullptr;}
        const sycl::queue* ComputeQueuePtr() const noexcept {return &Runtime_ ? &Runtime_->ControlQueue : nullptr;}

    #endif
    };

    struct Scheduler final
    {

    };

}