#pragma once
#include "ComputerDefAndVect.hpp"
#include <exception>

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

        bool Healthy() const noexcept
        {
            return !Failed->load(std::memory_order_relaxed);
        }

        void MarkFaild() noexcept
        {
            Failed->store(true, std::memory_order_relaxed);
        }

        bool Wait(sycl::queue& queue) noexcept
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

        bool Wait(sycl::event& event) noexcept
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

        // Completion and numerical health are distinct: async failure can
        // poison results even when every submitted command has completed.
        bool Drain() noexcept
        {
            bool compute = false, control = false;
            try {ComputeQueue.wait(); compute = true;} catch (...) {}
            try {ControlQueue.wait(); control = true;} catch (...) {}
            return compute && control;
        }

        ~Runtime() noexcept 
        {
            if (!Drain()) std::terminate();
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
            return true;
        #endif
        }

        bool SyncControlQ() noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return !Runtime_ || Runtime_->Wait(Runtime_->ControlQueue);
        #else
            return true;
        #endif
        }

        bool Sync() noexcept
        {
            const bool compute = SyncComputeQ();
            const bool control = SyncControlQ();
            return compute && control;
        }

        bool Available() const noexcept
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
        bool SupportsArithmeticType() const noexcept
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
            if constexpr (AtomicTypeSupport<DT>()) {
        #if defined (SN_ACPP_ENABLED)
            using Atomic = sycl::atomic_ref<U, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                            sycl::access::address_space::global_space>;
            try 
            {
                return NumericReady() && 
                    SupportsArithmeticType<DT>() &&
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
            } else {return false;}
        }

        bool Drain() noexcept
        {
        #if defined(SN_ACPP_ENABLED)
            return !Runtime_ || Runtime_->Drain();
        #else
            return true;
        #endif
        }

        bool SynchronizeCompute() noexcept {return SyncComputeQ();}
        bool SynchronizeControl() noexcept {return SyncControlQ();}
        bool Synchronize() noexcept {return Sync();}
        template<class T> static constexpr bool AtomicTypeSupported() noexcept {return AtomicTypeSupport<T>();}
        const std::string& Name() const noexcept {return Name_;}

    #if defined (SN_ACPP_ENABLED)
        std::shared_ptr<Runtime> RuntimeSharedPtr() const noexcept {return Runtime_;}
        std::shared_ptr<Runtime> AcquireRuntime() const noexcept {return Runtime_;}

        sycl::queue* ControlQueuePtr() noexcept {return Runtime_ ? &Runtime_->ControlQueue : nullptr;}

        sycl::queue* ComputeQueuePtr() noexcept {return Runtime_ ? &Runtime_->ComputeQueue : nullptr;}
        const sycl::queue* ComputeQueuePtr() const noexcept {return Runtime_ ? &Runtime_->ComputeQueue : nullptr;}

    #endif
    };

    struct Scheduler final
    {
        static constexpr uint64_t SaturatingMultiply(uint64_t value, uint64_t factor) noexcept
        {
            if (value == UNSIGNED_ZERO || factor == UNSIGNED_ZERO) 
            {
                return UNSIGNED_ZERO;
            }
            return value > UINT64_MAX / factor ? UINT64_MAX : value * factor;
        }

        static constexpr bool FallbackAllowed(
            const CDV::ComputerConf& config,
            const CDV::ComputeExecutionPolicy& policy
        ) noexcept
        {
            return config.AllowCPUFallback && policy.AllowCPUFallback;
        }

        static constexpr CDV::ComputerBackend Resolve(
            uint64_t work_units,
            const CDV::ComputerConf& config,
            const CDV::ComputeExecutionPolicy& policy,
            const Device& device
        ) noexcept
        {
            CDV::ComputerBackend requested = policy.Backend == CDV::ComputerBackend::AUTOMATIC ?
                config.DefaultBackend : policy.Backend;
            if (
                policy.Deterministic && 
                policy.WorkType == CDV::ComputerWorkType::REDUCTION
            ) 
            {
                return CDV::ComputerBackend::CPU;
            }
            if (requested == CDV::ComputerBackend::CPU) 
            {
                return CDV::ComputerBackend::CPU;
            }

            const bool accelerator_ready = device.NumericReady();
            if (requested == CDV::ComputerBackend::GPU) 
            {
                if (accelerator_ready) 
                {
                    return CDV::ComputerBackend::GPU;
                }
                return FallbackAllowed(config, policy) ? 
                    CDV::ComputerBackend::CPU : CDV::ComputerBackend::GPU;
            }

            if (!accelerator_ready) 
            {
                return CDV::ComputerBackend::CPU;
            }
            uint64_t threshold = std::max<uint64_t>(1u, config.GPUWorkThreshold);
            return work_units >= threshold ? 
                CDV::ComputerBackend::GPU : CDV::ComputerBackend::CPU;
        }
    };



}
