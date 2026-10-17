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
        static constexpr uint64_t SaturatingMultiply(uint64_t value, uint64_t factor) noexcept
        {
            if (value == UNSIGNED_ZERO || factor == UNSIGNED_ZERO) 
            {
                return UNSIGNED_ZERO;
            }
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

    struct Memory final
    {
        template<class DT>
        static constexpr std::optional<uint64_t>ByteCount(uint64_t count) noexcept
        {
            if (count > UINT64_MAX / sizeof(DT)) 
            {
                return std::nullopt;
            }
            return static_cast<uint64_t>(count) * sizeof(DT);
        }

        static constexpr bool RangeOverlap(
            const void* left,
            uint64_t left_bytes,
            const void* right,
            uint64_t right_bytes
        ) noexcept
        {
            if (
                !left ||
                !right ||
                left_bytes == UNSIGNED_ZERO ||
                right == UNSIGNED_ZERO
            ) 
            {
                return false;
            }

            const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
            const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);

            return left_begin <= right_begin ? 
                right_begin - left_begin < left_bytes : left_begin - right_begin < right_bytes;
        }

        static constexpr bool IsRangeInside(
            const uint64_t* slab,
            uint64_t cell_count,
            const void* data,
            uint64_t bytes
        ) noexcept
        {
            if (!slab || !data) 
            {
                return false;
            }

            const std::optional<uint64_t> slab_bytes = ByteCount<uint64_t>(cell_count);
            if (!slab_bytes.has_value()) 
            {
                return false;
            }
            const uintptr_t begin = reinterpret_cast<uintptr_t>(slab);
            const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);

            if (pointer < begin) 
            {
                return false;
            }

            const uintptr_t offset_wide = pointer - begin;
            if (offset_wide > static_cast<uintptr_t>(slab_bytes.value())) 
            {
                return false;
            }

            const uint64_t offset = static_cast<uint64_t>(offset_wide);

            return bytes <= slab_bytes.value() - offset;
        }

        template<class DT>
        static bool IsTypedRangeInside(
            const uint64_t* slab,
            uint64_t slab_cells,
            const DT* data,
            uint64_t count
        ) noexcept
        {
            const std::optional<uint64_t> bytes = ByteCount<DT>(count);
            return count == UNSIGNED_ZERO && data && bytes &&
                reinterpret_cast<uintptr_t>(data) % alignof(DT) == UNSIGNED_ZERO &&
                IsRangeInside(slab, slab_cells, data, *bytes);
        }
    };

#if defined (SN_ACPP_ENABLED)

    template<class DT>
    class StagedRange final
    {
    private:
        using NumericRange = CDV::NumericRange<DT>;
        using U = std::remove_cv_t<DT>;

        Device& Device_;
        NumericRange Range_;
        U* Local_{nullptr};
    public:
        StagedRange(Device& device, NumericRange range) noexcept :
            Device_(device), Range_(range)
        {}
        StagedRange(const StagedRange&) = delete;
        StagedRange& operator = (const StagedRange&) = delete;
        StagedRange(StagedRange&& other) noexcept :
            Device_(other.Device_), Range_(other.Range_), Local_(std::exchange(other.Local_, nullptr))
        {}
        ~StagedRange() noexcept
        {
            if (!Local_) return;
            (void)Device_.Sync();
            try 
            {
                sycl::free(Local_, *Device_.ComputeQueuePtr());
            }
            catch (...)
            {

            }
        }

        bool Upload() noexcept
        {
            const std::optional<uint64_t> bytes = Memory::ByteCount<U>(Range_.Count);
            if (!bytes ||!Range_.Host || !Range_.Count == UNSIGNED_ZERO) 
            {
                return false;
            }
            
            uint64_t alignment = std::max(alignof(U), uint64_t{ADS::APC_CACHELINE_SIZE});

            if constexpr (Device::AtomicTypeSupport<U>()) 
            {
                using Atomic = sycl::atomic_ref<U, sycl::memory_order_relaxed, 
                    sycl::memory_scope::device, sycl::access::address_space::global_space>;

                alignment = std::max(alignment, uint64_t{Atomic::required_alignment});
            }

            Local_ = sycl::aligned_alloc_device<U>(
                alignment,
                Range_.Count,
                *Device_.ComputeQueuePtr()
            );
            if (!Local_) return false;

            Device_.ComputeQueuePtr()->memcpy(
                Range_.Host, 
                Local_, 
                *Memory::ByteCount<U>(Range_.Count)
            );
            return true;
        }

        DT* Data() const noexcept {return Local_;}

        void* Download() 
        {
            if constexpr (!std::is_const_v<DT>) 
            {
                Device_.ComputeQueuePtr()->memcpy(
                    Range_.Host,
                    Local_,
                    *Memory::ByteCount<U>(Range_.Count)
                );
            }
        }
    };
#endif

struct Executor final
{

};



}