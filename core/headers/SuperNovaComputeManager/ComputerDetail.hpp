#pragma once
#include "ComputerDefAndVect.hpp"

namespace ComputerDetail 
{
    static constexpr uint8_t UNSIGNED_ZERO = BidirectionalInMemGraph::UNSIGNED_ZERO;
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

        bool Available() const noexcept {return Availavle_;}
        bool SharedUSM() const noexcept {return SharedUSM_;};
        bool ConcurrentSharedUSM() const noexcept {return ConcurrentSharedUSM_;}
        bool Atomic64() const noexcept {return Atomic64_;}
        bool FP16() const noexcept {return FP16_;}
        bool FP64() const noexcept {return FP64_;}

        template<class DT> bool SupportsArethmeticType() noexcept
        {
            using U = std::remove_cv_t<DT>;
        #if defined (SN_ACPP_ENABLED)
            if constexpr (std::is_same_v<U, sycl::half>) 
            {
                return FP16_;
            }
        #endif
            else if (std::is_same_v<U, double>) 
            {
                return FP64_;
            }
            else 
            {
                return true;
            }
        }

    #if defined (SN_ACPP_ENABLED)
        sycl::queue* Queue() noexcept
        {
            return Queue_.has_value() ? &Queue_.value() : nullptr;
        }

        const sycl::queue* Queue() const noexcept
        {
            return Queue_.has_value() ? &Queue_.value() : nullptr;
        }
    #endif    
    };


    struct SchedulerSN
    {
        using ComputerConf = BidirectionalInMemGraph::ComputerDefAndVect::ComputerConf;
        using ComputeExecutionPolicy = BidirectionalInMemGraph::ComputerDefAndVect::ComputeExecutionPolicy;
        using ComputerBackend = BidirectionalInMemGraph::ComputerDefAndVect::ComputerBackend;
        using ComputerWorkType = BidirectionalInMemGraph::ComputerDefAndVect::ComputerWorkType;
        using ComputerFutureUse = BidirectionalInMemGraph::ComputerDefAndVect::ComputerFutureUse;

        static constexpr uint64_t SaturateMult(uint64_t value, uint64_t factor) noexcept
        {
            if (value == UNSIGNED_ZERO || factor == UNSIGNED_ZERO) 
            {
                return UNSIGNED_ZERO;
            }

            return value > UINT64_MAX / factor ? UINT64_MAX : value * factor;
        }

        static constexpr bool FallBackAllowed(
            const ComputerConf& conf,
            const ComputeExecutionPolicy& policy
        ) noexcept
        {
            return conf.AllowCPUFallback && policy.AllowCPUFallback;
        }

        static constexpr std::optional<ComputerBackend> Resolve(
            uint64_t work_units,
            const ComputerConf& conf,
            const ComputeExecutionPolicy& policy,
            const DeviceSN& device,
            bool shared_slab
        ) noexcept
        {
            ComputerBackend requested = policy.Backend == ComputerBackend::AUTOMATIC ?
                conf.DefaultBackend : policy.Backend;

            if (
                policy.Deterministic && 
                policy.WorkType == ComputerWorkType::REDUCTION
            ) 
            {
                return ComputerBackend::CPU;
            }

            if (requested == ComputerBackend::CPU) 
            {
                return ComputerBackend::CPU;
            }

            const bool accel_device_ready = device.Available() && shared_slab;

            if (requested == ComputerBackend::GPU) 
            {
                if (accel_device_ready && device.ConcurrentSharedUSM()) 
                {
                    return ComputerBackend::GPU;
                }

                return FallBackAllowed(conf, policy) ? std::optional(ComputerBackend::CPU) : std::nullopt;
            }

            if (requested == ComputerBackend::COOPERATIVE) 
            {
                if (accel_device_ready && device.ConcurrentSharedUSM()) 
                {
                    return ComputerBackend::COOPERATIVE;
                }

                return FallBackAllowed(conf, policy) ? std::optional(ComputerBackend::CPU) : std::nullopt;
            }

            if (!accel_device_ready) 
            {
                return ComputerBackend::CPU;
            }

            uint64_t threshold = std::max<uint64_t>(1u, conf.GPUWorkThreshold);

            if (policy.FutureUse == ComputerFutureUse::HOST) 
            {
                threshold = threshold > UINT64_MAX / 2u ? UINT64_MAX : threshold * 2u;
            }

            else if (policy.FutureUse == ComputerFutureUse::ACCELERAROR) 
            {
                threshold = std::max<uint64_t>(1u, threshold / 2u);
            }

            return work_units >= threshold ? ComputerBackend::GPU : ComputerBackend::CPU;
        }

    };


    struct MemorySN final
    {
        static constexpr uint64_t BYTE_SIZE_LIMITER = UINT64_MAX / sizeof(uint64_t);

        static constexpr bool IsRangeInside(
            const uint64_t* slab,
            uint64_t cell_count,
            const void* data,
            uint64_t bytes
        ) noexcept
        {
            if (!slab || !data || cell_count > BYTE_SIZE_LIMITER) 
            {
                return false;
            }

            const uintptr_t slab_begin = reinterpret_cast<uintptr_t>(slab);
            const uintptr_t data_begin = reinterpret_cast<uintptr_t>(data);
            const uint64_t slab_bytes = cell_count * sizeof(uint64_t);

            if (data_begin < slab_begin) 
            {
                return false;
            }

            const uintptr_t offset = data_begin - slab_begin;

            return offset <= slab_bytes && bytes <= slab_bytes - static_cast<uint64_t>(offset);
        }
    };
}