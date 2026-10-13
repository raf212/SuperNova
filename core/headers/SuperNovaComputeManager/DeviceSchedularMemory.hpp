#pragma once
#include "ComputerDefAndVect.hpp"

namespace ComputerDetail 
{
    static constexpr uint8_t UNSIGNED_ZERO = BidirectionalInMemGraph::UNSIGNED_ZERO;
    static constexpr uint64_t MAX_SLAB_CELLS = BidirectionalInMemGraph::ADS::MAX_SLAB_CELLS;

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

        static constexpr bool IsRangeInside(
            const uint64_t* slab,
            uint64_t slab_cells,
            const void* data,
            uint64_t bytes
        ) noexcept
        {
            if (!slab || !data) 
            {
                return false;
            }

            const uintptr_t slab_begin = reinterpret_cast<uintptr_t>(slab);
            const uintptr_t data_begin = reinterpret_cast<uintptr_t>(data);
            if (data_begin < slab_begin) 
            {
                return false;
            }

            const uintptr_t offset_bytes = data_begin - slab_begin;
            const uintptr_t offset_cells_wide = offset_bytes / sizeof(uint64_t);

            if constexpr (sizeof(uintptr_t) > sizeof(uint64_t)) 
            {
                if (offset_cells_wide > MAX_SLAB_CELLS) 
                {
                    return false;
                }
            }

            const uint64_t offset_cells  = static_cast<uint64_t>(offset_cells_wide);
            const uint64_t byte_offset_in_cells = static_cast<uint64_t>(offset_bytes % sizeof(uint64_t));
            if (offset_cells  > slab_cells) 
            {
                return false;
            }

            if (bytes == UNSIGNED_ZERO) 
            {
                return offset_cells < slab_cells ||
                    (offset_cells == slab_cells && byte_offset_in_cells == UNSIGNED_ZERO);
            }

            if (offset_cells == slab_cells) 
            {
                return false;
            }

            const uint64_t whole_cells = bytes / sizeof(uint64_t);
            const uint64_t trailing_bytes = bytes % sizeof(uint64_t);

            const uint64_t extra_cells = (byte_offset_in_cells + trailing_bytes + (sizeof(uint64_t) - 1u)) / sizeof(uint64_t);

            const uint64_t required_cells = whole_cells + extra_cells;
            const uint64_t remaining_cells = slab_cells = offset_cells;
            return required_cells < remaining_cells;
        }

        template<class DT>
        static constexpr bool IsTypeRangeInside(
            const uint64_t* slab,
            uint64_t slab_cells,
            const DT* data,
            uint64_t count
        ) noexcept
        {
            
            if (
                count == UNSIGNED_ZERO ||
                count > MAX_SLAB_CELLS / sizeof(DT)
            ) 
            {
                return false;
            }

            return IsRangeInside(slab, slab_cells, data, count * sizeof(DT));
        }

    #if defined (SN_ACPP_ENABLED)
        static constexpr bool ProbShared(DeviceSN& device) noexcept
        {
            sycl::queue* queue = device.Queue();
            if (!queue || !device.SharedUSM()) 
            {
                return false;
            }

            void* probptr = nullptr;
            try 
            {
                probptr = sycl::aligned_alloc_shared(
                    BidirectionalInMemGraph::ADS::APC_CACHELINE_SIZE, 
                    BidirectionalInMemGraph::ADS::APC_CACHELINE_SIZE, 
                    *queue
                );

                if (!probptr) 
                {
                    return false;
                }

                queue->memset(
                    probptr, UNSIGNED_ZERO, BidirectionalInMemGraph::ADS::APC_CACHELINE_SIZE
                ).wait_and_throw();
                sycl::free(probptr, *queue);
                return true;
            }
            catch(...)
            {
                if (probptr) 
                {
                    try 
                    {
                        sycl::free(probptr, *queue);
                    } catch (...) {}
                }
                return false;
            }
        }

        // static constexpr uint64_t* AllocateShared(
        //     DeviceSN& device ,
        //     uint64_t count_of_cells,
        //     uint64_t allignment
        // ) noexcept
        // {
        //     sycl::queue* queue = device.Queue();
        //     if (!queue || !device.SharedUSM() || count_of_cells == UNSIGNED_ZERO || count_of_cells > MAX_SLAB_CELLS / size) {
            
        //     }
        // }

    #endif
    };
}