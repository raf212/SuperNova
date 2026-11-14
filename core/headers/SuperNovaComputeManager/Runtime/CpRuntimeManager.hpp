#pragma once
#include "ComputerRuntime.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{

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
            if (!Device_.Drain()) std::terminate();
            try 
            {
                sycl::free(Local_, *Device_.ComputeQueuePtr());
            }
            catch (...)
            {

            }
        }

        bool PrepareStage();

        DT* Data() const noexcept {return Local_;}

        bool Download();
    };
#endif

    struct Executor final
    {
        template<class OP>
        static constexpr bool CPU(uint64_t begin, uint64_t end, OP operation) noexcept
        {
            try 
            {
                for (size_t i = begin; i < end; i++) 
                {
                    operation(i);
                }
                return true;
            }
            catch(...)
            {
                return false;
            }
        }    

    #if defined (SN_ACPP_ENABLED)
        template<class OP>
        static std::optional<sycl::event> SubmitGPU(
            Device& device,
            uint64_t begin,
            uint64_t end,
            OP operation,
            std::span<const sycl::event> dependecies = {}
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>, "Not Trivially Copyable");
            if (
                !device.NumericReady() ||
                end <= begin
            ) 
            {
                return std::nullopt;
            }

            try 
            {
                return device.ComputeQueuePtr()->submit(
                    [&](sycl::handler& h)
                    {
                        for (const auto& e : dependecies) 
                        {
                            h.depends_on(e);
                        }
                        h.parallel_for(
                            sycl::range<1>{end - begin}, 
                            [=](sycl::id<1> i)
                            {
                                operation(begin + static_cast<uint64_t>(i[0]));
                            }  
                        );
                    }
                );
            }
            catch (...)
            {
                device.MarkField();
                return std::nullopt;
            }
        }
        
        template<class OP>
        static constexpr bool GPU(
            Device& device,
            uint64_t begin,
            uint64_t end,
            OP operation
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>, "Device operation must be trivially copyable.");
            if (end <= begin) 
            {
                return end == begin;
            }

            sycl::queue* queue = device.ComputeQueuePtr();
            if (!queue) return false;
            try 
            {
                const uint64_t count = end - begin;
                queue->parallel_for(
                    sycl::range<1>{count}, 
                    [=](sycl::id<1> id)
                    {
                        operation(begin + static_cast<size_t>(id[0]));
                    }
                );
                return device.Synchronize();
            }
            catch(...)
            {
                (void)device.Synchronize();
                return false;
            }
        }
    #endif

        template<class OP>
        static constexpr bool Run(
            Device& device,
            CDV::ComputerBackend backend,
            uint64_t count,
            OP operation
        ) noexcept
        {
        #if !defined (SN_ACPP_ENABLED)
            (void) device;
            if (backend == CDV::ComputerBackend::CPU || backend == CDV::ComputerBackend::AUTOMATIC) 
            {
                return CPU(0u, count, operation);
            }
            return false;
        #else
            switch (backend) 
            {
                case CDV::ComputerBackend::CPU:
                    return CPU( 0u, count, operation );
                case CDV::ComputerBackend::GPU:
                case CDV::ComputerBackend::AUTOMATIC:
                    return GPU(device, UNSIGNED_ZERO, count, operation);
            }
        #endif
        }

    };

    struct Reduction final
    {
        template<class DT, class Transform>
        static constexpr bool CPU(
            uint64_t count,
            Transform transform,
            DT& result
        ) noexcept
        {
            DT sum{};
            for (uint64_t i = 0; i < count; i++) 
            {
                sum += transform(i);
            }

            result = sum;
            return true;
        }

    #if defined (SN_ACPP_ENABLED)
        template<class DT, class Transform, class Combine = sycl::plus<DT>>
        static constexpr bool GPU(
            Device& device,
            uint64_t count,
            Transform transform,
            DT& result,
            DT identity = {},
            Combine combo = {}
        ) noexcept
        {
            static_assert(ScalerVC<DT>, "unsupported dtype");
            static_assert(std::is_trivially_copyable_v<Transform>, "Transform not tgrivially copyable");

            sycl::queue* queue = device.ComputeQueuePtr();
            if (!queue) return false;
            DT* scratch = nullptr;
            DT completed{};
            try 
            {
                scratch = sycl::malloc_device<DT>(1u, *queue);
                if (!scratch) return false;
                queue->submit([=](sycl::handler& handle)
                {
                    auto reduction = sycl::reduction(
                        scratch, 
                        identity, 
                        combo,
                        sycl::property_list{sycl::property::reduction::initialize_to_identity{}}
                    );

                    handle.parallel_for(
                        sycl::range<1>{count}, 
                        reduction,
                        [=](sycl::id<1>id, auto& value)
                        {
                            value.combine(transform(static_cast<uint64_t>(id[0])));
                        }
                    );
                });

                queue->memcpy(&completed, scratch, sizeof(DT));
                const bool ok = device.SyncComputeQ();
                sycl::free(std::exchange(scratch, nullptr), *queue);
                if (ok) result = completed;
                return ok;
            }
            catch (...)
            {
                (void)device.SyncComputeQ();
                if (scratch) 
                {
                    try 
                    {
                        sycl::free(scratch, *queue);
                    }
                    catch (...)
                    {
                    }
                }
                return false;
            }
        }

        template<class Pradicate>
        static bool AllOfGPU(
            Device& device,
            uint64_t count,
            Pradicate predicate,
            bool& result
        ) noexcept
        {
            int all = 1;
            const bool ok = GPU<int>(
                device,
                count,
                [=](uint64_t i) noexcept
                {
                    return static_cast<bool>(predicate(i)) ? 1 : 0;
                },
                all,
                1,
                sycl::minimum<int>{}
            );
            if (ok) result = (all != UNSIGNED_ZERO);
            return ok;
        }
    #endif
    };
}

#include "../../../templates/SuperNovaComputeManager/Runtime/CpRuntimeManager.tpp"
