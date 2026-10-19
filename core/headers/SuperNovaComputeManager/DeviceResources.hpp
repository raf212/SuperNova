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

        std::optional<sycl::event> Transfer_(
            DT* host,
            uint64_t count,
            uint64_t offset,
            bool upload,
            std::span<const sycl::event> dependencies 
        ) noexcept
        {
            if (
                !Data_ || !host || count == UNSIGNED_ZERO ||
                offset > Count_ || count > Count_ - offset || !Runtime_->Healthy()
            ) 
            {
                return std::nullopt;
            }

            try 
            {
                Tail_ = Runtime_->ControlQueue.submit(
                    [&](sycl::handler& h)
                    {
                        if (Tail_) h.depends_on(*Tail_);
                        for (const auto& e : dependencies) 
                        {
                            h.depends_on(e);
                        }
                        if (upload) 
                        {
                            h.memcpy(Data_ + offset, host, count *sizeof(DT));
                        }
                        else 
                        {
                            h.memcpy(host, Data_ + offset, count * sizeof(DT));
                        }
                    }
                );
                return Tail_;
            }
            catch (...)
            {
                Runtime_->MarkFaild();
                return std::nullopt;
            }
        }

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
            {
            }
        }

        bool Allocate(uint64_t count) noexcept
        {
            if (Data_ || !Runtime_ || !Memory::ByteCount<DT>(count).has_value()) 
            {
                return false;
            }

            try 
            {
                Data_ = sycl::aligned_alloc_device<DT>(
                    std::max(alignof(DT)),
                    ADS::APC_CACHELINE_SIZE,
                    count,
                    Runtime_->ComputeQueue
                );
                if (!Data_) return false;
                Count_ = count;
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        uint64_t Size() noexcept {return Count_;}

        std::optional<sycl::event> Upload(
            std::span<const DT> source, 
            uint64_t offset = UNSIGNED_ZERO,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            return Transfer_(
                const_cast<DT*>(source.data()), source.size(), offset, false, dependencies
            );
        }

        std::optional<sycl::event> Download(
            std::span<DT> destination,
            uint64_t offset = UNSIGNED_ZERO,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            return Transfer_(
                destination.data(),
                destination.size(),
                offset,
                false,
                dependencies
            );
        }

        template<class OP>
        std::optional<sycl::event> Submit(
            uint64_t work_items,
            OP operation,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>);
            if (!Data_ || work_items == UNSIGNED_ZERO || !Runtime_->Healthy()) 
            {
                return std::nullopt;
            }

            const auto data = Data_;
            const auto count = Count_;
            try 
            {
                Tail_ = Runtime_->ComputeQueue.submit(
                    [&](sycl::handler& h)
                    {
                        if (Tail_) h.depends_on(*Tail_);
                        for (const auto& e : dependencies) 
                        {
                            h.depends_on(e);
                        }
                        h.parallel_for(
                            sycl::range<1>{work_items},
                            [=](sycl::id<1> i)
                            {
                                operation(static_cast<uint64_t>(i[0]), data, count);
                            }
                        );
                    }
                );
                return Tail_;
            }
            catch (...)
            {
                Runtime_->MarkFaild();
                return std::nullopt;
            }
        }

        bool Wait() noexcept
        {
            return !Tail_ || Runtime_->Wait(*Tail_);
        }

    };  

    template<class Record>
    class DevicePlanDoubleBuffer
    {
    private:
        static constexpr uint8_t DOUBLE = 2u;

        static_assert(std::is_trivially_copyable_v<Record> && !std::is_const_v<Record>);
        std::shared_ptr<Runtime> Runtime_;
        Record* Bank_[DOUBLE] {nullptr, nullptr};
        void* Stage_[DOUBLE] {nullptr, nullptr};
        bool HostUsm_[DOUBLE] {false, false};
        uint64_t Count_[DOUBLE] {UNSIGNED_ZERO, UNSIGNED_ZERO};
        uint64_t Capacity_{UNSIGNED_ZERO};
        std::optional<sycl::event> Ready_[DOUBLE]{};
        std::optional<sycl::event> LastUse_[DOUBLE]{};
        unsigned Active_{UNSIGNED_ZERO};
        bool Published_{false};
        uint64_t Revision_{UNSIGNED_ZERO};

        bool Completed_(const std::optional<sycl::event>& e)
        {
            return !e ||
                e->get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
        }

        void Relese_() noexcept
        {
            if (!Runtime_) return;
            for (size_t i = 0u; i < DOUBLE; i++) 
            {
                if (Ready_[i]) (void) Runtime_->Wait(*Ready_[i]);
                if (LastUse_[i]) (void) Runtime_->Wait(*LastUse_[i]);
                try 
                {
                    if (Bank_[i]) sycl::free(Bank_[i], Runtime_->Contex);
                }
                catch (...)
                {
                }

                try 
                {
                    if (Stage_[i]) 
                    {
                        if (HostUsm_[i]) 
                        {
                            sycl::free(Stage_[i], Runtime_->Contex);
                        }
                        else 
                        {
                            ::operator delete(
                                Stage_[i],
                                std::align_val_t{std::max(alignof(Record), size_t{ADS::APC_CACHELINE_SIZE})}
                            );
                        }
                    }
                }
                catch (...)
                {
                }
                Bank_[i] = nullptr;
                Stage_[i] = nullptr;
            }
            Capacity_ = UNSIGNED_ZERO;
        }
    public:
        explicit DevicePlanDoubleBuffer(std::shared_ptr<Runtime> runtime) noexcept:
            Runtime_(std::move(runtime))
        {}
        DevicePlanDoubleBuffer(const DevicePlanDoubleBuffer&) = delete;
        DevicePlanDoubleBuffer& operator = (const DevicePlanDoubleBuffer&) = delete;
        ~DevicePlanDoubleBuffer() noexcept
        {
            Relese_();
        }
        
        bool Allocate(uint64_t capacity) noexcept
        {
            if (
                capacity == UNSIGNED_ZERO ||
                !Runtime_ ||
                Capacity_ != UNSIGNED_ZERO ||
                !Memory::ByteCount<Record>(capacity)
            ) 
            {
                return false;
            }

            const uint64_t alignment = std::max(alignof(Record), ADS::APC_CACHELINE_SIZE);
            try 
            {
                const bool host_usm = Runtime_->Gpu.has(sycl::aspect::usm_host_allocations);
                for (size_t i = 0; i < DOUBLE; i++) 
                {
                    Bank_[i] = sycl::aligned_alloc<Record>(alignment, capacity, Runtime_->ComputeQueue);
                    if (!Bank_[i])
                    {
                        Relese_();
                        return false;
                    }
                    if (host_usm) 
                    {
                        Stage_[i] = sycl::aligned_alloc_host(alignment, capacity * sizeof(Record), Runtime_->Contex);
                        HostUsm_[i] = Stage_[i] != nullptr;
                    }
                    if (!Stage_[i]) 
                    {
                        Stage_[i] = ::operator new(capacity * sizeof(Record), std::align_val_t{alignment});
                    }
                }
                Capacity_ = capacity;
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        PlanUpdateResult TryPublish(std::span<const Record> records) noexcept
        {
            if (
                Capacity_ == UNSIGNED_ZERO ||
                records.size() > Capacity_ ||
                (records.size() && !records.data()) ||
                Revision_ == FABRIC_CELL_SENTINAL
            ) 
            {
                return PlanUpdateResult::INVALID;
            }
            if (!Runtime_->Healthy()) 
            {
                return PlanUpdateResult::FAILED;
            }

            const unsigned next = Published_ ? (Active_ ^1u) : UNSIGNED_ZERO;
            try 
            {
                if (!Completed_(Ready_[next]) || !Completed_(LastUse_[next])) 
                {
                    return PlanUpdateResult::BUSY;
                }
                if (
                    (Ready_[next] && !Runtime_->Wait(*Ready_[next])) ||
                    (LastUse_[next && Runtime_->Wait(*LastUse_[next])])
                ) 
                {
                    return PlanUpdateResult::FAILED;
                }
                if (records.size()) 
                {
                    std::memcpy(Stage_[next], records.data(), records.size_bytes());
                }
                auto event = Runtime_->ControlQueue.submit(
                    [&](sycl::handler& h)
                    {
                        if (Ready_[next]) h.depends_on(*Ready_[next]);
                        if (LastUse_[next]) h.depends_on(*LastUse_[next]);
                        if (records.size()) h.memcpy(Bank_[next], Stage_[next], records.size_bytes());
                        else h.single_task([]{});
                    }
                );
                Ready_[next] = event;
                Count_[next] = records.size();
                Active_ = next;
                Published_ = true;
                ++Revision_;
                return PlanUpdateResult::ACCEPTED;
            }
            catch (...)
            {
                return PlanUpdateResult::FAILED;
            }
        }

        uint64_t Revision() const noexcept {return Revision_;}

        template<class OP>
        std::optional<sycl::event> Submit(
            uint64_t work_items,
            OP operation,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>);
            if (!Published_ || work_items == UNSIGNED_ZERO || !Runtime_->Healthy()) 
                return std::nullopt;
            const unsigned bank = Active_;
            const DevicePlanView<Record> view{Bank_[bank], Count_[bank], Revision_};
            try 
            {
                auto event = Runtime_->ComputeQueue.submit(
                    [&](sycl::handler& h)
                    {
                        h.depends_on(*Ready_[bank]);
                        for (const auto& e: dependencies) 
                        {
                            h.depends_on(e);
                        }
                        h.parallel_for(
                            sycl::range<1>{work_items}, 
                            [=](sycl::id<1> i)
                            {
                                operation(static_cast<uint64_t>(i[0], view));
                            }
                        );
                    }
                );
                LastUse_[bank] = event;
                return event;
            }
            catch (...)
            {
                Runtime_->MarkFaild();
                return std::nullopt;
            }
        }

        template<class DT, class OP>
        std::optional<sycl::event> Submit(
            uint64_t work_items,
            ResidentDeviceRegion<DT>& numeric,
            OP operation,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<OP>);
            if (numeric.Runtime_ != Runtime_ || !numeric.Data_) 
                return std::nullopt;
            try 
            {
                std::vector<sycl::event> deps(dependencies.begin(), dependencies.end());
                if (numeric.Tail_) 
                {
                    deps.push_back(*numeric.Tail_);
                }
                const auto data = numeric.Data_;
                const auto count = numeric.Count_;
                auto event = Submit(
                    work_items, 
                    [=](uint64_t i, DevicePlanView<DT>view)
                    {
                        operation(i, view, data, count);
                    },
                    std::span<const sycl::event>{deps}
                );
                if (event) numeric.Tail_ = event;
                return event;
            }
            catch(...)
            {
                Runtime_->MarkFaild();
                return std::nullopt;
            }
        }
        
    };
#endif

}