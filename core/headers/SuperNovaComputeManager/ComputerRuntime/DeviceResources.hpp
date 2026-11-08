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
        std::shared_ptr<RuntimeAcpp> Runtime_;
        DT* Data_{nullptr};
        uint64_t Count_{0};
        std::optional<sycl::event> Tail_{};

        std::optional<sycl::event> Transfer_(
            DT* host,
            uint64_t count,
            uint64_t offset,
            bool upload,
            std::span<const sycl::event> dependencies 
        ) noexcept;

    public:
        explicit ResidentDeviceRegion(std::shared_ptr<RuntimeAcpp> runtime) noexcept :
            Runtime_(std::move(runtime))
        {}
        ResidentDeviceRegion(const ResidentDeviceRegion&) = delete;
        ResidentDeviceRegion& operator = (const ResidentDeviceRegion&) = delete;
        ~ResidentDeviceRegion() noexcept
        {
            if (!Data_) return;
            if (!Runtime_->Drain()) std::terminate();
            try 
            {
                sycl::free(Data_, Runtime_->Contex);
            }
            catch (...)
            {
            }
        }

        bool Allocate(uint64_t count) noexcept;

        uint64_t Size() noexcept {return Count_;}
        DT* Data() const noexcept {return Data_;}

        std::optional<sycl::event> PrepareStage(
            std::span<const DT> source, 
            uint64_t offset = UNSIGNED_ZERO,
            std::span<const sycl::event> dependencies = {}
        ) noexcept
        {
            return Transfer_(
                const_cast<DT*>(source.data()), source.size(), offset, true, dependencies
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
        std::shared_ptr<RuntimeAcpp> Runtime_;
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

        void Relese_() noexcept;
    public:
        explicit DevicePlanDoubleBuffer(std::shared_ptr<RuntimeAcpp> runtime) noexcept:
            Runtime_(std::move(runtime))
        {}
        DevicePlanDoubleBuffer(const DevicePlanDoubleBuffer&) = delete;
        DevicePlanDoubleBuffer& operator = (const DevicePlanDoubleBuffer&) = delete;
        ~DevicePlanDoubleBuffer() noexcept
        {
            Relese_();
        }
        
        bool Allocate(uint64_t capacity) noexcept;

        PlanUpdateResult TryPublish(std::span<const Record> records) noexcept;

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
                                operation(static_cast<uint64_t>(i[0]), view);
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
                    [=](uint64_t i, DevicePlanView<Record> view)
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

#include "../templates/SuperNovaComputeManager/DeviceResources.tpp"
