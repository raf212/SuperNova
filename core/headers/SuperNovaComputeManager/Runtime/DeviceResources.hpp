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
        ) noexcept;

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

        enum class BankState
        {
            Empty,
            Pending,
            Ready
        };

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
        uint64_t Revision_{UNSIGNED_ZERO};
        BankState ActiveStateBank_{BankState::Empty};
        std::optional<unsigned> Staged_;
        
        bool Completed_(const std::optional<sycl::event>& e)
        {
            return !e ||
                e->get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
        }

        void Relese_() noexcept;

        PlanUpdateResult StageRecords_(std::span<const Record> records, bool require_idle_bank) noexcept;
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

        PlanUpdateResult TryStage(std::span<const Record> records) noexcept {return StageRecords_(records, false);}

        void CancelStage() noexcept {Staged_.reset();}

        // Call immediately after enqueueing an active-bank consumer on the associated in-order compute queue.
        void RecordUse(const sycl::event& tail) { LastUse_[Active_] = tail; }

        PlanUpdateResult TryActive(uint64_t revision, bool wait = false) noexcept;

        uint64_t Revision() const noexcept {return Revision_;}

        template<class OP>
        std::optional<sycl::event> Submit(
            uint64_t work_items,
            OP operation,
            std::span<const sycl::event> dependencies = {}
        ) noexcept;

        template<class DT, class OP>
        std::optional<sycl::event> Submit(
            uint64_t work_items,
            ResidentDeviceRegion<DT>& numeric,
            OP operation,
            std::span<const sycl::event> dependencies = {}
        ) noexcept;

        DevicePlanView<Record> ActiveView() const noexcept
        {
            return ActiveStateBank_ == BankState::Ready ?
                DevicePlanView<Record>{Bank_[Active_], Count_[Active_], Revision_} :
                DevicePlanView<Record>{};
        }
        
    };
#endif

}

#include "../../../templates/SuperNovaComputeManager/Runtime/DeviceResources.tpp"
