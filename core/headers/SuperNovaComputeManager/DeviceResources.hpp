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
            for (uint64_t i = 0u; i < DOUBLE; i++) 
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
        
        
    };


#endif

}