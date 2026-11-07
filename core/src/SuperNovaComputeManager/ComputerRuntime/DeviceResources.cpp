#include <SuperNova>
#if defined (SN_ACPP_ENABLED)
namespace BidirectionalInMemGraph::ComputerDetail 
{
    template<class DT>
    std::optional<sycl::event> ResidentDeviceRegion<DT>::Transfer_(
        DT* host,
        uint64_t count,
        uint64_t offset,
        bool upload,
        std::span<const sycl::event> dependencies 
    ) noexcept
    {
        if (
            !Runtime_ || !Data_ || !host || count == UNSIGNED_ZERO ||
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

    template<class DT>
    bool ResidentDeviceRegion<DT>::Allocate(uint64_t count) noexcept
    {
        if (Data_ || !Runtime_ || count == UNSIGNED_ZERO || !Memory::ByteCount<DT>(count).has_value()) 
        {
            return false;
        }

        try 
        {
            Data_ = sycl::aligned_alloc_device<DT>(
                std::max<size_t>(alignof(DT), ADS::CACELINE_SIZE_IN_BYTES),
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

    template<class Record>
    void DevicePlanDoubleBuffer<Record>::Relese_() noexcept
    {
        if (!Runtime_) return;
        if (!Runtime_->Drain()) std::terminate();
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
                            std::align_val_t{std::max(alignof(Record), size_t{ADS::CACELINE_SIZE_IN_BYTES})}
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


    template<class Record>
    bool DevicePlanDoubleBuffer<Record>::Allocate(uint64_t capacity) noexcept
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

        const size_t alignment = std::max<size_t>(alignof(Record), ADS::CACELINE_SIZE_IN_BYTES);
        try 
        {
            const bool host_usm = Runtime_->Gpu.has(sycl::aspect::usm_host_allocations);
            for (size_t i = 0; i < DOUBLE; i++) 
            {
                Bank_[i] = sycl::aligned_alloc_device<Record>(alignment, capacity, Runtime_->ComputeQueue);
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
            Relese_();
            return false;
        }
    }

    template<class Record>
    PlanUpdateResult DevicePlanDoubleBuffer<Record>::TryPublish(std::span<const Record> records) noexcept
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
                (LastUse_[next] && !Runtime_->Wait(*LastUse_[next]))
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
}
#endif