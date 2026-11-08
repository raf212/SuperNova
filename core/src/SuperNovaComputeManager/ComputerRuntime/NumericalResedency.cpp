#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{
    template<class T>
    bool T_NumericResidency<T>::EnsureDevice(Device& device, TransferStatics& stats) noexcept
    {
#if defined (SN_ACPP_ENABLED)
        if (Allocation_) return true;
        try 
        {
            auto alloc_u_ptr = std::make_unique<ResidentDeviceRegion<T>>(device.AcquireRuntime());
            if (!alloc_u_ptr->Allocate(UnitCount_())) 
            {
                return false;
            }
            Allocation_ = std::move(alloc_u_ptr);
            ++stats.Allocation;
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        (void) device; (void) stats;
        return false;
#endif
    }

    template<class T>
    bool T_NumericResidency<T>::Upload(TransferStatics& stats) noexcept
    {
#if defined (SN_ACPP_ENABLED)
        std::optional<sycl::event> evnt = Allocation_->Download({
            static_cast<T*>(this->Host),
            UnitCount_()
        });

        if (!evnt.has_value() || !Allocation_->Wait()) 
        {
            return false;
        }
        stats.UploadedBytes += this->LogicalRef.ByteCount;
        return true;
#else
        (void) stats;
        return false;
#endif
    }

    template<class T>
    bool T_NumericResidency<T>::Download(TransferStatics& stats) noexcept
    {
#if defined (SN_ACPP_ENABLED)
        std::optional<sycl::event> evnt = Allocation_->Download({
            static_cast<T*>(this->Host),
            UnitCount_()
        });

        if (!evnt.has_value() || !Allocation_->Wait()) 
        {
            return false;
        }

        stats.DownloadedfBytes += this->LogicalRef.ByteCount;
        return true;
#else
        (void) stats;
        return false;
#endif
    }

    template<class T>
    void* T_NumericResidency<T>::DevicePtr() noexcept
    {
#if defined (SN_ACPP_ENABLED)
        return  Allocation_ ? Allocation_->Data() : nullptr;
#else
        return nullptr;
#endif
    }
}

namespace BidirectionalInMemGraph::ComputerDetail 
{
    namespace CMF = CheckpointFormat;

    template<class T>
    using NR = Numerics::NumericalRegionRef<T>;

    bool NumericResidency::Reserve_(uint64_t bytes) noexcept
    {
        if (bytes > CacheBudget_) return false;
        uint64_t used = UNSIGNED_ZERO;

        for (std::unique_ptr<VirtualResidency>& rp : Records_) 
        {
            used += rp->LogicalRef.ByteCount;
        }
        return used <= CacheBudget_ - bytes;
    }

    bool NumericResidency::SetCacheBudget(uint64_t bytes) noexcept
    {
        if (Active() || UnpublishedWrites_ ||
            !Empty() || bytes == UNSIGNED_ZERO
        ) 
        {
            return false;
        }
        CacheBudget_ = bytes;
        return true;
    }

    bool NumericResidency::Begin(BackendC backend) noexcept
    {
        if (Active() || Failed_) return false;
        uint64_t id = NextEpoch_.load(std::memory_order_acquire);
        do 
        {
            if (id == UINT64_MAX) return false;
        }
        while (!NextEpoch_.compare_exchange_strong(id, id +1, std::memory_order_acq_rel, std::memory_order_release));
        CurEpoch_.store(id, std::memory_order_release);
        Backend_ = backend;
        Active_.clear();
        return true;
    }

    void NumericResidency::CommitMappings() noexcept
    {
        for (const size_t idx : Active_) 
        {
            VirtualResidency& vr = *Records_[idx];
            if (!vr.Pending) continue;

            ++vr.Version.CommittedRevision;
            UnpublishedWrites_ = true;
            vr.Version.HostRevision = vr.Version.CommittedRevision;
            vr.Version.Location = (Backend_ == BackendC::GPU) ? 
                MemoryLocation::GPU : MemoryLocation::HOST;
            vr.Pending = false;
        }
    }

    bool NumericResidency::PublishResidency(Device& device, uint64_t stamp) noexcept
    {
        if (Failed_ || !device.Sync()) return false;
        for (std::unique_ptr<VirtualResidency>& rp : Records_) 
        {
            VirtualResidency& vr = *rp;
            if (!vr.Pinned) continue;

            if (
                vr.Version.Location == MemoryLocation::GPU &&
                vr.Version.HostRevision != vr.Version.CommittedRevision
            ) 
            {
                if (!vr.Download(Stats_)) 
                {
                    Failed_ = true;
                    return false;
                }
                vr.Version.HostRevision = vr.Version.CommittedRevision;
            }
            vr.Version.Location = MemoryLocation::HOST;
            vr.Stamp = stamp;
            vr.Unpin();
        }
        UnpublishedWrites_ = false;
        return true;
    }

    bool NumericResidency::Finish(Device& device, bool publish, uint64_t stamp) noexcept
    {
        if (!Active()) return true;
        if (Failed_ || !device.Sync()) 
        {
            Failed_ = true;
            return false;
        }

        if (publish && !PublishResidency(device, stamp)) 
        {
            return false;
        }
        Active_.clear();
        CurEpoch_.store(UNSIGNED_ZERO, std::memory_order_release);
        return true;
    }


    bool NumericResidency::Dirty() const noexcept
    {
        if (UnpublishedWrites_) return true;
        for (const std::unique_ptr<VirtualResidency>& rp : Records_) 
        {
            if (rp->Pending) return true;
        }
        return false;
    }

    bool NumericResidency::Discard(Device& device) noexcept
    {
        if (!device.Drain()) return false;
        Records_.clear();
        Active_.clear();
        CurEpoch_.store(UNSIGNED_ZERO, std::memory_order_release);
        Failed_ = false;
        return true;
    }

    bool NumericResidency::EvictRegion(Device& device, const CheckpointFormat::RegionRef& ref) noexcept
    {
        if (Active() || !device.Drain()) return false;
        for (auto i = Records_.begin(); i != Records_.end(); i++) 
        {
            if ((*i)->LogicalRef != ref) continue;
            if ((*i)->Pinned || (*i)->Version.Location == MemoryLocation::GPU)
            {
                return false;
            }
            Records_.erase(i);
            return true;
        }
        return true;
    }


    template<class T>
    std::optional<NR<const T>> NumericResidency::RegisterDisk(
        Device& device,
        const CheckpointFormat::FabricCheckpoint& image,
        const CheckpointFormat::RegionRef& ref
    ) noexcept
    {
        static constexpr unsigned Init_HC_Revision = 1u;
        static constexpr unsigned Init_Device_Revision = UNSIGNED_ZERO;
        
        std::optional<CDV::CDT> type = CDV::DataTypeOfSN<T>();
        if (
            !Active() || Failed_ || !type.has_value() ||
            type.value() != ref.Type ||
            ref.ByteCount % sizeof(T) != UNSIGNED_ZERO
        ) 
        {
            return std::nullopt;
        }

        try 
        {
            size_t idx = 0;
            for (; idx < Records_.size(); idx++) 
            {
                if (
                    Records_[idx]->ReadOnly && 
                    Records_[idx]->LogicalRef == ref &&
                    Records_[idx]->DiskRevision == image.Revision() &&
                    Records_[idx]->DiskFingerPrint == image.Fingerprint()
                ) 
                {
                    break;
                }
            }

            for (const size_t a : Active_) 
            {
                if (a == idx) return std::nullopt;
            }

            if (idx == Records_.size()) 
            {
                if (!Reserve_(ref.ByteCount)) return std::nullopt;

                std::unique_ptr<T_NumericResidency<T>> entry = std::make_unique<T_NumericResidency<T>>();
                entry->LogicalRef = ref;
                entry->ReadOnly = true;
                entry->DiskRevision = image.Revision();
                entry->DiskFingerPrint = image.Fingerprint();
                entry->DiskStorage_.resize(entry->UnitCount_());

                if (!image.ReadExtent(ref, UNSIGNED_ZERO, std::as_writable_bytes(
                    std::span{entry->DiskStorage_}
                ))) 
                {
                    return std::nullopt;
                }

                entry->Host = entry->DiskStorage_.data();
                entry->Version = {
                    Init_HC_Revision, Init_HC_Revision, 
                    Init_Device_Revision, MemoryLocation::HOST
                };
                entry->ActiveWidth = ref.Width;
                Records_.pop_back(std::move(entry));
            }
            else 
            {
                if (!image.Validate(ref)) return std::nullopt;
                ++Stats_.CacheHits;
            }

            VirtualResidency& vr = *Records_[idx];
            vr.Pinned = true;
            if (Backend_ == BackendC::GPU) 
            {
                if (
                    !vr.EnsureDevice(device, Stats_) ||
                    (
                        vr.Version.DeviceRevision != vr.Version.CommittedRevision &&
                        !vr.Upload(Stats_)
                    )
                ) 
                {
                    Failed_ = true;
                    return std::nullopt;
                }
                vr.Version.DeviceRevision = vr.Version.CommittedRevision;
            }
            Active_.push_back(idx);

            return Numerics::NumericalRegionRef<class T>{
                CurEpoch_.load(std::memory_order_acquire),
                (Active_.size() - 1)
            };
        }
        catch (...)
        {
            Failed_ = true;
            return std::nullopt;
        }
    }
}