#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{
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

    std::optional<ResidencyInfo> NumericResidency::InspectResidency(const RR& ref) const noexcept
    {
        for (const std::unique_ptr<VirtualResidency>& vrpa : Records_) 
        {
            if (vrpa->LogicalRef == ref) return vrpa->Version;
        }
        return {};
    }

}