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
            static_cast<T*>(this->Host_),
            UnitCount_()
        });

        if (!evnt.has_value() || !Allocation_->Wait()) 
        {
            return false;
        }
        stats.UploadedBytes += this->LogicalRef_.ByteCount;
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
            static_cast<T*>(this->Host_),
            UnitCount_()
        });

        if (!evnt.has_value() || !Allocation_->Wait()) 
        {
            return false;
        }

        stats.DownloadedfBytes += this->LogicalRef_.ByteCount;
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
            used += rp->LogicalRef_.ByteCount;
        }
        return used <= CacheBudget_ - bytes;
    }


    // template<class T>
    // std::optional<NR<const T>> NumericResidency::RegisterDisk(
    //     Device& device,
    //     const CheckpointFormat::FabricCheckpoint& image,
    //     const CheckpointFormat::RegionRef& ref
    // ) noexcept
    // {
        
    // }
}