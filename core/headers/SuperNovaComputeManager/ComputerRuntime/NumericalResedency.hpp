#pragma once

#include "DeviceResources.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    struct VirtualResidency
    {
        CheckpointFormat::RegionRef LogicalRef_{};
        void* Host_{};
        ResidencyInfo Version_{};
        uint64_t Stamp_{};
        uint32_t ActiveWidth_{};
        bool Pending_{false};
        bool Pinned{false};
        bool ReadOnly{false};
        uint64_t DiskRevision{};
        uint64_t DiskFingerPrint{};

        virtual ~VirtualResidency() = default;
        virtual void Unpin() noexcept = 0;

        virtual bool EnsureDevice(Device&, TransferStatics&) noexcept = 0;

        virtual bool Upload(TransferStatics&) noexcept = 0;

        virtual bool Download(TransferStatics&) noexcept = 0;

        virtual void* DevicePtr() noexcept = 0;

    protected:
        virtual size_t UnitCount_() const noexcept = 0;
    };

    template<class T>
    struct T_NumericResidency final : VirtualResidency
    {
        std::optional<SNTOComputerBridge<T>> Pin_{};
        std::vector<T> DiskStorage_{};

    #if defined (SN_ACPP_ENABLED)
        std::unique_ptr<ResidentDeviceRegion<T>> Allocation_{};
    #endif
        
        bool EnsureDevice(Device& device, TransferStatics& stats) noexcept;

        bool Upload(TransferStatics& stats) noexcept;

        bool Download(TransferStatics& stats) noexcept;

        void* DevicePtr() noexcept;

    protected:
        size_t UnitCount_() const noexcept
        {
            return this->LogicalRef_.ByteCount / sizeof(T);
        }
    };


    class NumericResidency final
    {
        template<class T>
        using NR = Numerics::NumericalRegionRef<T>;

        std::vector<std::unique_ptr<VirtualResidency>> Records_{};
        std::vector<size_t> Active_{};
        uint64_t CacheBudget_{PER_REGION_DEVICE_CACHE_IN_MB * MB_TO_BYTE_MULTIPLYER};
        inline static std::atomic<uint64_t> NextEpoch_{1};
        std::atomic<uint64_t> CurEpoch_{0};
        BackendC Backend_{BackendC::CPU};
        bool Failed_{false};
        bool UnpublishedWrites_{false};
        TransferStatics Stats_{};

        bool Reserve_(uint64_t bytes) noexcept;

    public: 
        bool Active() const noexcept {return CurEpoch_ != UNSIGNED_ZERO;}

        bool Faild() const noexcept {return Failed_;}

        bool UnpublishedWrites() const noexcept {return UnpublishedWrites_;}

        BackendC Backend() const noexcept {return Backend_;}

        void SetFail() noexcept {Failed_ = true;}

        bool Empty() const noexcept {return Records_.empty();}

        TransferStatics Statistics() const noexcept {return Stats_;}

        // UNDEFINED
        bool SetCacheBudget(uint64_t bytes) noexcept;

        bool Begin(BackendC backend) noexcept;

        template<class T>
        std::optional<NR<const T>> RegisterDisk(
            Device& device,
            const CheckpointFormat::FabricCheckpoint& image,
            const CheckpointFormat::RegionRef& ref
        ) noexcept;

    };
}