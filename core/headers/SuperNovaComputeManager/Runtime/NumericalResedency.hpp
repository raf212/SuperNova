#pragma once

#include "DeviceResources.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    struct VirtualResidency
    {
        CheckpointFormat::RegionRef LogicalRef{};
        void* Host{};
        ResidencyInfo Version{};
        uint64_t Stamp{};
        uint32_t ActiveWidth{};
        bool Pending{false};
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

        virtual size_t UnitCount() const noexcept = 0;
    };

    template<class T>
    struct T_NumericResidency final : VirtualResidency
    {
        std::optional<SNTOComputerBridge<T>> PinT{};
        std::vector<T> DiskStorage{};

    #if defined (SN_ACPP_ENABLED)
        std::unique_ptr<ResidentDeviceRegion<T>> Allocation{};
    #endif
        
        bool EnsureDevice(Device& device, TransferStatics& stats) noexcept;

        bool Upload(TransferStatics& stats) noexcept;

        bool Download(TransferStatics& stats) noexcept;

        void* DevicePtr() noexcept;

        size_t UnitCount() const noexcept
        {
            return this->LogicalRef.ByteCount / sizeof(T);
        }

        void Unpin() noexcept override 
        {
            PinT.reset();
            this->Pinned = false;
        }
    };


    template<class T>
    struct WindowInfo
    {
        T* DataPtr{nullptr};
        uint64_t Stride{};

        explicit operator bool () const noexcept
        {
            return DataPtr != nullptr;
        }
    };


    class NumericResidency final
    {
        template<class T>
        using NR = Numerics::NumericalRegionRef<T>;
        using RR = CheckpointFormat::RegionRef;
        ///vrp means VirtualResidency pointer of an entry

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

        template<class T>
        WindowInfo<T> ResolveWindow_(
            NR<T> ref,
            uint32_t first_row,
            uint32_t rows,
            uint64_t first_column,
            uint64_t columns
        ) noexcept;

        template<class T>
        WindowInfo<VirtualResidency> FindActiveRange_(
            const T* host,
            uint64_t count
        ) const noexcept;

    public: 
        bool Active() const noexcept {return CurEpoch_ != UNSIGNED_ZERO;}

        bool Faild() const noexcept {return Failed_;}

        bool UnpublishedWrites() const noexcept {return UnpublishedWrites_;}

        BackendC Backend() const noexcept {return Backend_;}

        void SetFail() noexcept {Failed_ = true;}

        bool Empty() const noexcept {return Records_.empty();}

        TransferStatics Statistics() const noexcept {return Stats_;}

        ///@brief Current implimentation dosent chack device memory size
        bool SetCacheBudget(uint64_t bytes) noexcept;

        void CommitMappings() noexcept;

        bool PublishResidency(Device& device, uint64_t stamp) noexcept;

        bool Finish(Device& device, bool publish, uint64_t stamp) noexcept;

        bool Dirty() const noexcept;

        bool Discard(Device& device) noexcept;

        bool Begin(BackendC backend) noexcept;

        bool EvictRegion(Device& device, const RR& ref) noexcept;
        
        std::optional<ResidencyInfo> InspectResidency(const RR& ref) const noexcept;

        template<class T>
        std::optional<NR<T>> RegisterRegion(
            Device& device,
            SNTOComputerBridge<T>&& region,
            const FabricRuntimeBinding& binding
        ) noexcept;

        template<class T>
        std::optional<NR<const T>> RegisterDisk(
            Device& device,
            const CheckpointFormat::FabricCheckpoint& image,
            const RR& ref
        ) noexcept;

        template<class T>
        VirtualResidency* Find(NR<T> ref) noexcept;

        template<class T>
        std::span<T> Row(NR<T> ref, uint32_t row) noexcept;

        template<class T>
        T* Map(
            T* host, uint64_t count,
            OpenAccess access = std::is_const_v<T>() ? 
                OpenAccess::READ : OpenAccess::READ_WRITE
        ) noexcept;

        template<class T>
        std::optional<RR> Reference(NR<T> ref) noexcept
        {
            VirtualResidency* vr = Find(ref);
            return vr ? std::optional<RR>(vr->LogicalRef) :
                std::nullopt;
        }

        template<class T>
        WindowInfo<T> HostMatrix(Numerics::NumericalMetrixOperand<T> matrix) noexcept
        {
            return ResolveWindow_(
                matrix.Region, matrix.FirstRow, matrix.Rows,
                matrix.FirstColumn, matrix.Columns
            );
        }

        template<class T>
        T* HostOperand(Numerics::NumericOperand<T> op) noexcept
        {
            return ResolveWindow_(
                op.Region, op.FirstRow, 1,
                op.Offset, op.Count
            ).DataPtr;
        }

        template<class T>
        bool Contains(const T* host, uint64_t count) const noexcept
        {
            return static_cast<bool>(FindActiveRange_(host, count));
        }

        void CancelMapping() noexcept
        {
            for (const size_t idx : Active_) 
            {
                Records_[idx]->Pending = false;
            }
        }

        bool Evict(Device& device) noexcept
        {
            return !Active() && Discard(device);
        }
    };
}

#include "../../../templates/SuperNovaComputeManager/NumericalResedency.tpp"
