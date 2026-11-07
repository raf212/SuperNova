#pragma once

#include "DeviceResources.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    struct NumericResidencyVirtual
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

        virtual ~NumericResidencyVirtual() = default;
        virtual void Unpin() noexcept = 0;

        virtual bool EnsureDevice(Device&, TransferStatics&) noexcept = 0;

        virtual bool Upload(TransferStatics&) noexcept = 0;

        virtual bool Download(TransferStatics&) noexcept = 0;

        virtual void* DevicePtr() noexcept = 0;
    };
}