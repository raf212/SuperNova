#pragma once

namespace BidirectionalInMemGraph::ComputerDetail 
{
    namespace CMF = CheckpointFormat;
    static constexpr unsigned Init_HC_Revision = 1u;
    static constexpr unsigned Init_Device_Revision = UNSIGNED_ZERO;
    template<class T>
    using NR = Numerics::NumericalRegionRef<T>;

    template<class T>
    bool T_NumericResidency<T>::EnsureDevice(Device& device, TransferStatics& stats) noexcept
    {
#if defined (SN_ACPP_ENABLED)
        if (Allocation) return true;
        try 
        {
            auto alloc_u_ptr = std::make_unique<ResidentDeviceRegion<T>>(device.AcquireRuntime());
            if (!alloc_u_ptr->Allocate(UnitCount())) 
            {
                return false;
            }
            Allocation = std::move(alloc_u_ptr);
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
        std::optional<sycl::event> evnt = Allocation->Download({
            static_cast<T*>(this->Host),
            UnitCount()
        });

        if (!evnt.has_value() || !Allocation->Wait()) 
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
        std::optional<sycl::event> evnt = Allocation->Download({
            static_cast<T*>(this->Host),
            UnitCount()
        });

        if (!evnt.has_value() || !Allocation->Wait()) 
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
        return  Allocation ? Allocation->Data() : nullptr;
#else
        return nullptr;
#endif
    }

    template<class T>
    std::optional<NR<const T>> NumericResidency::RegisterRegion(
        Device& device,
        SNTOComputerBridge<T>&& region,
        const FabricRuntimeBinding& binding
    ) noexcept
    {
        static_assert(ComputerScalarV<T> && !std::is_const_v<T>);

        std::optional<std::span<T>> span = region.RawSpan();
        std::optional<CDV::CDT> type = CDV::DataTypeOfSN<T>();

        if (!Active() || Failed_ || !span.has_value() || !type.has_value() || span->empty()) 
        {
            return std::nullopt;
        }

        const std::optional<uint64_t> logical_offset = binding.LogicalByteOffset(
            span->data(), span->size_bytes()
        );
        if (!logical_offset) return std::nullopt;

        const uint64_t offset = *logical_offset;
        CMF::RegionRef key
        {
            binding.Id,
            offset,
            span->size_bytes(),
            region.Slot(),
            region.Generation(),
            region.RecordOrdinal(),
            region.Height(),
            region.CapacityWidth(),
            region.Column(),
            *type
        };

        for (const size_t idx : Active_) 
        {
            VirtualResidency& vr = *Records_[idx];
            if (Memory::RangesOverlap(
                vr.Host, vr.LogicalRef.ByteCount, 
                span->data(), span->size_bytes()
            )) 
            {
                return std::nullopt;
            }
        }

        try 
        {
            size_t idx = UNSIGNED_ZERO;
            for (; idx < Records_.size(); idx++)
            {
                if (
                    !Records_[idx]->ReadOnly &&
                    Records_[idx]->LogicalRef == key
                ) 
                {
                    break;
                }
            }

            if (idx == Records_.size()) 
            {
                if (!Reserve_(key.ByteCount)) return std::nullopt;
                std::unique_ptr<T_NumericResidency<T>> rp = std::make_unique<T_NumericResidency<T>>();
                rp->LogicalRef = key;
                rp->Version = {
                    Init_HC_Revision,
                    Init_HC_Revision,
                    Init_Device_Revision,
                    MemoryLocation::HOST
                };
                Records_.push_back(std::move(rp));
            }
            else 
            {
                ++Stats_.CacheHits;
            }

            T_NumericResidency<T>* entry = static_cast<T_NumericResidency<T>*>(Records_[idx].get());
            entry->Host = span->data();
            entry->ActiveWidth = region.ActiveWidth();
            if (
                !entry->Pinned &&
                entry->Stamp != binding.Gate->HostStamp()
            ) 
            {
                if (entry->Version.Location == MemoryLocation::GPU) return std::nullopt;
                ++entry->Version.CommittedRevision;
                entry->Version.HostRevision = entry->Version.CommittedRevision;
            }

            entry->PinT.emplace(std::move(region));
            entry->Pinned = true;
            if (Backend_ == BackendC::GPU) 
            {
                if (!entry->EnsureDevice(device, Stats_)) 
                {
                    Failed_ = true;
                    return std::nullopt;
                }
            }
            else if (entry->Version.HostRevision != entry->Version.CommittedRevision) 
            {
                if (!entry->Download(Stats_)) 
                {
                    Failed_ = true;
                    return std::nullopt;
                }

                entry->Version.HostRevision = entry->Version.CommittedRevision;
                entry->Version.Location = MemoryLocation::HOST;
            }

            entry->Stamp = binding.Gate->HostStamp();
            Active_.push_back(idx);
            return Numerics::NumericalRegionRef<T>{
                CurEpoch_.load(std::memory_order_acquire),
                (Active_.size() - 1)
            };

        } catch (...) {
            Failed_ = true;
            return std::nullopt;
        }

    }

    template<class T>
    std::optional<NR<const T>> NumericResidency::RegisterDisk(
        Device& device,
        const CheckpointFormat::FabricCheckpoint& image,
        const CheckpointFormat::RegionRef& ref
    ) noexcept
    {   
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
                entry->DiskStorage.resize(entry->UnitCount());

                if (!image.ReadExtent(ref, UNSIGNED_ZERO, std::as_writable_bytes(
                    std::span{entry->DiskStorage}
                ))) 
                {
                    return std::nullopt;
                }

                entry->Host = entry->DiskStorage.data();
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

    template<class T>
    VirtualResidency* NumericResidency::Find(NR<T> ref) noexcept
    {
        if (
            !Active() || Failed_ || ref.Epoch != CurEpoch_ ||
            ref.Index >= Active_.size()
        )
        {
            return nullptr;
        }

        VirtualResidency* vrp = Records_[Active_[ref.Index]].get();
        std::optional<CDV::CDT> type = CDV::DataTypeOfSN<T>();

        return type.has_value() && 
            type.value() == vrp->LogicalRef.Type ? vrp : nullptr;
    }

    template<class T>
    WindowInfo<T> NumericResidency::ResolveWindow_(
        NR<T> ref,
        uint32_t first_row,
        uint32_t rows,
        uint64_t first_column,
        uint64_t columns
    ) noexcept
    {
        VirtualResidency* vrp = Find(ref);
        if (!vrp || rows == UNSIGNED_ZERO || columns == UNSIGNED_ZERO) 
        {
            return WindowInfo<T>{};
        }

        if constexpr (std::is_const_v<T>) 
        {
            if (vrp->ReadOnly) return WindowInfo<T>{};
        }

        if (
            first_row >= vrp->LogicalRef.Height ||
            rows > (vrp->LogicalRef.Height - first_row) ||
            first_column > vrp->ActiveWidth ||
            columns > (vrp->ActiveWidth - first_column)
        ) 
        {
            return WindowInfo<T>{};
        }

        const uint64_t offset = (uint64_t{first_row} * vrp->LogicalRef.Width) + first_column;

        return {
            static_cast<T*>(vrp->Host) + offset,
            vrp->LogicalRef.Width
        };
    }

    template<class T>
    std::span<T> NumericResidency::Row(NR<T> ref, uint32_t row) noexcept
    {
        VirtualResidency* vrp = Find(ref);
        if (!vrp) return {};

        WindowInfo<T> info = ResolveWindow_(ref, row, 1, UNSIGNED_ZERO, vrp->ActiveWidth);
        
        return info ? std::span<T>{info.DataPtr, vrp->ActiveWidth} : std::span<T>{};
    }


    template<class T>
    WindowInfo<VirtualResidency> NumericResidency::FindActiveRange(
        const T* host,
        uint64_t count
    ) const noexcept
    {
        const std::optional<uint64_t> bytes = Memory::ByteCount<T>(count);
        std::optional<CDV::CDT> type = CDV::DataTypeOfSN<T>();
        if (
            !Active() || Failed_ || !host ||
            count == UNSIGNED_ZERO ||
            !bytes.has_value() ||
            !type.has_value()
        ) 
        {
            return {};
        }

        const uintptr_t address = reinterpret_cast<uintptr_t>(host);
        for (const size_t idx  : Active_) 
        {
            VirtualResidency& vr = *Records_[idx];
            const std::uintptr_t begin = reinterpret_cast<uintptr_t>(vr.Host);
            if (
                address < begin ||
                type.value() != vr.LogicalRef.Type
            ) 
            {
                continue;
            }

            const uint64_t offset = (address - begin);
            if (
                offset <= vr.LogicalRef.ByteCount &&
                bytes.value() <= (vr.LogicalRef.ByteCount - offset)
            ) 
            {
                return {&vr, offset};
            }
        }

        return {};
    }

    template<class T>
    T* NumericResidency::Map(
        const T* host, 
        uint64_t count,
        OpenAccess access
    ) noexcept
    {
        WindowInfo<VirtualResidency> info = FindActiveRange(host, count);
        if (!info) return nullptr;
        if constexpr (std::is_const_v<T>) 
        {
            if (access != OpenAccess::READ) return nullptr;
        }
        const bool overwrite_all = access == OpenAccess::WRITE &&
            info.Stride == UNSIGNED_ZERO && count == info.DataPtr->UnitCount();

        if (
            Backend_ == BackendC::GPU &&
            !overwrite_all &&
            info.DataPtr->DiskRevision != info.DataPtr->Version.CommittedRevision
        ) 
        {
            if (
                info.DataPtr->Version.HostRevision != info.DataPtr->Version.CommittedRevision ||
                !info.DataPtr->Upload(Stats_)
            ) 
            {
                Failed_ = true;
                return nullptr;
            }
            info.DataPtr->Version.DeviceRevision = info.DataPtr->Version.CommittedRevision;
        }

        if (access != OpenAccess::READ) info.DataPtr->Pending = true;

        if(Backend_ == BackendC::GPU)
        {
            return reinterpret_cast<T*>(
                static_cast<std::byte*>(info.DataPtr->DevicePtr()) + 
                info.Stride
            );
        }
        return host;
    }

}
