#pragma once
#include "Definition.hpp"

/// WORK - MEMORY - COMPUTER 
namespace BidirectionalInMemGraph::ComputerDetail
{
    //Later It can be optimized based on runtime availavlity
    static constexpr unsigned PER_REGION_DEVICE_CACHE_IN_MB = 256u;
    static constexpr unsigned MB_TO_BYTE_MULTIPLYER = 1024 * 1024;
    enum class MemoryLocation : uint8_t
    {
        HOST = 0,
        GPU = 1,
        DISK = 2
    };

    enum class OpenAccess : uint8_t
    {
        READ = 0,
        WRITE = 1,
        READ_WRITE = 2
    };

    struct ResidencyInfo
    {
        uint64_t CommittedRevision{};
        uint64_t HostRevision{};
        uint64_t DeviceRevision{};
        MemoryLocation Location{MemoryLocation::HOST};
    };

    struct TransferStatics
    {
        uint64_t Allocation{};
        uint64_t UploadedBytes{};
        uint64_t DownloadedfBytes{};
        uint64_t CacheHits{};
    };

    enum class Endianess : uint16_t
    {
        LITTLE = 0,
        BIG = 1
    };

    struct FabricRuntimeBinding
    {
        APCFinilizer* Owner{};
        std::span<uint64_t> Cells{};
        FabricExecutionGate* Gate{};
        CheckpointFormat::FabricId Id{};
        std::span<uint64_t> Numerics{};
        uint64_t NumericalBegainCell{};

        constexpr std::span<uint64_t> NumericalCells() const noexcept
        {
            return Numerics.empty() ? Cells : Numerics;
        }

        std::optional<uint64_t> LogicalByteOffset(const void* data, uint64_t bytes) const noexcept;

        uint64_t LogicalCellCount() const noexcept
        {
            return Numerics.empty() ? Cells.size() : (NumericalBegainCell + Numerics.size());
        }

        explicit operator bool() const noexcept
        {
            return !Cells.empty() && Gate && static_cast<bool>(Id);
        }
    };
}

/// WORK - MEMORY
namespace BidirectionalInMemGraph::ComputerDetail::Memory
{
    template<class DT>
    static constexpr std::optional<uint64_t>ByteCount(uint64_t count) noexcept
    {
        if (count > UINT64_MAX / sizeof(DT)) 
        {
            return std::nullopt;
        }
        return static_cast<uint64_t>(count) * sizeof(DT);
    }

    static constexpr bool RangesOverlap(
        const void* left,
        uint64_t left_bytes,
        const void* right,
        uint64_t right_bytes
    ) noexcept
    {
        if (
            !left ||
            !right ||
            left_bytes == UNSIGNED_ZERO ||
            right_bytes == UNSIGNED_ZERO
        ) 
        {
            return false;
        }

        const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
        const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);

        return left_begin <= right_begin ? 
            right_begin - left_begin < left_bytes : left_begin - right_begin < right_bytes;
    }

    static constexpr bool IsRangeInside(
        const uint64_t* slab,
        uint64_t cell_count,
        const void* data,
        uint64_t bytes
    ) noexcept
    {
        if (!slab || !data) 
        {
            return false;
        }

        const std::optional<uint64_t> slab_bytes = ByteCount<uint64_t>(cell_count);
        if (!slab_bytes.has_value()) 
        {
            return false;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(slab);
        const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);

        if (pointer < begin) 
        {
            return false;
        }

        const uintptr_t offset_wide = pointer - begin;
        if (offset_wide > static_cast<uintptr_t>(slab_bytes.value())) 
        {
            return false;
        }

        const uint64_t offset = static_cast<uint64_t>(offset_wide);

        return bytes <= slab_bytes.value() - offset;
    }

    template<class DT>
    static bool IsTypedRangeInside(
        const uint64_t* slab,
        uint64_t slab_cells,
        const DT* data,
        uint64_t count
    ) noexcept
    {
        const std::optional<uint64_t> bytes = ByteCount<DT>(count);
        return count != UNSIGNED_ZERO && data && bytes &&
            reinterpret_cast<uintptr_t>(data) % alignof(DT) == UNSIGNED_ZERO &&
            IsRangeInside(slab, slab_cells, data, *bytes);
    }

    template <class... Ts>
    bool IndependentNumericalRanges(NumericRange<Ts>... ranges) noexcept
    {
        struct Extent
        {
            const void* Pointer;
            uint64_t Bytes;
        };

        const std::array<Extent, sizeof...(Ts)> extents
        {{
            {
                ranges.Host,
                ByteCount<Ts>(ranges.Count).value_or(UINT64_MAX)
            }...
        }};
        for (size_t first = 0; first < extents.size(); first++) 
        {
            for (size_t second = first + 1; second < extents.size(); second++)
            {
                if (RangesOverlap(
                    extents[first].Pointer,
                    extents[first].Bytes,
                    extents[second].Pointer,
                    extents[second].Bytes
                )) 
                {
                    return false;
                }
            }
        }

        return true;
    }

    static constexpr std::optional<size_t> StrideElementCount(uint32_t rows, uint32_t columns, uint32_t stride) noexcept
    {
        if (
            rows == UNSIGNED_ZERO ||
            columns == UNSIGNED_ZERO ||
            stride < columns
        ) 
        {
            return std::nullopt;
        }

        const size_t row_offset = static_cast<size_t>(rows - 1);
        if (row_offset > SIZE_MAX / static_cast<size_t>(stride)) 
        {
            return std::nullopt;
        }

        const size_t base = row_offset * static_cast<size_t>(stride);
        if (base > (SIZE_MAX - static_cast<size_t>(stride))) 
        {
            return std::nullopt;
        }

        return base + static_cast<size_t>(columns);
    }

}

