#include <SuperNova>

namespace BidirectionalInMemGraph
{
    uint64_t* RawPackedCellAllocator::Allocate(size_t cells) const noexcept
    {
        if (!IsValid() || cells > (SIZE_MAX / sizeof(uint64_t)))
        {
            return nullptr;
        }

        const auto FN_ALLOC =  AllocatePackedCellStorage ? AllocatePackedCellStorage:DefaultAllocateAtomicCells;
        return FN_ALLOC(cells, EffectiveAllignment(), User);
    }

    void RawPackedCellAllocator::Free(uint64_t* data, size_t cells) const noexcept
    {
        if (!data) return;
        const auto FN_FREE =  FreePackedCellStorage ? FreePackedCellStorage:DefaultFreeAtomicCells;
        FN_FREE(data, cells, EffectiveAllignment(), User);
    }

    size_t RawPackedCellAllocator::AlignBiteCount_(size_t bytes, size_t alignment) noexcept
    {
        if (alignment == UNSIGNED_ZERO)
        {
            return bytes;
        }

        const size_t remaining_bytes = bytes % alignment;
        return remaining_bytes == UNSIGNED_ZERO ? bytes : bytes + (alignment - remaining_bytes);
    }

    uint64_t* RawPackedCellAllocator::DefaultAllocateAtomicCells(
        size_t count_of_packed_cell, size_t alignment, void*
    ) noexcept
    {
        if (count_of_packed_cell == UNSIGNED_ZERO)
        {
            return nullptr;
        }

        alignment = std::max<size_t>(alignment, alignof(uint64_t));
        if (
            !std::has_single_bit(alignment) ||
            count_of_packed_cell > (SIZE_MAX / sizeof(uint64_t))
        ) 
        {
            return nullptr;
        }

        const size_t byte_count = sizeof(uint64_t) * count_of_packed_cell;
        if (byte_count > (SIZE_MAX - (alignment - 1))) 
        {
            return nullptr;
        }
        
        const size_t aligned_bytes = AlignBiteCount_(byte_count, alignment);

#if defined(_MSC_VER)

        void* raw_packed_cell_memory = _aligned_malloc(aligned_bytes, alignment);
#else
        void* raw_packed_cell_memory = std::aligned_alloc(alignment, aligned_bytes);
#endif
        if (!raw_packed_cell_memory)
        {
            return nullptr;
        }
        std::memset(raw_packed_cell_memory, UNSIGNED_ZERO, aligned_bytes);
        return static_cast<uint64_t*>(raw_packed_cell_memory);
        
    }

    void RawPackedCellAllocator::DefaultFreeAtomicCells(
        uint64_t* packed_cell_storage_ptr, 
        size_t, size_t, void*
    ) noexcept
    {
        if (!packed_cell_storage_ptr)
        {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(packed_cell_storage_ptr);
#else
        std::free(packed_cell_storage_ptr);
#endif
    }


    bool PackedCellStorage::Allocate(size_t count, RawPackedCellAllocator allocator) noexcept
    {
        if (Data_) return false;
        Alloc_ = allocator;
        Data_ = Alloc_.Allocate(count);
        if (Data_) 
        {
            Count_ = count;
            Owned_ = true;
        }
        return Data_ != nullptr;
    }
}


#if defined(SN_ACPP_ENABLED)
namespace BidirectionalInMemGraph::ComputerDetail::Memory
{
    uint64_t* ACppFabricAlocator::AllocateUsm(size_t cells, size_t alignment ) 
    {
        if (
            !cells ||
            cells > LIMIT_OF_CELL_COUNT ||
            cells > SIZE_MAX / sizeof(uint64_t) ||
            !std::has_single_bit(alignment) ||
            alignment < alignof(uint64_t)
        ) 
        {
            return nullptr;
        }

        try 
        {
            return sycl::aligned_alloc<uint64_t>(
                alignment, cells, *Queue_, __Loc__
            );
        }
        catch (const std::runtime_error& e) {
            std::cerr << "sycl memory alocation error: " << e.what() << "\n";
            return nullptr;
        }
    }

    void ACppFabricAlocator::FreeUsm(uint64_t* data) noexcept
    {
        if (!data) return;
        try 
        {
            sycl::free(data, *Queue_);
        }
        catch(...)
        {
            std::terminate();
        }
    }
}
#endif