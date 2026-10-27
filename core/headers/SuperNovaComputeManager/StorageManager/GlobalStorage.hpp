#pragma once 
#include "../../AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

namespace BidirectionalInMemGraph
{
    struct RawPackedCellAllocator final
    {
        using AllocateFunction = uint64_t* (*)(
            size_t count_of_packed_cell, size_t alignment, void* user
        ) noexcept;

        using FreeFunction = void (*)(
            uint64_t* packed_cell_storage_ptr, 
            size_t count_of_cell, size_t alignment, void*user
        ) noexcept;

        AllocateFunction AllocatePackedCellStorage{nullptr};
        FreeFunction FreePackedCellStorage{nullptr};
        void* User{nullptr};
        size_t Alignment{BIT_COUNT_OF_UINT64_T};

        size_t EffectiveAllignment() const noexcept
        {
            return std::max<size_t>(ADS::CACELINE_SIZE_IN_BYTES, Alignment);
        }

        bool IsValid() const noexcept
        {
            return static_cast<bool>(AllocatePackedCellStorage) == static_cast<bool>(FreePackedCellStorage) &&
                std::has_single_bit(EffectiveAllignment());
        }

        uint64_t* Allocate(size_t cells) const noexcept
        {
            if (!IsValid() || cells > (SIZE_MAX / sizeof(uint64_t)))
            {
                return nullptr;
            }

            const auto FN_ALLOC =  AllocatePackedCellStorage ? AllocatePackedCellStorage:DefaultAllocateAtomicCells;
            return FN_ALLOC(cells, EffectiveAllignment(), User);
        }

        void Free(uint64_t* data, size_t cells) const noexcept
        {
            if (!data) return;
            const auto FN_FREE =  FreePackedCellStorage ? FreePackedCellStorage:DefaultFreeAtomicCells;
            FN_FREE(data, cells, EffectiveAllignment(), User);
        }

        static size_t AlignBiteCount_(size_t bytes, size_t alignment) noexcept
        {
            if (alignment == UNSIGNED_ZERO)
            {
                return bytes;
            }

            const size_t remaining_bytes = bytes % alignment;
            return remaining_bytes == UNSIGNED_ZERO ? bytes : bytes + (alignment - remaining_bytes);
        }

        static uint64_t* DefaultAllocateAtomicCells(
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

        static void DefaultFreeAtomicCells(
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
    };


    class PackedCellStorage final
    {
        uint64_t* Data_{};
        size_t Count_{};
        RawPackedCellAllocator Alloc_{};
        bool Owned_{false};
    
    public:
        PackedCellStorage() = default;
        PackedCellStorage(
            uint64_t* data,
            size_t count,
            RawPackedCellAllocator alloc,
            bool owned
        ) noexcept:
            Data_(data), Count_(count), Alloc_(alloc), Owned_(owned)
        {}
        PackedCellStorage(const PackedCellStorage&) = delete;
        PackedCellStorage& operator = (const PackedCellStorage&) = delete;
        PackedCellStorage(PackedCellStorage&& other) noexcept :
            Data_(other.Data_), Count_(other.Count_),
            Alloc_(other.Alloc_), Owned_(other.Owned_)
        {}
        ~PackedCellStorage()
        {
            if (Owned_) Alloc_.Free(Data_, Count_);
        }

        bool Allocate(size_t count, RawPackedCellAllocator allocator = {}) noexcept
        {
            if (Data_) return false;
            Alloc_ = allocator;
            Data_ = Alloc_.Allocate(count);
            return Data_ != nullptr;
        }

        size_t Count() const noexcept {return Count_;}

        uint64_t* Data() const noexcept {return Data_;}

        uint64_t* Relese() noexcept {return std::exchange(Data_, nullptr);}
    };

}