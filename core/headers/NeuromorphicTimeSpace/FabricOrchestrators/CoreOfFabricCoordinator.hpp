#pragma once 
#include "../../AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

namespace BidirectionalInMemGraph
{

    struct CoreOfFabricCoordinator
    {
        /// UNCHECKED
        static constexpr size_t RELATION_WIDTH_OF_FABRIC = 0u;
        static constexpr size_t DEVICE_PLANNER_RECORD_LEN = 0u;
        static constexpr size_t WORK_RECORD_WIDTH_OF_FABRIC = 0u;
        static constexpr size_t DEFAULT_FABRIC_CONTROLIO_LENGTH = 512u;
        ///--------------------------
        static constexpr size_t COMPILED_DAG_LEN = 2u;

        static constexpr uint32_t FABRIC_MAGIC = 0x41504643u;
        static constexpr uint32_t FABRIC_META_EOF = 0x41474946u;
        static constexpr uint8_t EACH_TABLE_RECORD_SENTINAL = UINT8_MAX;

        static constexpr uint8_t FORMAT_VERSION = 1u;

        enum class FabricBackigOwnership : uint8_t
        {
            NONE = 0,
            OWNED = 1,
            BORROWED = 2
        };

        struct alignas(uint64_t) FabricCache 
        {
            uint32_t FormateVersion_{UNSIGNED_ZERO};
            ///FABRIC CONSTRUCTION
            uint32_t PerAPCRuntimeCellCount_{UNSIGNED_ZERO};
            uint32_t CountOfAPC_{UNSIGNED_ZERO};
            uint64_t SlabCellCount_{UNSIGNED_ZERO};

            // RECORD BOOK / SEGMENT POOL
            uint64_t RecordBookBeginIndex_{UNSIGNED_ZERO};
            uint64_t RecordBookEndIndex_{UNSIGNED_ZERO};
            uint64_t SegmentPoolBegin_{UNSIGNED_ZERO};
            uint32_t FirstFreeIdx_{UNSIGNED_ZERO};

            // EDGE GEOMETRY
            uint8_t MaxDirectParentsPerAxis_{UNSIGNED_ZERO};
            uint16_t EdgeTableRecordWidth_{UNSIGNED_ZERO};

            ///MATRIX CONSTRUCTION
            uint8_t ActiveRegionCount_{UNSIGNED_ZERO};
            uint16_t ActiveRegionMask_{UNSIGNED_ZERO};
            uint16_t MatrixViewRowCellCount_{UNSIGNED_ZERO};
            uint32_t MatrixBatchCapacity_{UNSIGNED_ZERO};
            uint64_t RegionAlignmentCellCount_{UNSIGNED_ZERO};

            // HOT TABLE BEGIN INDICES
            uint64_t HorizontalEdgeBeginIdx_{UNSIGNED_ZERO};
            uint64_t VerticalEdgeBeginIdx_{UNSIGNED_ZERO};
            uint64_t CompiledDAGTableBeginIdx_{UNSIGNED_ZERO};
            uint64_t HandleTableBeginIndex_{UNSIGNED_ZERO};
            uint64_t MatrixViewTableBeginIndex_{UNSIGNED_ZERO};
            bool HasDefaultRegionTable_{false};

        };


        struct DetachFabric final
        {
            uint64_t* Slab_{nullptr};
            uint64_t CellCount_{UNSIGNED_ZERO};
            FabricBackigOwnership Ownership_ = FabricBackigOwnership::NONE;

            explicit constexpr operator bool() const noexcept
            {
                return Slab_ != nullptr && CellCount_ != UNSIGNED_ZERO;
            }
        };
        

        static_assert(sizeof(FabricCache) == 16 * sizeof(uint64_t));

        enum class RecordBookInternalIndexing : uint8_t
        {
            BEGIN64 = 0,
            END64 = 1,
        };
        static constexpr uint8_t RECORD_BOOK_WIDTH = static_cast<uint8_t>(RecordBookInternalIndexing::END64) + 1u;


        static constexpr uint8_t FABRIC_UNIT_COUNT = sizeof(FabricCache);

        static constexpr bool IsValidEdgeTable(FabricSegments table_class) noexcept
        {
            return
                table_class == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ||
                table_class == FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;
        }

        static constexpr size_t DefaultFabricAlignment16Cell_(size_t value) noexcept
        {
            const uint8_t alignment_value_15 = 16 - 1;
            return (value + alignment_value_15) & ~static_cast<size_t>(alignment_value_15);
        }




    
    };


    struct RecordBookConf
    {
        static constexpr uint8_t RECORD_BOOK_INTERNAL_SEGMENT_COUNT = static_cast<uint8_t>(FabricSegments::SEGMENT_POOL) + 1;

        struct FabricSegmentBounds
        {
            uint64_t BeginIndex = UNSIGNED_ZERO;
            uint64_t EndIndex = UNSIGNED_ZERO;
            FabricSegments OwnerTableOfTheBounds{};
            bool IsValid = false;  
        };

    };


    struct RawPackedCellAllocator
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
            const size_t byte_count = sizeof(uint64_t) * count_of_packed_cell;
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


}
