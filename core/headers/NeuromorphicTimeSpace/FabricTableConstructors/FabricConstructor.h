#pragma once 
#include <span>
#include "../FabricOrchestrators/HandleAndRetirement.hpp"

namespace BidirectionalInMemGraph
{

    class FabricConstructor
    {
        friend class RegionViewConstructor;
    protected:
        uint64_t* SlabBasePtr_{nullptr};

        uint32_t PerAPCRuntimeCellCount_{UNSIGNED_ZERO};
        uint64_t CountOfAPC_{UNSIGNED_ZERO};

        size_t SlabCellCount_{UNSIGNED_ZERO};
        size_t SegmentPoolBegin_{CoreOfFabricCoordinator::FABRIC_UNIT_COUNT};

        uint8_t MaxDirectParentsPerAxis_{UNSIGNED_ZERO};
        uint16_t EdgeTableRecordWidth_{UNSIGNED_ZERO};
    
        uint64_t HandleTableBeginIndex_{UNSIGNED_ZERO};


        std::atomic<bool> FabricInitialized_{false};
        std::atomic<bool> InitializationInProgress_{false};
        RawPackedCellAllocator AllocatorOfFabric_{};
        using SeqLockedOperation = FabricToAPCLinker::SeqLockedOperation;

        using DSA = DescriptionOfAPC;

        bool ReadAFabricU64Directly(
            size_t slab_index,
            uint64_t& return_value
        ) noexcept;

        bool AtomicallyLoadReadAUnit(
            size_t slab_index,
            uint64_t& return_value
        ) noexcept;
        
        void DirectlyStoreFabricUnit64(size_t slab_index, uint64_t fabric_unit) noexcept;

        void AtomicallyStoreU64Fab(
            size_t slab_index, uint64_t fabric_unit, 
            std::memory_order mem_order = std::memory_order_release
        ) noexcept;

        bool CompareExchangeStrongFromFabric(
            size_t slab_index, 
            uint64_t& expected_packed_cell, 
            uint64_t desired_packed_cell,
            std::memory_order mem_order_success = std::memory_order_acq_rel,
            std::memory_order mem_order_failure = std::memory_order_acquire
        ) noexcept;

        bool CompareExchangeWeakInSlab(
            size_t slab_index, 
            uint64_t& expected_packed_cell, 
            uint64_t desired_packed_cell,
            std::memory_order mem_order_success = std::memory_order_acq_rel,
            std::memory_order mem_order_failure = std::memory_order_acquire
        ) noexcept;

        bool ForceNxLenMemCopy(
            size_t slab_starting_idx, 
            size_t number_of_cells, 
            const uint64_t* desired_cells
        ) noexcept;

        constexpr bool IsDesiredIndexValidInSLab(size_t desired_idx) noexcept
        {
            if (SlabBasePtr_ && desired_idx < SlabCellCount_)
            {
                return true;
            }
            return false;
        }

        constexpr size_t SlotBegin_(uint32_t slot) noexcept
        {
            return SegmentPoolBegin_ + static_cast<size_t>(slot) * PerAPCRuntimeCellCount_;
        }

        template<typename T>
        bool IsInternalBuffer(const T* data, size_t count) noexcept
        {
            if (!SlabBasePtr_ || count == UNSIGNED_ZERO)
            {
                return false;
            }
            if (
                count > SIZE_MAX / sizeof(T) ||
                SlabCellCount_ > SIZE_MAX / sizeof(uint64_t)
            )
            {
                return true;
            }

            const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
            const uintptr_t slab = reinterpret_cast<uintptr_t>(SlabBasePtr_);
            const size_t bytes = count * sizeof(T);
            const size_t slab_bytes = SlabCellCount_ * sizeof(uint64_t);

            return begin >= slab
                ? begin - slab < slab_bytes
                : slab - begin < bytes;
        }
        
    };

    class MatrixViewConstructor : public FabricConstructor
    {
        friend class FabricToAPCLinker;
        friend class RegionViewConstructor;
    protected:
        using SD = SchemaDefinition;
        uint64_t MatrixViewTableBeginIndex_{UNSIGNED_ZERO};
        uint16_t ActiveRegionMask_{UNSIGNED_ZERO};
        uint8_t ActiveRegionCount_{UNSIGNED_ZERO};
        uint16_t MatrixViewRowCellCount_{UNSIGNED_ZERO};
        uint32_t MatrixBatchCapacity_{UNSIGNED_ZERO};

        std::span<SD::RegionSchemaRecord> MetrixViewRow_(uint32_t apc_slot) noexcept;

        bool ConstructMatrixViewRecords_(size_t table_begin, size_t table_end) noexcept;

        bool PrepareMatrixViewRow_(
            uint32_t apc_slot,
            const SD::RegionSchemaTable& requested
        ) noexcept;

        void ClearMatrixViewRow_(uint32_t apc_slot) noexcept;

        bool InitializeRegionProtocolStorage_(uint32_t apc_slot) noexcept;

    };

    class APCHandleAndRetirement : public MatrixViewConstructor
    {
    protected:
        uint64_t* GetAPCGenerationPtr_(uint32_t slot) noexcept;

        bool InitializeAPCGenerationTable_() noexcept;

        bool OpenAPCGeneration_(uint32_t slot, uint32_t generation) noexcept;

        bool CloseAPCGeneration_(uint32_t slot, uint32_t generation) noexcept;

        bool AdvanceClosedAPCGeneration_(uint32_t slot, uint32_t& generation_new) noexcept;

        std::optional<uint32_t> ReadFirstFreeAPCIdx_() noexcept;

        void UpdateFirstFreeIdx_(uint64_t& expected_value, uint64_t desired_value) noexcept;
    };




}