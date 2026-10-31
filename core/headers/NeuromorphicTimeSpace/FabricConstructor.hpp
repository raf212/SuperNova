#pragma once 
#include <span>
#include "FabricOrchestrators/HandleAndRetirement.hpp"
#include "../SuperNovaComputeManager/StorageManager/GlobalStorage.hpp"

namespace BidirectionalInMemGraph
{

    class FabricConstructor : protected StoragePointers
    {
        friend class AdaptivePackedCellContainer;
    public:
        FabricExecutionGate& ExecutionGate() noexcept {return ExecutionGate_;}
    protected:
        FabricExecutionGate ExecutionGate_{};
        
    protected:
        using SeqLockedOperation = FabricToAPCLinker::SeqLockedOperation;
        using DSA = DescriptionOfAPC;

        static constexpr uint8_t DEFAULT_INTERNAL_TRIES__ = 1u;

        CoreOfFabricCoordinator::FabricBackigOwnership BackingOwnership_ = CoreOfFabricCoordinator::FabricBackigOwnership::NONE;
        std::atomic<bool> FabricInitialized_{false};
        std::atomic<bool> InitializationInProgress_{false};

        void RefreshNumericalBase_() noexcept;

        uint64_t* ResolveCell(uint64_t index) noexcept;

        GMI::NodeModelDescription* NodeModel_(uint32_t slot) noexcept;

        constexpr bool IsCellStorageHot_(uint64_t index) noexcept;

        ADS::RangeOfAPC GetSegmentPoolRange(uint64_t single_description_index) noexcept;

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
            if (FabCache_ && SlabBasePtr_ && desired_idx < FabCache_->SlabCellCount_)
            {
                return true;
            }
            return false;
        }

        constexpr size_t SlotBegin_(uint32_t slot) noexcept
        {
            return FabCache_->SegmentPoolBegin_ + static_cast<size_t>(slot) * FabCache_->PerAPCRuntimeCellCount_;
        }

        template<typename T>
        bool IsInternalBuffer(const T* data, size_t count) noexcept
        {
            if (!FabCache_ || !SlabBasePtr_ || count == UNSIGNED_ZERO)
            {
                return false;
            }
            if (
                count > SIZE_MAX / sizeof(T) ||
                FabCache_->SlabCellCount_ > SIZE_MAX / sizeof(uint64_t)
            )
            {
                return true;
            }

            const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
            const uintptr_t slab = reinterpret_cast<uintptr_t>(SlabBasePtr_);
            const size_t bytes = count * sizeof(T);

            auto OverLaps___ = [&](const void* mem, uint64_t cells) noexcept -> bool
            {
                const auto base = reinterpret_cast<uintptr_t>(mem);
                return begin >= base ? (begin - base < cells * sizeof(uint64_t)) : 
                    (base - begin < bytes);
            };

            return OverLaps___(
                SlabBasePtr_, 
                NumericalBackingPtr_->DataPtr() ? FabCache_->SegmentPoolBegin_ : FabCache_->SlabCellCount_
            ) ||
            (
                NumericalBackingPtr_->DataPtr() &&
                OverLaps___(NumericalBackingPtr_->DataPtr(), FabCache_->SlabCellCount_ - FabCache_->SegmentPoolBegin_)
            );
        }
        
    };

    class MatrixViewConstructor : public FabricConstructor
    {
        friend class FabricToAPCLinker;
        friend class AdaptivePackedCellContainer;
    public:
        using SD = SchemaDefinition;
    private:
        SD::ConstRSSpan MutableMatrixViewRow_(uint32_t apc_slot) noexcept;
    protected:

        SD::ConstRSSpan MetrixViewRow_(uint32_t apc_slot) noexcept;

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
        friend class FabricToAPCLinker;

    protected:

        using HAS = HandleOfAPCStatic;

        HAS::StructuralHotRow* GetStructuralHotRow_(uint32_t slot) noexcept;

        HAS::ParentRowControl* ParentRowControl_(FabricSegments edge_table, uint32_t slot) noexcept;

        HAS::ChildListControl* ChildListControl_(FabricSegments edge_table, uint32_t slot) noexcept;

        uint32_t* ChildTailPtr_(FabricSegments edge_table, uint32_t slot) noexcept;

        uint64_t* GetAPCGenerationPtr_(uint32_t slot) noexcept;

        bool InitializeStructHotRowApcHandleTable_() noexcept;

        bool OpenAPCGeneration_(uint32_t slot, uint32_t generation) noexcept;

        bool CloseAPCGeneration_(uint32_t slot, uint32_t generation) noexcept;

        bool AdvanceClosedAPCGeneration_(uint32_t slot, uint32_t& generation_new) noexcept;

        std::optional<uint32_t> ReadFirstFreeAPCIdx_() noexcept;

        void UpdateFirstFreeIdx_(uint32_t& expected_value, uint32_t desired_value) noexcept;

        SeqLockedOperation SwitchDescriptionState(
            uint32_t slot,
            StateOfAPC updated_state,
            StateOfAPC desired_state,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        SeqLockedOperation ReadAPCStateAtomically_(
            uint32_t slot,
            HAS::LifeCycleControl& life_cycle,
            uint32_t max_tries = DEFAULT_INTERNAL_TRIES__
        ) noexcept;

        HAS::LifeCycleControl* LifeCycleControl_(uint32_t slot) noexcept
        {
            HAS::StructuralHotRow* const row = GetStructuralHotRow_(slot);
            return row ? &row->LifeCycle : nullptr;
        }

        bool IsOpenAPCGeneration_(
            uint32_t slot,
            uint32_t expected_generation
        ) noexcept
        {
            uint64_t* const cell = GetAPCGenerationPtr_(slot);
            return cell && HandleOfAPCStatic::IsOpenGeneration(
                std::atomic_ref<const uint64_t>(*cell).load(std::memory_order_acquire),
                expected_generation
            );
        }
    };




}