#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using HAS = HandleOfAPCStatic;

    bool FabricConstructor::ReadAFabricU64Directly(
        size_t slab_index,
        uint64_t& return_value
    ) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return false;
        }
        return_value = SlabBasePtr_[slab_index];
        return true;
    }

    bool FabricConstructor::AtomicallyLoadReadAUnit(
        size_t slab_index,
        uint64_t& return_value
    ) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return false;
        }
        std::atomic_ref<const uint64_t> fab_u64_ref(SlabBasePtr_[slab_index]);
        uint64_t desired_cell_raw = fab_u64_ref.load(std::memory_order_acquire);
        return_value = desired_cell_raw;
        return true;
    }

    void FabricConstructor::DirectlyStoreFabricUnit64(size_t slab_index, uint64_t fabric_unit) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return;
        }
        SlabBasePtr_[slab_index] = fabric_unit;
    }

    void FabricConstructor::AtomicallyStoreU64Fab(
        size_t slab_index, uint64_t fabric_unit,
        std::memory_order mem_order
    ) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return;
        }
        std::atomic_ref<uint64_t> fab_u64_ref(SlabBasePtr_[slab_index]);
        fab_u64_ref.store(fabric_unit, mem_order);
    }

    bool FabricConstructor::CompareExchangeStrongFromFabric(
        size_t slab_index, 
        uint64_t& expected_packed_cell, 
        uint64_t desired_packed_cell,
        std::memory_order mem_order_success,
        std::memory_order mem_order_failure
    ) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return false;
        }
        std::atomic_ref<uint64_t> fab_u64_ref(SlabBasePtr_[slab_index]);
        return fab_u64_ref.compare_exchange_strong(expected_packed_cell, desired_packed_cell, mem_order_success, mem_order_failure);
    }

    bool FabricConstructor::CompareExchangeWeakInSlab(  
        size_t slab_index, 
        uint64_t& expected_packed_cell, 
        uint64_t desired_packed_cell,
        std::memory_order mem_order_success,
        std::memory_order mem_order_failure
    ) noexcept
    {
        if (!IsDesiredIndexValidInSLab(slab_index))
        {
            return false;
        }
        std::atomic_ref<uint64_t> fab_u64_ref(SlabBasePtr_[slab_index]);
        return fab_u64_ref.compare_exchange_weak(expected_packed_cell, desired_packed_cell, mem_order_success, mem_order_failure);
    }

    bool FabricConstructor::ForceNxLenMemCopy(
        size_t slab_starting_idx, 
        size_t number_of_cells, 
        const uint64_t* desired_units
    ) noexcept
    {
        if (
            !IsDesiredIndexValidInSLab(slab_starting_idx + number_of_cells - 1) ||
            !desired_units ||
            number_of_cells == UNSIGNED_ZERO ||
            number_of_cells > FabCache_->SlabCellCount_ - slab_starting_idx
        )
        {
            return false;
        }

        try
        {
            uint64_t value_of_last_idx = desired_units[number_of_cells - 1];
            (void) value_of_last_idx;
        }
        catch(...)
        {
            return false;
        }

        std::memcpy(
            &SlabBasePtr_[slab_starting_idx],
            desired_units,
            number_of_cells * sizeof(uint64_t)
        );
        return true;
    }

    HAS::ParentRowControl* APCHandleAndRetirement::ParentRowControl_(FabricSegments edge_table, uint32_t slot) noexcept
    {
        HAS::StructuralHotRow* const row = GetStructuralHotRow_(slot);
        if (
            !row ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table)
        )
        {
            return nullptr;
        }
        
        return edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ?
            &row->ValueParentControl : &row->VolatileParentControl;
    }

    uint64_t* APCHandleAndRetirement::GetAPCGenerationPtr_(uint32_t slot) noexcept
    {
        HAS::StructuralHotRow* const row = GetStructuralHotRow_(slot);
        return row ? &row->GenerationControl : nullptr;
    }

    bool APCHandleAndRetirement::InitializeStructHotRowApcHandleTable_() noexcept
    {
        if (
            !SlabBasePtr_ ||
            !FabCache_
        )
        {
            return false;
        }

        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            const uint64_t begin = FabCache_->HandleTableBeginIndex_ + HAS::CellOffset(i);
            if (
                begin >= FabCache_->SlabCellCount_ ||
                HAS::HANDLE_TABLE_WIDTH > FabCache_->SlabCellCount_ - begin
            )
            {
                return false;
            }

            HAS::StructuralHotRow initial{};
            HAS::ControlValues generation{};
            generation.Generation = HAS::FIRST_GENERATION;
            generation.ActiveAccess = UNSIGNED_ZERO;
            generation.Closed = true;

            initial.GenerationControl = HAS::MakeControlCell(generation);
            initial.LifeCycle = HAS::LifeCycleControl{};
            initial.ValueParentControl = HAS::ParentRowControl{};
            initial.ValueChildControl = HAS::ChildListControl{};
            initial.VolatileParentControl = HAS::ParentRowControl{};
            initial.VolatileChildControl = HAS::ChildListControl{};
            initial.ValueChildTail = EdgeBuilder::RELATION_NULL;
            initial.VolatileChildTail = EdgeBuilder::RELATION_NULL;

            std::construct_at(
                reinterpret_cast<HAS::StructuralHotRow*>(SlabBasePtr_ + begin),
                initial
            );
        }
        return true;
    }


    bool APCHandleAndRetirement::OpenAPCGeneration_(uint32_t slot, uint32_t generation) noexcept
    {
        uint64_t* cell = GetAPCGenerationPtr_(slot);

        if (!cell || !HandleOfAPCStatic::IsGenerationValid(generation))
        {
            return false;
        }

        HandleOfAPCStatic::ControlValues values{};
        values.Generation = generation;
        values.ActiveAccess = UNSIGNED_ZERO;
        values.Closed = true;
        
        uint64_t expected = HandleOfAPCStatic::MakeControlCell(values);
        //desired
        values.Closed = false;

        return std::atomic_ref<uint64_t>(*cell).compare_exchange_strong(
            expected,
            HandleOfAPCStatic::MakeControlCell(values),
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );
    }

    bool APCHandleAndRetirement::AdvanceClosedAPCGeneration_(uint32_t slot, uint32_t& generation_new) noexcept
    {
        generation_new = UNSIGNED_ZERO;
        uint64_t* cell = GetAPCGenerationPtr_(slot);

        if (!cell)
        {
            return false;
        }

        std::atomic_ref<uint64_t> control(*cell);
        uint64_t observed = control.load(std::memory_order_acquire);

        const HandleOfAPCStatic::ControlValues values = HandleOfAPCStatic::ReadControlCell(observed);

        HandleOfAPCStatic::ControlValues desired_values{};


        const uint32_t desired_generation = HandleOfAPCStatic::NextGeneration(values.Generation);

        desired_values.Generation = desired_generation;
        desired_values.ActiveAccess = UNSIGNED_ZERO;
        desired_values.Closed = true;

        const uint64_t desired = HandleOfAPCStatic::MakeControlCell(desired_values);
        
        if (
            !values.Closed ||
            values.ActiveAccess != UNSIGNED_ZERO ||
            desired_generation == UNSIGNED_ZERO 
        )
        {
            return false;
        }
        
        if (
            !control.compare_exchange_strong(observed, desired, std::memory_order_acq_rel, std::memory_order_acquire)
        )
        {
            return false;
        }
        
        generation_new = desired_generation;
        return true;
    }


    bool APCHandleAndRetirement::CloseAPCGeneration_(uint32_t slot, uint32_t generation) noexcept
    {
        uint64_t* cell = GetAPCGenerationPtr_(slot);
        if (
            !cell ||
            !HandleOfAPCStatic::IsGenerationValid(generation)
        )
        {
            return false;
        }

        HandleOfAPCStatic::ControlValues values{};
        values.Generation = generation;
        values.ActiveAccess = UNSIGNED_ZERO;
        values.Closed = false;
        
        uint64_t expected = HandleOfAPCStatic::MakeControlCell(values);

        values.Closed = true;
        const uint64_t desired = HandleOfAPCStatic::MakeControlCell(values);

        return std::atomic_ref<uint64_t>(*cell).compare_exchange_strong(
            expected,
            desired,
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );  
    }


    std::optional<uint32_t> APCHandleAndRetirement::ReadFirstFreeAPCIdx_() noexcept
    {
        if (!FabCache_)
        {
            return std::nullopt;
        }

        const uint32_t first_free = std::atomic_ref<const uint32_t>(FabCache_->FirstFreeIdx_).load(std::memory_order_acquire);

        if (!ADS::IsValid32BitAPCUnit(first_free))
        {
            return std::nullopt;
        }
        
        return first_free;
    }

    void APCHandleAndRetirement::UpdateFirstFreeIdx_(uint32_t& expected_value, uint32_t desired_value) noexcept
    {
        if (!FabCache_)
        {
            return;
        }
        
        std::atomic_ref<uint32_t>(FabCache_->FirstFreeIdx_).compare_exchange_strong(
            expected_value,
            desired_value,
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );
    }

    HAS::StructuralHotRow* APCHandleAndRetirement::GetStructuralHotRow_(uint32_t slot) noexcept
    {
        if (!SlabBasePtr_ || !FabCache_ || slot >= FabCache_->CountOfAPC_)
        {
            return nullptr;
        }

        const uint64_t begin = static_cast<uint64_t>(FabCache_->HandleTableBeginIndex_) + HAS::CellOffset(slot);
        if (
            begin >= FabCache_->SlabCellCount_ ||
            HAS::HANDLE_TABLE_WIDTH > FabCache_->SlabCellCount_ - begin
        )
        {
            return nullptr;
        }

        return std::launder(reinterpret_cast<HAS::StructuralHotRow*>(SlabBasePtr_ + begin));
    }


}