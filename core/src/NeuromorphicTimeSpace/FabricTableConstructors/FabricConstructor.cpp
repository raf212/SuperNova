#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

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
            number_of_cells > FabCache_.SlabCellCount_ - slab_starting_idx
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

    uint64_t* APCHandleAndRetirement::GetAPCGenerationPtr_(uint32_t slot) noexcept
    {
        if (
            !SlabBasePtr_ ||
            slot >= FabCache_.CountOfAPC_ ||
            FabCache_.HandleTableBeginIndex_ >= FabCache_.SlabCellCount_
        )
        {
            return nullptr;
        }
        
        const size_t idx = FabCache_.HandleTableBeginIndex_ + HandleOfAPCStatic::CellOffset(slot);

        return idx < FabCache_.SlabCellCount_ ? &SlabBasePtr_[idx] : nullptr;
    }

    bool APCHandleAndRetirement::InitializeAPCGenerationTable_() noexcept
    {
        for (uint32_t slot = 0; slot < FabCache_.CountOfAPC_; slot++)
        {
            uint64_t* cell = GetAPCGenerationPtr_(slot);
            if (!cell)
            {
                return false;
            }

            HandleOfAPCStatic::ControlValues values{};
            values.Generation = HandleOfAPCStatic::FIRST_GENERATION;
            values.ActiveAccess = UNSIGNED_ZERO;
            values.Closed = true;


            std::atomic_ref<uint64_t>(*cell).store(
                HandleOfAPCStatic::MakeControlCell(values),
                std::memory_order_relaxed
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
        uint8_t first_free = static_cast<uint8_t>(CoreOfFabricCoordinator::FabricMetaIndicies::FIRST_FREE_IDX);
        if (!IsDesiredIndexValidInSLab(first_free))
        {
            return std::nullopt;
        }

        std::atomic_ref<const uint64_t> fab_u64_ref(SlabBasePtr_[first_free]);
        uint64_t first_free_apc = fab_u64_ref.load(std::memory_order_acquire);

        if (!ADS::IsValid32BitAPCUnit(first_free_apc))
        {
            return std::nullopt;
        }
        
        return static_cast<uint32_t>(first_free_apc);
    }

    void APCHandleAndRetirement::UpdateFirstFreeIdx_(uint64_t& expected_value, uint64_t desired_value) noexcept
    {
        uint8_t first_free = static_cast<uint8_t>(CoreOfFabricCoordinator::FabricMetaIndicies::FIRST_FREE_IDX);
        if (!IsDesiredIndexValidInSLab(first_free))
        {
            return;
        }
        std::atomic_ref<uint64_t> fab_u64_ref(SlabBasePtr_[first_free]);
        fab_u64_ref.compare_exchange_strong(expected_value, desired_value, std::memory_order_acq_rel, std::memory_order_acquire);
    }


}