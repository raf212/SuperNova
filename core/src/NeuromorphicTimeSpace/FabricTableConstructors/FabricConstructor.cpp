#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using HAS = HandleOfAPCStatic;


    ADS::RangeOfAPC FabricConstructor::GetSegmentPoolRange(uint64_t single_description_index) noexcept
    {
        ADS::RangeOfAPC desired_segment_pool_range{};

        if (
            single_description_index >= FabCache_->CountOfAPC_ ||
            FabCache_->PerAPCRuntimeCellCount_ == UNSIGNED_ZERO
        )
        {
            return desired_segment_pool_range;
        }

        const uint64_t apc_count_offset = single_description_index * FabCache_->PerAPCRuntimeCellCount_;
        desired_segment_pool_range.BeginIndex = FabCache_->SegmentPoolBegin_ + static_cast<size_t>(apc_count_offset);
        desired_segment_pool_range.EndIndex = desired_segment_pool_range.BeginIndex + static_cast<size_t>(FabCache_->PerAPCRuntimeCellCount_);
        desired_segment_pool_range.IsValid =
            desired_segment_pool_range.BeginIndex >= FabCache_->SegmentPoolBegin_ &&
            desired_segment_pool_range.BeginIndex < desired_segment_pool_range.EndIndex &&
            desired_segment_pool_range.EndIndex <= FabCache_->SlabCellCount_;

        return desired_segment_pool_range;
    }

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


}