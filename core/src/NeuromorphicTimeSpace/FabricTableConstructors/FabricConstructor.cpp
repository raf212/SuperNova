#include <SuperNova>

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
        return_value = *ResolveCell_(slab_index);
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
        std::atomic_ref<const uint64_t> fab_u64_ref(*ResolveCell_(slab_index));
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
        *ResolveCell_(slab_index) = fabric_unit;
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
        std::atomic_ref<uint64_t> fab_u64_ref(*ResolveCell_(slab_index));
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
        std::atomic_ref<uint64_t> fab_u64_ref(*ResolveCell_(slab_index));
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
        std::atomic_ref<uint64_t> fab_u64_ref(*ResolveCell_(slab_index));
        return fab_u64_ref.compare_exchange_weak(expected_packed_cell, desired_packed_cell, mem_order_success, mem_order_failure);
    }

    bool FabricConstructor::ForceNxLenMemCopy(
        size_t slab_starting_idx, 
        size_t number_of_cells, 
        const uint64_t* desired_units
    ) noexcept
    {
        if (
            !FabCache_ ||
            !desired_units ||
            number_of_cells == UNSIGNED_ZERO ||
            slab_starting_idx >= FabCache_->SlabCellCount_ ||
            number_of_cells > (FabCache_->SlabCellCount_ - slab_starting_idx)
        )
        {
            return false;
        }

        const size_t hot_end = FabCache_->SegmentPoolBegin_;
        const size_t hot_cells = slab_starting_idx < hot_end ?
            std::min(number_of_cells, (hot_end - slab_starting_idx)) : UNSIGNED_ZERO;
        
        if (hot_cells > UNSIGNED_ZERO) 
        {
            std::memcpy(
                (VolatileCache___.SlabBasePtr_ + slab_starting_idx),
                desired_units,
                hot_cells * sizeof(uint64_t)
            );
        }

        if (hot_cells < number_of_cells) 
        {
            const size_t numerical_index = slab_starting_idx + hot_cells;
            std::memcpy(
                (NumericalBase_() + (numerical_index - hot_end)),
                (desired_units + hot_cells),
                ((number_of_cells - hot_cells) * sizeof(uint64_t))
            );
        }
        return true;
    }


    void FabricConstructor::RefreshNumericalBase_() noexcept
    {
        VolatileCache___.StorageBasePtr_ = __NumericalBacking__.DataPtr() ? 
            __NumericalBacking__.DataPtr() :
            (VolatileCache___.SlabBasePtr_ && FabCache_ ? (VolatileCache___.SlabBasePtr_ + FabCache_->SegmentPoolBegin_) : nullptr);
    }

    uint64_t* FabricConstructor::ResolveCell_(uint64_t index) noexcept
    {
        if (
            !FabCache_ ||
            !VolatileCache___.SlabBasePtr_ ||
            !NumericalBase_() ||
            index >= FabCache_->SlabCellCount_
        )
        {
            return nullptr;
        }

        return index < FabCache_->SegmentPoolBegin_ ? (VolatileCache___.SlabBasePtr_ + index) :
            NumericalBase_() + (index - FabCache_->SegmentPoolBegin_);
    }

    GMI::NodeModelDescription* FabricConstructor::NodeModel_(uint32_t slot) noexcept
    {
        return FabCache_ && slot < FabCache_->CountOfAPC_ ?
            reinterpret_cast<GMI::NodeModelDescription*>(
                VolatileCache___.SlabBasePtr_ + FabCache_->ModelDiaphragm_.GlobalTableBeginIdx_ + slot
            ) : nullptr;
    }

    constexpr bool FabricConstructor::IsCellStorageHot_(uint64_t index) noexcept
    {
        if (
            FabCache_ &&
            VolatileCache___.SlabBasePtr_ &&
            index < FabCache_->SegmentPoolBegin_
        ) 
        {
            return true;
        }
        return false;
    }





}