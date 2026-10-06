#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using HAS = HandleOfAPCStatic;

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

    ADS::RangeOfAPC APCLifeCycle::GetSegmentPoolRange(uint64_t single_description_index) noexcept
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

    DescriptionOfAPC::SeqLockAndStateStruct APCLifeCycle::ReadAPCStateAtomically_(uint64_t apc_description_index) noexcept
    {
        DescriptionOfAPC::SeqLockAndStateStruct return_files{};
        std::optional<uint64_t> maybe_id_state_idx = GetDescriptionLockIdxInFabric_(apc_description_index);
        uint64_t state_of_apc_cell = FABRIC_CELL_SENTINAL;

        if (
            !maybe_id_state_idx.has_value() ||
            !AtomicallyLoadReadAUnit(maybe_id_state_idx.value(), state_of_apc_cell)
        )
        {
            return return_files;
        }
        DescriptionOfAPC::GetSeqLockAndLifeCycle(state_of_apc_cell, return_files);
        return return_files;
    }


    bool APCLifeCycle::SwitchDescriptionState(
        uint64_t description_idx,
        StateOfAPC updated_state,
        StateOfAPC desired_state,
        uint32_t max_tries
    ) noexcept
    {
        const std::optional<uint64_t> id_state_idx = GetDescriptionLockIdxInFabric_(description_idx);
        if (
            !id_state_idx.has_value()
        )
        {
            return false;
        }

        for (size_t i = 0; i < max_tries; i++)
        {
            DSA::SeqLockAndStateStruct current_id_st = ReadAPCStateAtomically_(description_idx);
            uint64_t current_id_state_value = DSA::ComposeSeqLockAndState(current_id_st);
            DSA::SeqLockAndStateStruct updated_files{};
            updated_files.SeqLock = current_id_st.SeqLock + 1u;
            updated_files.StateOfTheAPC = updated_state;
            uint64_t updated_id_state_value = DSA::ComposeSeqLockAndState(updated_files);

            if (
                !ADS::IsValidFabricUnit(current_id_state_value) ||
                !ADS::IsValidFabricUnit(updated_id_state_value) ||
                current_id_st.StateOfTheAPC != desired_state ||
                !DSA::IsTransitionStateLeagal(current_id_st.StateOfTheAPC, updated_state)
            )
            {
                return false;
            }

            if (
                !CompareExchangeStrongFromFabric(
                    id_state_idx.value(),
                    current_id_state_value,
                    updated_id_state_value
                )
            )
            {
                continue;
            }
            
            return true;
        }
        return false;
    }

    std::optional<uint64_t> APCLifeCycle::GetDescriptionLockIdxInFabric_(uint64_t description_idx) noexcept
    {
        const ADS::RangeOfAPC range_of_segmentpool = GetSegmentPoolRange(description_idx);

        if (!range_of_segmentpool.IsValid)
        {
            return std::nullopt;
        }
        const size_t state_cell_idx = range_of_segmentpool.BeginIndex + 
            static_cast<uint8_t>(ADS::HeaderIdentifierOfAPC::APC_LIFE_CYCLE);
        
        return state_cell_idx;
    }


}
