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

    HAS::ChildListControl* APCHandleAndRetirement::ChildListControl_(FabricSegments edge_table, uint32_t slot) noexcept
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
            &row->ValueChildControl : &row->VolatileChildControl;
    }

    uint32_t* APCHandleAndRetirement::ChildTailPtr_(FabricSegments edge_table, uint32_t slot) noexcept
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
            &row->ValueChildTail : &row->VolatileChildTail;
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


    APCHandleAndRetirement::SeqLockedOperation APCHandleAndRetirement::SwitchDescriptionState(
        uint32_t slot,
        StateOfAPC updated_state,
        StateOfAPC current_state,
        uint32_t max_tries 
    ) noexcept
    {
        if (
            slot > ADS::APC_INDEX_BOUND_SENTINAL ||
            slot >= FabCache_->CountOfAPC_
        )
        {
            return SeqLockedOperation::NONE;
        }

        HAS::LifeCycleControl* const ptr = LifeCycleControl_(slot);
        if (!ptr)
        {
            return SeqLockedOperation::NONE;
        }

        std::atomic_ref<HAS::LifeCycleControl> control(*ptr);

        for (uint32_t i = 0; i < max_tries; i++)
        {
            HAS::LifeCycleControl expected = control.load(std::memory_order_acquire);
            if (!HAS::ValidateLifeCycle(expected))
            {
                continue;
            }

            if (updated_state == expected.State)
            {
                return SeqLockedOperation::FOUND;
            }

            if (expected.State != current_state)
            {
                continue;
            }

            if (!DSA::IsTransitionStateLeagal(expected.State, updated_state))
            {
                return SeqLockedOperation::NONE;
            }
            
            HAS::LifeCycleControl desired = expected;
            ++desired.SeqLock;
            desired.State = updated_state;
            if (control.compare_exchange_strong(
                expected, desired,
                std::memory_order_acq_rel,
                std::memory_order_acquire
            ))
            {
                return SeqLockedOperation::FOUND;
            }
        }
        
        return SeqLockedOperation::RETRY;
    }

    APCHandleAndRetirement::SeqLockedOperation APCHandleAndRetirement::ReadAPCStateAtomically_(
        uint32_t slot,
        HAS::LifeCycleControl& life_cycle,
        uint32_t max_tries
    ) noexcept
    {
        life_cycle = {};
        if (
            slot > ADS::APC_INDEX_BOUND_SENTINAL ||
            slot >= FabCache_->CountOfAPC_
        )
        {
            return SeqLockedOperation::NONE;
        }

        HAS::LifeCycleControl* const ptr = LifeCycleControl_(slot);
        if (!ptr)
        {
            return SeqLockedOperation::NONE;
        }


        std::atomic_ref<HAS::LifeCycleControl> control(*ptr);
        for (uint32_t i = 0; i < max_tries; i++)
        {
            const HAS::LifeCycleControl before = control.load(std::memory_order_acquire);
            if (!HAS::ValidateLifeCycle(before))
            {
                continue;
            }

            const HAS::LifeCycleControl after = control.load(std::memory_order_acquire);
            if (before == after)
            {
                life_cycle = after;
                return SeqLockedOperation::FOUND;
            }
        }
        return SeqLockedOperation::RETRY;
    }

}
