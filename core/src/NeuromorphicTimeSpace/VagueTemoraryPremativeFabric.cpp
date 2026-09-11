#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    bool APCFinilizer::GetExistingAPC_(
        uint32_t slot,
        AdaptivePackedCellContainer& apc,
        APCUseScope& apc_use,
        std::optional<uint32_t> expected_generation
    ) noexcept
    {
        if (
            !IsFabricActive() ||
            slot >= FVolatileCache_.CountOfAPC_ ||
            apc.IsFabricBound_() ||
            apc_use ||
            (
                expected_generation.has_value() &&
                !HandleOfAPCStatic::IsGenerationValid(
                    expected_generation.value()
                )
            )
        )
        {
            return false;
        }

        const ADS::RangeOfAPC range = GetSegmentPoolRange(slot);
        uint64_t* generation_cell = GetAPCGenerationPtr_(slot);
        if (!generation_cell || !range.IsValid)
        {
            return false;
        }
        const uint64_t control_raw = std::atomic_ref<const uint64_t>(*generation_cell).load(std::memory_order_acquire);

        const HandleOfAPCStatic::ControlValues control_values = HandleOfAPCStatic::ReadControlCell(control_raw);
        if (
            control_values.Closed ||
            !HandleOfAPCStatic::IsGenerationValid(control_values.Generation) ||
            (
                expected_generation.has_value() &&
                control_values.Generation != expected_generation.value()
            )
        )
        {
            return false;
        }

        const uint32_t resolved_generation = control_values.Generation;

        if (!apc.BindExternalRawFabricBacking_(
            &SlabBasePtr_[range.BeginIndex],
            this,
            slot,
            generation_cell,
            resolved_generation
        ))
        {
            return false;
        }

        APCUseScope acquired_use = apc.AcquireAPCUse_();

        if (!acquired_use)
        {
            apc.ReleseFabricBindingOnly_();
            return false;
        }
        const DescriptionOfAPC::SeqLockAndStateStruct state = ReadAPCStateAtomically_(slot);
        if (
            !state.IsValid ||
            state.StateOfTheAPC != StateOfAPC::LIVE
        )
        {
            acquired_use.Release();
            apc.ReleseFabricBindingOnly_();
            return false;
        }
        apc_use = std::move(acquired_use);
        return true;
    }
    

    APCFinilizer::SeqLockedOperation APCFinilizer::ResolveChildLocator_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t locator,
        APCUseScope& use,
        AdaptivePackedCellContainer& child
    ) noexcept
    {
        child = AdaptivePackedCellContainer{};
        if (
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                locator,
                static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                FVolatileCache_.MaxDirectParentsPerAxis_
            )
        )
        {
            return SeqLockedOperation::NONE;
        }

        EdgeBuilder::ParentRelation relation{};
        const SeqLockedOperation read = ReadParentRelation_(
            edge_table,
            EdgeBuilder::RelationSlot(locator),
            EdgeBuilder::RelationOrdinal(locator),
            relation,
            DEFAULT_INTERNAL_TRIES__
        ); 

        if (read != SeqLockedOperation::FOUND)
        {
            return read;
        }

        if (
            relation.ParentHandle != EdgeBuilder::MakeParentHandle(parent_slot, parent_generation)
        )
        {
            return SeqLockedOperation::RETRY;
        }


        APCUseScope child_use{};
        if (
            !GetExistingAPC_(
                EdgeBuilder::RelationSlot(locator),
                child,
                child_use
            )
        )
        {
            return SeqLockedOperation::NONE;
        }
        
        if (
            child.APCCache_.FabricOwnerPtr_ != this ||
            child.APCCache_.APCSlotIdx_ != EdgeBuilder::RelationSlot(locator)
        )
        {
            return SeqLockedOperation::RETRY;
        }

        EdgeBuilder::ParentRelation confirmed{};
        const SeqLockedOperation confirm_read = ReadParentRelation_(
            edge_table,
            EdgeBuilder::RelationSlot(locator),
            EdgeBuilder::RelationOrdinal(locator),
            confirmed,
            DEFAULT_INTERNAL_TRIES__
        );
        if (confirm_read != SeqLockedOperation::FOUND)
        {
            return confirm_read;
        }
        if (
            confirmed.ParentHandle !=
            EdgeBuilder::MakeParentHandle(parent_slot, parent_generation)
        )
        {
            return SeqLockedOperation::RETRY;
        }

        use = std::move(child_use);
        return SeqLockedOperation::FOUND;
    }


    AdaptivePackedCellContainer APCFinilizer::FindParent_(
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        uint8_t relation_ordinal,
        FabricToAPCLinker::RelationOparation* result_ptr,
        uint32_t max_tries 
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        AdaptivePackedCellContainer parent{};
        
        if (
            child_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationOrdinal(
                relation_ordinal,
                FVolatileCache_.MaxDirectParentsPerAxis_
            )
        )
        {
            return parent;
        }
        
        for (uint32_t i = 0; i < max_tries; i++)
        {
            EdgeBuilder::ParentRelation relation{};
            const SeqLockedOperation read = ReadParentRelation_(
                edge_table,
                child_slot,
                relation_ordinal,
                relation,
                DEFAULT_INTERNAL_TRIES__
            );

            if (read == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (read != SeqLockedOperation::FOUND)
            {
                return parent;
            }

            APCUseScope parent_use{};
            if (
                !GetExistingAPC_(
                    EdgeBuilder::ParentSlot(relation),
                    parent,
                    parent_use,
                    EdgeBuilder::ParentGeneration(relation)
                )
            )
            {
                continue;
            }
            
            if (
                parent.APCCache_.FabricOwnerPtr_ != this ||
                parent.APCCache_.ExpectedGeneration_ != EdgeBuilder::ParentGeneration(relation)
            )
            {
                continue;
            }

            result.Use_ = std::move(parent_use);
            result.RelationLocator_ = EdgeBuilder::PackRelationLocator(
                child_slot,
                relation_ordinal
            );
            result.MutationOP_ = SeqLockedOperation::FOUND;

            if (result_ptr)
            {
                *result_ptr = std::move(result);
            }

            return parent;
        }
        
        result.MutationOP_ = SeqLockedOperation::RETRY;
        if (result_ptr)
        {
            *result_ptr = std::move(result);
        }
        
        return parent;
    }


    AdaptivePackedCellContainer APCFinilizer::FindFirstChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        FabricToAPCLinker::RelationOparation* result_ptr,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        AdaptivePackedCellContainer child{};


        if (
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table)
        )
        {
            return child;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            child = AdaptivePackedCellContainer{};
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return child;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return child;
            }
            if (before.TailLocator == EdgeBuilder::RELATION_NULL)
            {
                return child;
            }

            EdgeBuilder::ParentRelation tail_relation{};
            const SeqLockedOperation tail_read = ReadParentRelation_(
                edge_table,
                EdgeBuilder::RelationSlot(before.TailLocator),
                EdgeBuilder::RelationOrdinal(before.TailLocator),
                tail_relation,
                1u
            );

            if (tail_read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (
                tail_read != SeqLockedOperation::FOUND ||
                tail_relation.ParentHandle != EdgeBuilder::MakeParentHandle(
                    parent_slot,
                    parent_generation
                )
            )
            {
                return child;
            }

            const uint32_t first = EdgeBuilder::NextLocator(tail_relation);
            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                first,
                result.Use_,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                child = AdaptivePackedCellContainer{};
                return child;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.RelationLocator_ = first;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            if (result_ptr)
            {
                *result_ptr = std::move(result);
            }
            
            return child;
        }
        result.MutationOP_ = SeqLockedOperation::RETRY;

        if (result_ptr)
        {
            *result_ptr = std::move(result);
        }

        child = AdaptivePackedCellContainer{};
        return child;
    }

    AdaptivePackedCellContainer APCFinilizer::FindLastChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        FabricToAPCLinker::RelationOparation* result_ptr,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        AdaptivePackedCellContainer child{};
        if (
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table)
        )
        {
            return child;
        }
        
        for (uint32_t i = 0; i < max_tries; i++)
        {
            child = AdaptivePackedCellContainer{};
            EdgeBuilder::EdgeData before{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, before)
            )
            {
                return child;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return child;
            }
            
            if (before.TailLocator == EdgeBuilder::RELATION_NULL)
            {
                return child;
            }

            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                before.TailLocator,
                result.Use_,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                return child;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.RelationLocator_ = before.TailLocator;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            if (result_ptr)
            {
                *result_ptr = std::move(result);
            }
            return child;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;

        if (result_ptr)
        {
            *result_ptr = std::move(result);
        }

        child = AdaptivePackedCellContainer{};
        return child;
    }


    AdaptivePackedCellContainer APCFinilizer::FindNextChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        FabricToAPCLinker::RelationOparation* result_ptr,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        AdaptivePackedCellContainer child{};
        if (
            parent_slot >= FVolatileCache_.CountOfAPC_  ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                current_relation_locator,
                static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                FVolatileCache_.MaxDirectParentsPerAxis_
            )
        )
        {
            return child;
        }
        

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(parent_slot, parent_generation);

        for (uint32_t i = 0; i < max_tries; i++)
        {
            child = AdaptivePackedCellContainer{};
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return child;
            }
            
            if (
                before.Status == EdgeBuilder::EdgeStatus::RESERVED
            )
            {
                continue;
            }

            if (
                before.Status != EdgeBuilder::EdgeStatus::LIVE ||
                before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                return child;
            }

            EdgeBuilder::ParentRelation current{};
            const SeqLockedOperation current_read = ReadParentRelation_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator),
                EdgeBuilder::RelationOrdinal(current_relation_locator),
                current,
                DEFAULT_INTERNAL_TRIES__
            );

            if (current_read == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (
                current_read != SeqLockedOperation::FOUND ||
                current.ParentHandle != parent_handle
            )
            {
                return child;
            }
            
            if (current_relation_locator == before.TailLocator)
            {
                EdgeBuilder::EdgeData after{};
                if (
                    ReadEdgeHeader_(edge_table, parent_slot, after) &&
                    SameHeader_(before, after) 
                )
                {
                    return child;
                }
                continue;
            }
            
            const uint32_t next = EdgeBuilder::NextLocator(current);

            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                next,
                result.Use_,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (resolved != SeqLockedOperation::FOUND)
            {
                child = AdaptivePackedCellContainer{};
                return child;
            }

            EdgeBuilder::EdgeData after{};

            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after)
            )
            {
                continue;
            }

            result.RelationLocator_ = next;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            if (result_ptr)
            {
                *result_ptr = std::move(result);
            }
            
            return child;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;

        if (result_ptr)
        {
            *result_ptr = std::move(result);
        }

        child = AdaptivePackedCellContainer{};
        return child;
    }

    AdaptivePackedCellContainer APCFinilizer::FindPreviousChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        FabricToAPCLinker::RelationOparation* result_ptr,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        AdaptivePackedCellContainer child{};

        if (
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                current_relation_locator,
                static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                FVolatileCache_.MaxDirectParentsPerAxis_
            )
        )
        {
            return child;
        }

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            child = AdaptivePackedCellContainer{};
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return child;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            if (
                before.Status != EdgeBuilder::EdgeStatus::LIVE ||
                before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                return child;
            }

            EdgeBuilder::ParentRelation tail{};
            EdgeBuilder::ParentRelation current{};

            const SeqLockedOperation tail_read = ReadParentRelation_(
                edge_table,
                EdgeBuilder::RelationSlot(before.TailLocator),
                EdgeBuilder::RelationOrdinal(before.TailLocator),
                tail,
                1u
            );
            const SeqLockedOperation current_read = ReadParentRelation_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator),
                EdgeBuilder::RelationOrdinal(current_relation_locator),
                current,
                1u
            );

            if (
                tail_read == SeqLockedOperation::RETRY ||
                current_read == SeqLockedOperation::RETRY
            )
            {
                continue;
            }
            if (
                tail_read != SeqLockedOperation::FOUND ||
                current_read != SeqLockedOperation::FOUND ||
                tail.ParentHandle != parent_handle ||
                current.ParentHandle != parent_handle
            )
            {
                return child;
            }

            const uint32_t first = EdgeBuilder::NextLocator(tail);
            if (current_relation_locator == first)
            {
                EdgeBuilder::EdgeData after{};
                if (
                    ReadEdgeHeader_(edge_table, parent_slot, after) &&
                    ConstructDAGOnEachAxis::SameHeader_(before, after)
                )
                {
                    return child;
                }
                continue;
            }

            const uint32_t previous = EdgeBuilder::PreviousLocator(current);

            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                previous,
                result.Use_,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                child = AdaptivePackedCellContainer{};
                return child;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.RelationLocator_ = previous;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            if (result_ptr)
            {
                *result_ptr = std::move(result);
            }
            
            return child;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;

        if (result_ptr)
        {
            *result_ptr = std::move(result);
        }

        child = AdaptivePackedCellContainer{};
        return child;
    }


    bool APCFinilizer::CreateAPC(
        AdaptivePackedCellContainer& desired_apc,
        const SchemaDefinition::RegionSchemaTable& region_schemas,
        uint32_t internal_max_tries,
        bool override_table 
    ) noexcept
    {
        if (
            !IsFabricActive() ||
            desired_apc.IsFabricBound_()
        )
        {
            return false;
        }

        const std::optional<uint32_t> slot_new = GetASlotForNewAPCLink();
        if (!slot_new.has_value())
        {
            return false;
        }
        
        const uint32_t slot = slot_new.value();
        EdgeBuilder::EdgeData horizontal_before{};
        EdgeBuilder::EdgeData vertical_before{};

        bool horizontal_reserved = false;
        bool vertical_reserved = false;
        bool descriptor_live = false;
        bool matrix_view_prepared = false;

        auto AbortCreation___ = [&]()
        {
            if (descriptor_live)
            {
                SwitchDescriptionState(
                    slot,
                    StateOfAPC::RESERVED,
                    StateOfAPC::LIVE,
                    internal_max_tries
                );
            }
            
            if (vertical_reserved)
            {
                PublishReservedEdgeRow_(
                    FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                    slot,
                    vertical_before,
                    EdgeBuilder::RELATION_NULL,
                    EdgeBuilder::EdgeStatus::FREE
                );
            }

            if (horizontal_reserved)
            {
                PublishReservedEdgeRow_(
                    FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                    slot,
                    horizontal_before,
                    EdgeBuilder::RELATION_NULL,
                    EdgeBuilder::EdgeStatus::FREE
                );
            }

            if (matrix_view_prepared)
            {
                ClearMatrixViewRow_(slot);
            }
            

            SwitchDescriptionState(
                slot,
                StateOfAPC::FREE,
                StateOfAPC::RESERVED,
                internal_max_tries
            );
            desired_apc.ReleseFabricBindingOnly_();
        };

        uint64_t* generation_cell = GetAPCGenerationPtr_(slot);
        if (!generation_cell)
        {
            AbortCreation___();
            return false;
        }
        
        const uint64_t control_raw = std::atomic_ref<const uint64_t>(*generation_cell).load(std::memory_order_acquire);

        const HandleOfAPCStatic::ControlValues control_values = HandleOfAPCStatic::ReadControlCell(control_raw);

        if (
            !control_values.Closed ||
            control_values.ActiveAccess != UNSIGNED_ZERO ||
            !HandleOfAPCStatic::IsGenerationValid(control_values.Generation)
        )
        {
            AbortCreation___();
            return false;
        }

        if (
            !HasDefaultRegionTable_ ||
            override_table ||
            control_values.Generation != HandleOfAPCStatic::FIRST_GENERATION

        )
        {
            if (!PrepareMatrixViewRow_(slot, HasDefaultRegionTable_ && !override_table ? DefaultRegionTable_ : region_schemas))
            {
                AbortCreation___();
                return false;
            }
            matrix_view_prepared = true;

            if (!InitializeRegionProtocolStorage_(slot))
            {
                AbortCreation___();
                return false;
            }
        }
        
        const ADS::RangeOfAPC range = GetSegmentPoolRange(slot);
        if (
            !range.IsValid ||
            !desired_apc.BindExternalRawFabricBacking_(
                &SlabBasePtr_[range.BeginIndex],
                this,
                slot,
                generation_cell,
                control_values.Generation
            ) ||
            !desired_apc.InitiateAPCMetaHeader()
        )
        {
            AbortCreation___();
            return false;
        }
        
        if (
            ReserveEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                EdgeBuilder::EdgeStatus::FREE,
                horizontal_before,
                internal_max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            AbortCreation___();
            return false;
        }
        horizontal_reserved = true;

        if (
            ReserveEdgeRow_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                EdgeBuilder::EdgeStatus::FREE,
                vertical_before,
                internal_max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            AbortCreation___();
            return false;
        }
        vertical_reserved = true;

        auto ReservedRowIsEmpty___ = [&](FabricSegments table) noexcept -> bool
        {
            std::span<EdgeBuilder::ParentRelation> relations = ParentRelations_(table, slot);
            if (relations.size() != FVolatileCache_.MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0; ordinal < FVolatileCache_.MaxDirectParentsPerAxis_; ordinal++)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(relations[ordinal].ParentHandle).load(std::memory_order_relaxed);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(relations[ordinal].SiblingLocators).load(std::memory_order_relaxed);
                if (!EdgeBuilder::IsEmpty(relation))
                {
                    return false;
                }
            }
            return true;
        };

        if (
            horizontal_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            vertical_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            !ReservedRowIsEmpty___(FabricSegments::VALUE_PARENT_EDGE_TABLE_H) ||
            !ReservedRowIsEmpty___(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V)
        )
        {
            AbortCreation___();
            return false;
        }
        
        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::LIVE,
            StateOfAPC::RESERVED,
            internal_max_tries
        ))
        {
            AbortCreation___();
            return false;
        }
        
        descriptor_live = true;
        if (!OpenAPCGeneration_(slot, control_values.Generation))
        {
            AbortCreation___();
            return false;
        }
        

        PublishReservedEdgeRow_(
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            slot,
            horizontal_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::LIVE
        );
        horizontal_reserved = false;

        PublishReservedEdgeRow_(
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
            slot,
            vertical_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::LIVE
        );
        vertical_reserved = false;
        
        return true;
    }

    std::optional<uint32_t> APCFinilizer::GetASlotForNewAPCLink() noexcept
    {
        if (
            !FabricInitialized_.load(std::memory_order_acquire) ||
            !SlabBasePtr_ || 
            !ADS::IsCapacityOfAPCValid(FVolatileCache_.PerAPCRuntimeCellCount_)
        )
        {
            return std::nullopt;
        }

        std::optional<uint32_t> maybe_First_free = ReadFirstFreeAPCIdx_();

        if (maybe_First_free.has_value())
        {
            for (uint32_t description_idx = maybe_First_free.value(); description_idx < FVolatileCache_.CountOfAPC_; description_idx++)
            {
                const DSA::SeqLockAndStateStruct current = ReadAPCStateAtomically_(description_idx);
                if (
                    !current.IsValid ||
                    current.StateOfTheAPC != StateOfAPC::FREE
                )
                {
                    continue;
                }
                if (!SwitchDescriptionState(
                    description_idx,
                    StateOfAPC::RESERVED,
                    StateOfAPC::FREE
                ))
                {
                    continue;
                }
                uint64_t expected = maybe_First_free.value();
                UpdateFirstFreeIdx_(expected, description_idx);
                return description_idx;
            }
        }

        if (maybe_First_free.has_value())
        {
            uint64_t expected = maybe_First_free.value();
            UpdateFirstFreeIdx_(expected, FABRIC_CELL_SENTINAL);
        }
        
        for (uint32_t slot = 0; slot < FVolatileCache_.CountOfAPC_; slot++)
        {
            const DSA::SeqLockAndStateStruct current = ReadAPCStateAtomically_(slot);

            if (
                current.IsValid &&
                current.StateOfTheAPC == StateOfAPC::RETIRED &&
                ReclaimRetiredSlotTemp_(slot)
            )
            {
                return slot;
            }
        }
        return std::nullopt;
    }

    bool APCFinilizer::RetireAPC_(
        uint32_t slot,
        uint32_t generation,
        uint32_t max_tries
    ) noexcept
    {
        if (
            slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(generation)
        )
        {
            return false;
        }

        EdgeBuilder::EdgeData horizontal_before{};
        EdgeBuilder::EdgeData vertical_before{};

        if (
            ReserveEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                EdgeBuilder::EdgeStatus::LIVE,
                horizontal_before,
                max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            return false;
        }

        if (
            ReserveEdgeRow_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                EdgeBuilder::EdgeStatus::LIVE,
                vertical_before,
                max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            PublishReservedEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal_before,
                horizontal_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
            return false;
        }

        auto ReleaseRows___ = [&]() noexcept
        {
            PublishReservedEdgeRow_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                vertical_before,
                vertical_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
            PublishReservedEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal_before,
                horizontal_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
        };

        auto ReservedRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);

            if (relations.size() != FVolatileCache_.MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0u;
                ordinal < FVolatileCache_.MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(
                    relations[ordinal].ParentHandle
                ).load(std::memory_order_relaxed);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(
                    relations[ordinal].SiblingLocators
                ).load(std::memory_order_relaxed);

                if (!EdgeBuilder::IsEmpty(relation))
                {
                    return false;
                }
            }
            return true;
        };

        if (
            horizontal_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            vertical_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            !ReservedRowIsEmpty___(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H
            ) ||
            !ReservedRowIsEmpty___(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
            )
        )
        {
            ReleaseRows___();
            return false;
        }


        if (
            !CloseAPCGeneration_(slot, generation)
        )
        {
            ReleaseRows___();
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RESERVED,
            StateOfAPC::LIVE,
            max_tries
        ))
        {
            OpenAPCGeneration_(slot, generation);
            ReleaseRows___();
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RETIRED,
            StateOfAPC::RESERVED,
            max_tries
        ))
        {
            SwitchDescriptionState(
                slot,
                StateOfAPC::LIVE,
                StateOfAPC::RESERVED,
                max_tries
            );
            OpenAPCGeneration_(slot, generation);
            ReleaseRows___();
            return false;
        }

        PublishReservedEdgeRow_(
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
            slot,
            vertical_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        PublishReservedEdgeRow_(
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            slot,
            horizontal_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        return true;
    }

    bool APCFinilizer::ReclaimRetiredSlotTemp_(uint32_t slot) noexcept
    {
        if (slot >= FVolatileCache_.CountOfAPC_)
        {
            return false;
        }

        const ADS::RangeOfAPC range = GetSegmentPoolRange(slot);
        if (!range.IsValid)
        {
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RESERVED,
            StateOfAPC::RETIRED,
            DEFAULT_MAX_TRIES
        ))
        {
            return false;
        }

        auto RestoreRetired___ = [&]() noexcept
        {
            SwitchDescriptionState(
                slot,
                StateOfAPC::RETIRED,
                StateOfAPC::RESERVED,
                DEFAULT_MAX_TRIES
            );
        };

        EdgeBuilder::EdgeData horizontal{};
        EdgeBuilder::EdgeData vertical{};

        if (
            !ReadEdgeHeader_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal
            ) ||
            !ReadEdgeHeader_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                vertical
            ) ||
            horizontal.Status != EdgeBuilder::EdgeStatus::FREE ||
            vertical.Status != EdgeBuilder::EdgeStatus::FREE ||
            horizontal.TailLocator != EdgeBuilder::RELATION_NULL ||
            vertical.TailLocator != EdgeBuilder::RELATION_NULL
        )
        {
            RestoreRetired___();
            return false;
        }

        auto FreeRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);

            if (relations.size() != FVolatileCache_.MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0u;
                ordinal < FVolatileCache_.MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(
                    relations[ordinal].ParentHandle
                ).load(std::memory_order_acquire);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(
                    relations[ordinal].SiblingLocators
                ).load(std::memory_order_acquire);

                if (!EdgeBuilder::IsEmpty(relation))
                {
                    return false;
                }
            }
            return true;
        };

        if (
            !FreeRowIsEmpty___(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H
            ) ||
            !FreeRowIsEmpty___(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
            )
        )
        {
            RestoreRetired___();
            return false;
        }

        uint32_t new_generation = 0u;
        if (!AdvanceClosedAPCGeneration_(slot, new_generation))
        {
            RestoreRetired___();
            return false;
        }

        const size_t lifecycle_index =
            range.BeginIndex +
            static_cast<size_t>(ADS::HeaderIdentifierOfAPC::APC_LIFE_CYCLE);

        for (size_t idx = range.BeginIndex; idx < range.EndIndex; ++idx)
        {
            if (idx != lifecycle_index)
            {
                DirectlyStoreFabricUnit64(idx, 0u);
            }
        }

        return HandleOfAPCStatic::IsGenerationValid(new_generation);
    }


    constexpr bool APCFinilizer::IsNodePolicyReConfigurable_(const SD::RegionSchemaTable& table) noexcept
    {
        if (!HasDefaultRegionTable_)
        {
            return true;
        }

        for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
        {
            const SD::RegionSchemaRecord& expected = DefaultRegionTable_[i];

            const SD::RegionSchemaRecord& supplied = table[i];
            const bool expected_disabled = SD::HasSchemaFlag(expected.Flags, SD::SchemaFlags::REGION_DISABLED);
            const bool supplied_disabled = SD::HasSchemaFlag(supplied.Flags, SD::SchemaFlags::REGION_DISABLED);

            if (
                supplied.Region != expected.Region ||
                expected_disabled != supplied_disabled
            )
            {
                return false;
            }

            if (expected_disabled)
            {
                if (supplied.Flags != expected.Flags)
                {
                    return false;
                }
                continue;
            }
            
            if (
                supplied.Dtype != expected.Dtype ||
                supplied.MatrixHeight != expected.MatrixHeight ||
                supplied.MatrixWidth != expected.MatrixWidth ||
                supplied.CellCount != expected.CellCount ||
                supplied.Flags != expected.Flags
            )
            {
                return false;
            }

            // An exceptional node may change synchronization policy, but not
            // physical record count or vector geometry.
            switch (supplied.Protocol)
            {
            case SD::SchemaProtocols::PRIVATE_REGION:
            case SD::SchemaProtocols::IMMUTABLE_SNAPSHOT:
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
                break;

            default:
                return false;
            }

            return true;
            
        }
        
    }
}
