#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    void VagueTemoraryPremativeFabric::ShutDownFabricWithPtrTable() noexcept
    {
        const bool was_active = FabricInitialized_.exchange(false, std::memory_order_acq_rel);
        if (was_active && SlabBasePtr_)
        {
            for (uint32_t i = 0; i < CountOfAPC_; i++)
            {
                std::atomic_ref<uint64_t>(*GetAPCGenerationPtr_(i)).fetch_or(
                    HandleOfAPCStatic::CLOSED_MASK,
                    std::memory_order_acq_rel
                );
            }

            for (uint32_t i = 0; i < CountOfAPC_; i++)
            {
                std::atomic_ref<uint64_t> control(*GetAPCGenerationPtr_(i));
                while (HandleOfAPCStatic::ReadControlCell(control.load(std::memory_order_acquire)).ActiveAccess != UNSIGNED_ZERO)
                {
                    std::this_thread::yield();
                }
            }
        }

        uint64_t* old_ptr = SlabBasePtr_;
        const size_t old_count = SlabCellCount_;
        SlabBasePtr_ = nullptr;
        SlabCellCount_ = UNSIGNED_ZERO;
        if (old_ptr)
        {
            FreeRawPackedCells_(old_ptr,old_count);
        }
        CompiledDagTableBeginIdx_ = UNSIGNED_ZERO;
        CompiledDagRevision_.fetch_add(1, std::memory_order_release);
        ResetScalarsofTheFabric_();
    }


    bool VagueTemoraryPremativeFabric::BuildAPCRuntimePtrTable_() noexcept
    {
        if (CountOfAPC_ == UNSIGNED_ZERO)
        {
            return false;
        }
        APCRuntimePtrTable_.reset(new (std::nothrow) std::atomic<AdaptivePackedCellContainer*>[static_cast<size_t>(CountOfAPC_)]);

        if (!APCRuntimePtrTable_)
        {
            return false;
        }

        for (size_t i = 0; i < static_cast<size_t>(CountOfAPC_); i++)
        {
            APCRuntimePtrTable_[i].store(nullptr, std::memory_order_release);
        }
        
        return true;
    }

    void VagueTemoraryPremativeFabric::ClearAPCRuntimePtrTable_() noexcept
    {
        if (!APCRuntimePtrTable_)
        {
            return;
        }
        for (size_t i = 0; i < static_cast<size_t>(CountOfAPC_); i++)
        {   
            AdaptivePackedCellContainer* apc = APCRuntimePtrTable_[i].exchange(nullptr, std::memory_order_acq_rel);
            if (apc)
            {
                apc->ReleseFabricBindingOnly_();
            }            
        }
    }

    bool VagueTemoraryPremativeFabric::StoreAPCRuntimePtr(size_t apc_idx, AdaptivePackedCellContainer* apc_ptr) noexcept
    {
        if (!APCRuntimePtrTable_ || apc_idx >= CountOfAPC_)
        {
            return false;
        }

        APCRuntimePtrTable_[apc_idx].store(apc_ptr, std::memory_order_release);
        return true;
    }

    AdaptivePackedCellContainer* VagueTemoraryPremativeFabric::GetAPCRuntimePtrBySlotIndex_(size_t apc_idx) noexcept
    {
        if (!APCRuntimePtrTable_ || apc_idx >= CountOfAPC_)
        {
            return nullptr;
        }

        return APCRuntimePtrTable_[apc_idx].load(std::memory_order_acquire);
    }


    bool VagueTemoraryPremativeFabric::InitializeFabricWithPtrTable(
        uint32_t slot_count,
        uint32_t slot_cell_count,
        const SchemaDefinition::FabricRegionConfig& region_configuration,
        uint8_t max_direct_parents_per_axis 
    ) noexcept
    {
        ShutDownFabricWithPtrTable();
        const bool base_ok = InitializeFabric(
            slot_count,
            slot_cell_count,
            region_configuration,
            max_direct_parents_per_axis
        );

        if (!base_ok)
        {
            return false;
        }

        if (!BuildAPCRuntimePtrTable_())
        {
            ShutDownFabricWithPtrTable();
            return false;
        }
        
        return true;
    }

    VagueTemoraryPremativeFabric::SeqLockedOperation VagueTemoraryPremativeFabric::ResolveChildLocator_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t locator,
        AdaptivePackedCellContainer*& child
    ) noexcept
    {
        child = nullptr;
        if (
            parent_slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                locator,
                static_cast<uint32_t>(CountOfAPC_),
                MaxDirectParentsPerAxis_
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

        AdaptivePackedCellContainer* candidate = GetAPCRuntimePtrBySlotIndex_(EdgeBuilder::RelationSlot(locator));

        if (!candidate)
        {
            return SeqLockedOperation::RETRY;
        }
        
        APCUseScope child_use = candidate->AcquireAPCUse_();
        if (
            !child_use ||
            candidate->Cache_.FabricOwnerPtr_ != this ||
            candidate->Cache_.APCSlotIdx_ != EdgeBuilder::RelationSlot(locator)
        )
        {
            return SeqLockedOperation::RETRY;
        }
        
        child = candidate;

        return SeqLockedOperation::FOUND;
    }


    FabricToAPCLinker::RelationOparation VagueTemoraryPremativeFabric::FindParent_(
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        uint8_t relation_ordinal,
        uint32_t max_tries 
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        if (
            child_slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationOrdinal(
                relation_ordinal,
                MaxDirectParentsPerAxis_
            )
        )
        {
            return result;
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
                return result;
            }

            AdaptivePackedCellContainer* parent = GetAPCRuntimePtrBySlotIndex_(EdgeBuilder::ParentSlot(relation));

            if (!parent)
            {
                continue;
            }
            
            APCUseScope parent_use = parent->AcquireAPCUse_();
            if (
                !parent_use ||
                parent->Cache_.FabricOwnerPtr_ != this ||
                parent->Cache_.ExpectedGeneration_ != EdgeBuilder::ParentGeneration(relation)
            )
            {
                continue;
            }

            result.APCPtr_ = parent;
            result.RelationLocator_ = EdgeBuilder::PackRelationLocator(child_slot, relation_ordinal);

            result.MutationOP_ = SeqLockedOperation::FOUND;
            return result;
        }
        result.MutationOP_ = SeqLockedOperation::RETRY;
        return result;
    }


    FabricToAPCLinker::RelationOparation VagueTemoraryPremativeFabric::FindFirstChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t max_tries 
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};

        if (
            parent_slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table)
        )
        {
            return result;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return result;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return result;
            }
            if (before.TailLocator == EdgeBuilder::RELATION_NULL)
            {
                return result;
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
                return result;
            }

            const uint32_t first =
                EdgeBuilder::NextLocator(tail_relation);
            AdaptivePackedCellContainer* child = nullptr;
            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                first,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                return result;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.APCPtr_ = child;
            result.RelationLocator_ = first;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            return result;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;
        return result;
    }

    FabricToAPCLinker::RelationOparation VagueTemoraryPremativeFabric::FindLastChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        if (
            parent_slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table)
        )
        {
            return result;
        }
        
        for (uint32_t i = 0; i < max_tries; i++)
        {
            EdgeBuilder::EdgeData before{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, before)
            )
            {
                return result;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return result;
            }
            
            if (before.TailLocator == EdgeBuilder::RELATION_NULL)
            {
                return result;
            }

            AdaptivePackedCellContainer* child = nullptr;
            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                before.TailLocator,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                return result;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.APCPtr_ = child;
            result.RelationLocator_ = before.TailLocator;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            return result;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;
        return result;
    }


    FabricToAPCLinker::RelationOparation VagueTemoraryPremativeFabric::FindNextChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};
        if (
            parent_slot >= CountOfAPC_  ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                current_relation_locator,
                static_cast<uint32_t>(CountOfAPC_),
                MaxDirectParentsPerAxis_
            )
        )
        {
            return result;
        }
        

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(parent_slot, parent_generation);

        for (uint32_t i = 0; i < max_tries; i++)
        {
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return result;
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
                return result;
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
                return result;
            }
            
            if (current_relation_locator == before.TailLocator)
            {
                EdgeBuilder::EdgeData after{};
                if (
                    ReadEdgeHeader_(edge_table, parent_slot, after) &&
                    SameHeader_(before, after) 
                )
                {
                    return result;
                }
                continue;
            }
            
            const uint32_t next = EdgeBuilder::NextLocator(current);
            AdaptivePackedCellContainer* child = nullptr;

            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                next,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (resolved != SeqLockedOperation::FOUND)
            {
                return result;
            }

            EdgeBuilder::EdgeData after{};

            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after)
            )
            {
                continue;
            }

            result.APCPtr_ = child;
            result.RelationLocator_ = next;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            return result;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;
        return result;
    }

    FabricToAPCLinker::RelationOparation VagueTemoraryPremativeFabric::FindPreviousChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        uint32_t max_tries
    ) noexcept
    {
        FabricToAPCLinker::RelationOparation result{};

        if (
            parent_slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                current_relation_locator,
                static_cast<uint32_t>(CountOfAPC_),
                MaxDirectParentsPerAxis_
            )
        )
        {
            return result;
        }

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return result;
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
                return result;
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
                return result;
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
                    return result;
                }
                continue;
            }

            const uint32_t previous = EdgeBuilder::PreviousLocator(current);
            AdaptivePackedCellContainer* child = nullptr;
            const SeqLockedOperation resolved = ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                previous,
                child
            );

            if (resolved == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (resolved != SeqLockedOperation::FOUND)
            {
                return result;
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !ConstructDAGOnEachAxis::SameHeader_(before, after)
            )
            {
                continue;
            }

            result.APCPtr_ = child;
            result.RelationLocator_ = previous;
            result.MutationOP_ = SeqLockedOperation::FOUND;
            return result;
        }

        result.MutationOP_ = SeqLockedOperation::RETRY;
        return result;
    }


    bool VagueTemoraryPremativeFabric::CreateAPC(
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
        bool pointer_stored = false;
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

            if (pointer_stored)
            {
                StoreAPCRuntimePtr(slot, nullptr);
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
            if (relations.size() != MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0; ordinal < MaxDirectParentsPerAxis_; ordinal++)
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
            !ReservedRowIsEmpty___(FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V) ||
            !StoreAPCRuntimePtr(slot, &desired_apc)
        )
        {
            AbortCreation___();
            return false;
        }

        pointer_stored = true;
        
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

    std::optional<uint32_t> VagueTemoraryPremativeFabric::GetASlotForNewAPCLink() noexcept
    {
        if (
            !FabricInitialized_.load(std::memory_order_acquire) ||
            !SlabBasePtr_ || 
            !ADS::IsCapacityOfAPCValid(PerAPCRuntimeCellCount_)
        )
        {
            return std::nullopt;
        }

        std::optional<uint32_t> maybe_First_free = ReadFirstFreeAPCIdx_();

        if (maybe_First_free.has_value())
        {
            for (uint32_t description_idx = maybe_First_free.value(); description_idx < CountOfAPC_; description_idx++)
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
        
        for (uint32_t slot = 0; slot < CountOfAPC_; slot++)
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


}
