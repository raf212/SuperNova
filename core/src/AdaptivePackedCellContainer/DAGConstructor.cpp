#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using MutationResult = AdaptivePackedCellContainer::MutationResult;

    bool DAGMutationConf::ValidateConditionalParentPublication_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        ConditionalParentPublication* publication
    ) noexcept
    {
        if (!publication)
            return true;

        DAGRowParticipant* const row = FindRowParticipant_(
            transaction,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );

        if (!row || !row->Reserved)
            return false;

        if (row->Before.SeqLock != publication->ExpectedRowSequence)
        {
            publication->SequenceMismatch = true;
            return false;
        }
        return true;
    }

    void DAGMutationConf::PrepareConditionalParentPublication_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint32_t relation_ordinal,
        ConditionalParentPublication* publication
    ) noexcept
    {
        if (!publication)
            return;

        DAGRowParticipant* const row = FindRowParticipant_(
            transaction,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );

        publication->PublishedOrdinal = relation_ordinal;
        publication->PublishedRowSequence = EdgeBuilder::NextSequence(
            EdgeBuilder::NextSequence(row->Before.SeqLock)
        );

        if (publication->Publish)
            publication->Publish(publication->Context, relation_ordinal);
    }

    void CompiledDAGTableConstructor::CompiledDAGRelation_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint32_t relation_ordinal,
        const EdgeBuilder::ParentRelation& relation
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            child_slot >= FabCache_->CountOfAPC_ ||
            relation_ordinal >= FabCache_->MaxDirectParentsPerAxis_ ||
            EdgeBuilder::IsPartiallyEmpty(relation)
        )
        {
            return;
        }

        EB::PMSpan mask = ParentMask_(edge_table, child_slot);
        const uint32_t word_index = relation_ordinal / EB::PARENT_MASK_BITS_PER_BLOCK;
        const uint32_t bit_index = relation_ordinal % EB::PARENT_MASK_BITS_PER_BLOCK;
        if (word_index >= mask.size())
        {
            return;
        }
        
        EB::ParentMaskBlock block = std::atomic_ref<const EB::ParentMaskBlock>(mask[word_index]).load(std::memory_order_relaxed);
        std::bitset<EB::PARENT_MASK_BITS_PER_BLOCK> bits(block.Block);

        bits.set(bit_index, !EB::IsParentEmpty(relation));
        block.Block = bits.to_ullong();
        std::atomic_ref<EB::ParentMaskBlock>(mask[word_index]).store(
            block,
            std::memory_order_relaxed
        );
    }

    CompiledDAGTableConstructor::SeqLockedOperation CompiledDAGTableConstructor::ReadCompiledDAGParentMask_(
        FabricSegments edge_table,
        uint32_t slot,
        uint64_t& return_mask,
        uint32_t max_tries
    ) noexcept
    {
        return_mask = 0u;

        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            slot >= FabCache_->CountOfAPC_ ||
            FabCache_->MaxDirectParentsPerAxis_ > 64u ||
            max_tries == 0u
        )
        {
            return SeqLockedOperation::NONE;
        }

        HAS::ParentRowControl* const control = ParentRowControl_(edge_table, slot);
        EB::PMSpan mask = ParentMask_(edge_table, slot);

        if (!control || mask.empty())
        {
            return SeqLockedOperation::NONE;
        }

        for (size_t i = 0; i < max_tries; i++)
        {
            const HAS::ParentRowControl before = std::atomic_ref<const HAS::ParentRowControl>(*control).load(std::memory_order_acquire);

            if (!HAS::ValidParentControl(before))
            {
                return SeqLockedOperation::NONE;
            }

            if (before.Status == EB::EdgeStatus::RESERVED)
            {
                continue;
            }

            if (before.Status != EB::EdgeStatus::LIVE)
            {
                return SeqLockedOperation::NONE;
            }

            const EB::ParentMaskBlock observed = std::atomic_ref<const EB::ParentMaskBlock>(mask.front()).load(std::memory_order_acquire);

            const HAS::ParentRowControl after = std::atomic_ref<const HAS::ParentRowControl>(*control).load(std::memory_order_acquire);

            if (before != after)
            {
                continue;
            }
            
            return_mask = observed.Block;
            return SeqLockedOperation::FOUND;
        }
        
        return SeqLockedOperation::RETRY;
    }

    bool DAGMutationConf::AddRowParticipant_(
        DAGMutationTransaction& transaction,
        uint32_t slot,
        EdgeBuilder::EdgeDomain domain
    ) noexcept
    {
        if (slot >= FabCache_->CountOfAPC_)
        {
            return false;
        }

        uint8_t insert_at = transaction.RowCount;
        for (uint8_t i = 0u; i < transaction.RowCount; ++i)
        {
            DAGRowParticipant& current = transaction.Rows[i];
            if (current.Slot == slot && current.Domain == domain)
            {
                return true;
            }
            if (
                slot < current.Slot ||
                (
                    slot == current.Slot &&
                    static_cast<uint8_t>(domain) <
                        static_cast<uint8_t>(current.Domain)
                )
            )
            {
                insert_at = i;
                break;
            }
        }

        if (transaction.RowCount >= DAG_MAX_ROW_PARTICIPANTS)
        {
            return false;
        }
        for (uint8_t i = transaction.RowCount; i > insert_at; --i)
        {
            transaction.Rows[i] = transaction.Rows[i - 1u];
        }
        transaction.Rows[insert_at] = DAGRowParticipant{};
        transaction.Rows[insert_at].Slot = slot;
        transaction.Rows[insert_at].Domain = domain;
        ++transaction.RowCount;
        return true;
    }



    DAGMutationConf::DAGRowParticipant*
    DAGMutationConf::FindRowParticipant_(
        DAGMutationTransaction& transaction,
        uint32_t slot,
        EdgeBuilder::EdgeDomain domain
    ) noexcept
    {
        for (uint8_t i = 0u; i < transaction.RowCount; ++i)
        {
            if (
                transaction.Rows[i].Slot == slot &&
                transaction.Rows[i].Domain == domain
            )
            {
                return &transaction.Rows[i];
            }
        }
        return nullptr;
    }

    FabricToAPCLinker::SeqLockedOperation DAGMutationConf::ReserveAllRows_(
        DAGMutationTransaction& transaction,
        uint32_t max_tries
    ) noexcept
    {
        for (uint8_t i = 0u; i < transaction.RowCount; ++i)
        {
            DAGRowParticipant& row = transaction.Rows[i];
            SeqLockedOperation op = ReserveEdgeDomain_(
                transaction.EdgeTable,
                row.Slot,
                row.Domain,
                EdgeBuilder::EdgeStatus::LIVE,
                row.Before,
                max_tries
            );

            if (op != SeqLockedOperation::FOUND)
            {
                AbortRowTransaction_(transaction);
                return op;
            }
            row.WorkTail = row.Before.TailLocator;
            row.Reserved = true;
        }
        return SeqLockedOperation::FOUND;
    }

    DAGMutationConf::DAGRelationDelta* DAGMutationConf::FindOrInsertRelationDelta_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint32_t ordinal
    ) noexcept
    {
        if (
            child_slot >= FabCache_->CountOfAPC_ ||
            !EdgeBuilder::IsValidRelationOrdinal(
                ordinal,
                FabCache_->MaxDirectParentsPerAxis_
            )
        )
        {
            return nullptr;
        }

        for (uint8_t i = 0u; i < transaction.RelationCount; ++i)
        {
            DAGRelationDelta& delta = transaction.Relations[i];
            if (delta.ChildSlot == child_slot && delta.Ordinal == ordinal)
            {
                return &delta;
            }
        }
        if (transaction.RelationCount >= DAG_MAX_RELATION_DELTAS)
        {
            return nullptr;
        }

        const std::span<EdgeBuilder::ParentRelation> relations =
            EdgeRelationsPerSlot_(transaction.EdgeTable, child_slot);
        if (relations.size() != FabCache_->MaxDirectParentsPerAxis_)
        {
            return nullptr;
        }

        DAGRelationDelta& inserted =
            transaction.Relations[transaction.RelationCount++];
        inserted.ChildSlot = child_slot;
        inserted.Ordinal = ordinal;
        inserted.Before.Parent = std::atomic_ref<const EB::ParentIDGeneration>(relations[ordinal].Parent).load(std::memory_order_relaxed);
        inserted.Before.Siblings = std::atomic_ref<const EB::SiblingLinks>(relations[ordinal].Siblings).load(std::memory_order_relaxed);
        inserted.Work = inserted.Before;
        return &inserted;
    }

    DAGMutationConf::DAGRelationDelta*
    DAGMutationConf::EditReservedParentHandle_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint32_t ordinal
    ) noexcept
    {
        DAGRowParticipant* const owner = FindRowParticipant_(
            transaction,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );
        if (!owner || !owner->Reserved)
        {
            return nullptr;
        }
        DAGRelationDelta* const delta = FindOrInsertRelationDelta_(
            transaction,
            child_slot,
            ordinal
        );
        if (delta)
        {
            delta->ParentHandleDirty = true;
        }
        return delta;
    }

    DAGMutationConf::DAGRelationDelta*
    DAGMutationConf::EditReservedSiblingLocators_(
        DAGMutationTransaction& transaction,
        uint32_t owner_parent_slot,
        uint32_t relation_locator
    ) noexcept
    {
        DAGRowParticipant* const owner = FindRowParticipant_(
            transaction,
            owner_parent_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST
        );
        if (
            !owner ||
            !owner->Reserved ||
            !EdgeBuilder::IsValidRelationLocator(
                relation_locator,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            )
        )
        {
            return nullptr;
        }

        DAGRelationDelta* const delta = FindOrInsertRelationDelta_(
            transaction,
            EdgeBuilder::RelationSlot(relation_locator, FabCache_->MaxDirectParentsPerAxis_),
            EdgeBuilder::RelationOrdinal(relation_locator, FabCache_->MaxDirectParentsPerAxis_)
        );
        if (delta)
        {
            delta->SiblingLocatorsDirty = true;
        }
        return delta;
    }


    void DAGMutationConf::CommitRowTransaction_(
        DAGMutationTransaction& transaction
    ) noexcept
    {
        for (uint8_t i = transaction.RelationCount; i > 0u; --i)
        {
            const DAGRelationDelta& delta = transaction.Relations[i - 1u];
            if (delta.ParentHandleDirty)
            {
                StoreReservedParentHandle_(
                    transaction.EdgeTable,
                    delta.ChildSlot,
                    delta.Ordinal,
                    delta.Work.Parent
                );
                if (
                    EdgeBuilder::IsParentEmpty(delta.Before) !=
                    EdgeBuilder::IsParentEmpty(delta.Work)
                )
                {
                    CompiledDAGRelation_(
                        transaction.EdgeTable,
                        delta.ChildSlot,
                        delta.Ordinal,
                        delta.Work
                    );
                }
            }
            if (delta.SiblingLocatorsDirty)
            {
                StoreReservedSiblingLocators_(
                    transaction.EdgeTable,
                    delta.ChildSlot,
                    delta.Ordinal,
                    delta.Work.Siblings
                );
            }
        }

        if (TrackDAGRevision_.load(std::memory_order_relaxed))
        {
            SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        }

        const auto PublishDomain___ = [this, &transaction](
            EdgeBuilder::EdgeDomain domain
        ) noexcept -> void
        {
            for (uint8_t i = transaction.RowCount; i > 0u; --i)
            {
                DAGRowParticipant& row = transaction.Rows[i - 1u];
                if (row.Domain != domain)
                {
                    continue;
                }

                PublishReservedEdgeDomain_(
                    transaction.EdgeTable,
                    row.Slot,
                    row.Domain,
                    row.Before,
                    row.WorkTail,
                    EdgeBuilder::EdgeStatus::LIVE
                );

                row.Reserved = false;
            }
        };

        // Parent identity is the linearization publication. Child-list readers
        // remain excluded until their complete reverse-list state is publishable.
        PublishDomain___(EdgeBuilder::EdgeDomain::PARENT_RELATIONS);
        PublishDomain___(EdgeBuilder::EdgeDomain::CHILD_LIST);
    }

    void DAGMutationConf::AbortRowTransaction_(
        DAGMutationTransaction& transaction
    ) noexcept
    {
        for (uint8_t i = transaction.RowCount; i > 0u; --i)
        {
            DAGRowParticipant& row = transaction.Rows[i - 1u];
            if (!row.Reserved)
            {
                continue;
            }
            PublishReservedEdgeDomain_(
                transaction.EdgeTable,
                row.Slot,
                row.Domain,
                row.Before,
                row.Before.TailLocator,
                row.Before.Status
            );
            row.Reserved = false;
        }
    }



    bool ConstructDAGOnEachAxis::ScanReservedParentRow_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        EB::ParentIDGeneration wanted_parent_handle,
        EB::ParentIDGeneration other_parent_handle,
        ParentRowScan& scan
    ) noexcept
    {
        scan = ParentRowScan{};
        DAGRowParticipant* const owner = FindRowParticipant_(
            transaction,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );
        const std::span<EdgeBuilder::ParentRelation> relations =
            EdgeRelationsPerSlot_(transaction.EdgeTable, child_slot);
        if (
            !owner ||
            !owner->Reserved ||
            relations.size() != FabCache_->MaxDirectParentsPerAxis_
        )
        {
            return false;
        }

        for (uint32_t ordinal = 0u;
            ordinal < FabCache_->MaxDirectParentsPerAxis_;
            ++ordinal)
        {
            const EB::ParentIDGeneration handle = std::atomic_ref<const EB::ParentIDGeneration>(relations[ordinal].Parent).load(std::memory_order_relaxed);
            if (!HAS::IsGenerationValid(handle.Generation) || !APCDataStructure::IsValid32BitAPCUnit(handle.Slot))
            {
                if (scan.EmptyOrdinal == EB::RELATION_NULL)
                {
                    scan.EmptyOrdinal = ordinal;
                }
                continue;
            }
            if (handle == wanted_parent_handle)
            {
                if (scan.MatchOrdinal != EB::RELATION_NULL)
                {
                    return false;
                }
                scan.MatchOrdinal = ordinal;
                scan.MatchParent = handle;
            }
            if (
                APCDataStructure::IsValid32BitAPCUnit(handle.Slot) &&
                handle == other_parent_handle
            )
            {
                if (scan.OtherOrdinal != EB::RELATION_NULL)
                {
                    return false;
                }
                scan.OtherOrdinal = ordinal;
            }
        }
        return true;
    }

    MutationResult ConstructDAGOnEachAxis::AddParentRelation_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries,
        uint32_t internal_max_tries
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            parent_slot >= FabCache_->CountOfAPC_ ||
            child_slot >= FabCache_->CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !EdgeBuilder::CanInsertCombinedDAGRelation(parent_slot, child_slot)
        )
        {
            return MutationResult::INVALID;
        }

        const EB::ParentIDGeneration parent_handle{parent_generation, parent_slot};
        const EB::ParentIDGeneration other{EB::RELATION_NULL, EB::RELATION_NULL};

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;
            if (
                !AddRowParticipant_(
                    transaction,
                    child_slot,
                    EdgeBuilder::EdgeDomain::PARENT_RELATIONS
                ) ||
                !AddRowParticipant_(
                    transaction,
                    parent_slot,
                    EdgeBuilder::EdgeDomain::CHILD_LIST
                )
            )
            {
                return MutationResult::INVALID;
            }
            
            SeqLockedOperation op = ReserveAllRows_(transaction, internal_max_tries);
            if (op == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (op != SeqLockedOperation::FOUND)
            {
                return MutationResult::INVALID;
            }
            
            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            if (
                !IsOpenAPCGeneration_(parent_slot, parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            ParentRowScan scan{};
            DAGRowParticipant* const parent_list = FindRowParticipant_(
                transaction,
                parent_slot,
                EdgeBuilder::EdgeDomain::CHILD_LIST
            );
            if (
                !parent_list ||
                !ScanReservedParentRow_(
                    transaction,
                    child_slot,
                    parent_handle,
                    other,
                    scan
                ) ||
                scan.MatchOrdinal != EB::RELATION_NULL ||
                scan.EmptyOrdinal == EB::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.EmptyOrdinal,
                FabCache_->MaxDirectParentsPerAxis_
            );
            DAGRelationDelta* const moving_parent = EditReservedParentHandle_(
                transaction,
                child_slot,
                scan.EmptyOrdinal
            );
            DAGRelationDelta* const moving_siblings =
                EditReservedSiblingLocators_(transaction, parent_slot, self);
            if (
                !moving_parent ||
                moving_parent != moving_siblings ||
                !EdgeBuilder::IsParentEmpty(moving_parent->Before) ||
                !EdgeBuilder::IsSiblingEmpty(moving_parent->Before)
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t old_tail = parent_list->Before.TailLocator;
            moving_parent->Work.Parent = parent_handle;

            if (old_tail == EdgeBuilder::RELATION_NULL)
            {
                moving_parent->Work.Siblings = EB::SiblingLinks{self, self};

                parent_list->WorkTail = self;

                PrepareConditionalParentPublication_(
                    transaction,
                    child_slot,
                    scan.EmptyOrdinal,
                    publication
                );

                CommitRowTransaction_(transaction);
                return MutationResult::COMMITTED;
            }

            if (!EdgeBuilder::IsValidRelationLocator(
                old_tail,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            ))
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            DAGRelationDelta* const tail = EditReservedSiblingLocators_(
                transaction,
                parent_slot,
                old_tail
            );
            if (
                !tail ||
                tail->Before.Parent != parent_handle ||
                EdgeBuilder::IsSiblingEmpty(tail->Before)
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t first = tail->Before.Siblings.Next;

            if (!EdgeBuilder::IsValidRelationLocator(
                first,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            ))
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }
            DAGRelationDelta* const first_delta = EditReservedSiblingLocators_(
                transaction,
                parent_slot,
                first
            );
            if (
                !first_delta ||
                first_delta->Before.Parent != parent_handle ||
                first_delta->Before.Siblings.Previous != old_tail
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            moving_parent->Work.Siblings = EB::SiblingLinks{old_tail, first};
            tail->Work.Siblings = EB::SiblingLinks{tail->Work.Siblings.Previous, self};
            first_delta->Work.Siblings = EB::SiblingLinks{self, first_delta->Work.Siblings.Next};

            parent_list->WorkTail = self;
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.EmptyOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);
            return MutationResult::COMMITTED;
        }
        return MutationResult::RETRY;
    }

    MutationResult ConstructDAGOnEachAxis::RemoveParentRelation_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries,
        uint32_t internal_max_tries
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            parent_slot >= FabCache_->CountOfAPC_ ||
            child_slot >= FabCache_->CountOfAPC_ ||
            parent_slot == child_slot ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation)
        )
        {
            return MutationResult::INVALID;
        }

        const EB::ParentIDGeneration parent_handle{parent_generation, parent_slot};
        const EB::ParentIDGeneration other{EB::RELATION_NULL, EB::RELATION_NULL};

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;
            if (
                !AddRowParticipant_(
                    transaction,
                    child_slot,
                    EdgeBuilder::EdgeDomain::PARENT_RELATIONS
                ) ||
                !AddRowParticipant_(
                    transaction,
                    parent_slot,
                    EdgeBuilder::EdgeDomain::CHILD_LIST
                )
            )
            {
                return MutationResult::INVALID;
            }

            SeqLockedOperation op = ReserveAllRows_(transaction, internal_max_tries);
            if (op == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (op != SeqLockedOperation::FOUND)
            {
                return MutationResult::INVALID;
            }

            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            if (
                !IsOpenAPCGeneration_(parent_slot, parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            ParentRowScan scan{};
            DAGRowParticipant* const parent_list = FindRowParticipant_(
                transaction,
                parent_slot,
                EdgeBuilder::EdgeDomain::CHILD_LIST
            );
            if (
                !parent_list ||
                !ScanReservedParentRow_(
                    transaction,
                    child_slot,
                    parent_handle,
                    other,
                    scan
                ) ||
                scan.MatchOrdinal == EB::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.MatchOrdinal,
                FabCache_->MaxDirectParentsPerAxis_
            );

            DAGRelationDelta* const moving_parent = EditReservedParentHandle_(
                transaction,
                child_slot,
                scan.MatchOrdinal
            );
            DAGRelationDelta* const moving_siblings =
                EditReservedSiblingLocators_(transaction, parent_slot, self);
            if (
                !moving_parent ||
                moving_parent != moving_siblings ||
                moving_parent->Before.Parent != parent_handle ||
                EdgeBuilder::IsSiblingEmpty(moving_parent->Before) ||
                parent_list->Before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t previous = moving_parent->Before.Siblings.Previous;
            const uint32_t next = moving_parent->Before.Siblings.Next;

            if (
                !EdgeBuilder::IsValidRelationLocator(
                    previous,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ) ||
                !EdgeBuilder::IsValidRelationLocator(
                    next,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                )
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const bool singleton = previous == self && next == self;
            if (
                (singleton && parent_list->Before.TailLocator != self) ||
                (!singleton && (previous == self || next == self))
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            if (!singleton)
            {
                DAGRelationDelta* const previous_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        parent_slot,
                        previous
                    );
                DAGRelationDelta* const next_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        parent_slot,
                        next
                    );
                if (
                    !previous_delta ||
                    !next_delta ||
                    previous_delta->Before.Parent != parent_handle ||
                    next_delta->Before.Parent != parent_handle ||
                    previous_delta->Before.Siblings.Next != self ||
                    previous_delta->Before.Siblings.Previous != self
                )
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }
                previous_delta->Work.Siblings = EB::SiblingLinks{previous_delta->Work.Siblings.Previous, next};
                next_delta->Work.Siblings = EB::SiblingLinks{previous, next_delta->Work.Siblings.Next};
                if (parent_list->Before.TailLocator == self)
                {
                    parent_list->WorkTail = previous;
                }
            }
            else
            {
                parent_list->WorkTail = EdgeBuilder::RELATION_NULL;
            }

            moving_parent->Work.Parent = EB::ParentIDGeneration{EB::RELATION_NULL, EB::RELATION_NULL};
            moving_parent->Work.Siblings = EB::SiblingLinks{EB::RELATION_NULL, EB::RELATION_NULL};
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.MatchOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);
            return MutationResult::COMMITTED;
        }
        return MutationResult::RETRY;
    }

    MutationResult ConstructDAGOnEachAxis::ReplaceParentRelation_(
        uint32_t old_parent_slot,
        uint32_t old_parent_generation,
        uint32_t new_parent_slot,
        uint32_t new_parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries,
        uint32_t internal_max_tries
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            old_parent_slot >= FabCache_->CountOfAPC_ ||
            new_parent_slot >= FabCache_->CountOfAPC_ ||
            child_slot >= FabCache_->CountOfAPC_ ||
            old_parent_slot == new_parent_slot ||
            !HandleOfAPCStatic::IsGenerationValid(old_parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(new_parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !EdgeBuilder::CanInsertCombinedDAGRelation(
                new_parent_slot,
                child_slot
            )
        )
        {
            return MutationResult::INVALID;
        }

        const EB::ParentIDGeneration old_parent_handle {old_parent_generation, old_parent_slot};
        const EB::ParentIDGeneration new_parent_handle {new_parent_generation, new_parent_slot};

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;
            if (
                !AddRowParticipant_(
                    transaction,
                    child_slot,
                    EdgeBuilder::EdgeDomain::PARENT_RELATIONS
                ) ||
                !AddRowParticipant_(
                    transaction,
                    old_parent_slot,
                    EdgeBuilder::EdgeDomain::CHILD_LIST
                ) ||
                !AddRowParticipant_(
                    transaction,
                    new_parent_slot,
                    EdgeBuilder::EdgeDomain::CHILD_LIST
                ) 
            )
            {
                return MutationResult::INVALID;
            }

            SeqLockedOperation op = ReserveAllRows_(transaction, internal_max_tries);
            if (op == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (op != SeqLockedOperation::FOUND)
            {
                return MutationResult::INVALID;
            }

            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            if (
                !IsOpenAPCGeneration_(old_parent_slot, old_parent_generation) ||
                !IsOpenAPCGeneration_(new_parent_slot, new_parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            ParentRowScan scan{};
            DAGRowParticipant* const old_list = FindRowParticipant_(
                transaction,
                old_parent_slot,
                EdgeBuilder::EdgeDomain::CHILD_LIST
            );
            DAGRowParticipant* const new_list = FindRowParticipant_(
                transaction,
                new_parent_slot,
                EdgeBuilder::EdgeDomain::CHILD_LIST
            );
            if (
                !old_list ||
                !new_list ||
                !ScanReservedParentRow_(
                    transaction,
                    child_slot,
                    old_parent_handle,
                    new_parent_handle,
                    scan
                ) ||
                scan.MatchOrdinal == EB::RELATION_NULL ||
                scan.OtherOrdinal != EB::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.MatchOrdinal,
                FabCache_->MaxDirectParentsPerAxis_
            );

            DAGRelationDelta* const moving_parent = EditReservedParentHandle_(
                transaction,
                child_slot,
                scan.MatchOrdinal
            );
            DAGRelationDelta* const moving_old =
                EditReservedSiblingLocators_(
                    transaction,
                    old_parent_slot,
                    self
                );
            DAGRelationDelta* const moving_new =
                EditReservedSiblingLocators_(
                    transaction,
                    new_parent_slot,
                    self
                );
            if (
                !moving_parent ||
                moving_parent != moving_old ||
                moving_parent != moving_new ||
                moving_parent->Before.Parent != old_parent_handle ||
                EdgeBuilder::IsSiblingEmpty(moving_parent->Before) ||
                old_list->Before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const uint32_t old_previous = moving_parent->Before.Siblings.Previous;
            const uint32_t old_next = moving_parent->Before.Siblings.Next;
            if (
                !EdgeBuilder::IsValidRelationLocator(
                    old_previous,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ) ||
                !EdgeBuilder::IsValidRelationLocator(
                    old_next,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                )
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            const bool old_singleton =
                old_previous == self && old_next == self;
            if (
                (old_singleton && old_list->Before.TailLocator != self) ||
                (!old_singleton &&
                    (old_previous == self || old_next == self))
            )
            {
                AbortRowTransaction_(transaction);
                return MutationResult::INVALID;
            }

            if (old_singleton)
            {
                old_list->WorkTail = EdgeBuilder::RELATION_NULL;
            }
            else
            {
                DAGRelationDelta* const previous_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        old_parent_slot,
                        old_previous
                    );
                DAGRelationDelta* const next_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        old_parent_slot,
                        old_next
                    );
                if (
                    !previous_delta ||
                    !next_delta ||
                    previous_delta->Before.Parent != old_parent_handle ||
                    next_delta->Before.Parent != old_parent_handle ||
                    previous_delta->Before.Siblings.Next != self ||
                    next_delta->Before.Siblings.Previous != self
                )
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }

                previous_delta->Work.Siblings = EB::SiblingLinks{previous_delta->Work.Siblings.Previous, old_next};
                next_delta->Work.Siblings = EB::SiblingLinks{old_previous, next_delta->Work.Siblings.Next};
                if (old_list->Before.TailLocator == self)
                {
                    old_list->WorkTail = old_previous;
                }
            }

            const uint32_t new_tail = new_list->Before.TailLocator;
            if (new_tail == EdgeBuilder::RELATION_NULL)
            {
                moving_parent->Work.Siblings = EB::SiblingLinks{self, self};
            }
            else
            {
                if (!EdgeBuilder::IsValidRelationLocator(
                    new_tail,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ))
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }
                DAGRelationDelta* const tail_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        new_parent_slot,
                        new_tail
                    );
                if (
                    !tail_delta ||
                    tail_delta->Before.Parent != new_parent_handle ||
                    EdgeBuilder::IsSiblingEmpty(tail_delta->Before)
                )
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }
                const uint32_t new_first = tail_delta->Before.Siblings.Next;

                if (!EdgeBuilder::IsValidRelationLocator(
                    new_first,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ))
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }
                DAGRelationDelta* const first_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        new_parent_slot,
                        new_first
                    );
                if (
                    !first_delta ||
                    first_delta->Before.Parent != new_parent_handle ||
                    first_delta->Before.Siblings.Previous != new_tail
                )
                {
                    AbortRowTransaction_(transaction);
                    return MutationResult::INVALID;
                }
                tail_delta->Work.Siblings = EB::SiblingLinks{tail_delta->Work.Siblings.Previous, self};
                first_delta->Work.Siblings = EB::SiblingLinks{self, first_delta->Work.Siblings.Next};
                moving_parent->Work.Siblings = EB::SiblingLinks{new_tail, new_first};
            }

            moving_parent->Work.Parent = new_parent_handle;
            new_list->WorkTail = self;
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.MatchOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);            
            return MutationResult::COMMITTED;
        }
        return MutationResult::RETRY;
    }


}