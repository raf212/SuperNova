#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
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
        uint8_t relation_ordinal,
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

    CompiledDAGTableConstructor::CompiledDAGRecord* CompiledDAGTableConstructor::CompiledDAGRow_(uint32_t row_slot) noexcept
    {
        if (!SlabBasePtr_ || row_slot >= FabCache_->CountOfAPC_)
        {
            return nullptr;
        }
        
        const size_t row_begin = static_cast<size_t>(FabCache_->CompiledDAGTableBeginIdx_) +
            (static_cast<size_t>(row_slot) * CoreOfFabricCoordinator::COMPILED_DAG_LEN);
        
        if (
            row_begin >= FabCache_->SlabCellCount_ ||
            CoreOfFabricCoordinator::COMPILED_DAG_LEN > FabCache_->SlabCellCount_ - row_begin
        )
        {
            return nullptr;
        }

        return std::launder(reinterpret_cast<CompiledDAGRecord*>(SlabBasePtr_ + row_begin));
    }


    bool CompiledDAGTableConstructor::InitializeCompiledDAGTAble_() noexcept
    {
        RecordBookConf::FabricSegmentBounds bounds{};
        if (!GetRecordMapCarrierRanges_(FabricSegments::COMPILED_DAG_TABLE, bounds))
        {
            return false;
        }

        const size_t required_cells = static_cast<size_t>(FabCache_->CountOfAPC_) * CoreOfFabricCoordinator::COMPILED_DAG_LEN;

        if (
            bounds.EndIndex - bounds.BeginIndex != required_cells ||
            bounds.EndIndex > FabCache_->SlabCellCount_
        )
        {
            return false;
        }

        FabCache_->CompiledDAGTableBeginIdx_ = bounds.BeginIndex;

        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
        {
            const size_t row_begin = static_cast<size_t>(FabCache_->CompiledDAGTableBeginIdx_) +
                (static_cast<size_t>(i) * CoreOfFabricCoordinator::COMPILED_DAG_LEN);

            std::construct_at(reinterpret_cast<CompiledDAGRecord*>(SlabBasePtr_ + row_begin), CompiledDAGRecord{});
        }
        
        return true;
    }


    void CompiledDAGTableConstructor::CompiledDAGRelation_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
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

        CompiledDAGRecord* record = CompiledDAGRow_(child_slot);

        if (!record)
        {
            return;
        }

        uint64_t& stored_mask = edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ? 
            record->ValueParentMask : record->VolatileParentMask;

        std::atomic_ref<uint64_t> mask(stored_mask);

        const uint64_t relation_bit = EdgeBuilder::DirtyBit(relation_ordinal);

        if (EdgeBuilder::IsEmpty(relation))
        {
            mask.fetch_and(~relation_bit, std::memory_order_release);
        }
        else
        {
            mask.fetch_or(relation_bit, std::memory_order_release);
        }
    }


    CompiledDAGTableConstructor::SeqLockedOperation
    CompiledDAGTableConstructor::ReadCompiledDAGParentMask_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint64_t& return_mask,
        uint32_t max_tries
    ) noexcept
    {
        return_mask = 0u;
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            child_slot >= FabCache_->CountOfAPC_
        )
        {
            return SeqLockedOperation::NONE;
        }

        CompiledDAGRecord* const record = CompiledDAGRow_(child_slot);
        const size_t control_index = EdgeControlCellIndex_(
            edge_table,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );
        if (!record || control_index == SIZE_MAX)
        {
            return SeqLockedOperation::NONE;
        }

        uint64_t& stored_mask =
            edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                ? record->ValueParentMask
                : record->VolatileParentMask;
        std::atomic_ref<const uint64_t> seq_lock_ref(
            SlabBasePtr_[control_index]
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            const uint64_t before_raw =
                seq_lock_ref.load(std::memory_order_acquire);
            const EdgeBuilder::EdgeData before =
                EdgeBuilder::UnpackEdgeHeader(before_raw);
            if (!before.IsValid)
            {
                return SeqLockedOperation::NONE;
            }
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return SeqLockedOperation::NONE;
            }

            const uint64_t observed_mask = std::atomic_ref<const uint64_t>(
                stored_mask
            ).load(std::memory_order_relaxed);
            if (before_raw != seq_lock_ref.load(std::memory_order_acquire))
            {
                continue;
            }

            return_mask = observed_mask;
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

    bool DAGMutationConf::ReserveAllRows_(
        DAGMutationTransaction& transaction,
        uint32_t max_tries
    ) noexcept
    {
        for (uint8_t i = 0u; i < transaction.RowCount; ++i)
        {
            DAGRowParticipant& row = transaction.Rows[i];
            if (
                ReserveEdgeDomain_(
                    transaction.EdgeTable,
                    row.Slot,
                    row.Domain,
                    EdgeBuilder::EdgeStatus::LIVE,
                    row.Before,
                    max_tries
                ) != SeqLockedOperation::FOUND
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }
            row.WorkTail = row.Before.TailLocator;
            row.Reserved = true;
        }
        return true;
    }

    DAGMutationConf::DAGRelationDelta*
    DAGMutationConf::FindOrInsertRelationDelta_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint8_t ordinal
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
            ParentRelations_(transaction.EdgeTable, child_slot);
        if (relations.size() != FabCache_->MaxDirectParentsPerAxis_)
        {
            return nullptr;
        }

        DAGRelationDelta& inserted =
            transaction.Relations[transaction.RelationCount++];
        inserted.ChildSlot = child_slot;
        inserted.Ordinal = ordinal;
        inserted.Before.ParentHandle = std::atomic_ref<const uint64_t>(
            relations[ordinal].ParentHandle
        ).load(std::memory_order_relaxed);
        inserted.Before.SiblingLocators = std::atomic_ref<const uint64_t>(
            relations[ordinal].SiblingLocators
        ).load(std::memory_order_relaxed);
        inserted.Work = inserted.Before;
        return &inserted;
    }

    DAGMutationConf::DAGRelationDelta*
    DAGMutationConf::EditReservedParentHandle_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint8_t ordinal
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
            EdgeBuilder::RelationSlot(relation_locator),
            EdgeBuilder::RelationOrdinal(relation_locator)
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
                    delta.Work.ParentHandle
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
                    delta.Work.SiblingLocators
                );
            }
        }

        if (TrackDAGRevision_.load(std::memory_order_relaxed))
        {
            SealedDAGRevision_.fetch_add(1u, std::memory_order_release);
        }

        const auto PublishDomain___ = [this, &transaction](
            EdgeBuilder::EdgeDomain domain
        ) noexcept
        {
            for (uint8_t i = 0u; i < transaction.RowCount; ++i)
            {
                DAGRowParticipant& row = transaction.Rows[i];
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
        uint64_t wanted_parent_handle,
        uint64_t other_parent_handle,
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
            ParentRelations_(transaction.EdgeTable, child_slot);
        if (
            !owner ||
            !owner->Reserved ||
            relations.size() != FabCache_->MaxDirectParentsPerAxis_
        )
        {
            return false;
        }

        for (uint8_t ordinal = 0u;
            ordinal < FabCache_->MaxDirectParentsPerAxis_;
            ++ordinal)
        {
            const uint64_t handle = std::atomic_ref<const uint64_t>(
                relations[ordinal].ParentHandle
            ).load(std::memory_order_relaxed);
            if (handle == FABRIC_CELL_SENTINAL)
            {
                if (scan.EmptyOrdinal == UINT8_MAX)
                {
                    scan.EmptyOrdinal = ordinal;
                }
                continue;
            }
            if (handle == wanted_parent_handle)
            {
                if (scan.MatchOrdinal != UINT8_MAX)
                {
                    return false;
                }
                scan.MatchOrdinal = ordinal;
                scan.MatchParentHandle = handle;
            }
            if (
                other_parent_handle != FABRIC_CELL_SENTINAL &&
                handle == other_parent_handle
            )
            {
                if (scan.OtherOrdinal != UINT8_MAX)
                {
                    return false;
                }
                scan.OtherOrdinal = ordinal;
            }
        }
        return true;
    }

    bool ConstructDAGOnEachAxis::AddParentRelation_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries
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
            return false;
        }

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

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
                ) ||
                !ReserveAllRows_(transaction, DEFAULT_INTERNAL_TRIES__)
            )
            {
                continue;
            }

            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            if (
                !IsOpenAPCGeneration_(parent_slot, parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return false;
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
                    FABRIC_CELL_SENTINAL,
                    scan
                ) ||
                scan.MatchOrdinal != UINT8_MAX ||
                scan.EmptyOrdinal == UINT8_MAX
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.EmptyOrdinal
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
                !EdgeBuilder::AreSiblingsEmpty(moving_parent->Before)
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t old_tail = parent_list->Before.TailLocator;
            moving_parent->Work.ParentHandle = parent_handle;

            if (old_tail == EdgeBuilder::RELATION_NULL)
            {
                EdgeBuilder::SetSiblingLocators(
                    moving_parent->Work,
                    self,
                    self
                );
                parent_list->WorkTail = self;

                PrepareConditionalParentPublication_(
                    transaction,
                    child_slot,
                    scan.EmptyOrdinal,
                    publication
                );

                CommitRowTransaction_(transaction);
                return true;
            }

            if (!EdgeBuilder::IsValidRelationLocator(
                old_tail,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            ))
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            DAGRelationDelta* const tail = EditReservedSiblingLocators_(
                transaction,
                parent_slot,
                old_tail
            );
            if (
                !tail ||
                tail->Before.ParentHandle != parent_handle ||
                EdgeBuilder::AreSiblingsEmpty(tail->Before)
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t first = EdgeBuilder::NextLocator(tail->Before);
            if (!EdgeBuilder::IsValidRelationLocator(
                first,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            ))
            {
                AbortRowTransaction_(transaction);
                return false;
            }
            DAGRelationDelta* const first_delta = EditReservedSiblingLocators_(
                transaction,
                parent_slot,
                first
            );
            if (
                !first_delta ||
                first_delta->Before.ParentHandle != parent_handle ||
                EdgeBuilder::PreviousLocator(first_delta->Before) != old_tail
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            EdgeBuilder::SetSiblingLocators(
                moving_parent->Work,
                old_tail,
                first
            );
            EdgeBuilder::SetSiblingLocators(
                tail->Work,
                EdgeBuilder::PreviousLocator(tail->Work),
                self
            );
            EdgeBuilder::SetSiblingLocators(
                first_delta->Work,
                self,
                EdgeBuilder::NextLocator(first_delta->Work)
            );
            parent_list->WorkTail = self;
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.EmptyOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);
            return true;
        }
        return false;
    }

    bool ConstructDAGOnEachAxis::RemoveParentRelation_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries
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
            return false;
        }

        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );
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
                ) ||
                !ReserveAllRows_(transaction, DEFAULT_INTERNAL_TRIES__)
            )
            {
                continue;
            }
            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            if (
                !IsOpenAPCGeneration_(parent_slot, parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return false;
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
                    FABRIC_CELL_SENTINAL,
                    scan
                ) ||
                scan.MatchOrdinal == UINT8_MAX
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.MatchOrdinal
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
                moving_parent->Before.ParentHandle != parent_handle ||
                EdgeBuilder::AreSiblingsEmpty(moving_parent->Before) ||
                parent_list->Before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t previous =
                EdgeBuilder::PreviousLocator(moving_parent->Before);
            const uint32_t next =
                EdgeBuilder::NextLocator(moving_parent->Before);
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
                return false;
            }

            const bool singleton = previous == self && next == self;
            if (
                (singleton && parent_list->Before.TailLocator != self) ||
                (!singleton && (previous == self || next == self))
            )
            {
                AbortRowTransaction_(transaction);
                return false;
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
                    previous_delta->Before.ParentHandle != parent_handle ||
                    next_delta->Before.ParentHandle != parent_handle ||
                    EdgeBuilder::NextLocator(previous_delta->Before) != self ||
                    EdgeBuilder::PreviousLocator(next_delta->Before) != self
                )
                {
                    AbortRowTransaction_(transaction);
                    return false;
                }
                EdgeBuilder::SetSiblingLocators(
                    previous_delta->Work,
                    EdgeBuilder::PreviousLocator(previous_delta->Work),
                    next
                );
                EdgeBuilder::SetSiblingLocators(
                    next_delta->Work,
                    previous,
                    EdgeBuilder::NextLocator(next_delta->Work)
                );
                if (parent_list->Before.TailLocator == self)
                {
                    parent_list->WorkTail = previous;
                }
            }
            else
            {
                parent_list->WorkTail = EdgeBuilder::RELATION_NULL;
            }

            moving_parent->Work.ParentHandle = FABRIC_CELL_SENTINAL;
            moving_parent->Work.SiblingLocators = FABRIC_CELL_SENTINAL;
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.MatchOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);
            return true;
        }
        return false;
    }

    bool ConstructDAGOnEachAxis::ReplaceParentRelation_(
        uint32_t old_parent_slot,
        uint32_t old_parent_generation,
        uint32_t new_parent_slot,
        uint32_t new_parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        ConditionalParentPublication* publication,
        uint32_t max_tries
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
            return false;
        }

        const uint64_t old_parent_handle = EdgeBuilder::MakeParentHandle(
            old_parent_slot,
            old_parent_generation
        );
        const uint64_t new_parent_handle = EdgeBuilder::MakeParentHandle(
            new_parent_slot,
            new_parent_generation
        );

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
                ) ||
                !ReserveAllRows_(transaction, DEFAULT_INTERNAL_TRIES__)
            )
            {
                continue;
            }

            if (!ValidateConditionalParentPublication_(
                    transaction,
                    child_slot,
                    publication))
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            if (
                !IsOpenAPCGeneration_(old_parent_slot, old_parent_generation) ||
                !IsOpenAPCGeneration_(new_parent_slot, new_parent_generation) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                AbortRowTransaction_(transaction);
                return false;
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
                scan.MatchOrdinal == UINT8_MAX ||
                scan.OtherOrdinal != UINT8_MAX
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(
                child_slot,
                scan.MatchOrdinal
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
                moving_parent->Before.ParentHandle != old_parent_handle ||
                EdgeBuilder::AreSiblingsEmpty(moving_parent->Before) ||
                old_list->Before.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                AbortRowTransaction_(transaction);
                return false;
            }

            const uint32_t old_previous =
                EdgeBuilder::PreviousLocator(moving_parent->Before);
            const uint32_t old_next =
                EdgeBuilder::NextLocator(moving_parent->Before);
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
                return false;
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
                return false;
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
                    previous_delta->Before.ParentHandle != old_parent_handle ||
                    next_delta->Before.ParentHandle != old_parent_handle ||
                    EdgeBuilder::NextLocator(previous_delta->Before) != self ||
                    EdgeBuilder::PreviousLocator(next_delta->Before) != self
                )
                {
                    AbortRowTransaction_(transaction);
                    return false;
                }
                EdgeBuilder::SetSiblingLocators(
                    previous_delta->Work,
                    EdgeBuilder::PreviousLocator(previous_delta->Work),
                    old_next
                );
                EdgeBuilder::SetSiblingLocators(
                    next_delta->Work,
                    old_previous,
                    EdgeBuilder::NextLocator(next_delta->Work)
                );
                if (old_list->Before.TailLocator == self)
                {
                    old_list->WorkTail = old_previous;
                }
            }

            const uint32_t new_tail = new_list->Before.TailLocator;
            if (new_tail == EdgeBuilder::RELATION_NULL)
            {
                EdgeBuilder::SetSiblingLocators(
                    moving_parent->Work,
                    self,
                    self
                );
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
                    return false;
                }
                DAGRelationDelta* const tail_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        new_parent_slot,
                        new_tail
                    );
                if (
                    !tail_delta ||
                    tail_delta->Before.ParentHandle != new_parent_handle ||
                    EdgeBuilder::AreSiblingsEmpty(tail_delta->Before)
                )
                {
                    AbortRowTransaction_(transaction);
                    return false;
                }
                const uint32_t new_first =
                    EdgeBuilder::NextLocator(tail_delta->Before);
                if (!EdgeBuilder::IsValidRelationLocator(
                    new_first,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ))
                {
                    AbortRowTransaction_(transaction);
                    return false;
                }
                DAGRelationDelta* const first_delta =
                    EditReservedSiblingLocators_(
                        transaction,
                        new_parent_slot,
                        new_first
                    );
                if (
                    !first_delta ||
                    first_delta->Before.ParentHandle != new_parent_handle ||
                    EdgeBuilder::PreviousLocator(first_delta->Before) != new_tail
                )
                {
                    AbortRowTransaction_(transaction);
                    return false;
                }
                EdgeBuilder::SetSiblingLocators(
                    tail_delta->Work,
                    EdgeBuilder::PreviousLocator(tail_delta->Work),
                    self
                );
                EdgeBuilder::SetSiblingLocators(
                    first_delta->Work,
                    self,
                    EdgeBuilder::NextLocator(first_delta->Work)
                );
                EdgeBuilder::SetSiblingLocators(
                    moving_parent->Work,
                    new_tail,
                    new_first
                );
            }

            moving_parent->Work.ParentHandle = new_parent_handle;
            new_list->WorkTail = self;
            PrepareConditionalParentPublication_(
                transaction,
                child_slot,
                scan.MatchOrdinal,
                publication
            );
            CommitRowTransaction_(transaction);            
            return true;
        }
        return false;
    }


}