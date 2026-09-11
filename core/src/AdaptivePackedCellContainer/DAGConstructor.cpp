#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{


    CompiledDAGTableConstructor::CompiledDAGRecord* CompiledDAGTableConstructor::CompiledDAGRow_(uint32_t row_slot) noexcept
    {
        if (!SlabBasePtr_ || row_slot >= FVolatileCache_.CountOfAPC_)
        {
            return nullptr;
        }
        
        const size_t row_begin = static_cast<size_t>(FVolatileCache_.CompiledDagTableBeginIdx_) +
            (static_cast<size_t>(row_slot) * CoreOfFabricCoordinator::COMPILED_DAG_LEN);
        
        if (
            row_begin >= FVolatileCache_.SlabCellCount_ ||
            CoreOfFabricCoordinator::COMPILED_DAG_LEN > FVolatileCache_.SlabCellCount_ - row_begin
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

        const size_t required_cells = static_cast<size_t>(FVolatileCache_.CountOfAPC_) * CoreOfFabricCoordinator::COMPILED_DAG_LEN;

        if (
            bounds.EndIndex - bounds.BeginIndex != required_cells ||
            bounds.EndIndex > FVolatileCache_.SlabCellCount_
        )
        {
            return false;
        }

        FVolatileCache_.CompiledDagTableBeginIdx_ = bounds.BeginIndex;

        for (uint32_t i = 0; i < FVolatileCache_.CountOfAPC_; i++)
        {
            const size_t row_begin = static_cast<size_t>(FVolatileCache_.CompiledDagTableBeginIdx_) +
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
            child_slot >= FVolatileCache_.CountOfAPC_ ||
            relation_ordinal >= FVolatileCache_.MaxDirectParentsPerAxis_ ||
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


    CompiledDAGTableConstructor::SeqLockedOperation CompiledDAGTableConstructor::ReadCompiledDAGParentMask_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint64_t& return_mask,
        uint32_t max_tries
    ) noexcept
    {
        return_mask = UNSIGNED_ZERO;
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            child_slot >= FVolatileCache_.CountOfAPC_
        )
        {
            return SeqLockedOperation::NONE;
        }

        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, child_slot);

        CompiledDAGRecord* record = CompiledDAGRow_(child_slot);

        if (
            !range.IsValid ||
            !record
        )
        {
            return SeqLockedOperation::NONE;
        }

        uint64_t& stored_mask = edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ?
            record->ValueParentMask : record->VolatileParentMask;


        std::atomic_ref<const uint64_t>seq_lock_ref(SlabBasePtr_[range.BeginIndex]);
        for (size_t i = 0; i < max_tries; i++)
        {
            uint64_t before_raw = seq_lock_ref.load(std::memory_order_acquire);

            const EdgeBuilder::EdgeData before = EdgeBuilder::UnpackEdgeHeader(before_raw);

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

            const uint64_t observed_mask = std::atomic_ref<const uint64_t>(stored_mask).load(std::memory_order_acquire);
            const uint64_t after_raw = seq_lock_ref.load(std::memory_order_acquire);
            if (before_raw != after_raw)
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
        bool is_parent_anchor
    ) noexcept
    {
        if (slot >= FVolatileCache_.CountOfAPC_)
        {
            return false;
        }
        
        uint8_t insert_at = transaction.RowCount;
        for (uint8_t i = 0; i < transaction.RowCount; i++)
        {
            DAGRowParticipant& current = transaction.Rows[i];
            if (current.Slot == slot)
            {
                current.IsParentAnchor = current.IsParentAnchor || is_parent_anchor;
                return true;
            }
            if (slot < current.Slot)
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
            transaction.Rows[i] = transaction.Rows[i - 1];
        }
        transaction.Rows[insert_at] = DAGRowParticipant{};
        transaction.Rows[insert_at].Slot = slot;
        transaction.Rows[insert_at].IsParentAnchor = is_parent_anchor;
        ++transaction.RowCount;
        return true;
    }


    DAGMutationConf::DAGRowParticipant* DAGMutationConf::FindRowParticipant_(
        DAGMutationTransaction& transaction,
        uint32_t slot
    ) noexcept
    {
        for (size_t i = 0; i < transaction.RowCount; i++)
        {
            if (transaction.Rows[i].Slot == slot)
            {
                return &transaction.Rows[i];
            }
        }
        return nullptr;
    }

    bool DAGMutationConf::ReserveAllRows_(
        DAGMutationTransaction& transaction,
        EdgeBuilder::EdgeStatus required_status,
        uint32_t max_tries
    ) noexcept
    {
        for (uint8_t i = 0; i < transaction.RowCount; i++)
        {
            DAGRowParticipant& row = transaction.Rows[i];
            if (
                ReserveEdgeRow_(
                    transaction.EdgeTable,
                    row.Slot,
                    required_status,
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
    DAGMutationConf::EditReservedRelation_(
        DAGMutationTransaction& transaction,
        uint32_t child_slot,
        uint8_t ordinal
    ) noexcept
    {
        DAGRowParticipant* row =
            FindRowParticipant_(transaction, child_slot);

        if (
            !row ||
            !row->Reserved ||
            !EdgeBuilder::IsValidRelationOrdinal(
                ordinal,
                FVolatileCache_.MaxDirectParentsPerAxis_
            )
        )
        {
            return nullptr;
        }

        for (uint8_t i = 0u; i < transaction.RelationCount; ++i)
        {
            DAGRelationDelta& delta = transaction.Relations[i];
            if (
                delta.ChildSlot == child_slot &&
                delta.Ordinal == ordinal
            )
            {
                return &delta;
            }
        }

        if (transaction.RelationCount >= DAG_MAX_RELATION_DELTAS)
        {
            return nullptr;
        }

        std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(transaction.EdgeTable, child_slot);

        if (relations.size() != FVolatileCache_.MaxDirectParentsPerAxis_)
        {
            return nullptr;
        }

        DAGRelationDelta& inserted =
            transaction.Relations[transaction.RelationCount++];
        inserted.ChildSlot = child_slot;
        inserted.Ordinal = ordinal;
        
        inserted.Before.ParentHandle = std::atomic_ref<uint64_t>(
            relations[ordinal].ParentHandle
        ).load(std::memory_order_relaxed);

        inserted.Before.SiblingLocators = std::atomic_ref<uint64_t>(
            relations[ordinal].SiblingLocators
        ).load(std::memory_order_relaxed);

        inserted.Work = inserted.Before;
        return &inserted;
    }



    void DAGMutationConf::CommitRowTransaction_(
        DAGMutationTransaction& transaction,
        EdgeBuilder::EdgeStatus final_status
    ) noexcept
    {
        for (uint8_t i = transaction.RelationCount; i > UNSIGNED_ZERO; i--)
        {
            const DAGRelationDelta& delta = transaction.Relations[i - 1];

            StoreReservedParentRelation_(
                transaction.EdgeTable,
                delta.ChildSlot,
                delta.Ordinal,
                delta.Work
            );

            CompiledDAGRelation_(
                transaction.EdgeTable,
                delta.ChildSlot,
                delta.Ordinal,
                delta.Work
            );
        }
        
        CompiledDagRevision_.fetch_add(1u, std::memory_order_release);
    
        auto Publish___ = [&](bool publish_anchor) noexcept -> void
        {
            for (uint8_t i = 0; i < transaction.RowCount; i++)
            {
                DAGRowParticipant& row = transaction.Rows[i];

                if (row.IsParentAnchor != publish_anchor)
                {
                    continue;
                }

                PublishReservedEdgeRow_(
                    transaction.EdgeTable,
                    row.Slot,
                    row.Before,
                    row.WorkTail,
                    final_status
                );

                row.Reserved = false;
            }
        };

        Publish___(false);
        Publish___(true);
    }

    void DAGMutationConf::AbortRowTransaction_(
        DAGMutationTransaction& transaction
    ) noexcept
    {
        for (uint8_t i = transaction.RowCount; i > UNSIGNED_ZERO; i--)
        {
            DAGRowParticipant& row = transaction.Rows[i - 1];
            if (!row.Reserved)
            {
                continue;
            }
            
            PublishReservedEdgeRow_(
                transaction.EdgeTable,
                row.Slot,
                row.Before,
                row.Before.TailLocator,
                row.Before.Status
            );
            row.Reserved = false;
        }
    }

    ConstructDAGOnEachAxis::SeqLockedOperation ConstructDAGOnEachAxis::ScanParentRow_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint64_t wanted_parent_handle,
        uint64_t other_parent_handle,
        ParentRowScan& scan,
        uint32_t max_tries 
    ) noexcept
    {
        scan = ParentRowScan{};
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, child_slot);
        std::span<EdgeBuilder::ParentRelation> stored = ParentRelations_(edge_table, child_slot);

        if (
            !range.IsValid ||
            stored.size() != FVolatileCache_.MaxDirectParentsPerAxis_
        )
        {
            return SeqLockedOperation::NONE;
        }

        for (size_t i = 0; i < max_tries; i++)
        {
            const uint64_t before_raw = std::atomic_ref<uint64_t>(SlabBasePtr_[range.BeginIndex]).load(std::memory_order_acquire);

            const EdgeBuilder::EdgeData before = EdgeBuilder::UnpackEdgeHeader(before_raw);

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
            
            ParentRowScan observed{};
            observed.Header = before;
            bool malformed = false;

            for (uint8_t ordinal = 0; ordinal < FVolatileCache_.MaxDirectParentsPerAxis_; ordinal++)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(stored[ordinal].ParentHandle).load(std::memory_order_acquire);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(stored[ordinal].SiblingLocators).load(std::memory_order_relaxed);  

                if (EdgeBuilder::IsPartiallyEmpty(relation))
                {
                    malformed = true;
                    continue;
                }

                if (EdgeBuilder::IsEmpty(relation))
                {
                    if (observed.EmptyOrdinal == UINT8_MAX)
                    {
                        observed.EmptyOrdinal = ordinal;
                    }
                    continue;
                }
                
                if (relation.ParentHandle == wanted_parent_handle)
                {
                    if (observed.MatchOrdinal != UINT8_MAX)
                    {
                        malformed = true;
                    }
                    else
                    {
                        observed.MatchOrdinal = ordinal;
                        observed.Match = relation;
                    }
                }

                if (
                    other_parent_handle != FABRIC_CELL_SENTINAL &&
                    relation.ParentHandle == other_parent_handle
                )
                {
                    if (observed.OtherOrdinal != UINT8_MAX)
                    {
                        malformed = true;
                    }
                    else
                    {
                        observed.OtherOrdinal = ordinal;
                    }
                }
            }
            
            const uint64_t after_raw = std::atomic_ref<uint64_t>(SlabBasePtr_[range.BeginIndex]).load(std::memory_order_acquire);
            if (before_raw != after_raw)
            {
                continue;
            }
            
            if (malformed)
            {
                return SeqLockedOperation::NONE;
            }

            scan = observed;
            return SeqLockedOperation::FOUND;
        }
        
        return SeqLockedOperation::RETRY;
    }

    bool ConstructDAGOnEachAxis::AddParentRelation_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            child_slot >= FVolatileCache_.CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !EdgeBuilder::CanInsertCombinedDAGRelation(
                parent_slot,
                child_slot
            )
        )
        {
            return false;
        }
        
        const uint64_t parent_handle = EdgeBuilder::MakeParentHandle(parent_slot, parent_generation);

        for (size_t i = 0; i < max_tries; i++)
        {
            ParentRowScan child_scan{};
            const SeqLockedOperation child_read = ScanParentRow_(
                edge_table,
                child_slot,
                parent_handle,
                FABRIC_CELL_SENTINAL,
                child_scan,
                DEFAULT_INTERNAL_TRIES__
            );

            if (child_read == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (
                child_read != SeqLockedOperation::FOUND ||
                child_scan.MatchOrdinal != UINT8_MAX ||
                child_scan.EmptyOrdinal == UINT8_MAX
            )
            {
                return false;
            }
            
            EdgeBuilder::EdgeData parent_header{};

            if (!ReadEdgeHeader_(edge_table, parent_slot, parent_header))
            {
                return false;
            }

            if (parent_header.Status != EdgeBuilder::EdgeStatus::LIVE)
            {
                return false;
            }
            
            const uint32_t self = EdgeBuilder::PackRelationLocator(child_slot, child_scan.EmptyOrdinal);

            EdgeBuilder::ParentRelation old_tail_relation{};
            EdgeBuilder::ParentRelation first_relation{};

            uint32_t old_tail = EdgeBuilder::RELATION_NULL;
            uint32_t first = EdgeBuilder::RELATION_NULL;

            if (parent_header.TailLocator != EdgeBuilder::RELATION_NULL)
            {
                old_tail = parent_header.TailLocator;

                if (!EdgeBuilder::IsValidRelationLocator(
                    old_tail,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                ))
                {
                    return false;
                }
                const SeqLockedOperation tail_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(old_tail),
                    EdgeBuilder::RelationOrdinal(old_tail),
                    old_tail_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                if (
                    tail_read == SeqLockedOperation::RETRY ||
                    tail_read != SeqLockedOperation::FOUND ||
                    old_tail_relation.ParentHandle != parent_handle
                )
                {
                    continue;
                }

                first = EdgeBuilder::NextLocator(old_tail_relation);
                if (!EdgeBuilder::IsValidRelationLocator(
                    first,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                ))
                {
                    return false;
                }
                
                const SeqLockedOperation first_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(first),
                    EdgeBuilder::RelationOrdinal(first),
                    first_relation,
                    DEFAULT_INTERNAL_TRIES__
                );
                if (
                    first_read != SeqLockedOperation::FOUND ||
                    first_relation.ParentHandle != parent_handle ||
                    EdgeBuilder::PreviousLocator(first_relation) != old_tail
                )
                {
                    continue;
                }
            }

            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;
            if (
                !AddRowParticipant_(transaction, child_slot) ||
                !AddRowParticipant_(transaction, parent_slot, true) ||
                (
                    old_tail != EdgeBuilder::RELATION_NULL &&
                    (
                        !AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(old_tail)
                        ) ||
                        !AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(first)
                        )
                    )
                ) ||
                !ReserveAllRows_(
                    transaction,
                    EdgeBuilder::EdgeStatus::LIVE,
                    DEFAULT_INTERNAL_TRIES__
                )
            )
            {
                continue;
            }

            DAGRowParticipant* child_row = FindRowParticipant_(transaction, child_slot);
            DAGRowParticipant* parent_row = FindRowParticipant_(transaction, parent_slot);

            if (
                !child_row ||
                !parent_row ||
                !SameHeader_(child_row->Before, child_scan.Header) ||
                !SameHeader_(parent_row->Before, parent_header)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }

            DAGRelationDelta* inserted = EditReservedRelation_(
                transaction,
                child_slot,
                child_scan.EmptyOrdinal
            );

            if(
                !inserted ||
                !EdgeBuilder::IsEmpty(inserted->Before)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }
            
            if (old_tail == EdgeBuilder::RELATION_NULL)
            {
                inserted->Work = EdgeBuilder::MakeParentRelation(
                    parent_slot,
                    parent_generation,
                    self,
                    self
                );

                parent_row->WorkTail = self;
                CommitRowTransaction_(transaction);
                return true;
            }

            DAGRelationDelta* tail_delta = EditReservedRelation_(
                transaction,
                EdgeBuilder::RelationSlot(old_tail),
                EdgeBuilder::RelationOrdinal(old_tail)
            );

            DAGRelationDelta* first_delta = EditReservedRelation_(
                transaction,
                EdgeBuilder::RelationSlot(first),
                EdgeBuilder::RelationOrdinal(first)
            );

            if (
                !tail_delta ||
                !first_delta ||
                !SameRelation_(tail_delta->Before, old_tail_relation) ||
                !SameRelation_(first_delta->Before, first_relation) ||
                tail_delta->Before.ParentHandle != parent_handle ||
                first_delta->Before.ParentHandle != parent_handle ||
                EdgeBuilder::NextLocator(tail_delta->Before) != first ||
                EdgeBuilder::PreviousLocator(first_delta->Before) != old_tail
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }
            
            inserted->Work = EdgeBuilder::MakeParentRelation(
                parent_slot,
                parent_generation,
                old_tail,
                first
            );


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

            parent_row->WorkTail = self,
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
        uint32_t max_tries
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            parent_slot >= FVolatileCache_.CountOfAPC_ ||
            child_slot >= FVolatileCache_.CountOfAPC_ ||
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

        for (size_t i = 0; i < max_tries; i++)
        {
            ParentRowScan child_scan{};
            const SeqLockedOperation child_read = ScanParentRow_(
                edge_table,
                child_slot,
                parent_handle,
                FABRIC_CELL_SENTINAL,
                child_scan,
                DEFAULT_INTERNAL_TRIES__
            );

            if (child_read == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (
                child_read != SeqLockedOperation::FOUND ||
                child_scan.MatchOrdinal == UINT8_MAX
            )
            {
                return false;
            }
            
            const uint32_t self = EdgeBuilder::PackRelationLocator(child_slot, child_scan.MatchOrdinal);
            const uint32_t previous = EdgeBuilder::PreviousLocator(child_scan.Match);
            const uint32_t next = EdgeBuilder::NextLocator(child_scan.Match);
            if (
                !EdgeBuilder::IsValidRelationLocator(
                    previous,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                ) ||
                !EdgeBuilder::IsValidRelationLocator(
                    next,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                )
            )
            {
                return false;
            }

            EdgeBuilder::EdgeData parent_header{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, parent_header)
            )
            {
                return false;
            }
            
            if (parent_header.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }

            if (
                parent_header.Status != EdgeBuilder::EdgeStatus::LIVE ||
                parent_header.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                return false;
            }

            const bool singleton = previous == self && next == self;

            if (
                singleton &&
                parent_header.TailLocator != self
            )
            {
                continue;
            }
            
            if (
                !singleton &&
                (previous == self || next == self)
            )
            {
                return false;
            }
            
            EdgeBuilder::ParentRelation previous_relation{};
            EdgeBuilder::ParentRelation next_relation{};
            if (!singleton)
            {
                const SeqLockedOperation previous_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(previous),
                    EdgeBuilder::RelationOrdinal(previous),
                    previous_relation,
                    DEFAULT_INTERNAL_TRIES__
                );
                const SeqLockedOperation next_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(next),
                    EdgeBuilder::RelationOrdinal(next),
                    next_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                if (
                    previous_read != SeqLockedOperation::FOUND ||
                    next_read != SeqLockedOperation::FOUND ||
                    previous_relation.ParentHandle != parent_handle ||
                    next_relation.ParentHandle != parent_handle ||
                    EdgeBuilder::NextLocator(previous_relation) != self ||
                    EdgeBuilder::PreviousLocator(next_relation) != self
                )
                {
                    continue;
                }
            }
            
            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;
            if (
                !AddRowParticipant_(transaction, child_slot) ||
                !AddRowParticipant_(transaction, parent_slot, true) ||
                (
                    !singleton &&
                    (
                        !AddRowParticipant_(transaction, EdgeBuilder::RelationSlot(previous)) ||
                        !AddRowParticipant_(transaction, EdgeBuilder::RelationSlot(next))
                    )
                ) ||
                !ReserveAllRows_(
                    transaction,
                    EdgeBuilder::EdgeStatus::LIVE,
                    DEFAULT_INTERNAL_TRIES__
                )
            )
            {
                continue;
            }

            DAGRowParticipant* child_row = FindRowParticipant_(transaction, child_slot);
            DAGRowParticipant* parent_row = FindRowParticipant_(transaction, parent_slot);

            if (
                !child_row ||
                !parent_row ||
                !SameHeader_(child_row->Before, child_scan.Header) ||
                !SameHeader_(parent_row->Before, parent_header)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }

            DAGRelationDelta* moving = EditReservedRelation_(
                transaction,
                child_slot,
                child_scan.MatchOrdinal
            );

            if (
                !moving ||
                !SameRelation_(moving->Before, child_scan.Match)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }

            if (singleton)
            {
                EdgeBuilder::Clear(moving->Work);
                parent_row->WorkTail = EdgeBuilder::RELATION_NULL;
                CommitRowTransaction_(transaction);
                return true;
            }
            
            DAGRelationDelta* previous_delta = EditReservedRelation_(
                transaction,
                EdgeBuilder::RelationSlot(previous),
                EdgeBuilder::RelationOrdinal(previous)
            );

            DAGRelationDelta* next_delta = EditReservedRelation_(
                transaction,
                EdgeBuilder::RelationSlot(next),
                EdgeBuilder::RelationOrdinal(next)
            );

            if (
                !previous_delta ||
                !next_delta ||
                !SameRelation_(previous_delta->Before, previous_relation) ||
                !SameRelation_(next_delta->Before, next_relation) ||
                previous_delta->Before.ParentHandle != parent_handle ||
                next_delta->Before.ParentHandle != parent_handle ||
                EdgeBuilder::NextLocator(previous_delta->Before) != self ||
                EdgeBuilder::PreviousLocator(next_delta->Before) != self
            )
            {
                AbortRowTransaction_(transaction);
                continue;
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
            EdgeBuilder::Clear(moving->Work);

            if (parent_row->Before.TailLocator == self)
            {
                parent_row->WorkTail = previous;
            }
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
        uint32_t max_tries 
    ) noexcept
    {
        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            old_parent_slot >= FVolatileCache_.CountOfAPC_ ||
            new_parent_slot >= FVolatileCache_.CountOfAPC_ ||
            child_slot >= FVolatileCache_.CountOfAPC_ ||
            old_parent_slot == new_parent_slot ||
            !HandleOfAPCStatic::IsGenerationValid(old_parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(new_parent_generation) ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !EdgeBuilder::CanInsertCombinedDAGRelation(new_parent_slot, child_slot)
        )
        {
            return false;
        }

        const uint64_t old_parent_handle = EdgeBuilder::MakeParentHandle(old_parent_slot, old_parent_generation);
        const uint64_t new_parent_handle = EdgeBuilder::MakeParentHandle(new_parent_slot, new_parent_generation);

        for (uint32_t i = 0; i < max_tries; i++)
        {
            ParentRowScan child_scan{};
            const SeqLockedOperation child_read = ScanParentRow_(
                edge_table,
                child_slot,
                old_parent_handle,
                new_parent_handle,
                child_scan,
                DEFAULT_INTERNAL_TRIES__
            );

            if (child_read == SeqLockedOperation::RETRY)
            {
                continue;
            }

            if (
                child_read != SeqLockedOperation::FOUND ||
                child_scan.MatchOrdinal == UINT8_MAX ||
                child_scan.OtherOrdinal != UINT8_MAX
            )
            {
                return false;
            }

            const uint32_t self = EdgeBuilder::PackRelationLocator(child_slot, child_scan.MatchOrdinal);
            const uint32_t old_previous = EdgeBuilder::PreviousLocator(child_scan.Match);
            const uint32_t old_next = EdgeBuilder::NextLocator(child_scan.Match);

            if (
                !EdgeBuilder::IsValidRelationLocator(
                    old_previous,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                ) ||
                !EdgeBuilder::IsValidRelationLocator(
                    old_next,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                )
            )
            {
                return false;
            }
            
            EdgeBuilder::EdgeData old_parent_header{};
            EdgeBuilder::EdgeData new_parent_header{};

            if (
                !ReadEdgeHeader_(
                    edge_table,
                    old_parent_slot,
                    old_parent_header
                ) ||
                !ReadEdgeHeader_(
                    edge_table,
                    new_parent_slot,
                    new_parent_header
                )
            )
            {
                return false;
            }
            
            if(
                old_parent_header.Status == EdgeBuilder::EdgeStatus::RESERVED ||
                new_parent_header.Status == EdgeBuilder::EdgeStatus::RESERVED
            )
            {
                continue;
            }

            if (
                old_parent_header.Status != EdgeBuilder::EdgeStatus::LIVE ||
                new_parent_header.Status != EdgeBuilder::EdgeStatus::LIVE ||
                old_parent_header.TailLocator == EdgeBuilder::RELATION_NULL
            )
            {
                return false;
            }

            const bool old_singleton = old_previous == self && old_next == self;
            if (
                (old_singleton && old_parent_header.TailLocator != self) ||
                (
                    !old_singleton &&
                    (old_previous == self || old_next == self)
                )
            )
            {
                continue;
            }

            EdgeBuilder::ParentRelation old_previous_relation{};
            EdgeBuilder::ParentRelation old_next_relation{};

            if(!old_singleton)
            {
                const SeqLockedOperation previous_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(old_previous),
                    EdgeBuilder::RelationOrdinal(old_previous),
                    old_previous_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                const SeqLockedOperation next_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(old_next),
                    EdgeBuilder::RelationOrdinal(old_next),
                    old_next_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                if (
                    previous_read != SeqLockedOperation::FOUND ||
                    next_read != SeqLockedOperation::FOUND ||
                    old_previous_relation.ParentHandle != old_parent_handle ||
                    old_next_relation.ParentHandle != old_parent_handle ||
                    EdgeBuilder::NextLocator(old_previous_relation) != self ||
                    EdgeBuilder::PreviousLocator(old_next_relation) != self
                )
                {
                    continue;
                }
            }

            uint32_t new_tail = EdgeBuilder::RELATION_NULL;
            uint32_t new_first = EdgeBuilder::RELATION_NULL;
            EdgeBuilder::ParentRelation new_tail_relation{};
            EdgeBuilder::ParentRelation new_first_relation{};

            if (new_parent_header.TailLocator != EdgeBuilder::RELATION_NULL)
            {
                new_tail = new_parent_header.TailLocator;
                if (!EdgeBuilder::IsValidRelationLocator(
                    new_tail,
                    static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                    FVolatileCache_.MaxDirectParentsPerAxis_
                ))
                {
                    return false;
                }
                
                const SeqLockedOperation tail_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(new_tail),
                    EdgeBuilder::RelationOrdinal(new_tail),
                    new_tail_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                if (
                    tail_read != SeqLockedOperation::FOUND ||
                    new_tail_relation.ParentHandle != new_parent_handle
                )
                {
                    continue;
                }

                new_first = EdgeBuilder::NextLocator(new_tail_relation);
                if (
                    !EdgeBuilder::IsValidRelationLocator(
                        new_first,
                        static_cast<uint32_t>(FVolatileCache_.CountOfAPC_),
                        FVolatileCache_.MaxDirectParentsPerAxis_
                    )
                )
                {
                    return false;
                }
                
                const SeqLockedOperation first_read = ReadParentRelation_(
                    edge_table,
                    EdgeBuilder::RelationSlot(new_first),
                    EdgeBuilder::RelationOrdinal(new_first),
                    new_first_relation,
                    DEFAULT_INTERNAL_TRIES__
                );

                if (
                    first_read != SeqLockedOperation::FOUND ||
                    new_first_relation.ParentHandle != new_parent_handle ||
                    EdgeBuilder::PreviousLocator(new_first_relation) != new_tail
                )
                {
                    continue;
                }
            }

            DAGMutationTransaction transaction{};
            transaction.EdgeTable = edge_table;

            const bool participants_ok =
                AddRowParticipant_(transaction, child_slot) &&
                AddRowParticipant_(transaction, old_parent_slot, true) &&
                AddRowParticipant_(transaction, new_parent_slot, true) &&
                (
                    old_singleton ||
                    (
                        AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(old_previous)
                        ) &&
                        AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(old_next)
                        )
                    )
                ) &&
                (
                    new_tail == EdgeBuilder::RELATION_NULL ||
                    (
                        AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(new_tail)
                        ) &&
                        AddRowParticipant_(
                            transaction,
                            EdgeBuilder::RelationSlot(new_first)
                        )
                    )
                );

            if (
                !participants_ok ||
                !ReserveAllRows_(
                    transaction,
                    EdgeBuilder::EdgeStatus::LIVE,
                    DEFAULT_INTERNAL_TRIES__
                )
            )
            {
                continue;
            }
            
            DAGRowParticipant* child_row = FindRowParticipant_(transaction, child_slot);
            DAGRowParticipant* old_parent_row = FindRowParticipant_(transaction, old_parent_slot);
            DAGRowParticipant* new_parent_row = FindRowParticipant_(transaction, new_parent_slot);

            if (
                !child_row ||
                !old_parent_row ||
                !new_parent_row ||
                !SameHeader_(child_row->Before, child_scan.Header) ||
                !SameHeader_(old_parent_row->Before, old_parent_header) ||
                !SameHeader_(new_parent_row->Before, new_parent_header)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }
            
            DAGRelationDelta* moving = EditReservedRelation_(
                transaction,
                child_slot,
                child_scan.MatchOrdinal
            );

            if (
                !moving ||
                !SameRelation_(moving->Before, child_scan.Match)
            )
            {
                AbortRowTransaction_(transaction);
                continue;
            }
            
            if (old_singleton)
            {
                old_parent_row->WorkTail = EdgeBuilder::RELATION_NULL;
            }
            else
            {
                DAGRelationDelta* previous_delta = EditReservedRelation_(
                    transaction,
                    EdgeBuilder::RelationSlot(old_previous),
                    EdgeBuilder::RelationOrdinal(old_previous)
                );
                DAGRelationDelta* next_delta = EditReservedRelation_(
                    transaction,
                    EdgeBuilder::RelationSlot(old_next),
                    EdgeBuilder::RelationOrdinal(old_next)
                );

                if (
                    !previous_delta ||
                    !next_delta ||
                    !SameRelation_(previous_delta->Before, old_previous_relation) ||
                    !SameRelation_(next_delta->Before, old_next_relation) ||
                    previous_delta->Before.ParentHandle != old_parent_handle ||
                    next_delta->Before.ParentHandle != old_parent_handle ||
                    EdgeBuilder::NextLocator(previous_delta->Before) != self ||
                    EdgeBuilder::PreviousLocator(next_delta->Before) != self
                )
                {
                    AbortRowTransaction_(transaction);
                    continue;
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

                if (old_parent_row->Before.TailLocator == self)
                {
                    old_parent_row->WorkTail = old_previous;
                }
            }

            if (new_tail == EdgeBuilder::RELATION_NULL)
            {
                moving->Work = EdgeBuilder::MakeParentRelation(
                    new_parent_slot,
                    new_parent_generation,
                    self,
                    self
                );
                new_parent_row->WorkTail = self;
            }
            else
            {
                DAGRelationDelta* tail_delta = EditReservedRelation_(
                    transaction,
                    EdgeBuilder::RelationSlot(new_tail),
                    EdgeBuilder::RelationOrdinal(new_tail)
                );

                DAGRelationDelta* first_delta = EditReservedRelation_(
                    transaction,
                    EdgeBuilder::RelationSlot(new_first),
                    EdgeBuilder::RelationOrdinal(new_first)
                );

                if (
                    !tail_delta ||
                    !first_delta ||
                    !SameRelation_(tail_delta->Before, new_tail_relation) ||
                    !SameRelation_(first_delta->Before, new_first_relation) ||
                    tail_delta->Before.ParentHandle != new_parent_handle ||
                    first_delta->Before.ParentHandle != new_parent_handle ||
                    EdgeBuilder::NextLocator(tail_delta->Before) != new_first ||
                    EdgeBuilder::PreviousLocator(first_delta->Before) != new_tail
                )
                {
                    AbortRowTransaction_(transaction);
                    continue;
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

                moving->Work = EdgeBuilder::MakeParentRelation(
                    new_parent_slot,
                    new_parent_generation,
                    new_tail,
                    new_first
                );
                new_parent_row->WorkTail = self;
            }
            
            CommitRowTransaction_(transaction);
            return true;
        }
        
        return false;
    }

}