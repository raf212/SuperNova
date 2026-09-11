#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using EdgeTableRange = ADS::RangeOfAPC;

    EdgeTableConstructor::EdgeTableRange
    EdgeTableConstructor::ReadAnEdgeTableRange_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        EdgeTableRange range{};

        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            row_slot >= CountOfAPC_ ||
            !EdgeBuilder::IsValidConfigurableParentCapacity(
                MaxDirectParentsPerAxis_
            ) ||
            EdgeTableRecordWidth_ !=
                EdgeBuilder::EdgeTableRecordWidth(
                    MaxDirectParentsPerAxis_
                )
        )
        {
            return range;
        }

        const uint64_t table_begin =
            edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                ? HorizontalEdgeBeginIdx_
                : VerticalEdgeBeginIdx_;

        range.BeginIndex = table_begin +
            static_cast<uint64_t>(row_slot) * EdgeTableRecordWidth_;
        range.EndIndex = range.BeginIndex + EdgeTableRecordWidth_;
        range.IsValid =
            range.BeginIndex >= table_begin &&
            range.BeginIndex < range.EndIndex &&
            range.EndIndex <= SlabCellCount_;
        return range;
    }

    std::span<EdgeBuilder::ParentRelation>
    EdgeTableConstructor::ParentRelations_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, row_slot);

        if (!range.IsValid)
        {
            return {};
        }

        auto* first = std::launder(
            reinterpret_cast<EdgeBuilder::ParentRelation*>(
                SlabBasePtr_ + range.BeginIndex + 1u
            )
        );

        return {
            first,
            static_cast<size_t>(MaxDirectParentsPerAxis_)
        };
    }

    bool EdgeTableConstructor::ConstructParentRelationObjects_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, row_slot);

        if (!range.IsValid)
        {
            return false;
        }

        auto* first = reinterpret_cast<EdgeBuilder::ParentRelation*>(
            SlabBasePtr_ + range.BeginIndex + 1u
        );

        for (uint8_t ordinal = 0u;
            ordinal < MaxDirectParentsPerAxis_;
            ++ordinal)
        {
            std::construct_at(first + ordinal);
        }

        return true;
    }

    bool EdgeTableConstructor::InitializeEdgeTable_(
        FabricSegments edge_table
    ) noexcept
    {
        if (!CoreOfFabricCoordinator::IsValidEdgeTable(edge_table))
        {
            return false;
        }

        for (uint32_t row_slot = 0u;
            row_slot < CountOfAPC_;
            ++row_slot)
        {
            const EdgeTableRange range =
                ReadAnEdgeTableRange_(edge_table, row_slot);

            if (
                !range.IsValid ||
                !ConstructParentRelationObjects_(edge_table, row_slot)
            )
            {
                return false;
            }

            EdgeBuilder::EdgeData empty{};
            empty.TailLocator = EdgeBuilder::RELATION_NULL;
            empty.SeqLock = 0u;
            empty.Status = EdgeBuilder::EdgeStatus::FREE;
            empty.IsValid = true;

            SlabBasePtr_[range.BeginIndex] =
                EdgeBuilder::PackEdgeHeader(empty);
        }

        return true;
    }

    bool EdgeTableConstructor::ReadEdgeHeader_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeData& edge
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, row_slot);

        if (!range.IsValid)
        {
            edge = {};
            return false;
        }

        const uint64_t raw = std::atomic_ref<uint64_t>(
            SlabBasePtr_[range.BeginIndex]
        ).load(std::memory_order_acquire);

        edge = EdgeBuilder::UnpackEdgeHeader(raw);
        return edge.IsValid;
    }

    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReadParentRelation_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        EdgeBuilder::ParentRelation& relation,
        uint32_t max_tries
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, child_slot);
        std::span<EdgeBuilder::ParentRelation> stored =
            ParentRelations_(edge_table, child_slot);

        if (
            !range.IsValid ||
            stored.size() != MaxDirectParentsPerAxis_ ||
            !EdgeBuilder::IsValidRelationOrdinal(
                relation_ordinal,
                MaxDirectParentsPerAxis_
            )
        )
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            const uint64_t before_raw = std::atomic_ref<uint64_t>(
                SlabBasePtr_[range.BeginIndex]
            ).load(std::memory_order_acquire);

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

            EdgeBuilder::ParentRelation observed{};
            observed.ParentHandle = std::atomic_ref<uint64_t>(
                stored[relation_ordinal].ParentHandle
            ).load(std::memory_order_relaxed);
            observed.SiblingLocators = std::atomic_ref<uint64_t>(
                stored[relation_ordinal].SiblingLocators
            ).load(std::memory_order_relaxed);

            const uint64_t after_raw = std::atomic_ref<uint64_t>(
                SlabBasePtr_[range.BeginIndex]
            ).load(std::memory_order_acquire);

            if (before_raw != after_raw)
            {
                continue;
            }

            relation = observed;
            if (EdgeBuilder::IsPartiallyEmpty(relation))
            {
                return SeqLockedOperation::NONE;
            }
            return EdgeBuilder::IsEmpty(relation)
                ? SeqLockedOperation::NONE
                : SeqLockedOperation::FOUND;
        }

        return SeqLockedOperation::RETRY;
    }

        
    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReserveEdgeRow_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeStatus required_status,
        EdgeBuilder::EdgeData& before,
        uint32_t max_tries
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, row_slot);

        if (!range.IsValid)
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            uint64_t observed_raw = std::atomic_ref<uint64_t>(
                SlabBasePtr_[range.BeginIndex]
            ).load(std::memory_order_acquire);

            EdgeBuilder::EdgeData observed =
                EdgeBuilder::UnpackEdgeHeader(observed_raw);

            if (!observed.IsValid)
            {
                return SeqLockedOperation::NONE;
            }
            if (observed.Status == EdgeBuilder::EdgeStatus::RESERVED)
            {
                continue;
            }
            if (observed.Status != required_status)
            {
                return SeqLockedOperation::NONE;
            }

            EdgeBuilder::EdgeData reserved = observed;
            reserved.SeqLock = EdgeBuilder::NextSequence(
                observed.SeqLock
            );
            reserved.Status = EdgeBuilder::EdgeStatus::RESERVED;
            reserved.IsValid = true;

            const uint64_t desired_raw =
                EdgeBuilder::PackEdgeHeader(reserved);

            if (CompareExchangeWeakInSlab(
                range.BeginIndex,
                observed_raw,
                desired_raw
            ))
            {
                before = observed;
                return SeqLockedOperation::FOUND;
            }
        }

        return SeqLockedOperation::RETRY;
    }

    void EdgeTableConstructor::StoreReservedParentRelation_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        const EdgeBuilder::ParentRelation& relation
    ) noexcept
    {
        std::span<EdgeBuilder::ParentRelation> stored =
            ParentRelations_(edge_table, child_slot);

        std::atomic_ref<uint64_t>(
            stored[relation_ordinal].ParentHandle
        ).store(relation.ParentHandle, std::memory_order_relaxed);

        std::atomic_ref<uint64_t>(
            stored[relation_ordinal].SiblingLocators
        ).store(relation.SiblingLocators, std::memory_order_relaxed);
    }

    void EdgeTableConstructor::PublishReservedEdgeRow_(
        FabricSegments edge_table,
        uint32_t row_slot,
        const EdgeBuilder::EdgeData& before,
        uint32_t desired_tail,
        EdgeBuilder::EdgeStatus desired_status
    ) noexcept
    {
        const EdgeTableRange range =
            ReadAnEdgeTableRange_(edge_table, row_slot);

        EdgeBuilder::EdgeData published{};
        published.TailLocator = desired_tail;
        published.SeqLock = EdgeBuilder::NextSequence(
            EdgeBuilder::NextSequence(before.SeqLock)
        );
        published.Status = desired_status;
        published.IsValid = true;

        std::atomic_ref<uint64_t>(
            SlabBasePtr_[range.BeginIndex]
        ).store(
            EdgeBuilder::PackEdgeHeader(published),
            std::memory_order_release
        );
    }
}