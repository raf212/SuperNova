#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using EdgeTableRange = ADS::RangeOfAPC;

    size_t EdgeTableConstructor::EdgeControlCellIndex_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain
    ) noexcept
    {
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, row_slot);
        return range.IsValid
            ? range.BeginIndex + EdgeBuilder::ControlOffset(domain)
            : SIZE_MAX;
    }


    EdgeTableConstructor::EdgeTableRange
    EdgeTableConstructor::ReadAnEdgeTableRange_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        EdgeTableRange range{};

        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            row_slot >= FabCache_->CountOfAPC_
        )
        {
            return range;
        }

        const uint64_t table_begin =
            edge_table == FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                ? FabCache_->HorizontalEdgeBeginIdx_
                : FabCache_->VerticalEdgeBeginIdx_;

        range.BeginIndex = table_begin +
            static_cast<uint64_t>(row_slot) * FabCache_->EdgeTableRecordWidth_;
        range.EndIndex = range.BeginIndex + FabCache_->EdgeTableRecordWidth_;
        range.IsValid = true;
        return range;
    }

    std::span<EdgeBuilder::ParentRelation>
    EdgeTableConstructor::ParentRelations_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, row_slot);
        if (!range.IsValid)
        {
            return {};
        }

        auto* const first = std::launder(
            reinterpret_cast<EdgeBuilder::ParentRelation*>(
                SlabBasePtr_ + range.BeginIndex +
                EdgeBuilder::PARENT_RELATION_ARRAY_OFFSET
            )
        );
        return {first, static_cast<size_t>(FabCache_->MaxDirectParentsPerAxis_)};
    }

    bool EdgeTableConstructor::ConstructParentRelationObjects_(
        FabricSegments edge_table,
        uint32_t row_slot
    ) noexcept
    {
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, row_slot);
        if (!range.IsValid)
        {
            return false;
        }

        auto* const first = reinterpret_cast<EdgeBuilder::ParentRelation*>(
            SlabBasePtr_ + range.BeginIndex +
            EdgeBuilder::PARENT_RELATION_ARRAY_OFFSET
        );
        for (uint8_t ordinal = 0u;
            ordinal < FabCache_->MaxDirectParentsPerAxis_;
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
            row_slot < FabCache_->CountOfAPC_;
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

            EdgeBuilder::EdgeData child_list{};
            child_list.TailLocator = EdgeBuilder::RELATION_NULL;
            child_list.Status = EdgeBuilder::EdgeStatus::FREE;
            child_list.IsValid = true;

            EdgeBuilder::EdgeData parent_relations{};
            parent_relations.TailLocator = EdgeBuilder::RELATION_NULL;
            parent_relations.Status = EdgeBuilder::EdgeStatus::LIVE;
            parent_relations.IsValid = true;

            SlabBasePtr_[range.BeginIndex +
                EdgeBuilder::CHILD_LIST_CONTROL_OFFSET] =
                EdgeBuilder::PackEdgeHeader(child_list);
            SlabBasePtr_[range.BeginIndex +
                EdgeBuilder::PARENT_RELATION_CONTROL_OFFSET] =
                EdgeBuilder::PackEdgeHeader(parent_relations);
        }
        return true;
    }

    bool EdgeTableConstructor::ReadEdgeControl_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain,
        EdgeBuilder::EdgeData& edge
    ) noexcept
    {
        const size_t index = EdgeControlCellIndex_(edge_table, row_slot, domain);
        if (index == SIZE_MAX)
        {
            edge = {};
            return false;
        }

        edge = EdgeBuilder::UnpackEdgeHeader(
            std::atomic_ref<const uint64_t>(SlabBasePtr_[index]).load(
                std::memory_order_acquire
            )
        );
        return edge.IsValid;
    }


    bool EdgeTableConstructor::ReadEdgeHeader_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeData& edge
    ) noexcept
    {
        return ReadEdgeControl_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            edge
        );
    }

    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReadParentHandle_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        uint64_t& parent_handle,
        uint32_t max_tries
    ) noexcept
    {
        parent_handle = FABRIC_CELL_SENTINAL;
        const size_t control_index = EdgeControlCellIndex_(
            edge_table,
            child_slot,
            EdgeBuilder::EdgeDomain::PARENT_RELATIONS
        );
        const std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(edge_table, child_slot);
        if (
            control_index == SIZE_MAX ||
            relations.size() != FabCache_->MaxDirectParentsPerAxis_ ||
            !EdgeBuilder::IsValidRelationOrdinal(
                relation_ordinal,
                FabCache_->MaxDirectParentsPerAxis_
            )
        )
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            const uint64_t before_raw = std::atomic_ref<const uint64_t>(
                SlabBasePtr_[control_index]
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

            const uint64_t observed = std::atomic_ref<const uint64_t>(
                relations[relation_ordinal].ParentHandle
            ).load(std::memory_order_relaxed);
            const uint64_t after_raw = std::atomic_ref<const uint64_t>(
                SlabBasePtr_[control_index]
            ).load(std::memory_order_acquire);
            if (before_raw != after_raw)
            {
                continue;
            }

            parent_handle = observed;
            return observed == FABRIC_CELL_SENTINAL
                ? SeqLockedOperation::NONE
                : SeqLockedOperation::FOUND;
        }
        return SeqLockedOperation::RETRY;
    }
        
    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReserveEdgeDomain_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain,
        EdgeBuilder::EdgeStatus required_status,
        EdgeBuilder::EdgeData& before,
        uint32_t max_tries
    ) noexcept
    {
        const size_t index = EdgeControlCellIndex_(edge_table, row_slot, domain);
        if (index == SIZE_MAX)
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            uint64_t observed_raw = std::atomic_ref<const uint64_t>(
                SlabBasePtr_[index]
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
            reserved.SeqLock = EdgeBuilder::NextSequence(observed.SeqLock);
            reserved.Status = EdgeBuilder::EdgeStatus::RESERVED;
            reserved.IsValid = true;
            if (CompareExchangeWeakInSlab(
                index,
                observed_raw,
                EdgeBuilder::PackEdgeHeader(reserved)
            ))
            {
                before = observed;
                return SeqLockedOperation::FOUND;
            }
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
        return ReserveEdgeDomain_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            required_status,
            before,
            max_tries
        );
    }

    void EdgeTableConstructor::StoreReservedParentHandle_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        uint64_t parent_handle
    ) noexcept
    {
        std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(edge_table, child_slot);
        std::atomic_ref<uint64_t>(relations[relation_ordinal].ParentHandle).store(
            parent_handle,
            std::memory_order_relaxed
        );
    }

    void EdgeTableConstructor::PublishReservedEdgeDomain_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain,
        const EdgeBuilder::EdgeData& before,
        uint32_t desired_tail,
        EdgeBuilder::EdgeStatus desired_status
    ) noexcept
    {
        const size_t index = EdgeControlCellIndex_(edge_table, row_slot, domain);
        EdgeBuilder::EdgeData published{};
        published.TailLocator = desired_tail;
        published.SeqLock = EdgeBuilder::NextSequence(
            EdgeBuilder::NextSequence(before.SeqLock)
        );
        published.Status = desired_status;
        published.IsValid = true;
        std::atomic_ref<uint64_t>(SlabBasePtr_[index]).store(
            EdgeBuilder::PackEdgeHeader(published),
            std::memory_order_release
        );
    }

    void EdgeTableConstructor::PublishReservedEdgeRow_(
        FabricSegments edge_table,
        uint32_t row_slot,
        const EdgeBuilder::EdgeData& before,
        uint32_t desired_tail,
        EdgeBuilder::EdgeStatus desired_status
    ) noexcept
    {
        PublishReservedEdgeDomain_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            before,
            desired_tail,
            desired_status
        );
    }

    void EdgeTableConstructor::StoreReservedSiblingLocators_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        uint64_t sibling_locators
    ) noexcept
    {
        std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(edge_table, child_slot);
        std::atomic_ref<uint64_t>(relations[relation_ordinal].SiblingLocators).store(
            sibling_locators,
            std::memory_order_relaxed
        );
    }
}