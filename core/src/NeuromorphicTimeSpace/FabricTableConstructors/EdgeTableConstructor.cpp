#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    using EdgeTableRange = ADS::RangeOfAPC;
    using EB = EdgeBuilder;
    using EdgeTableRowView = EB::EdgeTableRowView;

    EdgeTableRowView EdgeTableConstructor::EdgeTableRow_(FabricSegments edge_table, uint32_t row_slot) noexcept
    {
        EdgeTableRowView view{};
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, row_slot);
        if (!range.IsValid)
        {
            return view;
        }

        const uint32_t k = FabCache_->MaxDirectParentsPerAxis_;
        const uint64_t mask_words = EB::ParentMaskWordCount(k);

        EB::ParentMaskBlock* const mask_begin = std::launder(reinterpret_cast<EB::ParentMaskBlock*>(
            SlabBasePtr_ + range.BeginIndex));
        EB::ParentRelation* const relations_begin = std::launder(reinterpret_cast<EB::ParentRelation*>(
            SlabBasePtr_ + range.BeginIndex + mask_words
        ));

        view.Masks = std::span<EB::ParentMaskBlock>(mask_begin, mask_words);
        view.Relations = std::span<EB::ParentRelation>(relations_begin, k);
        return view;
    }

    bool EdgeTableConstructor::ConstructEdgeTableBySlot_(FabricSegments edge_table, uint32_t slot) noexcept
    {
        const EdgeTableRange range = ReadAnEdgeTableRange_(edge_table, slot);
        if (!range.IsValid)
        {
            return false;
        }
        
        const uint32_t k = FabCache_->MaxDirectParentsPerAxis_;
        const uint32_t mask_words = EB::ParentMaskWordCount(k);

        EB::ParentMaskBlock* const mask_begin = reinterpret_cast<EB::ParentMaskBlock*>(
            SlabBasePtr_ + range.BeginIndex
        );

        EB::ParentRelation* const relation_begin = reinterpret_cast<EB::ParentRelation*>(
            SlabBasePtr_ + range.BeginIndex + mask_words
        );

        for (uint32_t i = 0; i < mask_words; i++)
        {
            std::construct_at(mask_begin + i, EB::ParentMaskBlock{});
        }

        for (uint32_t i = 0; i < k; i++)
        {
            std::construct_at(relation_begin + i, EB::ParentRelation{});
        }

        return true;
    }


    EdgeTableConstructor::EdgeTableRange
    EdgeTableConstructor::ReadAnEdgeTableRange_(
        FabricSegments edge_table,
        uint32_t row_slot) noexcept
    {
        EdgeTableRange range{};

        if (
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            row_slot >= FabCache_->CountOfAPC_)
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


    bool EdgeTableConstructor::InitializeEdgeTable_(
        FabricSegments edge_table) noexcept
    {
        if (!CoreOfFabricCoordinator::IsValidEdgeTable(edge_table))
        {
            return false;
        }

        for (uint32_t row_slot = 0u;
             row_slot < FabCache_->CountOfAPC_;
             ++row_slot)
        {
            if (!ConstructEdgeTableBySlot_(edge_table, row_slot))
            {
                return false;
            }
        }
        return true;
    }

    bool EdgeTableConstructor::ReadEdgeControl_(
        FabricSegments edge_table,
        uint32_t slot,
        EdgeBuilder::EdgeDomain domain,
        EdgeBuilder::EdgeData &edge) noexcept
    {
        edge = {};
        if (domain == EB::EdgeDomain::PARENT_RELATIONS)
        {
            HAS::ParentRowControl* const p_row_ptr = ParentRowControl_(edge_table, slot);
            if (!p_row_ptr)
            {
                return false;
            }
            
            const HAS::ParentRowControl value = std::atomic_ref<const HAS::ParentRowControl>(*p_row_ptr).load(std::memory_order_acquire);
            if (!HAS::ValidParentControl(value))
            {
                return false;
            }

            edge.TailLocator = EdgeBuilder::RELATION_NULL;
            edge.SeqLock = value.SeqLock;
            edge.Status = value.Status;
            edge.IsValid = true;
            return true;
        }


        HAS::ChildListControl* child_list_ptr = ChildListControl_(edge_table, slot);
        uint32_t* const tail_ptr = ChildTailPtr_(edge_table, slot);

        if (!child_list_ptr || !tail_ptr)
        {
            return false;
        }

        HAS::ChildListControl before = std::atomic_ref<const HAS::ChildListControl>(*child_list_ptr).load(std::memory_order_acquire);

        if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
        {
            edge.TailLocator = EdgeBuilder::RELATION_NULL;
            edge.SeqLock = before.SeqLockChild;
            edge.Status = before.Status;
            edge.IsValid = true;
            return true;
        }

        const uint32_t tail = std::atomic_ref<const uint32_t>(*tail_ptr).load(std::memory_order_relaxed);
        HAS::ChildListControl after = std::atomic_ref<const HAS::ChildListControl>(*child_list_ptr).load(std::memory_order_acquire);

        if (before != after)
        {
            edge.TailLocator = EdgeBuilder::RELATION_NULL;
            edge.SeqLock = after.SeqLockChild;
            edge.Status = after.Status;
            edge.IsValid = true;
            return true;
        }
        
        if (!HAS::ValidChildControl(before, tail))
        {
            return false;
        }
        
        edge.TailLocator = tail;
        edge.SeqLock = before.SeqLockChild;
        edge.Status = before.Status;
        edge.IsValid = true;
        return true;
    }

    bool EdgeTableConstructor::ReadChildDomainControl_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeData &edge) noexcept
    {
        return ReadEdgeControl_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            edge);
    }

    EdgeTableConstructor::SeqLockedOperation EdgeTableConstructor::ReadParentHandle_(
        FabricSegments edge_table,
        uint32_t slot,
        uint32_t relation_ordinal,
        EB::ParentIDGeneration& parent_handle,
        uint32_t max_tries
    ) noexcept
    {
        parent_handle = {};
        HAS::ParentRowControl* const control = ParentRowControl_(edge_table, slot);
        std::span<EB::ParentRelation> relations = ParentRelations_(edge_table, slot);

        if (
            !control ||
            relations.size() != FabCache_->MaxDirectParentsPerAxis_ ||
            !EB::IsValidRelationOrdinal(relation_ordinal, FabCache_->MaxDirectParentsPerAxis_)
        )
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t i = 0; i < max_tries; i++)
        {
            const HAS::ParentRowControl before = std::atomic_ref<const HAS::ParentRowControl>(*control).load(std::memory_order_acquire);

            if (!HAS::ValidParentControl(before))
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

            const EB::ParentIDGeneration observed = std::atomic_ref<const EB::ParentIDGeneration>(relations[relation_ordinal].Parent).load(std::memory_order_relaxed);
            
            const HAS::ParentRowControl after = std::atomic_ref<const HAS::ParentRowControl>(*control).load(std::memory_order_acquire);

            if (before != after)
            {
                continue;
            }

            if (EB::IsParentEmpty(observed))
            {
                return SeqLockedOperation::NONE;
            }
            
            parent_handle = observed;
            return SeqLockedOperation::FOUND;
        }
        
        return SeqLockedOperation::RETRY;
    }

    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReserveEdgeDomain_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain,
        EdgeBuilder::EdgeStatus required_status,
        EdgeBuilder::EdgeData &before,
        uint32_t max_tries) noexcept
    {
        const size_t index = EdgeControlCellIndex_(edge_table, row_slot, domain);
        if (index == SIZE_MAX)
        {
            return SeqLockedOperation::NONE;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            uint64_t observed_raw = std::atomic_ref<const uint64_t>(SlabBasePtr_[index]).load(std::memory_order_acquire);
            EdgeBuilder::EdgeData observed = EdgeBuilder::UnpackEdgeHeader(observed_raw);
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
            /// Can Fail Spontenuiusly if used compare_exchange_weak()
            if (CompareExchangeStrongFromFabric(
                    index,
                    observed_raw,
                    EdgeBuilder::PackEdgeHeader(reserved)))
            {
                before = observed;
                return SeqLockedOperation::FOUND;
            }
            ///
        }
        return SeqLockedOperation::RETRY;
    }

    EdgeTableConstructor::SeqLockedOperation
    EdgeTableConstructor::ReserveEdgeRow_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeStatus required_status,
        EdgeBuilder::EdgeData &before,
        uint32_t max_tries) noexcept
    {
        return ReserveEdgeDomain_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            required_status,
            before,
            max_tries);
    }

    void EdgeTableConstructor::StoreReservedParentHandle_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        uint64_t parent_handle) noexcept
    {
        std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(edge_table, child_slot);
        std::atomic_ref<uint64_t>(relations[relation_ordinal].ParentHandle).store(parent_handle, std::memory_order_relaxed);
    }

    void EdgeTableConstructor::PublishReservedEdgeDomain_(
        FabricSegments edge_table,
        uint32_t row_slot,
        EdgeBuilder::EdgeDomain domain,
        const EdgeBuilder::EdgeData &before,
        uint32_t desired_tail,
        EdgeBuilder::EdgeStatus desired_status) noexcept
    {
        const size_t index = EdgeControlCellIndex_(edge_table, row_slot, domain);
        EdgeBuilder::EdgeData published{};
        published.TailLocator = desired_tail;
        published.SeqLock = EdgeBuilder::NextSequence(
            EdgeBuilder::NextSequence(before.SeqLock));
        published.Status = desired_status;
        published.IsValid = true;
        std::atomic_ref<uint64_t>(SlabBasePtr_[index]).store(EdgeBuilder::PackEdgeHeader(published), std::memory_order_release);
    }

    void EdgeTableConstructor::PublishReservedEdgeRow_(
        FabricSegments edge_table,
        uint32_t row_slot,
        const EdgeBuilder::EdgeData &before,
        uint32_t desired_tail,
        EdgeBuilder::EdgeStatus desired_status) noexcept
    {
        PublishReservedEdgeDomain_(
            edge_table,
            row_slot,
            EdgeBuilder::EdgeDomain::CHILD_LIST,
            before,
            desired_tail,
            desired_status);
    }

    void EdgeTableConstructor::StoreReservedSiblingLocators_(
        FabricSegments edge_table,
        uint32_t child_slot,
        uint8_t relation_ordinal,
        uint64_t sibling_locators) noexcept
    {
        std::span<EdgeBuilder::ParentRelation> relations =
            ParentRelations_(edge_table, child_slot);
        std::atomic_ref<uint64_t>(relations[relation_ordinal].SiblingLocators).store(sibling_locators, std::memory_order_relaxed);
    }
}