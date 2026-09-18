#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    bool APCFinilizer::BindExistingAPCSnapshot_(
        uint32_t slot,
        AdaptivePackedCellContainer& apc,
        std::optional<uint32_t> expected_generation 
    ) noexcept
    {
        if (
            slot >= FabCache_->CountOfAPC_ ||
            apc.IsFabricBound_() ||
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

        uint64_t * const generation_cell = SlabBasePtr_ + FabCache_->HandleTableBeginIndex_ + HandleOfAPCStatic::CellOffset(slot);
        const HandleOfAPCStatic::ControlValues control = HandleOfAPCStatic::ReadControlCell(
            std::atomic_ref<const uint64_t>(*generation_cell).load(std::memory_order_acquire)
        );
        if (
            control.Closed ||
            !HandleOfAPCStatic::IsGenerationValid(control.Generation) ||
            (
                expected_generation.has_value() &&
                control.Generation != expected_generation.value()
            )
        )
        {
            return false;
        }
        
        apc.APCCache_.FabricOwnerPtr_ = this;
        apc.APCCache_.RawAPCBasePtr_ = reinterpret_cast<std::byte*>(
            SlabBasePtr_ +
            FabCache_->SegmentPoolBegin_ +
            static_cast<size_t>(slot) * FabCache_->PerAPCRuntimeCellCount_
        );
        apc.APCCache_.APCSlotIdx_ = slot;
        apc.APCCache_.GenerationCellPtr_ = generation_cell;
        apc.APCCache_.CurrentGeneration_ = control.Generation;
        return true;
    }



    bool APCFinilizer::GetExistingAPC_(
        uint32_t slot,
        AdaptivePackedCellContainer& apc,
        APCUseScope& apc_use,
        std::optional<uint32_t> expected_generation
    ) noexcept
    {
        if (!IsFabricActive() || apc_use)
        {
            return false;
        }

        if (!BindExistingAPCSnapshot_(slot, apc, expected_generation))
        {
            return false;
        }

        APCUseScope acquired_use = apc.AcquireAPCUse_();
        if (!acquired_use)
        {
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
        AdaptivePackedCellContainer& child
    ) noexcept
    {
        child = AdaptivePackedCellContainer{};

        if (
            parent_slot >= FabCache_->CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(parent_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationLocator(
                locator,
                static_cast<uint32_t>(FabCache_->CountOfAPC_),
                FabCache_->MaxDirectParentsPerAxis_
            )
        )
        {
            return SeqLockedOperation::NONE;
        }

        return BindExistingAPCSnapshot_(
            EdgeBuilder::RelationSlot(locator),
            child
        )
            ? SeqLockedOperation::FOUND
            : SeqLockedOperation::NONE;
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
            child_slot >= FabCache_->CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(child_generation) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge_table) ||
            !EdgeBuilder::IsValidRelationOrdinal(
                relation_ordinal,
                FabCache_->MaxDirectParentsPerAxis_
            )
        )
        {
            return parent;
        }

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            if (!IsOpenAPCGeneration_(child_slot, child_generation))
            {
                return {};
            }

            uint64_t parent_handle = FABRIC_CELL_SENTINAL;
            const SeqLockedOperation read = ReadParentHandle_(
                edge_table,
                child_slot,
                relation_ordinal,
                parent_handle,
                1u
            );
            if (read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (read != SeqLockedOperation::FOUND)
            {
                return {};
            }

            parent = {};
            if (
                !BindExistingAPCSnapshot_(
                    TwinU32ToU64::ExtractLow32Of64(parent_handle),
                    parent,
                    TwinU32ToU64::ExtractHigh32Of64(parent_handle)
                ) ||
                !IsOpenAPCGeneration_(child_slot, child_generation)
            )
            {
                continue;
            }

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
        return {};
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
        const uint64_t expected_parent = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            if (!IsOpenAPCGeneration_(parent_slot, parent_generation))
            {
                return {};
            }
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return {};
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
                return {};
            }

            const uint32_t tail = before.TailLocator;
            uint64_t tail_parent = FABRIC_CELL_SENTINAL;
            const SeqLockedOperation owner_read = ReadParentHandle_(
                edge_table,
                EdgeBuilder::RelationSlot(tail),
                EdgeBuilder::RelationOrdinal(tail),
                tail_parent,
                1u
            );
            if (owner_read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (owner_read != SeqLockedOperation::FOUND ||
                tail_parent != expected_parent)
            {
                return {};
            }

            const std::span<EdgeBuilder::ParentRelation> tail_row =
                ParentRelations_(edge_table, EdgeBuilder::RelationSlot(tail));
            const uint64_t sibling_raw = std::atomic_ref<const uint64_t>(
                tail_row[EdgeBuilder::RelationOrdinal(tail)].SiblingLocators
            ).load(std::memory_order_relaxed);
            const uint32_t first = TwinU32ToU64::ExtractHigh32Of64(sibling_raw);

            AdaptivePackedCellContainer child{};
            if (
                !EdgeBuilder::IsValidRelationLocator(
                    first,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ) ||
                ResolveChildLocator_(
                    parent_slot,
                    parent_generation,
                    edge_table,
                    first,
                    child
                ) != SeqLockedOperation::FOUND
            )
            {
                return {};
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after) ||
                !IsOpenAPCGeneration_(parent_slot, parent_generation)
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
        return {};
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
        const uint64_t expected_parent = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            if (!IsOpenAPCGeneration_(parent_slot, parent_generation))
            {
                return {};
            }
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return {};
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
                return {};
            }

            uint64_t owner = FABRIC_CELL_SENTINAL;
            const uint32_t tail = before.TailLocator;
            const SeqLockedOperation owner_read = ReadParentHandle_(
                edge_table,
                EdgeBuilder::RelationSlot(tail),
                EdgeBuilder::RelationOrdinal(tail),
                owner,
                1u
            );
            if (owner_read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (owner_read != SeqLockedOperation::FOUND || owner != expected_parent)
            {
                return {};
            }

            AdaptivePackedCellContainer child{};
            if (ResolveChildLocator_(
                parent_slot,
                parent_generation,
                edge_table,
                tail,
                child
            ) != SeqLockedOperation::FOUND)
            {
                return {};
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after) ||
                !IsOpenAPCGeneration_(parent_slot, parent_generation)
            )
            {
                continue;
            }

            result.RelationLocator_ = tail;
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
        return {};
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
        if (!EdgeBuilder::IsValidRelationLocator(
            current_relation_locator,
            static_cast<uint32_t>(FabCache_->CountOfAPC_),
            FabCache_->MaxDirectParentsPerAxis_
        ))
        {
            return {};
        }
        const uint64_t expected_parent = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            if (!IsOpenAPCGeneration_(parent_slot, parent_generation))
            {
                return {};
            }
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return {};
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
                return {};
            }

            uint64_t owner = FABRIC_CELL_SENTINAL;
            const SeqLockedOperation owner_read = ReadParentHandle_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator),
                EdgeBuilder::RelationOrdinal(current_relation_locator),
                owner,
                1u
            );
            if (owner_read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (owner_read != SeqLockedOperation::FOUND || owner != expected_parent)
            {
                return {};
            }

            if (current_relation_locator == before.TailLocator)
            {
                EdgeBuilder::EdgeData after{};
                if (
                    ReadEdgeHeader_(edge_table, parent_slot, after) &&
                    SameHeader_(before, after) &&
                    IsOpenAPCGeneration_(parent_slot, parent_generation)
                )
                {
                    return {};
                }
                continue;
            }

            const std::span<EdgeBuilder::ParentRelation> row = ParentRelations_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator)
            );
            const uint64_t sibling_raw = std::atomic_ref<const uint64_t>(
                row[EdgeBuilder::RelationOrdinal(current_relation_locator)]
                    .SiblingLocators
            ).load(std::memory_order_relaxed);
            const uint32_t next = TwinU32ToU64::ExtractHigh32Of64(sibling_raw);

            AdaptivePackedCellContainer child{};
            if (
                !EdgeBuilder::IsValidRelationLocator(
                    next,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ) ||
                ResolveChildLocator_(
                    parent_slot,
                    parent_generation,
                    edge_table,
                    next,
                    child
                ) != SeqLockedOperation::FOUND
            )
            {
                return {};
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after) ||
                !IsOpenAPCGeneration_(parent_slot, parent_generation)
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
        return {};
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
        if (!EdgeBuilder::IsValidRelationLocator(
            current_relation_locator,
            static_cast<uint32_t>(FabCache_->CountOfAPC_),
            FabCache_->MaxDirectParentsPerAxis_
        ))
        {
            return {};
        }
        const uint64_t expected_parent = EdgeBuilder::MakeParentHandle(
            parent_slot,
            parent_generation
        );

        for (uint32_t attempt = 0u; attempt < max_tries; ++attempt)
        {
            if (!IsOpenAPCGeneration_(parent_slot, parent_generation))
            {
                return {};
            }
            EdgeBuilder::EdgeData before{};
            if (!ReadEdgeHeader_(edge_table, parent_slot, before))
            {
                return {};
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
                return {};
            }

            uint64_t owner = FABRIC_CELL_SENTINAL;
            const SeqLockedOperation owner_read = ReadParentHandle_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator),
                EdgeBuilder::RelationOrdinal(current_relation_locator),
                owner,
                1u
            );
            if (owner_read == SeqLockedOperation::RETRY)
            {
                continue;
            }
            if (owner_read != SeqLockedOperation::FOUND || owner != expected_parent)
            {
                return {};
            }

            const std::span<EdgeBuilder::ParentRelation> row = ParentRelations_(
                edge_table,
                EdgeBuilder::RelationSlot(current_relation_locator)
            );
            const uint64_t sibling_raw = std::atomic_ref<const uint64_t>(
                row[EdgeBuilder::RelationOrdinal(current_relation_locator)]
                    .SiblingLocators
            ).load(std::memory_order_relaxed);
            const uint32_t previous = TwinU32ToU64::ExtractLow32Of64(sibling_raw);

            if (previous == before.TailLocator)
            {
                EdgeBuilder::EdgeData after{};
                if (
                    ReadEdgeHeader_(edge_table, parent_slot, after) &&
                    SameHeader_(before, after) &&
                    IsOpenAPCGeneration_(parent_slot, parent_generation)
                )
                {
                    return {};
                }
                continue;
            }

            AdaptivePackedCellContainer child{};
            if (
                !EdgeBuilder::IsValidRelationLocator(
                    previous,
                    static_cast<uint32_t>(FabCache_->CountOfAPC_),
                    FabCache_->MaxDirectParentsPerAxis_
                ) ||
                ResolveChildLocator_(
                    parent_slot,
                    parent_generation,
                    edge_table,
                    previous,
                    child
                ) != SeqLockedOperation::FOUND
            )
            {
                return {};
            }

            EdgeBuilder::EdgeData after{};
            if (
                !ReadEdgeHeader_(edge_table, parent_slot, after) ||
                !SameHeader_(before, after) ||
                !IsOpenAPCGeneration_(parent_slot, parent_generation)
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
        return {};
    }

    bool APCFinilizer::CreateAPC(
        AdaptivePackedCellContainer& desired_apc,
        const SD::RegionSchemaTable& region_schemas,
        uint32_t internal_max_tries,
        bool override_table
    ) noexcept
    {
        if (!IsFabricActive() || desired_apc.IsFabricBound_())
        {
            return false;
        }

        const std::optional<uint32_t> slot_new = GetASlotForNewAPCLink();
        if (!slot_new.has_value())
        {
            return false;
        }

        using Domain = EdgeBuilder::EdgeDomain;
        constexpr FabricSegments H =
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H;
        constexpr FabricSegments V =
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;
        const uint32_t slot = slot_new.value();

        EdgeBuilder::EdgeData h_parent_before{};
        EdgeBuilder::EdgeData h_list_before{};
        EdgeBuilder::EdgeData v_parent_before{};
        EdgeBuilder::EdgeData v_list_before{};
        bool h_parent_reserved = false;
        bool h_list_reserved = false;
        bool v_parent_reserved = false;
        bool v_list_reserved = false;
        bool descriptor_live = false;
        bool matrix_view_prepared = false;

        const auto ReleaseDomains___ = [&](EdgeBuilder::EdgeStatus list_status) noexcept
        {
            if (v_list_reserved)
            {
                PublishReservedEdgeDomain_(
                    V, slot, Domain::CHILD_LIST, v_list_before,
                    EdgeBuilder::RELATION_NULL, list_status
                );
                v_list_reserved = false;
            }
            if (v_parent_reserved)
            {
                PublishReservedEdgeDomain_(
                    V, slot, Domain::PARENT_RELATIONS, v_parent_before,
                    EdgeBuilder::RELATION_NULL, EdgeBuilder::EdgeStatus::LIVE
                );
                v_parent_reserved = false;
            }
            if (h_list_reserved)
            {
                PublishReservedEdgeDomain_(
                    H, slot, Domain::CHILD_LIST, h_list_before,
                    EdgeBuilder::RELATION_NULL, list_status
                );
                h_list_reserved = false;
            }
            if (h_parent_reserved)
            {
                PublishReservedEdgeDomain_(
                    H, slot, Domain::PARENT_RELATIONS, h_parent_before,
                    EdgeBuilder::RELATION_NULL, EdgeBuilder::EdgeStatus::LIVE
                );
                h_parent_reserved = false;
            }
        };

        const auto AbortCreation___ = [&]() noexcept
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
            ReleaseDomains___(EdgeBuilder::EdgeStatus::FREE);
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

        uint64_t* const generation_cell = GetAPCGenerationPtr_(slot);
        if (!generation_cell)
        {
            AbortCreation___();
            return false;
        }
        const HandleOfAPCStatic::ControlValues control_values =
            HandleOfAPCStatic::ReadControlCell(
                std::atomic_ref<const uint64_t>(*generation_cell).load(
                    std::memory_order_acquire
                )
            );
        if (
            !control_values.Closed ||
            control_values.ActiveAccess != 0u ||
            !HandleOfAPCStatic::IsGenerationValid(control_values.Generation)
        )
        {
            AbortCreation___();
            return false;
        }

        if (
            !FabCache_->HasDefaultRegionTable_ ||
            override_table ||
            control_values.Generation != HandleOfAPCStatic::FIRST_GENERATION
        )
        {
            const SD::RegionSchemaTable& schemas =
                FabCache_->HasDefaultRegionTable_ && !override_table
                    ? DefaultRegionTable_
                    : region_schemas;
            if (!PrepareMatrixViewRow_(slot, schemas))
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

        if (ReserveEdgeDomain_(
            H, slot, Domain::PARENT_RELATIONS,
            EdgeBuilder::EdgeStatus::LIVE,
            h_parent_before, internal_max_tries
        ) != SeqLockedOperation::FOUND)
        {
            AbortCreation___();
            return false;
        }
        h_parent_reserved = true;
        if (ReserveEdgeDomain_(
            H, slot, Domain::CHILD_LIST,
            EdgeBuilder::EdgeStatus::FREE,
            h_list_before, internal_max_tries
        ) != SeqLockedOperation::FOUND)
        {
            AbortCreation___();
            return false;
        }
        h_list_reserved = true;
        if (ReserveEdgeDomain_(
            V, slot, Domain::PARENT_RELATIONS,
            EdgeBuilder::EdgeStatus::LIVE,
            v_parent_before, internal_max_tries
        ) != SeqLockedOperation::FOUND)
        {
            AbortCreation___();
            return false;
        }
        v_parent_reserved = true;
        if (ReserveEdgeDomain_(
            V, slot, Domain::CHILD_LIST,
            EdgeBuilder::EdgeStatus::FREE,
            v_list_before, internal_max_tries
        ) != SeqLockedOperation::FOUND)
        {
            AbortCreation___();
            return false;
        }
        v_list_reserved = true;

        const auto ReservedRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            const std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);
            if (relations.size() != FabCache_->MaxDirectParentsPerAxis_)
            {
                return false;
            }
            for (uint8_t ordinal = 0u;
                ordinal < FabCache_->MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                if (
                    std::atomic_ref<const uint64_t>(
                        relations[ordinal].ParentHandle
                    ).load(std::memory_order_relaxed) != FABRIC_CELL_SENTINAL ||
                    std::atomic_ref<const uint64_t>(
                        relations[ordinal].SiblingLocators
                    ).load(std::memory_order_relaxed) != FABRIC_CELL_SENTINAL
                )
                {
                    return false;
                }
            }
            return true;
        };

        if (
            h_list_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            v_list_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            !ReservedRowIsEmpty___(H) ||
            !ReservedRowIsEmpty___(V) ||
            !SwitchDescriptionState(
                slot,
                StateOfAPC::LIVE,
                StateOfAPC::RESERVED,
                internal_max_tries
            )
        )
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

        ReleaseDomains___(EdgeBuilder::EdgeStatus::LIVE);
        return true;
    }

    std::optional<uint32_t> APCFinilizer::GetASlotForNewAPCLink() noexcept
    {
        if (
            !FabricInitialized_.load(std::memory_order_acquire) ||
            !SlabBasePtr_ || 
            !ADS::IsCapacityOfAPCValid(FabCache_->PerAPCRuntimeCellCount_)
        )
        {
            return std::nullopt;
        }

        std::optional<uint32_t> maybe_First_free = ReadFirstFreeAPCIdx_();

        if (maybe_First_free.has_value())
        {
            for (uint32_t description_idx = maybe_First_free.value(); description_idx < FabCache_->CountOfAPC_; description_idx++)
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
                uint32_t expected = maybe_First_free.value();
                UpdateFirstFreeIdx_(expected, description_idx);
                return description_idx;
            }
        }

        if (maybe_First_free.has_value())
        {
            uint32_t expected = maybe_First_free.value();
            UpdateFirstFreeIdx_(expected, ADS::APC_INDEX_BOUND_SENTINAL);
        }
        
        for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; slot++)
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
            slot >= FabCache_->CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(generation)
        )
        {
            return false;
        }

        using Domain = EdgeBuilder::EdgeDomain;
        constexpr FabricSegments H =
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H;
        constexpr FabricSegments V =
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;

        EdgeBuilder::EdgeData h_parent_before{};
        EdgeBuilder::EdgeData h_list_before{};
        EdgeBuilder::EdgeData v_parent_before{};
        EdgeBuilder::EdgeData v_list_before{};
        bool h_parent_reserved = false;
        bool h_list_reserved = false;
        bool v_parent_reserved = false;
        bool v_list_reserved = false;

        const auto ReleaseDomains___ = [&]() noexcept
        {
            if (v_list_reserved)
            {
                PublishReservedEdgeDomain_(
                    V,
                    slot,
                    Domain::CHILD_LIST,
                    v_list_before,
                    v_list_before.TailLocator,
                    EdgeBuilder::EdgeStatus::LIVE
                );
                v_list_reserved = false;
            }
            if (v_parent_reserved)
            {
                PublishReservedEdgeDomain_(
                    V,
                    slot,
                    Domain::PARENT_RELATIONS,
                    v_parent_before,
                    EdgeBuilder::RELATION_NULL,
                    EdgeBuilder::EdgeStatus::LIVE
                );
                v_parent_reserved = false;
            }
            if (h_list_reserved)
            {
                PublishReservedEdgeDomain_(
                    H,
                    slot,
                    Domain::CHILD_LIST,
                    h_list_before,
                    h_list_before.TailLocator,
                    EdgeBuilder::EdgeStatus::LIVE
                );
                h_list_reserved = false;
            }
            if (h_parent_reserved)
            {
                PublishReservedEdgeDomain_(
                    H,
                    slot,
                    Domain::PARENT_RELATIONS,
                    h_parent_before,
                    EdgeBuilder::RELATION_NULL,
                    EdgeBuilder::EdgeStatus::LIVE
                );
                h_parent_reserved = false;
            }
        };

        if (ReserveEdgeDomain_(
            H,
            slot,
            Domain::PARENT_RELATIONS,
            EdgeBuilder::EdgeStatus::LIVE,
            h_parent_before,
            max_tries
        ) != SeqLockedOperation::FOUND)
        {
            return false;
        }
        h_parent_reserved = true;

        if (ReserveEdgeDomain_(
            H,
            slot,
            Domain::CHILD_LIST,
            EdgeBuilder::EdgeStatus::LIVE,
            h_list_before,
            max_tries
        ) != SeqLockedOperation::FOUND)
        {
            ReleaseDomains___();
            return false;
        }
        h_list_reserved = true;

        if (ReserveEdgeDomain_(
            V,
            slot,
            Domain::PARENT_RELATIONS,
            EdgeBuilder::EdgeStatus::LIVE,
            v_parent_before,
            max_tries
        ) != SeqLockedOperation::FOUND)
        {
            ReleaseDomains___();
            return false;
        }
        v_parent_reserved = true;

        if (ReserveEdgeDomain_(
            V,
            slot,
            Domain::CHILD_LIST,
            EdgeBuilder::EdgeStatus::LIVE,
            v_list_before,
            max_tries
        ) != SeqLockedOperation::FOUND)
        {
            ReleaseDomains___();
            return false;
        }
        v_list_reserved = true;

        const auto ReservedRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            const std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);
            if (relations.size() != FabCache_->MaxDirectParentsPerAxis_)
            {
                return false;
            }
            for (uint8_t ordinal = 0u;
                ordinal < FabCache_->MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                const uint64_t parent = std::atomic_ref<const uint64_t>(
                    relations[ordinal].ParentHandle
                ).load(std::memory_order_relaxed);
                const uint64_t siblings = std::atomic_ref<const uint64_t>(
                    relations[ordinal].SiblingLocators
                ).load(std::memory_order_relaxed);
                if (
                    parent != FABRIC_CELL_SENTINAL ||
                    siblings != FABRIC_CELL_SENTINAL
                )
                {
                    return false;
                }
            }
            return true;
        };

        if (
            h_list_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            v_list_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            !ReservedRowIsEmpty___(H) ||
            !ReservedRowIsEmpty___(V) ||
            !CloseAPCGeneration_(slot, generation)
        )
        {
            ReleaseDomains___();
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
            ReleaseDomains___();
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
            ReleaseDomains___();
            return false;
        }

        PublishReservedEdgeDomain_(
            V,
            slot,
            Domain::CHILD_LIST,
            v_list_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        v_list_reserved = false;
        PublishReservedEdgeDomain_(
            V,
            slot,
            Domain::PARENT_RELATIONS,
            v_parent_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::LIVE
        );
        v_parent_reserved = false;
        PublishReservedEdgeDomain_(
            H,
            slot,
            Domain::CHILD_LIST,
            h_list_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        h_list_reserved = false;
        PublishReservedEdgeDomain_(
            H,
            slot,
            Domain::PARENT_RELATIONS,
            h_parent_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::LIVE
        );
        h_parent_reserved = false;
        return true;
    }
    bool APCFinilizer::ReclaimRetiredSlotTemp_(uint32_t slot) noexcept
    {
        if (slot >= FabCache_->CountOfAPC_)
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

            if (relations.size() != FabCache_->MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0u;
                ordinal < FabCache_->MaxDirectParentsPerAxis_;
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
        if (!FabCache_->HasDefaultRegionTable_)
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
        }
        return true;
    }
}
