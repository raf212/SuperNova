#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 
    using GM = GHGFLayerModel;

    void GHGFStructralLearningModel::PublishCoupling_(
        void* raw_context,
        uint8_t ordinal
    ) noexcept
    {
        auto* const context =
            static_cast<CouplingPublicationContext*>(raw_context);
        if (!context || !context->Model)
            return;

        const uint32_t index = GM::CouplingIndex(
            context->Edge,
            ordinal,
            context->Model->Profile_.MaxDirectParentPerAxis
        );

        std::atomic_ref<float>(
            context->Model->GHGFWeight_(context->Child)[index]
        ).store(context->Coupling, std::memory_order_relaxed);
    }

    GM::GHGFConcurrentOperation
    GHGFStructralLearningModel::ReadStructureSnapshotConcurrently(
        uint32_t child,
        FabricSegments edge,
        GM::GHGFStructureSnapshot& snapshot,
        uint32_t max_tries
    ) noexcept
    {
        snapshot = {};
        if (
            !IsGHGFModelReady_() ||
            !GHGFCache_.StructuralLearningActive_ ||
            max_tries == UNSIGNED_ZERO
        )
        {
            return GM::GHGFConcurrentOperation::REJECTED;
        }

        GM::GHGFParentExecutionSnapshot execution{};
        const FabricToAPCLinker::SeqLockedOperation read =
            ReadGHGFParentExecutionSnapshot_(
                child, edge, execution, max_tries);

        if (read == FabricToAPCLinker::SeqLockedOperation::RETRY)
            return GM::GHGFConcurrentOperation::RETRY;
        if (read != FabricToAPCLinker::SeqLockedOperation::FOUND)
            return GM::GHGFConcurrentOperation::REJECTED;

        snapshot.Child = child;
        snapshot.Edge = edge;
        snapshot.RowSequence = execution.RowSequence;
        snapshot.ParentMask = execution.ParentMask;
        snapshot.IsValid = true;
        return GM::GHGFConcurrentOperation::SUCCESS;
    }

    GM::GHGFStructureMutationResult
    GHGFStructralLearningModel::TryApplyGHGFStructureMutation(
        const GM::GHGFStructureMutation& mutation,
        uint32_t max_tries
    ) noexcept
    {
        using Concurrent = GM::GHGFConcurrentOperation;
        using Operation = EdgeBuilder::StructureOperation;

        GM::GHGFStructureMutationResult result{};
        if (
            !IsGHGFModelReady_() ||
            !GHGFCache_.StructuralLearningActive_ ||
            max_tries == UNSIGNED_ZERO ||
            mutation.Child >= FabCache_->CountOfAPC_ ||
            mutation.ExpectedRowSequence == UINT32_MAX ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(mutation.Edge)
        )
        {
            return result;
        }

        const bool adds = mutation.Operation == Operation::ADD_PARENT;
        const bool removes = mutation.Operation == Operation::REMOVE_PARENT;
        const bool replaces = mutation.Operation == Operation::REPLACE_PARENT;
        if (!adds && !removes && !replaces)
            return result;
        if ((adds || replaces) && !std::isfinite(mutation.Coupling))
            return result;

        GM::GHGFParentExecutionSnapshot current{};
        const auto read = ReadGHGFParentExecutionSnapshot_(
            mutation.Child, mutation.Edge, current, max_tries);
        if (read == FabricToAPCLinker::SeqLockedOperation::RETRY)
        {
            result.Result = Concurrent::RETRY;
            return result;
        }
        if (read != FabricToAPCLinker::SeqLockedOperation::FOUND)
            return result;
        if (current.RowSequence != mutation.ExpectedRowSequence)
        {
            result.Result = Concurrent::STALE;
            return result;
        }

        const auto FindOrdinal___ = [&](uint32_t wanted) noexcept -> uint8_t
        {
            for (uint64_t mask = current.ParentMask; mask; mask &= mask - 1u)
            {
                const uint8_t ordinal =
                    static_cast<uint8_t>(std::countr_zero(mask));
                if (TwinU32ToU64::ExtractLow32Of64(
                        current.ParentHandles[ordinal]) == wanted)
                {
                    return ordinal;
                }
            }
            return UINT8_MAX;
        };

        const uint8_t old_ordinal =
            removes || replaces ? FindOrdinal___(mutation.OldParent) : UINT8_MAX;
        const uint8_t new_ordinal =
            adds || replaces ? FindOrdinal___(mutation.NewParent) : UINT8_MAX;

        if (
            ((removes || replaces) && old_ordinal == UINT8_MAX) ||
            ((adds || replaces) && new_ordinal != UINT8_MAX) ||
            (adds && std::popcount(current.ParentMask) >=
                FabCache_->MaxDirectParentsPerAxis_)
        )
        {
            return result;
        }

        GM::GHGFNodeRole child_role{};
        uint32_t child_generation = UNSIGNED_ZERO;
        if (!ReadGHGFNodeIdentity_(
                mutation.Child, child_role, child_generation))
        {
            return result;
        }

        if (
            child_role == GM::GHGFNodeRole::OBSERVATION &&
            mutation.Edge != FabricSegments::VALUE_PARENT_EDGE_TABLE_H
        )
        {
            return result;
        }
        if (
            removes &&
            child_role == GM::GHGFNodeRole::OBSERVATION &&
            std::popcount(current.ParentMask) == 1u
        )
        {
            return result;
        }

        GM::GHGFNodeRole old_role{};
        GM::GHGFNodeRole new_role{};
        uint32_t old_generation = UNSIGNED_ZERO;
        uint32_t new_generation = UNSIGNED_ZERO;

        if ((removes || replaces) &&
            !ReadGHGFNodeIdentity_(
                mutation.OldParent, old_role, old_generation))
        {
            return result;
        }
        if ((adds || replaces) &&
            (!ReadGHGFNodeIdentity_(
                mutation.NewParent, new_role, new_generation) ||
            new_role == GM::GHGFNodeRole::OBSERVATION ||
            !EdgeBuilder::CanInsertCombinedDAGRelation(
                mutation.NewParent, mutation.Child)))
        {
            return result;
        }

        const float cleared_coupling =
            mutation.Edge == FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                ? GM::StorageConst::INITIAL_VALUE_COUPLING
                : GM::StorageConst::INITIAL_VOLATILITY_COUPLING;

        CouplingPublicationContext context{
            this,
            mutation.Child,
            mutation.Edge,
            removes ? cleared_coupling : mutation.Coupling
        };

        ConditionalParentPublication publication{};
        publication.ExpectedRowSequence = mutation.ExpectedRowSequence;
        publication.Context = &context;
        publication.Publish = &GHGFStructralLearningModel::PublishCoupling_;

        bool committed = false;
        switch (mutation.Operation)
        {
        case Operation::ADD_PARENT:
            committed = AddParentRelation_(
                mutation.NewParent,
                new_generation,
                mutation.Child,
                child_generation,
                mutation.Edge,
                &publication,
                max_tries
            );
            break;

        case Operation::REMOVE_PARENT:
            committed = RemoveParentRelation_(
                mutation.OldParent,
                old_generation,
                mutation.Child,
                child_generation,
                mutation.Edge,
                &publication,
                max_tries
            );
            break;

        case Operation::REPLACE_PARENT:
            committed = ReplaceParentRelation_(
                mutation.OldParent,
                old_generation,
                mutation.NewParent,
                new_generation,
                mutation.Child,
                child_generation,
                mutation.Edge,
                &publication,
                max_tries
            );
            break;
        }

        if (publication.SequenceMismatch)
        {
            result.Result = Concurrent::STALE;
            return result;
        }
        if (!committed)
        {
            result.Result = Concurrent::RETRY;
            return result;
        }

        result.Result = Concurrent::SUCCESS;
        result.PublishedRowSequence = publication.PublishedRowSequence;
        result.PublishedOrdinal = publication.PublishedOrdinal;
        return result;
    }
}