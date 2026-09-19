#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    bool AdaptivePackedCellContainer::IsOpenGeneration_() noexcept
    {
        return HandleOfAPCStatic::IsOpenGeneration(
            std::atomic_ref<uint64_t>(*APCCache_.GenerationCellPtr_).load(std::memory_order_acquire),
            APCCache_.CurrentGeneration_
        );
    }

    bool AdaptivePackedCellContainer::AddParent(
        AdaptivePackedCellContainer& parent,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        return
            IsFabricBound_() &&
            parent.IsFabricBound_() &&
            APCCache_.FabricOwnerPtr_ == parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->AddParentRelation_(
                parent.APCCache_.APCSlotIdx_,
                parent.APCCache_.CurrentGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                max_tries
            );
    }

    bool AdaptivePackedCellContainer::RemoveParent(
        AdaptivePackedCellContainer& parent,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        return
            IsFabricBound_() &&
            parent.IsFabricBound_() &&
            APCCache_.FabricOwnerPtr_ == parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->RemoveParentRelation_(
                parent.APCCache_.APCSlotIdx_,
                parent.APCCache_.CurrentGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                max_tries
            );
    }

    bool AdaptivePackedCellContainer::ReplaceParent(
        AdaptivePackedCellContainer& old_parent,
        AdaptivePackedCellContainer& new_parent,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        return
            &old_parent != &new_parent &&
            IsFabricBound_() &&
            old_parent.IsFabricBound_() &&
            new_parent.IsFabricBound_() &&
            APCCache_.FabricOwnerPtr_ == old_parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_ == new_parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->ReplaceParentRelation_(
                old_parent.APCCache_.APCSlotIdx_,
                old_parent.APCCache_.CurrentGeneration_,
                new_parent.APCCache_.APCSlotIdx_,
                new_parent.APCCache_.CurrentGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                max_tries
            );
    }

    bool AdaptivePackedCellContainer::AttachMyChild(
        AdaptivePackedCellContainer& child,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        return child.AddParent(*this, edge_table, max_tries);
    }

    bool AdaptivePackedCellContainer::DetachMyChild(
        AdaptivePackedCellContainer& child,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        return child.RemoveParent(*this, edge_table, max_tries);
    }
    
    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindParent(
        FabricSegments edge_table,
        uint8_t relation_ordinal,
        RelationOparation* parent_relation,
        uint32_t max_tries
    ) noexcept
    {
        return IsFabricBound_()
            ? APCCache_.FabricOwnerPtr_->FindParent_(
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                relation_ordinal,
                parent_relation,
                max_tries
            )
            : AdaptivePackedCellContainer{};
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindFirstChild(
        FabricSegments edge_table,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        return IsFabricBound_()
            ? APCCache_.FabricOwnerPtr_->FindFirstChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                child_relation,
                max_tries
            )
            : AdaptivePackedCellContainer{};
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindLastChild(
        FabricSegments edge_table,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        return IsFabricBound_()
            ? APCCache_.FabricOwnerPtr_->FindLastChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                child_relation,
                max_tries
            )
            : AdaptivePackedCellContainer{};
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindNextChild(
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        return IsFabricBound_()
            ? APCCache_.FabricOwnerPtr_->FindNextChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                current_relation_locator,
                child_relation,
                max_tries
            )
            : AdaptivePackedCellContainer{};
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindPreviousChild(
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        return IsFabricBound_()
            ? APCCache_.FabricOwnerPtr_->FindPreviousChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.CurrentGeneration_,
                edge_table,
                current_relation_locator,
                child_relation,
                max_tries
            )
            : AdaptivePackedCellContainer{};
    }

    bool AdaptivePackedCellContainer::Retire(
        uint32_t max_tries
    ) noexcept
    {
        if (!IsFabricBound_())
        {
            return false;
        }

        APCFinilizer* owner = APCCache_.FabricOwnerPtr_;
        const uint32_t slot = APCCache_.APCSlotIdx_;
        const uint32_t generation = APCCache_.CurrentGeneration_;

        if (!owner->RetireAPC_(slot, generation, max_tries))
        {
            return false;
        }
        ReleseFabricBindingOnly_();
        return true;
    }

}