#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{


    bool AdaptivePackedCellContainer::AddParent(
        AdaptivePackedCellContainer& parent,
        FabricSegments edge_table,
        uint32_t max_tries
    ) noexcept
    {
        APCUseScope child_use = AcquireAPCUse_();
        APCUseScope parent_use = parent.AcquireAPCUse_();

        return 
            child_use &&
            parent_use &&
            APCCache_.FabricOwnerPtr_ == parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->AddParentRelation_(
                parent.APCCache_.APCSlotIdx_,
                parent.APCCache_.ExpectedGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
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
        APCUseScope child_use = AcquireAPCUse_();
        APCUseScope parent_use = parent.AcquireAPCUse_();

        return
            child_use &&
            parent_use &&
            APCCache_.FabricOwnerPtr_ == parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->RemoveParentRelation_(
                parent.APCCache_.APCSlotIdx_,
                parent.APCCache_.ExpectedGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
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
        if (&old_parent == &new_parent)
        {
            return false;
        }

        APCUseScope child_use = AcquireAPCUse_();
        APCUseScope old_parent_use = old_parent.AcquireAPCUse_();
        APCUseScope new_parent_use = new_parent.AcquireAPCUse_();

        return
            child_use &&
            old_parent_use &&
            new_parent_use &&
            APCCache_.FabricOwnerPtr_ == old_parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_ == new_parent.APCCache_.FabricOwnerPtr_ &&
            APCCache_.FabricOwnerPtr_->ReplaceParentRelation_(
                old_parent.APCCache_.APCSlotIdx_,
                old_parent.APCCache_.ExpectedGeneration_,
                new_parent.APCCache_.APCSlotIdx_,
                new_parent.APCCache_.ExpectedGeneration_,
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
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
        APCUseScope use = AcquireAPCUse_();

        if (!use)
        {
            return AdaptivePackedCellContainer{};
        }

        return 
            APCCache_.FabricOwnerPtr_->FindParent_(
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
                edge_table,
                relation_ordinal,
                parent_relation,
                max_tries
            );
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindFirstChild(
        FabricSegments edge_table,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        APCUseScope use = AcquireAPCUse_();

        if (!use)
        {
            return AdaptivePackedCellContainer{};
        }

        return 
            APCCache_.FabricOwnerPtr_->FindFirstChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
                edge_table,
                child_relation,
                max_tries
            );
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindLastChild(
        FabricSegments edge_table,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        APCUseScope use = AcquireAPCUse_();

        if (!use)
        {
            return AdaptivePackedCellContainer{};
        }

        return 
            APCCache_.FabricOwnerPtr_->FindLastChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
                edge_table,
                child_relation,
                max_tries
            );
    }

    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindNextChild(
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        APCUseScope use = AcquireAPCUse_();

        if (!use)
        {
            return AdaptivePackedCellContainer{};
        }

        return 
            APCCache_.FabricOwnerPtr_->FindNextChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
                edge_table,
                current_relation_locator,
                child_relation,
                max_tries
            );
    }


    AdaptivePackedCellContainer AdaptivePackedCellContainer::FindPreviousChild(
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        RelationOparation* child_relation,
        uint32_t max_tries
    ) noexcept
    {
        APCUseScope use = AcquireAPCUse_();

        if (!use)
        {
            return AdaptivePackedCellContainer{};
        }

        return 
            APCCache_.FabricOwnerPtr_->FindPreviousChild_(
                APCCache_.APCSlotIdx_,
                APCCache_.ExpectedGeneration_,
                edge_table,
                current_relation_locator,
                child_relation,
                max_tries
            );
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
        const uint32_t generation = APCCache_.ExpectedGeneration_;

        if (!owner->RetireAPC_(slot, generation, max_tries))
        {
            return false;
        }
        ReleseFabricBindingOnly_();
        return true;
    }

}