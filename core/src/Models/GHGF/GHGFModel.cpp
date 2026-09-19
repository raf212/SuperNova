#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 

    bool GHGFModel::IsGHGFPlanCurrent_() noexcept
    {
        return IsFabricActive() && GHGFCache_.ModelPrepared_ &&
            GHGFCache_.PreparedRevision_ == SealedDAGRevision_.load(std::memory_order_acquire);
    }

    float* GHGFModel::GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept
    {
        return RegionT_<float>(slot, cell_offset);
    }

    float* GHGFModel::GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.StateCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    float* GHGFModel::GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.ErrorCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    float* GHGFModel::GHGFWeight_(uint32_t slot) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.WeightCellOffset_);
    }

    float* GHGFModel::FFRowGHGF_(uint32_t slot, GM::GHGFMessageFForward row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.FFCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    float* GHGFModel::FBRowGHGF_(uint32_t slot, GM::GHGFMessageFBackward row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.FBCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    void GHGFModel::InvalidateGHGFModel_() noexcept
    {
        GHGFCache_.ModelPrepared_ = false;
        GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
        GHGFCache_.ActiveBatch_ = UNSIGNED_ZERO;
    }

    uint64_t GHGFModel::GHGFParentMask_(uint32_t slot, FabricSegments axis) noexcept
    {
        CompiledDAGRecord* record = CompiledDAGRow_(slot);
        return axis == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ?
            record->ValueParentMask : record->VolatileParentMask;
    }


    bool GHGFModel::GetGHGFNode_(
        uint32_t slot, GHGFNode& node, APCUseScope& use) noexcept
    {
        if (!GetExistingAPC_(slot, node, use))
        {
            return false;
        }
        node.GHGFFabric_ = this;
        if (!node.GHGFRole_().has_value())
        {
            use.Release();
            node.ReleseFabricBindingOnly_();
            node.GHGFFabric_ = nullptr;
            return false;
        }
        return true;
    }

    bool GHGFModel::ConnectGHGFParent(const GM::GHGFConnection& connection) noexcept
    {
        GHGFNode parent, child;
        APCUseScope parent_use, child_use;

        if (
            !GetGHGFNode_(connection.Parent, parent, parent_use) ||
            !GetGHGFNode_(connection.Child, child, child_use) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(connection.Edge) ||
            !std::isfinite(connection.Coupling) ||
            parent.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION
        )
        {
            return false;
        }
        
        if (
            child.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION &&
            connection.Edge != FabricSegments::VALUE_PARENT_EDGE_TABLE_H 
        )
        {
            return false;
        }

        if (!child.AddParent(parent, connection.Edge))
        {
            return false;
        }
        
        InvalidateGHGFModel_();

        const std::span<EdgeBuilder::ParentRelation> retations = ParentRelations_(connection.Edge, connection.Child);
        for (uint8_t i = 0; i < FabCache_->MaxDirectParentsPerAxis_; i++)
        {
            if (
                !EdgeBuilder::IsEmpty(retations[i]) &&
                EdgeBuilder::ParentSlot(retations[i]) == connection.Parent
            )
            {
                GHGFRegion_(connection.Child, GHGFCache_.WeightCellOffset_)[GM::CouplingIndex(connection.Edge, i, FabCache_->MaxDirectParentsPerAxis_)] = connection.Coupling;
                return true;
            }
        }
        child.RemoveParent(parent, connection.Edge);
        return false;
    }


    bool GHGFModel::RemoveParent(const GM::GHGFConnection& connection) noexcept
    {
        GHGFNode parent, child;
        APCUseScope parent_use, child_use;
        if (
            !GetGHGFNode_(connection.Parent, parent, parent_use) ||
            !GetGHGFNode_(connection.Child, child, child_use) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(connection.Edge)
        )
        {
            return false;
        }

        if (!child.RemoveParent(parent, connection.Edge))
        {
            return false;
        }
        InvalidateGHGFModel_();
        return true;
    }

    std::optional<float> GHGFModel::GetGHGFParameter_(uint32_t slot, uint32_t index) noexcept
    {
        GHGFNode node;
        APCUseScope use;
        if (!GetGHGFNode_(slot, node, use) || index >= Profile_.ParameterCount)
        {
            return std::nullopt;
        }
        return GHGFRegion_(slot, GHGFCache_.WeightCellOffset_)[index];
    }

    bool GHGFModel::SetGHGFParameter_(
        uint32_t slot,
        uint32_t index,
        float value
    ) noexcept
    {
        using EI = GM::GHGFErrorValueIndexing;

        GHGFNode node;
        APCUseScope use;

        if (
            !GetGHGFNode_(slot, node, use) ||
            index >= Profile_.ParameterCount ||
            !std::isfinite(value)
        )
        {
            return false;
        }

        const std::optional<GM::GHGFNodeRole> role = node.GHGFRole_();

        if (!role.has_value())
        {
            return false;
        }

        if (role.value() == GM::GHGFNodeRole::OBSERVATION)
        {
            const uint32_t h_begin = GM::FIRST_COUPLING_INDEX;

            const uint32_t h_end =
                h_begin +
                static_cast<uint32_t>(
                    Profile_.MaxDirectParentPerAxis
                );

            const bool allowed =
                index == static_cast<uint32_t>(EI::TONIC_DRIFT) ||
                (
                    index >= h_begin &&
                    index < h_end
                );

            if (!allowed)
            {
                return false;
            }
        }

        GHGFWeight_(slot)[index] = value;
        GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
        return true;
    }
}