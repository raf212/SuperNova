#pragma once
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 

    bool GHGFModelConstructor::CreateNodeOfGHGF(
        GHGFNode& desired_apc,
        GM::GHGFNodeRole role
    ) noexcept
    {
        if (
            !IsFabricActive() ||
            !HasDefaultRegionTable_ ||
            !CreateAPC(desired_apc, DefaultRegionTable_)
        )
        {
            return false;
        }

        desired_apc.GHGFFabric_ = this;
        if (desired_apc.InitializeGHGFNode(role))
        {
            return true;
        }
        
        InvalidateGHGFModel_();
        if (desired_apc.Retire())
        {
            desired_apc.GHGFFabric_ = nullptr;
        }
        return false;
    }

    bool GHGFModelConstructor::IsGHGFPlanCurrent_() noexcept
    {
        return IsFabricActive() && HGFCache_.ModelPrepared_ &&
            HGFCache_.PreparedRevision_ == CompiledDagRevision_.load(std::memory_order_acquire);
    }

    float* GHGFModelConstructor::GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept
    {
        return RegionT_<float>(slot, cell_offset);
    }

    float* GHGFModelConstructor::GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept
    {
        return GHGFRegion_(slot, HGFCache_.StateCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    float* GHGFModelConstructor::GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept
    {
        return GHGFRegion_(slot, HGFCache_.ErrorCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    uint64_t GHGFModelConstructor::GHGFParentMask_(uint32_t slot, FabricSegments axis) noexcept
    {
        CompiledDAGRecord* record = CompiledDAGRow_(slot);
        return axis == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ?
            record->ValueParentMask : record->VolatileParentMask;
    }

    void GHGFModelConstructor::InvalidateGHGFModel_() noexcept
    {
        HGFCache_.ModelPrepared_ = false;
        HGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
        HGFCache_.ActiveBatch_ = UNSIGNED_ZERO;
    }



    bool GHGFModelConstructor::InitializeGHGFFabric(
        uint32_t slot_count,
        const GHGFLayerModel::GHGFStorageProfile& profile
    ) noexcept
    {
        if (
            IsFabricActive() || 
            slot_count == UNSIGNED_ZERO || 
            !GM::IsValidStoregeProfile(profile)
        )
        {
            return false;
        }
        
        InvalidateGHGFModel_();
        HGFCache_.NodeCount_ = UNSIGNED_ZERO;
        HGFCache_.ObservationCount_ = UNSIGNED_ZERO;
        DefaultRegionTable_ = profile.DefaultSchemaTable;
        HasDefaultRegionTable_ = true;

        if (
            !InitializeFabric(
                slot_count,
                profile.RequiredAPCCells,
                profile.FabricConfig,
                profile.MaxDirectParentPerAxis
            )
        )
        {
            return false;
        }

        Profile_ = profile;

        for (const SD::RegionSchemaRecord& record : MetrixViewRow_(0u))
        {
            switch (record.Region)
            {
            case MacroColumnOfAPC::STATE_SLOT: HGFCache_.StateCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::ERROR_SLOT: HGFCache_.ErrorCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::WEIGHT_SLOT: HGFCache_.WeightCellOffset_ = record.CellOffset; break;
            default: 
                break;
            }
        }
        return true;
    }



    bool GHGFModelConstructor::ConstructGHGFModel(
        GHGFModelConstructionValues& model_values,
        const GHGFLayerModel::GHGFStorageProfile& profile
    ) noexcept
    {
        if (
            IsFabricActive() ||
            !GM::IsValidStoregeProfile(profile) ||
            model_values.APCNodes.empty() ||
            model_values.APCNodes.size() != model_values.RoleSpan.size() ||
            model_values.APCNodes.size() >= GM::StorageConst::INVALID_SLOT
        )
        {
            return false;
        }
        for (AdaptivePackedCellContainer& apc : model_values.APCNodes)
        {
            if (apc.IsActiveAPC())
            {
                return false;
            }
        }

        if (!InitializeGHGFFabric(static_cast<uint32_t>(model_values.APCNodes.size()), profile))
        {
            return false;
        }

        const auto AbortConstruction___ = [&]() noexcept -> void
        {
            ShutDownFabric();
            for (GHGFNode& node : model_values.APCNodes)
            {
                node.ReleseFabricBindingOnly_();
                node.GHGFFabric_ = nullptr;
            }
            InvalidateGHGFModel_();
        };
        
        for (uint32_t i = 0; i < FVolatileCache_.CountOfAPC_; i++)
        {
            if (!CreateNodeOfGHGF(model_values.APCNodes[i], model_values.RoleSpan[i]))
            {
                AbortConstruction___();
                return false;
            }
        }
        
        for (const GM::GHGFConnection& connection : model_values.ConnectionSpan)
        {
            if (!ConnectGHGFParent(connection))
            {
                AbortConstruction___();
                return false;
            }
        }
        
        if (!CompileGHGFModel() ||!ResetGHGFState())
        {
            AbortConstruction___();
            return false;
        }
        return true;
    }

    bool GHGFModelConstructor::GetGHGFNode_(uint32_t slot, GHGFNode& node, APCUseScope& use) noexcept
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

    // bool GHGFModelConstructor::ConnectGHGFParent(const GM::GHGFConnection& connection) noexcept
    // {
    //     GHGFNode parent, child;
    //     APCUseScope parent_use, child_use;

    //     if (
    //         !GetGHGFNode_(connection.parent, parent, parent_use) ||
    //         !GetGHGFNode_(connection.Child, parent, parent_use) ||
    //     )
    //     {
    //         /* code */
    //     }
        
    // }


}