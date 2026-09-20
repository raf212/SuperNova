#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 

    bool GHGFModel::IsGHGFModelReady_() noexcept
    {
        return IsFabricActive() && GHGFCache_.ModelPrepared_;
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
        if (IsGHGFModelReady_() && GHGFCache_.StructuralLearningActive_)
            return false;
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
        if (IsGHGFModelReady_() && GHGFCache_.StructuralLearningActive_)
            return false;
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
        return true;
    }

    bool GHGFModel::InitializeGHGFFabric(
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
        GHGFCache_.NodeCount_ = UNSIGNED_ZERO;
        GHGFCache_.ObservationCount_ = UNSIGNED_ZERO;
        DefaultRegionTable_ = profile.DefaultSchemaTable;
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
            case MacroColumnOfAPC::BOTTOM_UP_SLOT: GHGFCache_.FFCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::TOP_DOWN_SLOT: GHGFCache_.FBCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::STATE_SLOT: GHGFCache_.StateCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::ERROR_SLOT: GHGFCache_.ErrorCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::WEIGHT_SLOT: GHGFCache_.WeightCellOffset_ = record.CellOffset; break;
            default: 
                break;
            }
        }
        return true;
    }

    bool GHGFModel::ResetGHGFState() noexcept
    {
        if (!IsGHGFModelReady_())
        {
            return false;
        }
        
        for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(slot, node, use))
            {
                node.ResetAPCGHGFStateRegion_();
            }
        }
        return true;
    }

    bool GHGFModel::CreateNodeOfGHGF(
        GHGFNode& desired_apc,
        GM::GHGFNodeRole role
    ) noexcept
    {
        if (
            !IsFabricActive() ||
            !FabCache_->HasDefaultRegionTable_ ||
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

    bool GHGFModel::ReadGHGFNodeIdentity_(
        uint32_t slot,
        GM::GHGFNodeRole& role,
        uint32_t& generation
    ) noexcept
    {
        GHGFNode node{};
        APCUseScope use{};
        if (!GetGHGFNode_(slot, node, use))
            return false;

        const std::optional<GM::GHGFNodeRole> maybe_role = node.GHGFRole_();
        if (!maybe_role.has_value())
            return false;

        role = maybe_role.value();
        generation = node.APCCache_.CurrentGeneration_;
        return HandleOfAPCStatic::IsGenerationValid(generation);
    }

    FabricToAPCLinker::SeqLockedOperation GHGFModel::ReadGHGFParentExecutionSnapshot_(
        uint32_t child,
        FabricSegments edge,
        GM::GHGFParentExecutionSnapshot& snapshot,
        uint32_t max_tries
    ) noexcept
    {
        using Operation = FabricToAPCLinker::SeqLockedOperation;
        using Domain = EdgeBuilder::EdgeDomain;

        snapshot = {};
        if (
            !IsGHGFModelReady_() ||
            child >= FabCache_->CountOfAPC_ ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(edge) ||
            max_tries == UNSIGNED_ZERO
        )
        {
            return Operation::NONE;
        }

        const std::span<EdgeBuilder::ParentRelation> relations = ParentRelations_(edge, child);
        CompiledDAGRecord* const compiled = CompiledDAGRow_(child);
        float* const weights = GHGFWeight_(child);

        if (
            !compiled || !weights ||
            relations.size() != FabCache_->MaxDirectParentsPerAxis_
        )
        {
            return Operation::NONE;
        }

        uint64_t& stored_mask =
            edge == FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                ? compiled->ValueParentMask
                : compiled->VolatileParentMask;

        // Construction-time static model: preserve the direct compiled-table path.
        if (!GHGFCache_.StructuralLearningActive_)
        {
            snapshot.ParentMask = stored_mask;
            for (uint8_t ordinal = 0;
                ordinal < FabCache_->MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                if ((snapshot.ParentMask & EdgeBuilder::DirtyBit(ordinal)) == 0u)
                    continue;

                snapshot.ParentHandles[ordinal] = relations[ordinal].ParentHandle;
                snapshot.Couplings[ordinal] = weights[GM::CouplingIndex(edge, ordinal, Profile_.MaxDirectParentPerAxis)];
            }
            return Operation::FOUND;
        }

        const size_t control_index = EdgeControlCellIndex_(
            edge,
            child,
            Domain::PARENT_RELATIONS
        );
        if (control_index == SIZE_MAX)
            return Operation::NONE;

        std::atomic_ref<const uint64_t> control(SlabBasePtr_[control_index]);
        const uint64_t allowed_mask = MaskLowBitsForU64(FabCache_->MaxDirectParentsPerAxis_);

        for (uint32_t attempt = 0; attempt < max_tries; ++attempt)
        {
            const uint64_t before_raw = control.load(std::memory_order_acquire);
            const EdgeBuilder::EdgeData before =
                EdgeBuilder::UnpackEdgeHeader(before_raw);

            if (!before.IsValid)
                return Operation::NONE;
            if (before.Status == EdgeBuilder::EdgeStatus::RESERVED)
                continue;
            if (before.Status != EdgeBuilder::EdgeStatus::LIVE)
                return Operation::NONE;

            GM::GHGFParentExecutionSnapshot local{};
            local.RowSequence = before.SeqLock;
            local.ParentMask = std::atomic_ref<const uint64_t>(stored_mask).load(
                std::memory_order_relaxed);

            if ((local.ParentMask & ~allowed_mask) != 0u)
                return Operation::NONE;

            bool valid = true;
            for (uint8_t ordinal = 0;
                ordinal < FabCache_->MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                if ((local.ParentMask & EdgeBuilder::DirtyBit(ordinal)) == 0u)
                    continue;

                local.ParentHandles[ordinal] =
                    std::atomic_ref<const uint64_t>(
                        relations[ordinal].ParentHandle).load(
                            std::memory_order_relaxed);

                const uint32_t parameter = GM::CouplingIndex(edge, ordinal, Profile_.MaxDirectParentPerAxis);
                local.Couplings[ordinal] = std::atomic_ref<const float>(weights[parameter]).load(std::memory_order_relaxed);

                const uint32_t parent = TwinU32ToU64::ExtractLow32Of64(local.ParentHandles[ordinal]);
                valid = valid &&
                    local.ParentHandles[ordinal] != FABRIC_CELL_SENTINAL &&
                    parent < child &&
                    std::isfinite(local.Couplings[ordinal]);
            }

            const uint64_t after_raw = control.load(std::memory_order_acquire);
            if (before_raw != after_raw)
                continue;

            if (!valid)
                return Operation::NONE;

            snapshot = local;
            return Operation::FOUND;
        }
        return Operation::RETRY;
    }

}