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
        return IsFabricActive() && GHGFCache_.ModelPrepared_ &&
            GHGFCache_.PreparedRevision_ == SealedDAGRevision_.load(std::memory_order_acquire);
    }

    float* GHGFModelConstructor::GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept
    {
        return RegionT_<float>(slot, cell_offset);
    }

    float* GHGFModelConstructor::GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.StateCellOffset_) +
            static_cast<size_t>(row) * Profile_.BatchCapacity;
    }

    float* GHGFModelConstructor::GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.ErrorCellOffset_) +
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
        GHGFCache_.ModelPrepared_ = false;
        GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
        GHGFCache_.ActiveBatch_ = UNSIGNED_ZERO;
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
        GHGFCache_.NodeCount_ = UNSIGNED_ZERO;
        GHGFCache_.ObservationCount_ = UNSIGNED_ZERO;
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
            case MacroColumnOfAPC::STATE_SLOT: GHGFCache_.StateCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::ERROR_SLOT: GHGFCache_.ErrorCellOffset_ = record.CellOffset; break;
            case MacroColumnOfAPC::WEIGHT_SLOT: GHGFCache_.WeightCellOffset_ = record.CellOffset; break;
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
        
        for (uint32_t i = 0; i < FabCache_.CountOfAPC_; i++)
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
        
        if (!SealGHGFModel_() ||!ResetGHGFState())
        {
            AbortConstruction___();
            return false;
        }
        return true;
    }

    bool GHGFModelConstructor::GetGHGFNode_(
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

    bool GHGFModelConstructor::ConnectGHGFParent(const GM::GHGFConnection& connection) noexcept
    {
        GHGFNode parent, child;
        APCUseScope parent_use, child_use;

        if (
            !GetGHGFNode_(connection.Parent, parent, parent_use) ||
            !GetGHGFNode_(connection.Child, parent, parent_use) ||
            !CoreOfFabricCoordinator::IsValidEdgeTable(connection.Edge) ||
            !std::isfinite(connection.Coupling) ||
            parent.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION
        )
        {
            return false;
        }
        
        if (
            child.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION &&
            (
                connection.Edge != FabricSegments::VALUE_PARENT_EDGE_TABLE_H ||
                connection.Coupling != GM::StorageConst::ONE
            )
        )
        {
            return false;
        }

        InvalidateGHGFModel_();

        const std::span<EdgeBuilder::ParentRelation> retations = ParentRelations_(connection.Edge, connection.Child);
        for (uint8_t i = 0; i < FabCache_.MaxDirectParentsPerAxis_; i++)
        {
            if (
                !EdgeBuilder::IsEmpty(retations[i]) &&
                EdgeBuilder::ParentSlot(retations[i]) == connection.Parent
            )
            {
                GHGFRegion_(connection.Child, GHGFCache_.WeightCellOffset_)[GM::CouplingIndex(connection.Edge, i, FabCache_.MaxDirectParentsPerAxis_)] = connection.Coupling;
                return true;
            }
        }
        child.RemoveParent(parent, connection.Edge);
        return false;
    }

    bool GHGFModelConstructor::RemoveParent(const GM::GHGFConnection& connection) noexcept
    {
        GHGFNode parent, child;
        APCUseScope parent_use, child_use;
        if (
            !GetGHGFNode_(connection.Parent, parent, parent_use) ||
            !GetGHGFNode_(connection.Child, parent, parent_use) ||
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


    bool GHGFModelConstructor::SealGHGFModel_() noexcept
    {
        if (
            !IsFabricActive() || 
            !GM::IsValidStoregeProfile(Profile_)
        )
        {
            return false;
        }

        const uint64_t revision = SealedDAGRevision_.load(std::memory_order_acquire);
        const bool same_topology = GHGFCache_.ModelPrepared_ && GHGFCache_.PreparedRevision_ == revision;
        GHGFCache_.ModelPrepared_ = false;
        GHGFCache_.NodeCount_ = GHGFCache_.ObservationCount_ = UNSIGNED_ZERO;
        const uint64_t allowed_mask = MaskLowBitsForU64(FabCache_.MaxDirectParentsPerAxis_);

        for (uint32_t i = 0; i < FabCache_.CountOfAPC_; i++)
        {

            GHGFNode node;
            APCUseScope use;

            if (!GetGHGFNode_(i, node, use))
            {
                return false;
            }
            
            const HandleOfAPCStatic::ControlValues control = HandleOfAPCStatic::ReadControlCell(
                std::atomic_ref<uint64_t>(*node.APCCache_.GenerationCellPtr_).load(std::memory_order_acquire)
            );
            if (control.ActiveAccess != UNSIGNED_ZERO)
            {
                return false;
            }
            
            if (control.Closed)
            {
                continue;
            }


            const std::span<SD::RegionSchemaRecord> schemas = MetrixViewRow_(i);
            if (schemas.size() != std::popcount(Profile_.ActiveRegionMask))
            {
                return false;
            }
            for (const SD::RegionSchemaRecord& record : schemas)
            {
                const size_t region = static_cast<size_t>(record.Region);
                if (region >= Profile_.DefaultSchemaTable.size())
                {
                    return false;
                }

                const SD::RegionSchemaRecord& expected = Profile_.DefaultSchemaTable[region];
                const uint32_t offset = record.Region == MacroColumnOfAPC::STATE_SLOT ?
                    GHGFCache_.StateCellOffset_ : record.Region == MacroColumnOfAPC::ERROR_SLOT ?
                    GHGFCache_.ErrorCellOffset_ : GHGFCache_.WeightCellOffset_;
                if (record.Dtype != expected.Dtype || record.Protocol != expected.Protocol ||
                    record.MatrixHeight != expected.MatrixHeight || record.MatrixWidth != expected.MatrixWidth ||
                    record.Flags != expected.Flags || record.CellOffset != offset)
                {
                    return false;
                }                
            }
            ++GHGFCache_.NodeCount_;
            const bool observation = node.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION;
            if (observation)
            {
                ++GHGFCache_.ObservationCount_;
            }
            
            const float* weight = GHGFRegion_(i, GHGFCache_.WeightCellOffset_);
            for (uint32_t param = 0; param < Profile_.ParameterCount; param++)
            {
                if (!std::isfinite(weight[param]))
                {
                    return false;
                }
            }

            bool has_child = false;

            for (uint8_t e = 0; e < EDGE_COUNT; e++)
            {
                FabricSegments edge_table = e == UNSIGNED_ZERO ? 
                    FabricSegments::VALUE_PARENT_EDGE_TABLE_H : FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;
                EdgeBuilder::EdgeData header{};

                if (
                    !ReadEdgeHeader_(edge_table, i, header) ||
                    header.Status != EdgeBuilder::EdgeStatus::LIVE
                )
                {
                    return false;
                }

                has_child = has_child || header.TailLocator != EdgeBuilder::RELATION_NULL;
                const uint64_t mask = GHGFParentMask_(i, edge_table);
                const std::span<EdgeBuilder::ParentRelation> relations = ParentRelations_(edge_table, i);
                auto ValidMask___ = [&]() noexcept -> bool {return (mask & ~allowed_mask) == UNSIGNED_ZERO;};
                if (
                    !ValidMask___() ||
                    relations.size() != FabCache_.MaxDirectParentsPerAxis_
                )
                {
                    return false;
                }

                for (uint8_t ordinal = 0; ordinal < FabCache_.MaxDirectParentsPerAxis_; ordinal++)
                {
                    const EdgeBuilder::ParentRelation& relation = relations[ordinal];
                    const bool occupied = (mask & EdgeBuilder::DirtyBit(ordinal)) != UNSIGNED_ZERO;
                    if (
                        EdgeBuilder::IsPartiallyEmpty(relation) ||
                        occupied == EdgeBuilder::IsEmpty(relation)
                    )
                    {
                        return false;
                    }
                    if (!occupied)
                    {
                        continue;
                    }
                    
                    const uint32_t parent = EdgeBuilder::ParentSlot(relation);
                    GHGFNode parent_node;
                    APCUseScope parent_use;
                    auto ValidParent___ = [&]() noexcept -> bool {return parent <= i;};
                    if (
                        !ValidParent___() ||
                        !GetGHGFNode_(parent, parent_node, parent_use) ||
                        parent_node.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION
                    )
                    {
                        return false;
                    }

                    if (!parent_node.IsOpenGeneration_())
                    {
                        return false;
                    }                    
                    if (
                        observation && 
                        (edge_table != FabricSegments::VALUE_PARENT_EDGE_TABLE_H ||
                        weight[GM::CouplingIndex(edge_table, ordinal, FabCache_.MaxDirectParentsPerAxis_)] != GM::StorageConst::ONE)
                    )
                    {
                        return false;
                    }
                }
            }
            if (observation)
            {
                if (
                    has_child ||
                    GHGFParentMask_(i, FabricSegments::VALUE_PARENT_EDGE_TABLE_H) == UNSIGNED_ZERO
                )
                {
                    return false;
                }
                else if (!has_child)
                {
                    return false;
                }
            }
        }
        
        if (
            GHGFCache_.NodeCount_ == UNSIGNED_ZERO || GHGFCache_.ObservationCount_ == UNSIGNED_ZERO ||
            revision != SealedDAGRevision_.load(std::memory_order_acquire)
        )
        {
            return false;
        }
        
        GHGFCache_.PreparedRevision_ = revision;
        GHGFCache_.ModelPrepared_ = true;
        if (!same_topology)
        {
            GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
            GHGFCache_.ActiveBatch_ = UNSIGNED_ZERO;
        }
        return true;
    }

    bool GHGFModelConstructor::ResetGHGFState() noexcept
    {
        if (!IsGHGFPlanCurrent_())
        {
            return false;
        }
        
        for (uint32_t slot = 0; slot < FabCache_.CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(slot, node, use))
            {
                node.ResetAPCGHGFStateRegion_();
            }
        }
        GHGFCache_.ActiveBatch_ = UNSIGNED_ZERO;
        GHGFCache_.Phase_ = GM::GHGFPhase::READY;
        return true;
    }

    float GHGFModelConstructor::GetGHGFParameter_(uint32_t slot, uint32_t index) noexcept
    {
        return GHGFRegion_(slot, GHGFCache_.WeightCellOffset_)[index];
    }


    bool GHGFModelConstructor::PredictBatchNONVectorized_(uint32_t batch) noexcept
    {
        for (uint32_t i = 0; i < FabCache_.CountOfAPC_; i++)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(i, node, use) && !node.PredictGHGFNodenNONVectorized_(batch))
            {
                return false;
            }
        }
        return true;
    }

    bool GHGFModelConstructor::UpdateGHGFBatchNONVectorized_(std::span<const float> observations, uint32_t batch) noexcept
    {
        using SR = GM::GHGFStateRow;
        uint32_t observation = UNSIGNED_ZERO;
        for (uint32_t slot = 0; slot < FabCache_.CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (!GetGHGFNode_(slot, node, use))
            {
                continue;
            }
            float* precision = GHGFStateRow_(slot, SR::PRECISION);
            const float* marginal = GHGFStateRow_(slot, SR::EXPECTED_PRECISION);
            float* error = GHGFErrorRow_(slot, GM::GHGFErrorRow::VALUE_PREDICTION_ERROR);
            std::copy_n(marginal, batch, precision);
            if (node.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION)
            {
                float* mean = GHGFStateRow_(slot, SR::MEAN);
                const float* predicted = GHGFStateRow_(slot, SR::EXPECTED_MEAN);
                for (uint32_t lane = 0; lane < batch; ++lane)
                {
                    mean[lane] = observations[static_cast<size_t>(observation) * batch + lane];
                    error[lane] = (mean[lane] - predicted[lane]) / marginal[lane];
                }
                ++observation;
            }
            else
            {
                std::fill_n(error, batch, GM::StorageConst::ZERO); // Sum of children's weighted corrections.
            }
        }
        for (uint32_t reverse = static_cast<uint32_t>(FabCache_.CountOfAPC_); reverse > 0; --reverse)
        {
            const uint32_t slot = reverse - 1u;
            GHGFNode node;
            APCUseScope use;
            if (!GetGHGFNode_(slot, node, use))
            {
                continue;
            }
            if (node.GHGFRole_() != GM::GHGFNodeRole::OBSERVATION && !node.UpdateGHGFNodeNONVectorized_(batch))
            {
                return false;
            }
            if (!node.PropogateGHGFErrorNONVectorized_(slot, batch))
            {
                return false;
            }
        }
        return true;
    }

    bool GHGFModelConstructor::CopyGHGFPredictionNONVectorized_(std::span<float> predictions, uint32_t batch) noexcept
    {
        uint32_t observation = UNSIGNED_ZERO;
        for (uint32_t slot = 0; slot < FabCache_.CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(slot, node, use) && node.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION)
            {
                std::copy_n(GHGFStateRow_(slot, GM::GHGFStateRow::EXPECTED_MEAN), batch,
                    predictions.data() + static_cast<size_t>(observation++) * batch);
            }
        }
        return observation == GHGFCache_.ObservationCount_;
    }

    bool GHGFModelConstructor::PredictGHGFNONVectorized(uint32_t batch, std::span<float> predictions) noexcept
    {
        if (!IsGHGFPlanCurrent_() || GHGFCache_.Phase_ != GM::GHGFPhase::READY ||
            batch == UNSIGNED_ZERO || batch > Profile_.BatchCapacity ||
            (GHGFCache_.ActiveBatch_ != UNSIGNED_ZERO && GHGFCache_.ActiveBatch_ != batch) ||
            predictions.size() != static_cast<size_t>(GHGFCache_.ObservationCount_) * batch ||
            IsInternalBuffer(predictions.data(), predictions.size()))
        {
            return false;
        }
        if (!PredictGHGFBatchNONVectorized_(batch))
        {
            GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
            return false;
        }
        GHGFCache_.ActiveBatch_ = batch;
        GHGFCache_.Phase_ = GM::GHGFPhase::PREDICTED;
        if (!CopyGHGFPredictionNONVectorized_(predictions, batch))
        {
            GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
            return false;
        }
        return true;
    }

    bool GHGFModelConstructor::UpdateGHGFNONVectorized(uint32_t batch, FCSpan observations) noexcept
    {
        if (!IsGHGFPlanCurrent_() || GHGFCache_.Phase_ != GM::GHGFPhase::PREDICTED ||
            batch != GHGFCache_.ActiveBatch_ || observations.size() != static_cast<size_t>(GHGFCache_.ObservationCount_) * batch ||
            IsInternalBuffer(observations.data(), observations.size()))
        {
            return false;
        }
        for (const float value : observations)
        {
            if (value != GM::StorageConst::ZERO && value != GM::StorageConst::ONE)
            {
                return false;
            }
        }
        if (!UpdateGHGFBatchNONVectorized_(observations, batch))
        {
            GHGFCache_.Phase_ = GM::GHGFPhase::NEEDS_RESET;
            return false;
        }
        GHGFCache_.Phase_ = GM::GHGFPhase::READY;
        return true;
    }

    bool GHGFModelConstructor::PredictGHGFBatchNONVectorized_(uint32_t batch) noexcept
    {
        for (uint32_t slot = 0; slot < FabCache_.CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(slot, node, use) && !node.PredictGHGFNodenNONVectorized_(batch))
            {
                return false;
            }
        }
        return true;
    }

}