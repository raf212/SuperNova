#pragma once
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 

    void GHGFNode::PublishFBackwardMessageGHGF_(uint32_t batch) noexcept
    {
        const uint32_t slot = APCCache_.APCSlotIdx_;
        std::copy_n(
            GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::MEAN),
            batch,
            GHGFFabric_->FBRowGHGF_(slot, GM::GHGFMessageFBackward::MEAN)
        );

        std::copy_n(
            GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::EXPECTED_MEAN),
            batch,
            GHGFFabric_->FBRowGHGF_(slot, GM::GHGFMessageFBackward::EXPECTED_MEAN)
        );

        std::copy_n(
            GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::EXPECTED_PRECISION),
            batch,
            GHGFFabric_->FBRowGHGF_(slot, GM::GHGFMessageFBackward::EXPECTED_PRECISION)
        );
    }

    bool GHGFNode::PublishFForwardMessageGHGF_(uint32_t batch) noexcept
    {
        const uint32_t slot = APCCache_.APCSlotIdx_;
        const std::optional<GM::GHGFNodeRole> role = GHGFRole_();

        const float* marginal = GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::EFFECTIVE_PRECISION);
        const float* precision = GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::PRECISION);
        const float* conditional = GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::CONDITIONAL_EXPECTED_PRECISION);
        const float* effictive = GHGFFabric_->GHGFStateRow_(slot, GM::GHGFStateRow::EFFECTIVE_PRECISION);
        const float* value_error = GHGFFabric_->GHGFErrorRow_(slot, GM::GHGFErrorRow::VALUE_PREDICTION_ERROR);
        const float* volatile_error = GHGFFabric_->GHGFErrorRow_(slot, GM::GHGFErrorRow::VOLATILE_PREDICTION_ERROR);

        float* value_precision = GHGFFabric_->FFRowGHGF_(slot, GM::GHGFMessageFForward::VALUE_PRECISION);
        float* value_correction = GHGFFabric_->FFRowGHGF_(slot, GM::GHGFMessageFForward::VALUE_CORRECTION);
        float* volatile_precision = GHGFFabric_->FFRowGHGF_(slot, GM::GHGFMessageFForward::VOLATILE_PRECISION);
        float* volatile_correction = GHGFFabric_->FFRowGHGF_(slot, GM::GHGFMessageFForward::VOLATILE_CORRECTION);

        if (role.value() == GM::GHGFNodeRole::OBSERVATION)
        {
            for (uint32_t i = 0; i < batch; i++)
            {
                value_precision[i] = marginal[i];
                value_correction[i] = marginal[i] * value_error[i];
                volatile_precision[i] = GM::StorageConst::ZERO;
                volatile_correction[i] = GM::StorageConst::ZERO;
            }
            return true;
        }
        
        for (uint32_t i = 0; i < batch; i++)
        {
            const float information = precision[i] - marginal[i];
            const float denominator = conditional[i] + information;
            if (
                !std::isfinite(denominator) ||
                denominator <= GM::StorageConst::ZERO
            )
            {
                return false;
            }

            const float factor = conditional[i] * (information / denominator);
            const float gain = conditional[i] * (precision[i] / denominator);

            value_precision[i] = factor;
            value_correction[i] = gain * value_error[i];

            const float effictive_value = effictive[i];
            const float volatine_value = volatile_error[i];

            volatile_precision[i] = (GM::StorageConst::HALF * effictive_value * effictive_value) +
                (effictive_value * effictive_value * volatine_value) -
                (GM::StorageConst::HALF * effictive_value * volatine_value);

            volatile_correction[i] = GM::StorageConst::HALF * effictive_value * volatine_value;
        }
        
        return true;
    }

    bool GHGFNode::InitializeGHGFNode(
        GHGFLayerModel::GHGFNodeRole role
    ) noexcept
    {
        using GMC = GM::StorageConst;

        if (
            !GHGFFabric_ ||
            !GHGFFabric_->IsFabricActive() ||
            APCCache_.FabricOwnerPtr_ != static_cast<APCFinilizer*>(GHGFFabric_) ||
            !GM::IsValidStoregeProfile(GHGFFabric_->Profile_) || !IsActiveAPC()
        )
        {
            return false;
        }

        std::optional<RegionView<float>> state_view = BuildAViewOverRegion<float>(MacroColumnOfAPC::STATE_SLOT);
        std::optional<RegionView<float>> error_view = BuildAViewOverRegion<float>(MacroColumnOfAPC::ERROR_SLOT);
        std::optional<RegionView<float>> weight_view = BuildAViewOverRegion<float>(MacroColumnOfAPC::WEIGHT_SLOT);
        
        if (
            !state_view.has_value() ||
            !error_view.has_value() ||
            !weight_view.has_value()
        )
        {
            return false;
        }
        
        std::optional<std::span<float>> state = state_view.value().RawMutableSpan();
        std::optional<std::span<float>> error = error_view.value().RawMutableSpan();
        std::optional<std::span<float>> weight = weight_view.value().RawMutableSpan();
        

        if (
            !state.has_value() ||
            !error.has_value() ||
            !weight.has_value() ||
            state.value().size() != static_cast<size_t>(GM::STATE_ROW_COUNT_HEIGHT) * GHGFFabric_->Profile_.BatchCapacity ||
            error.value().size() != static_cast<size_t>(GM::ERROR_ROW_COUNT_HEIGHT) * GHGFFabric_->Profile_.BatchCapacity ||
            weight.value().size() != GHGFFabric_->Profile_.ParameterCount
        )
        {
            return false;
        }
        
        std::fill(state.value().begin(), state.value().end(), GMC::INITIAL_STORAGE_VALUE);
        std::fill(error.value().begin(), error.value().end(), GMC::INITIAL_STORAGE_VALUE);
        std::fill(weight.value().begin(), weight.value().end(), GMC::INITIAL_STORAGE_VALUE);

        const auto StateIndex___ = [&](GM::GHGFStateRow row, uint32_t batch) noexcept -> size_t
        {
            return (static_cast<size_t>(row) * GHGFFabric_->Profile_.BatchCapacity) + batch;
        };
        
        for (uint32_t batch = 0; batch < GHGFFabric_->Profile_.BatchCapacity; batch++)
        {
            (state.value())[StateIndex___(GM::GHGFStateRow::MEAN, batch)] = GMC::INITIAL_STORAGE_VALUE;
            if (role == GM::GHGFNodeRole::OBSERVATION)
            {
                (state.value())[StateIndex___(GM::GHGFStateRow::EXPECTED_MEAN, batch)] = GMC::INITIAL_BINARY_PROBABILITY;
            }
            else
            {
                (state.value())[StateIndex___(GM::GHGFStateRow::EXPECTED_MEAN, batch)] = GMC::INITIAL_STORAGE_VALUE;
            }

            (state.value())[StateIndex___(GM::GHGFStateRow::PRECISION, batch)] = GMC::INITIAL_PRECISION;
            (state.value())[StateIndex___(GM::GHGFStateRow::EXPECTED_PRECISION, batch)] = GMC::INITIAL_PRECISION;
            (state.value())[StateIndex___(GM::GHGFStateRow::CONDITIONAL_EXPECTED_PRECISION, batch)] = GMC::INITIAL_PRECISION;
            (state.value())[StateIndex___(GM::GHGFStateRow::OBSERVED, batch)] = GMC::INITIAL_PRECISION;
            (state.value())[StateIndex___(GM::GHGFStateRow::CURRENT_VARIANCE, batch)] = GMC::INITIAL_PRECISION;
            (state.value())[StateIndex___(GM::GHGFStateRow::EFFECTIVE_PRECISION, batch)] = GMC::INITIAL_STORAGE_VALUE;
        }

        (weight.value())[static_cast<size_t>(GM::GHGFErrorValueIndexing::TONIC_VOLATILE)] = role == GM::GHGFNodeRole::OBSERVATION?
            GMC::OBSERVATION_TONIC_LOG_VOLATILITY : GMC::INITIAL_TONIC_LOG_VOLATILITY;
        (weight.value())[static_cast<size_t>(GM::GHGFErrorValueIndexing::TONIC_DRIFT)] = GMC::INITIAL_STORAGE_VALUE;
        (weight.value())[static_cast<size_t>(GM::GHGFErrorValueIndexing::AUTO_CONNECTION)] = role == GM::GHGFNodeRole::OBSERVATION?
            GMC::INITIAL_STORAGE_VALUE : GMC::INITIAL_AUTO_CONNECTION;

        for (uint8_t i = 0; i < GHGFFabric_->Profile_.MaxDirectParentPerAxis; i++)
        {
            const uint32_t value_index = GM::CouplingIndex(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                i,
                GHGFFabric_->Profile_.MaxDirectParentPerAxis
            );

            const uint32_t volatile_index = GM::CouplingIndex(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                i,
                GHGFFabric_->Profile_.MaxDirectParentPerAxis
            );

            if (
                value_index >= weight.value().size() ||
                volatile_index >= weight.value().size()
            )
            {
                return false;
            }

            (weight.value())[value_index] = GMC::INITIAL_VALUE_COUPLING;
            (weight.value())[volatile_index] = GMC::INITIAL_VOLATILITY_COUPLING;
        }
        
        GHGFFabric_->AtomicallyStoreU64Fab(
            GHGFFabric_->SlotBegin_(APCCache_.APCSlotIdx_) + static_cast<size_t>(ADS::HeaderIdentifierOfAPC::GHGF_ROLE_CELL),
            static_cast<uint64_t>(role)
        );

        GHGFFabric_->InvalidateGHGFModel_();
        return true;
    }


    void GHGFNode::ResetAPCGHGFStateRegion_() noexcept
    {
        std::fill_n(
            GHGFFabric_->GHGFRegion_(APCCache_.APCSlotIdx_, GHGFFabric_->GHGFCache_.StateCellOffset_),
            static_cast<size_t>(GM::STATE_ROW_COUNT_HEIGHT) * GHGFFabric_->Profile_.BatchCapacity, GM::StorageConst::ZERO
        );
        std::fill_n(
            GHGFFabric_->GHGFRegion_(APCCache_.APCSlotIdx_, GHGFFabric_->GHGFCache_.ErrorCellOffset_),
            static_cast<size_t>(GM::ERROR_ROW_COUNT_HEIGHT) * GHGFFabric_->Profile_.BatchCapacity, GM::StorageConst::ZERO
        );
        const float initial_mean = GHGFRole_() == GM::GHGFNodeRole::OBSERVATION ?
            GM::StorageConst::INITIAL_BINARY_PROBABILITY : GM::StorageConst::ZERO;

        for (uint32_t lane = 0; lane < GHGFFabric_->Profile_.BatchCapacity; ++lane)
        {
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::EXPECTED_MEAN)[lane] = initial_mean;
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::PRECISION)[lane] = GM::StorageConst::INITIAL_PRECISION;
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::EXPECTED_PRECISION)[lane] = GM::StorageConst::INITIAL_PRECISION;
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::CONDITIONAL_EXPECTED_PRECISION)[lane] = GM::StorageConst::INITIAL_PRECISION;
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::OBSERVED)[lane] = GM::StorageConst::ONE;
            GHGFFabric_->GHGFStateRow_(APCCache_.APCSlotIdx_, GM::GHGFStateRow::CURRENT_VARIANCE)[lane] = GM::StorageConst::ONE;
        }
    }

    std::optional<GHGFLayerModel::GHGFNodeRole> GHGFNode::GHGFRole_() noexcept
    {
        uint64_t value{};
        if (
            !ReadAPCMetaUnit(ADS::HeaderIdentifierOfAPC::GHGF_ROLE_CELL, value)||
            value < static_cast<uint8_t>(GM::GHGFNodeRole::OBSERVATION) ||
            value > static_cast<uint8_t>(GM::GHGFNodeRole::VOLATILE)
        )
        {
            return std::nullopt;
        }
        return static_cast<GM::GHGFNodeRole>(value);
    }

    bool GHGFNode::PredictGHGFNodenNONVectorized_(uint32_t batch) noexcept
    {
        using SR = GM::GHGFStateRow;
        using ER = GM::GHGFErrorRow;
        using FR = GM::GHGFMessageFBackward;
        using EI = GM::GHGFErrorValueIndexing;
        using SC = GM::StorageConst;

        const uint32_t slot = APCCache_.APCSlotIdx_;
        const auto value_axis = FabricSegments::VALUE_PARENT_EDGE_TABLE_H;
        const auto volatile_axis = FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;

        float* predicted = GHGFFabric_->GHGFStateRow_(slot, SR::EXPECTED_MEAN);
        float* marginal = GHGFFabric_->GHGFStateRow_(slot, SR::EXPECTED_PRECISION);

        const auto value_relations = GHGFFabric_->ParentRelations_(value_axis, slot);

        // Observation node: Bernoulli prediction from value parents.
        if (GHGFRole_() == GM::GHGFNodeRole::OBSERVATION)
        {
            std::fill_n(predicted, batch, SC::ZERO);

            for (
                uint64_t mask = GHGFFabric_->GHGFParentMask_(slot, value_axis);
                mask;
                mask &= mask - 1u
            )
            {
                const uint8_t ordinal = static_cast<uint8_t>(std::countr_zero(mask));
                const uint32_t parent = EdgeBuilder::ParentSlot(value_relations[ordinal]);
                const float* parent_mean = GHGFFabric_->FBRowGHGF_(parent, FR::EXPECTED_MEAN);

                for (uint32_t lane = 0; lane < batch; ++lane)
                    predicted[lane] += parent_mean[lane];
            }

            for (uint32_t lane = 0; lane < batch; ++lane)
            {
                if (!std::isfinite(predicted[lane]))
                    return false;

                const float exponent = std::exp(-std::abs(predicted[lane]));
                const float probability =
                    predicted[lane] >= SC::ZERO
                        ? SC::ONE / (SC::ONE + exponent)
                        : exponent / (SC::ONE + exponent);

                predicted[lane] = std::clamp(probability, SC::BINARY_CLIP, SC::ONE - SC::BINARY_CLIP);

                marginal[lane] = predicted[lane] * (SC::ONE - predicted[lane]);
            }

            PublishFBackwardMessageGHGF_(batch);
            return true;
        }

        const float* mean = GHGFFabric_->GHGFStateRow_(slot, SR::MEAN);
        const float* precision = GHGFFabric_->GHGFStateRow_(slot, SR::PRECISION);
        const float* weight =
            GHGFFabric_->GHGFRegion_(slot, GHGFFabric_->GHGFCache_.WeightCellOffset_);

        float* conditional =
            GHGFFabric_->GHGFStateRow_(slot, SR::CONDITIONAL_EXPECTED_PRECISION);
        float* current_variance =
            GHGFFabric_->GHGFStateRow_(slot, SR::CURRENT_VARIANCE);
        float* effective =
            GHGFFabric_->GHGFStateRow_(slot, SR::EFFECTIVE_PRECISION);
        float* value_variance =
            GHGFFabric_->GHGFErrorRow_(slot, ER::VALUE_PREDICTION_ERROR);
        float* log_volatility =
            GHGFFabric_->GHGFErrorRow_(slot, ER::VOLATILE_PREDICTION_ERROR);

        const float auto_connection = weight[static_cast<size_t>(EI::AUTO_CONNECTION)];
        const float tonic_drift = weight[static_cast<size_t>(EI::TONIC_DRIFT)];
        const float tonic_volatility = weight[static_cast<size_t>(EI::TONIC_VOLATILE)];

        for (uint32_t lane = 0; lane < batch; ++lane)
        {
            predicted[lane] = auto_connection * mean[lane] + tonic_drift;
            current_variance[lane] = SC::ONE / precision[lane];
            value_variance[lane] = SC::ZERO;
            log_volatility[lane] = tonic_volatility;
        }

        // H/value-parent contribution.
        for (
            uint64_t mask = GHGFFabric_->GHGFParentMask_(slot, value_axis);
            mask;
            mask &= mask - 1u
        )
        {
            const uint8_t ordinal = static_cast<uint8_t>(std::countr_zero(mask));
            const uint32_t parent = EdgeBuilder::ParentSlot(value_relations[ordinal]);
            const float coupling = weight[
                GM::CouplingIndex(
                    value_axis,
                    ordinal,
                    GHGFFabric_->Profile_.MaxDirectParentPerAxis)];

            const float* parent_mean =
                GHGFFabric_->FBRowGHGF_(parent, FR::EXPECTED_MEAN);
            const float* parent_precision =
                GHGFFabric_->FBRowGHGF_(parent, FR::EXPECTED_PRECISION);

            const float coupling_squared = coupling * coupling;

            for (uint32_t lane = 0; lane < batch; ++lane)
            {
                predicted[lane] += coupling * parent_mean[lane];
                value_variance[lane] += coupling_squared / parent_precision[lane];
            }
        }

        // V/volatility-parent contribution.
        const auto volatile_relations =
            GHGFFabric_->ParentRelations_(volatile_axis, slot);

        for (uint64_t mask = GHGFFabric_->GHGFParentMask_(slot, volatile_axis);
            mask;
            mask &= mask - 1u)
        {
            const uint8_t ordinal = static_cast<uint8_t>(std::countr_zero(mask));
            const uint32_t parent = EdgeBuilder::ParentSlot(volatile_relations[ordinal]);
            const float coupling = weight[
                GM::CouplingIndex(
                    volatile_axis,
                    ordinal,
                    GHGFFabric_->Profile_.MaxDirectParentPerAxis)];

            const float* parent_mean =
                GHGFFabric_->FBRowGHGF_(parent, FR::MEAN);
            const float* parent_precision =
                GHGFFabric_->FBRowGHGF_(parent, FR::EXPECTED_PRECISION);

            const float half_coupling_squared = SC::HALF * coupling * coupling;

            for (uint32_t lane = 0; lane < batch; ++lane)
            {
                log_volatility[lane] +=
                    coupling * parent_mean[lane] +
                    half_coupling_squared / parent_precision[lane];
            }
        }

        // Final predicted precision terms.
        for (uint32_t lane = 0; lane < batch; ++lane)
        {
            const float volatility = std::exp(log_volatility[lane]);
            const float variance = current_variance[lane] + volatility;

            conditional[lane] = SC::ONE / variance;
            marginal[lane] = SC::ONE / (variance + value_variance[lane]);
            effective[lane] = volatility * marginal[lane];

            if (
                !std::isfinite(predicted[lane]) ||
                !std::isfinite(volatility) ||
                volatility <= SC::ZERO ||
                !std::isfinite(conditional[lane]) ||
                conditional[lane] <= SC::ZERO ||
                !std::isfinite(marginal[lane]) ||
                marginal[lane] <= SC::ZERO ||
                !std::isfinite(effective[lane]))
            {
                return false;
            }
        }

        PublishFBackwardMessageGHGF_(batch);
        return true;
    }


    bool GHGFNode::UpdateGHGFNodeNONVectorized_(uint32_t batch) noexcept
    {
        using SR = GM::GHGFStateRow;
        using SC = GM::StorageConst;

        const uint32_t slot = APCCache_.APCSlotIdx_;
        float* mean = GHGFFabric_->GHGFStateRow_(slot, SR::MEAN);
        float* precision = GHGFFabric_->GHGFStateRow_(slot, SR::PRECISION);
        const float* predicted = GHGFFabric_->GHGFStateRow_(slot, SR::EXPECTED_MEAN);
        const float* marginal = GHGFFabric_->GHGFStateRow_(slot, SR::EXPECTED_PRECISION);
        float* value_error = GHGFFabric_->GHGFErrorRow_(slot, GM::GHGFErrorRow::VALUE_PREDICTION_ERROR);
        float* volatile_error = GHGFFabric_->GHGFErrorRow_(slot, GM::GHGFErrorRow::VOLATILE_PREDICTION_ERROR);
        const unsigned volatile_parents = std::popcount(
            GHGFFabric_->GHGFParentMask_(slot, FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V));
        const float divisor = static_cast<float>(std::max(1u, volatile_parents));
        for (uint32_t lane = 0; lane < batch; ++lane)
        {
            if (!std::isfinite(precision[lane]) || !std::isfinite(value_error[lane]))
            {
                return false;
            }
            precision[lane] = std::clamp(precision[lane], SC::MIN_PRECISION, SC::MAX_PRECISION);
            mean[lane] = predicted[lane] + value_error[lane] / precision[lane];
            value_error[lane] = mean[lane] - predicted[lane];
            volatile_error[lane] = (marginal[lane] / precision[lane] +
                marginal[lane] * value_error[lane] * value_error[lane] - SC::ONE) / divisor;
            if (!std::isfinite(mean[lane]) || !std::isfinite(volatile_error[lane]))
            {
                return false;
            }
        }
        return true;
    }
    
    bool GHGFNode::PropogateGHGFErrorNONVectorized_(uint32_t child, uint32_t batch) noexcept
    {
        using SR = GM::GHGFStateRow;
        using SC = GM::StorageConst;

        if (child != APCCache_.APCSlotIdx_) { return false; }
        const bool gaussian = GHGFRole_() != GM::GHGFNodeRole::OBSERVATION;
        const float* weight = GHGFFabric_->GHGFRegion_(child, GHGFFabric_->GHGFCache_.WeightCellOffset_);
        const float* child_marginal = GHGFFabric_->GHGFStateRow_(child, SR::EXPECTED_PRECISION);
        const float* child_precision = GHGFFabric_->GHGFStateRow_(child, SR::PRECISION);
        const float* child_conditional = GHGFFabric_->GHGFStateRow_(child, SR::CONDITIONAL_EXPECTED_PRECISION);
        const float* child_effective = GHGFFabric_->GHGFStateRow_(child, SR::EFFECTIVE_PRECISION);
        const float* child_value_error = GHGFFabric_->GHGFErrorRow_(child, GM::GHGFErrorRow::VALUE_PREDICTION_ERROR);
        const float* child_volatile_error = GHGFFabric_->GHGFErrorRow_(child, GM::GHGFErrorRow::VOLATILE_PREDICTION_ERROR);
        for (const auto axis : {FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                               FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V})
        {
            const auto relations = GHGFFabric_->ParentRelations_(axis, child);
            for (uint64_t mask = GHGFFabric_->GHGFParentMask_(child, axis); mask; mask &= mask - 1u)
            {
                const uint8_t ordinal = static_cast<uint8_t>(std::countr_zero(mask));
                const uint32_t parent = EdgeBuilder::ParentSlot(relations[ordinal]);
                const float coupling = weight[GM::CouplingIndex(axis, ordinal, GHGFFabric_->Profile_.MaxDirectParentPerAxis)];
                float* parent_precision = GHGFFabric_->GHGFStateRow_(parent, SR::PRECISION);
                float* parent_correction = GHGFFabric_->GHGFErrorRow_(parent, GM::GHGFErrorRow::VALUE_PREDICTION_ERROR);
                if (axis == FabricSegments::VALUE_PARENT_EDGE_TABLE_H)
                {
                    for (uint32_t lane = 0; lane < batch; ++lane)
                    {
                        float factor = child_marginal[lane];
                        float gain = factor;
                        if (gaussian)
                        {
                            const float information = child_precision[lane] - child_marginal[lane];
                            const float denominator = child_conditional[lane] + information;
                            if (!std::isfinite(denominator) || denominator <= SC::ZERO)
                            {
                                return false;
                            }
                            factor = child_conditional[lane] * information / denominator;
                            gain = child_conditional[lane] * child_precision[lane] / denominator;
                        }
                        parent_precision[lane] += coupling * coupling * factor;
                        parent_correction[lane] += coupling * gain * child_value_error[lane];
                    }
                }
                else
                {
                    for (uint32_t lane = 0; lane < batch; ++lane)
                    {
                        const float weighted_effective = coupling * child_effective[lane];
                        parent_precision[lane] += SC::HALF * weighted_effective * weighted_effective +
                            weighted_effective * weighted_effective * child_volatile_error[lane] -
                            SC::HALF * coupling * coupling * child_effective[lane] * child_volatile_error[lane];
                        parent_correction[lane] += SC::HALF * weighted_effective * child_volatile_error[lane];
                    }
                }
            }
        }
        return true;
    }
}