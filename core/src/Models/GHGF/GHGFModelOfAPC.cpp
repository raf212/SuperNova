#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 

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
        GHGFCache_.ModelPrepared_ = false;
        GHGFCache_.NodeCount_ = GHGFCache_.ObservationCount_ = UNSIGNED_ZERO;
        const uint64_t allowed_mask = MaskLowBitsForU64(FabCache_->MaxDirectParentsPerAxis_);

        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
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
            if (control.ActiveAccess != 1u)
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

                uint32_t offset = UNSIGNED_ZERO;

                switch (record.Region)
                {
                case MacroColumnOfAPC::BOTTOM_UP_SLOT:
                    offset = GHGFCache_.FFCellOffset_;
                    break;

                case MacroColumnOfAPC::TOP_DOWN_SLOT:
                    offset = GHGFCache_.FBCellOffset_;
                    break;

                case MacroColumnOfAPC::STATE_SLOT:
                    offset = GHGFCache_.StateCellOffset_;
                    break;

                case MacroColumnOfAPC::ERROR_SLOT:
                    offset = GHGFCache_.ErrorCellOffset_;
                    break;

                case MacroColumnOfAPC::WEIGHT_SLOT:
                    offset = GHGFCache_.WeightCellOffset_;
                    break;

                default:
                    return false;
                }

                if (
                    record.Dtype != expected.Dtype ||
                    record.Protocol != expected.Protocol ||
                    record.MatrixHeight != expected.MatrixHeight ||
                    record.MatrixWidth != expected.MatrixWidth ||
                    record.Flags != expected.Flags ||
                    record.CellOffset != offset
                )
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
                    relations.size() != FabCache_->MaxDirectParentsPerAxis_
                )
                {
                    return false;
                }

                for (uint8_t ordinal = 0; ordinal < FabCache_->MaxDirectParentsPerAxis_; ordinal++)
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
                    auto ValidParent___ = [&]() noexcept -> bool {return parent < i;};
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
                        edge_table != FabricSegments::VALUE_PARENT_EDGE_TABLE_H
                    )
                    {
                        return false;
                    }
                }
            }
            if (
                observation &&
                (
                    has_child ||
                    GHGFParentMask_(i, FabricSegments::VALUE_PARENT_EDGE_TABLE_H) == UNSIGNED_ZERO
                )
            )
            {
                return false;
            }
        }
        
        if (
            GHGFCache_.NodeCount_ == UNSIGNED_ZERO || GHGFCache_.ObservationCount_ == UNSIGNED_ZERO ||
            revision != SealedDAGRevision_.load(std::memory_order_acquire)
        )
        {
            return false;
        }
        
        GHGFCache_.ModelPrepared_ = true;
        return true;
    }

    bool GHGFModelConstructor::UpdateGHGFBatchNONVectorized_(
        std::span<const float> observations,
        uint32_t batch
    ) noexcept
    {
        using SR = GM::GHGFStateRow;
        using ER = GM::GHGFErrorRow;
        using FR = GM::GHGFMessageFForward;

        uint32_t observation = UNSIGNED_ZERO;

        for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;

            if (!GetGHGFNode_(slot, node, use))
                continue;

            float* precision = GHGFStateRow_(slot, SR::PRECISION);
            const float* marginal = GHGFStateRow_(slot, SR::EXPECTED_PRECISION);
            float* error = GHGFErrorRow_(slot, ER::VALUE_PREDICTION_ERROR);

            std::copy_n(marginal, batch, precision);

            if (node.GHGFRole_() != GM::GHGFNodeRole::OBSERVATION)
            {
                std::fill_n(error, batch, GM::StorageConst::ZERO);
                continue;
            }

            float* message = FFRowGHGF_(slot, FR::OBSERVATION);
            float* mean = GHGFStateRow_(slot, SR::MEAN);
            const float* predicted = GHGFStateRow_(slot, SR::EXPECTED_MEAN);

            std::copy_n(
                observations.data() + static_cast<size_t>(observation) * batch,
                batch,
                message
            );

            for (uint32_t lane = 0; lane < batch; ++lane)
            {
                mean[lane] = message[lane];
                error[lane] = (mean[lane] - predicted[lane]) / marginal[lane];
            }

            if (!node.PublishFForwardMessageGHGF_(batch))
                return false;

            ++observation;
        }

        for (uint32_t reverse = static_cast<uint32_t>(FabCache_->CountOfAPC_); reverse > 0; --reverse)
        {
            const uint32_t slot = reverse - 1u;

            GHGFNode node;
            APCUseScope use;

            if (!GetGHGFNode_(slot, node, use))
                continue;

            if (
                node.GHGFRole_() != GM::GHGFNodeRole::OBSERVATION &&
                !node.UpdateGHGFNodeNONVectorized_(batch)
            )
            {
                return false;
            }

            if (!node.PropogateGHGFErrorNONVectorized_(slot, batch))
                return false;
        }

        return observation == GHGFCache_.ObservationCount_;
    }

    bool GHGFModelConstructor::CopyGHGFPredictionNONVectorized_(std::span<float> predictions, uint32_t batch) noexcept
    {
        uint32_t observation = UNSIGNED_ZERO;
        for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; ++slot)
        {
            GHGFNode node;
            APCUseScope use;
            if (GetGHGFNode_(slot, node, use) && node.GHGFRole_() == GM::GHGFNodeRole::OBSERVATION)
            {
                std::copy_n(
                    FBRowGHGF_(
                        slot,
                        GM::GHGFMessageFBackward::EXPECTED_MEAN
                    ),
                    batch,
                    predictions.data() + static_cast<size_t>(observation++) * batch
                );
            }
        }
        return observation == GHGFCache_.ObservationCount_;
    }

    bool GHGFModelConstructor::PredictModelNONVectorized(uint32_t batch, std::span<float> predictions) noexcept
    {
        if (!IsGHGFPlanCurrent_() ||
            batch == UNSIGNED_ZERO || batch > Profile_.BatchCapacity ||
            predictions.size() != static_cast<size_t>(GHGFCache_.ObservationCount_) * batch ||
            IsInternalBuffer(predictions.data(), predictions.size()))
        {
            return false;
        }
        if (!PredictGHGFBatchNONVectorized_(batch))
        {
            return false;
        }
        if (!CopyGHGFPredictionNONVectorized_(predictions, batch))
        {
            return false;
        }
        return true;
    }

    bool GHGFModelConstructor::UpdateModelNONVectorized(uint32_t batch, FCSpan observations) noexcept
    {
        if (!IsGHGFPlanCurrent_() ||
            observations.size() != static_cast<size_t>(GHGFCache_.ObservationCount_) * batch ||
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
            return false;
        }
        return true;
    }

    bool GHGFModelConstructor::PredictGHGFBatchNONVectorized_(uint32_t batch) noexcept
    {
        for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; ++slot)
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


    std::optional<double> GHGFModelConstructor::RunGHGFSequence(
        FCSpan observations,
        uint32_t time_count,
        uint32_t batch_count,
        std::span<float> predictions,
        bool reset_state
    ) noexcept
    {
        if (!IsGHGFPlanCurrent_() || time_count == UNSIGNED_ZERO ||
            batch_count == UNSIGNED_ZERO || batch_count > Profile_.BatchCapacity)
        {
            return std::nullopt;
        }
        const size_t step_size = static_cast<size_t>(GHGFCache_.ObservationCount_) * batch_count;
        if (step_size > SIZE_MAX / time_count)
        {
            return std::nullopt;
        }
        const size_t count = step_size * time_count;
        if (observations.size() != count || (!predictions.empty() && predictions.size() != count) ||
            IsInternalBuffer(observations.data(), observations.size()) ||
            IsInternalBuffer(predictions.data(), predictions.size()))
        {
            return std::nullopt;
        }
        if (!predictions.empty())
        {
            const uintptr_t input = reinterpret_cast<uintptr_t>(observations.data());
            const uintptr_t output = reinterpret_cast<uintptr_t>(predictions.data());
            const size_t bytes = observations.size_bytes();
            if (input <= output ? output - input < bytes : input - output < bytes)
            {
                return std::nullopt;
            }
        }
        for (const float value : observations)
        {
            if (value != GM::StorageConst::ZERO && value != GM::StorageConst::ONE)
            {
                return std::nullopt;
            }
        }
        if (reset_state && !ResetGHGFState())
        {
            return std::nullopt;
        }

        double loss = 0.0;
        for (uint32_t time = 0; time < time_count; ++time)
        {
            if (!PredictGHGFBatchNONVectorized_(batch_count))
            {
                return std::nullopt;
            }
            const size_t step_begin = static_cast<size_t>(time) * step_size;
            uint32_t observation = UNSIGNED_ZERO;
            for (uint32_t slot = 0; slot < FabCache_->CountOfAPC_; ++slot)
            {
                GHGFNode node;
                APCUseScope use;
                if (!GetGHGFNode_(slot, node, use) || node.GHGFRole_() != GM::GHGFNodeRole::OBSERVATION)
                {
                    continue;
                }
                const float* probability = FBRowGHGF_(slot, GM::GHGFMessageFBackward::EXPECTED_MEAN);
                const size_t begin = step_begin + static_cast<size_t>(observation++) * batch_count;
                for (uint32_t lane = 0; lane < batch_count; ++lane)
                {
                    const double predicted = probability[lane];
                    loss -= observations[begin + lane] == GM::StorageConst::ONE ?
                        std::log(predicted) : std::log1p(-predicted);
                    if (!predictions.empty())
                    {
                        predictions[begin + lane] = probability[lane];
                    }
                }
            }
            // Score before the current observation changes any belief.
            if (!UpdateGHGFBatchNONVectorized_(observations.subspan(step_begin, step_size), batch_count))
            {
                return std::nullopt;
            }
        }
        return loss / static_cast<double>(count);
    }


    std::optional<double> GHGFModelConstructor::FitGHGFParameters(
        FCSpan observations, 
        uint32_t time_count, 
        uint32_t batch_count,
        std::span<const GM::GHGFParameterRange> parameters, 
        uint32_t passes
    ) noexcept
    {
        using SC = GM::StorageConst;

        if (!IsGHGFPlanCurrent_() || parameters.empty() || passes == UNSIGNED_ZERO ||
            IsInternalBuffer(parameters.data(), parameters.size()))
        {
            return std::nullopt;
        }
        for (size_t index = 0; index < parameters.size(); ++index)
        {
            const auto& parameter = parameters[index];
            GHGFNode node;
            APCUseScope use;
            std::optional<float> current = GetGHGFParameter_(parameter.Slot, parameter.Index);
            if (
                !current.has_value() ||
                !GetGHGFNode_(parameter.Slot, node, use) ||
                !std::isfinite(parameter.Lower) ||
                !std::isfinite(parameter.Upper) ||
                parameter.Lower >= parameter.Upper ||
                current.value() < parameter.Lower ||
                current.value() > parameter.Upper
            )
            {
                return std::nullopt;
            }
            for (size_t previous = 0; previous < index; ++previous)
            {
                if (parameters[previous].Slot == parameter.Slot && parameters[previous].Index == parameter.Index)
                {
                    return std::nullopt;
                }
            }
        }
        std::optional<double> mean_log_loss = RunGHGFSequence(observations, time_count, batch_count, {}, true);

        if (!mean_log_loss.has_value())
        {
            return std::nullopt;
        }
        float step = SC::INITIAL_SEARCH_STEP;
        for (uint32_t pass = 0; pass < passes && step >= SC::MIN_SEARCH_STEP; ++pass)
        {
            for (const auto& parameter : parameters)
            {
                std::optional<float> center = GetGHGFParameter_(parameter.Slot, parameter.Index);
                if (!center.has_value())
                {
                    return std::nullopt;
                }
                
                float best = center.value();
                double best_loss = mean_log_loss.value();
                for (const float direction : {-SC::ONE, SC::ONE})
                {
                    const float candidate = std::clamp(center.value() + direction * step, parameter.Lower, parameter.Upper);
                    if (candidate == center)
                    {
                        continue;
                    }
                    if (!SetGHGFParameter_(parameter.Slot, parameter.Index, candidate))
                    {
                        return std::nullopt;
                    }
                    
                    std::optional<double> candidate_loss = RunGHGFSequence(observations, time_count, batch_count, {}, true);
                    if (candidate_loss.has_value() &&
                        candidate_loss < best_loss)
                    {
                        best = candidate;
                        best_loss = candidate_loss.value();
                    }
                }
                if (!SetGHGFParameter_(parameter.Slot, parameter.Index, best))
                {
                    return std::nullopt;
                }
                
                mean_log_loss = best_loss;
            }
            step *= SC::HALF;
        }
        // Leave beliefs corresponding to the selected parameters, not the last trial.
        return RunGHGFSequence(observations, time_count, batch_count, {}, true);
    }


    bool GHGFModelConstructor::LearnGHGFBatchNONVectorized_(
        uint32_t batch,
        const GHGFLearningConfig& learning
    ) noexcept
    {
        using EI = GM::GHGFErrorValueIndexing;
        using FF = GM::GHGFMessageFForward;
        using FB = GM::GHGFMessageFBackward;
        using SR = GM::GHGFStateRow;
        using SC = GM::StorageConst;

        const auto ValidRate___ = [](float value) noexcept
        {
            return std::isfinite(value) && value >= 0.0f;
        };

        if (!IsGHGFPlanCurrent_() ||
            batch == UNSIGNED_ZERO ||
            batch > Profile_.BatchCapacity ||
            !ValidRate___(learning.HCouplingLearningRate) ||
            !ValidRate___(learning.DriftLearningRate) ||
            !ValidRate___(learning.VolatilityLearningRate) ||
            !ValidRate___(learning.VCouplingLearningRate) ||
            !ValidRate___(learning.AutoConnectionLearningRate) ||
            !std::isfinite(learning.GradientClip) ||
            learning.GradientClip <= 0.0f ||
            !std::isfinite(learning.MinTonicLogVolatility) ||
            !std::isfinite(learning.MaxTonicLogVolatility) ||
            learning.MinTonicLogVolatility >= learning.MaxTonicLogVolatility)
        {
            return false;
        }

        const float inverse_batch = SC::ONE / static_cast<float>(batch);

        for (uint32_t child = 0; child < FabCache_->CountOfAPC_; ++child)
        {
            GHGFNode node;
            APCUseScope use;

            if (!GetGHGFNode_(child, node, use))
                continue;

            const auto maybe_role = node.GHGFRole_();
            if (!maybe_role.has_value())
                return false;

            const GM::GHGFNodeRole role = maybe_role.value();
            const bool observation = role == GM::GHGFNodeRole::OBSERVATION;

            float* weight = GHGFRegion_(child, GHGFCache_.WeightCellOffset_);

            const float* observed =
                GHGFStateRow_(child, SR::OBSERVED);

            const float* value_signal =
                FFRowGHGF_(child, FF::VALUE_LEARNING_SIGNAL);

            const float* effective_precision =
                FFRowGHGF_(child, FF::EFFECTIVE_PRECISION);

            const float* volatile_error =
                FFRowGHGF_(child, FF::VOLATILE_ERROR);

            const float* previous_mean =
                FBRowGHGF_(child, FB::MEAN);

            const auto ApplyUpdate___ =
                [&](uint32_t index, float learning_rate,
                    float accumulated_direction) noexcept -> bool
            {
                if (learning_rate == 0.0f)
                    return true;

                if (index >= Profile_.ParameterCount ||
                    !std::isfinite(accumulated_direction))
                {
                    return false;
                }

                float direction = accumulated_direction * inverse_batch;
                direction = std::clamp(
                    direction, -learning.GradientClip, learning.GradientClip);

                const float updated =
                    weight[index] + learning_rate * direction;

                if (!std::isfinite(updated))
                    return false;

                weight[index] = updated;
                return true;
            };

            // -----------------------------------------------------
            // 1. TONIC DRIFT / BIAS
            // -----------------------------------------------------

            float drift_direction = 0.0f;

            for (uint32_t lane = 0; lane < batch; ++lane)
                drift_direction += value_signal[lane];

            if (!ApplyUpdate___(
                    static_cast<uint32_t>(EI::TONIC_DRIFT),
                    learning.DriftLearningRate,
                    drift_direction))
            {
                return false;
            }

            // Observation nodes currently have no temporal or
            // volatility dynamics of their own.
            if (!observation)
            {
                // -------------------------------------------------
                // 2. TEMPORAL / AUTO CONNECTION
                // -------------------------------------------------

                float temporal_direction = 0.0f;

                for (uint32_t lane = 0; lane < batch; ++lane)
                    temporal_direction += value_signal[lane] * previous_mean[lane];

                const uint32_t auto_connection =
                    static_cast<uint32_t>(EI::AUTO_CONNECTION);

                if (!ApplyUpdate___(
                        auto_connection,
                        learning.AutoConnectionLearningRate,
                        temporal_direction))
                {
                    return false;
                }

                weight[auto_connection] =
                    std::clamp(weight[auto_connection], 0.0f, 1.0f);

                // -------------------------------------------------
                // 3. TONIC VOLATILITY
                // -------------------------------------------------

                float volatility_direction = 0.0f;

                for (uint32_t lane = 0; lane < batch; ++lane)
                {
                    const float gate =
                        observed[lane] != SC::ZERO ? SC::ONE : SC::ZERO;

                    const float volatility_signal =
                        SC::HALF *
                        effective_precision[lane] *
                        volatile_error[lane];

                    volatility_direction += gate * volatility_signal;
                }

                const uint32_t tonic_volatile =
                    static_cast<uint32_t>(EI::TONIC_VOLATILE);

                if (!ApplyUpdate___(
                        tonic_volatile,
                        learning.VolatilityLearningRate,
                        volatility_direction))
                {
                    return false;
                }

                weight[tonic_volatile] = std::clamp(
                    weight[tonic_volatile],
                    learning.MinTonicLogVolatility,
                    learning.MaxTonicLogVolatility);
            }

            // -----------------------------------------------------
            // 4. H COUPLING LEARNING
            // -----------------------------------------------------

            const auto h_axis =
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H;

            const auto h_relations = ParentRelations_(h_axis, child);

            for (uint64_t mask = GHGFParentMask_(child, h_axis);
                mask;
                mask &= mask - 1u)
            {
                const uint8_t ordinal =
                    static_cast<uint8_t>(std::countr_zero(mask));

                const uint32_t parent =
                    EdgeBuilder::ParentSlot(h_relations[ordinal]);

                const uint32_t parameter_index = GM::CouplingIndex(
                    h_axis,
                    ordinal,
                    Profile_.MaxDirectParentPerAxis);

                const float* parent_feature =
                    FBRowGHGF_(parent, FB::EXPECTED_MEAN);

                float coupling_direction = 0.0f;

                for (uint32_t lane = 0; lane < batch; ++lane)
                    coupling_direction +=
                        value_signal[lane] * parent_feature[lane];

                if (!ApplyUpdate___(
                        parameter_index,
                        learning.HCouplingLearningRate,
                        coupling_direction))
                {
                    return false;
                }
            }

            if (observation)
                continue;

            // -----------------------------------------------------
            // 5. V COUPLING LEARNING
            // -----------------------------------------------------

            const auto v_axis =
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;

            const auto v_relations = ParentRelations_(v_axis, child);

            for (uint64_t mask = GHGFParentMask_(child, v_axis);
                mask;
                mask &= mask - 1u)
            {
                const uint8_t ordinal =
                    static_cast<uint8_t>(std::countr_zero(mask));

                const uint32_t parent =
                    EdgeBuilder::ParentSlot(v_relations[ordinal]);

                const uint32_t parameter_index = GM::CouplingIndex(
                    v_axis,
                    ordinal,
                    Profile_.MaxDirectParentPerAxis);

                const float coupling = weight[parameter_index];

                const float* parent_mean =
                    FBRowGHGF_(parent, FB::MEAN);

                const float* parent_precision =
                    FBRowGHGF_(parent, FB::EXPECTED_PRECISION);

                float coupling_direction = 0.0f;

                for (uint32_t lane = 0; lane < batch; ++lane)
                {
                    if (!std::isfinite(parent_precision[lane]) ||
                        parent_precision[lane] <= SC::ZERO)
                    {
                        return false;
                    }

                    const float gate =
                        observed[lane] != SC::ZERO ? SC::ONE : SC::ZERO;

                    const float volatility_signal =
                        SC::HALF *
                        effective_precision[lane] *
                        volatile_error[lane];

                    const float feature =
                        parent_mean[lane] +
                        coupling / parent_precision[lane];

                    coupling_direction +=
                        gate * volatility_signal * feature;
                }

                if (!ApplyUpdate___(
                        parameter_index,
                        learning.VCouplingLearningRate,
                        coupling_direction))
                {
                    return false;
                }
            }
        }

        return true;
    }


    bool GHGFModelConstructor::TrainModelNONVectorized(
        uint32_t batch,
        FCSpan observations,
        const GHGFLearningConfig& learning
    ) noexcept
    {
        if (
            !IsGHGFPlanCurrent_() ||
            observations.size() !=
                static_cast<size_t>(
                    GHGFCache_.ObservationCount_
                ) * batch ||
            IsInternalBuffer(
                observations.data(),
                observations.size()
            )
        )
        {
            return false;
        }

        for (const float value : observations)
        {
            if (
                value != GM::StorageConst::ZERO &&
                value != GM::StorageConst::ONE
            )
            {
                return false;
            }
        }

        if (!UpdateGHGFBatchNONVectorized_(
            observations,
            batch
        ))
        {
            return false;
        }

        if (!LearnGHGFBatchNONVectorized_(
            batch,
            learning
        ))
        {
            return false;
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

        GHGFCache_.StructuralLearningActive_ = model_values.StructuralLearningActive;

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
        
        for (uint32_t i = 0; i < FabCache_->CountOfAPC_; i++)
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
}