#pragma once
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include <span>

namespace BidirectionalInMemGraph
{ 
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

}