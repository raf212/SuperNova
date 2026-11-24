#pragma once

namespace BidirectionalInMemGraph 
{
    template<class DT>
    SNTOComputerBridge<DT> APCFinilizer::PinNumericRegion(
        uint32_t slot,
        MacroColumnOfAPC column,
        uint32_t ordinal,
        std::optional<uint32_t> generation
    ) noexcept
    {
        ReadLease admission = ExecutionGate().TryTransientRead(true);
        AdaptivePackedCellContainer apc{};
        if (
            !admission ||
            !BindExistingAPCSnapshot_(slot, apc, generation)
        ) 
        {
            return {};
        }

        SNTOComputerBridge<DT> region = apc.template BuildAViewOverRegion<DT>(column, ordinal);

        if (
            region.GetProtocol() != SD::SchemaProtocols::PRIVATE_REGION
        ) 
        {
            return {};
        }
        return region;
    }
}