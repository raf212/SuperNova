#pragma once

namespace BidirectionalInMemGraph 
{
    template<class DT> 
    SNTOComputerBridge<DT>  AdaptivePackedCellContainer::BuildAViewOverRegion(MacroColumnOfAPC column, uint32_t record_ordinal) noexcept
    {
        static_assert(std::is_trivially_copyable_v<DT>);
        using Cache = ASG::CacheOfSNCB<DT>;
        const std::optional<SD::DataTypeOfMacroColumn> expected_dtype = SD::CppTypeToRegionDType<DT>();

        APCUseScope use = AcquireAPCUse_();
        if (!use || !expected_dtype.has_value()) return {};

        ResolveRegionBiteView resolved{};

        if (!ResolveBridgeRegion(column, record_ordinal, resolved)) 
        {
            return {};
        }

        switch (resolved.Schema->Protocol) 
        {
            case SD::SchemaProtocols::PRIVATE_REGION:
            case SD::SchemaProtocols::IMMUTABLE_SNAPSHOT:
            {
                if (!ASG::CanInstallTypedSpan<DT>(resolved, expected_dtype.value())) 
                {
                    return {};
                }
                break;
            }
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
            {
                if (!ASG::CanInstallAtomicSpan<DT>(resolved, expected_dtype.value())) 
                {
                    return {};
                }
                break;
            }
            default:
                return {};
        }

        Cache cache{};
        cache.Data_ = reinterpret_cast<DT*>(resolved.Bytes.data());
        cache.Height_ = resolved.Schema->MatrixHeight;
        cache.ActiveWidth = resolved.Schema->MatrixWidth;
        cache.CapacityWidth_ = resolved.Schema->MatrixWidth;
        cache.Slot_ = APCCache_.APCSlotIdx_;
        cache.Generation_ = APCCache_.CurrentGeneration_;
        cache.RecordOrdinal_ = record_ordinal;
        cache.Column_ = column;
        cache.Protocol_ = resolved.Schema->Protocol;
        cache.Use_ = std::move(use);

        return SNTOComputerBridge<DT>(std::move(cache));
    }

    template<class DT>
    bool AdaptivePackedCellContainer::ZeroARegion(MacroColumnOfAPC column) noexcept
    {
        SNTOComputerBridge<DT> bridge = BuildAViewOverRegion<DT>(column);
        if (!bridge.IsValid()) 
        {
            return false;
        }

        switch (bridge.GetProtocol()) 
        {
            case SD::SchemaProtocols::PRIVATE_REGION:
            {
                std::optional<std::span<DT>> maybe_span = bridge.RawSpan();
                if (!maybe_span.has_value()) return false;
                for (
                    DT& value :
                    maybe_span.value()
                ) 
                {
                    value = DT{};
                }
                return true;
            }
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
            {
                for (uint64_t i = 0u; i < bridge.Size(); i++) 
                {
                    if (!bridge.AtomicStore(i, DT{}, std::memory_order_relaxed)) 
                    {
                        return false;
                    }
                }
                return true;
            }

            default:
                return false;
        }
    }
}