#pragma once
#include <functional>
#include <utility>
#include "APCOrchestrators/ViewOrchestrator.hpp"

namespace BidirectionalInMemGraph
{

    class AdaptivePackedCellContainer;

    class FabricToAPCLinker 
    {
        friend class VagueTemoraryPremativeFabric;
    public:
        enum class SeqLockedOperation : uint8_t
        {
            FOUND = 0,
            NONE = 1,
            RETRY = 2
        };

        uint32_t GetThisSlotIdx() noexcept
        {
            return
                IsActiveAPC() ? Cache_.APCSlotIdx_ : ADS::APC_INDEX_BOUND_SENTINAL;
        }

        bool IsActiveAPC() noexcept;

    protected:

        ADS::CacheOfAPC Cache_{};

        struct RelationOparation 
        {
            APCUseScope Use_{};
            uint32_t RelationLocator_ = UINT32_MAX;
            SeqLockedOperation MutationOP_ = SeqLockedOperation::NONE;
        };

        bool IsFabricBound_() const noexcept;

        APCUseScope AcquireAPCUse_() noexcept;

        void ReleseFabricBindingOnly_() noexcept;

        bool BindExternalRawFabricBacking_(
            uint64_t* raw_cells_ptr,
            VagueTemoraryPremativeFabric* fabric_owner,
            uint64_t fabric_slot_idx,
            uint64_t* generation_cell,
            uint32_t expected_generation
        ) noexcept;

        bool InitiateAPCMetaHeader() noexcept;

        bool ReadAPCMetaUnit(
            ADS::HeaderIdentifierOfAPC meta_idx,
            uint64_t& return_value
        ) noexcept;

    };


    class RegionViewConstructor : public FabricToAPCLinker
    {   
    private:
        bool ResolveRegionView_(
            MacroColumnOfAPC column_name,
            uint32_t record_ordinal,
            ResolveRegionBiteView& out
        ) noexcept;

    public:
        using SD = SchemaDefinition;

        template<class DType>
        std::optional<RegionView<DType>> BuildAViewOverRegion(
            MacroColumnOfAPC macro_column,
            uint32_t record_ordinal = UNSIGNED_ZERO
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<DType>);

            APCUseScope use = AcquireAPCUse_();
            if (!use)
            {
                return std::nullopt;
            }

            ResolveRegionBiteView resolved{};
            if (!ResolveRegionView_(macro_column, record_ordinal, resolved))
            {
                return std::nullopt;
            }


            switch (resolved.Schema->Protocol)
            {
            case SD::SchemaProtocols::PRIVATE_REGION:
            case SD::SchemaProtocols::IMMUTABLE_SNAPSHOT:
                if (!APCStorageGeometry::CanInstallTypedSpan<DType>(resolved))
                {
                    return std::nullopt;
                }
                break;
            
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
                if (!APCStorageGeometry::CanInstallAtomicSpan<DType>(resolved))
                {
                    return std::nullopt;
                }
                break;
            
            default:
                return std::nullopt;
            }
            
            DType* type_based = reinterpret_cast<DType*>(resolved.Bytes.data());
            const size_t element_count = static_cast<uint64_t>(resolved.Schema->MatrixHeight) * resolved.Schema->MatrixWidth;

            if (element_count > SIZE_MAX)
            {
                return std::nullopt;
            }
            
            return RegionView<DType>(
                std::span<DType>(type_based, element_count),
                resolved.Schema->Protocol,
                std::move(use)
            );
        }

        template<class DType>
        bool ZeroARegion(MacroColumnOfAPC macro_column) noexcept
        {
            std::optional<RegionView<DType>> maybe_view = BuildAViewOverRegion<DType>(macro_column);
            if (!maybe_view.has_value())
            {
                return false;
            }

            RegionView<DType>& view = maybe_view.value();

            using SD = SchemaDefinition;

            switch (view.GetProtocol())
            {
            case SD::SchemaProtocols::PRIVATE_REGION:
            {
                std::optional<std::span<DType>> maybe_mutable_span = view.RawMutableSpan();
                if (!maybe_mutable_span.has_value())
                {
                    return false;
                }
                
                for (DType& value : maybe_mutable_span.value())
                {
                    value = DType{};
                }
                return true;
            }
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
                for (size_t i = 0; i < view.Size(); i++)
                {
                    if (!view.AtomicStore(i, DType{}, std::memory_order_relaxed))
                    {
                        return false;
                    }
                }
                return true;
            
            default:
                return false;
            }
            
        }

    };
        
    
}