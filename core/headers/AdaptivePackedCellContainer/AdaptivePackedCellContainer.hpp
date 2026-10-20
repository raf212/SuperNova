#pragma once 
#include "FabricToAPCLinker.hpp"

namespace BidirectionalInMemGraph
{
static_assert(__cpp_lib_atomic_wait, "C++ must suppoet atomic wait/notify");


    class AdaptivePackedCellContainer : public FabricToAPCLinker
    {
        friend class GHGFModelOfAPC;
        using SD = SchemaDefinition;
        using ASG = APCStorageGeometry;
    protected:
        bool IsOpenGeneration_() noexcept;

        bool ResolveBridgeRegion(
            MacroColumnOfAPC column,
            uint32_t record_ordinal,
            ResolveRegionBiteView& out
        ) noexcept;
        
    public:
        static constexpr uint8_t REALTION_FIND_TRIES = 1u;
        //Do not change wherever it is default until proven
        static constexpr uint8_t INTERNAL_RECURSION = 1u;

        enum class MutationResult : uint8_t
        {
            COMMITTED = 0,
            INVALID = 1,
            RETRY = 2
        };

        MutationResult AddParent(
            AdaptivePackedCellContainer& parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_recursion = INTERNAL_RECURSION
        ) noexcept;

        MutationResult RemoveParent(
            AdaptivePackedCellContainer& parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_recursion = INTERNAL_RECURSION
        ) noexcept;

        MutationResult ReplaceParent(
            AdaptivePackedCellContainer& old_parent,
            AdaptivePackedCellContainer& new_parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_recursion = INTERNAL_RECURSION
        ) noexcept;

        MutationResult AttachMyChild(
            AdaptivePackedCellContainer& child,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_recursion = INTERNAL_RECURSION
        ) noexcept;

        MutationResult DetachMyChild(
            AdaptivePackedCellContainer& child,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_recursion = INTERNAL_RECURSION
        ) noexcept;

        AdaptivePackedCellContainer FindParent(
            FabricSegments edge_table,
            uint32_t relation_ordinal,
            RelationOparation* parent_relation = nullptr,
            uint32_t max_tries = REALTION_FIND_TRIES
        ) noexcept;

        AdaptivePackedCellContainer FindFirstChild(
            FabricSegments edge_table,
            RelationOparation* child_relation = nullptr,
            uint32_t max_tries = REALTION_FIND_TRIES
        ) noexcept;

        AdaptivePackedCellContainer FindLastChild(
            FabricSegments edge_table,
            RelationOparation* child_relation = nullptr,
            uint32_t max_tries = REALTION_FIND_TRIES
        ) noexcept;

        AdaptivePackedCellContainer FindNextChild(
            FabricSegments edge_table,
            uint32_t current_relation_locator,
            RelationOparation* child_relation = nullptr,
            uint32_t max_tries = REALTION_FIND_TRIES
        ) noexcept;

        AdaptivePackedCellContainer FindPreviousChild(
            FabricSegments edge_table,
            uint32_t current_relation_locator,
            RelationOparation* child_relation = nullptr,
            uint32_t max_tries = REALTION_FIND_TRIES
        ) noexcept;

        bool Retire(
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        AdaptivePackedCellContainer* MyAPCPtr() noexcept
        {
            return this;
        }

        template<class DT> 
        SNTOComputerBridge<DT>  BuildAViewOverRegion(MacroColumnOfAPC column, uint32_t record_ordinal = UNSIGNED_ZERO) noexcept
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
            cache.ActiveWidth_ = resolved.Schema->MatrixWidth;
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
        bool ZeroARegion(MacroColumnOfAPC column) noexcept
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
    };


}