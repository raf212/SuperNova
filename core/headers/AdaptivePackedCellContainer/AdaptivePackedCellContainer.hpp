#pragma once 
#include "FabricToAPCLinker.hpp"

namespace BidirectionalInMemGraph
{
static_assert(__cpp_lib_atomic_wait, "C++ must suppoet atomic wait/notify");


    class AdaptivePackedCellContainer : public RegionViewConstructor
    {
        friend class GHGFModelOfAPC;
    protected:
        bool IsOpenGeneration_() noexcept;
    public:
        static constexpr uint8_t REALTION_FIND_TRIES = 1u;

        bool AddParent(
            AdaptivePackedCellContainer& parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool RemoveParent(
            AdaptivePackedCellContainer& parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool ReplaceParent(
            AdaptivePackedCellContainer& old_parent,
            AdaptivePackedCellContainer& new_parent,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool AttachMyChild(
            AdaptivePackedCellContainer& child,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool DetachMyChild(
            AdaptivePackedCellContainer& child,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        AdaptivePackedCellContainer FindParent(
            FabricSegments edge_table,
            uint8_t relation_ordinal,
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

        template<typename T>
        T* DataRegionPtrTest(MacroColumnOfAPC column) noexcept
            {
            using SD = SchemaDefinition;

            std::span<SchemaDefinition::RegionSchemaRecord> region_row = APCCache_.FabricOwnerPtr_->MetrixViewRow_(
                static_cast<uint32_t>(APCCache_.APCSlotIdx_)
            );

            if (
                !IsActiveAPC() ||
                !APCCache_.RawAPCBasePtr_  ||
                region_row.size() != APCCache_.FabricOwnerPtr_->FabCache_->ActiveRegionCount_
            )
            {
                return false;
            }

            const std::optional<uint8_t> compact_index = ADS::CompactRegionIndex(APCCache_.FabricOwnerPtr_->FabCache_->ActiveRegionMask_, column);
            if (!compact_index.has_value())
            {
                return false;
            }
            

            const SD::RegionSchemaRecord& stored = region_row[compact_index.value()];
            const auto expected_dtype = SD::CppTypeToRegionDType<T>();
            if (
                !expected_dtype.has_value() ||
                stored.Region != column ||
                stored.Dtype != expected_dtype.value() ||
                !SD::ValidateStortedRegionSchema(
                    stored,
                    APCCache_.FabricOwnerPtr_->FabCache_->PerAPCRuntimeCellCount_,
                    APCCache_.FabricOwnerPtr_->FabCache_->MatrixBatchCapacity_
                )
            )
            {
                return nullptr;
            }

            return RegionT_<T>(APCCache_.APCSlotIdx_, stored.CellOffset);
        }
    };


}  