#pragma once 
#include "FabricToAPCLinker.hpp"

namespace BidirectionalInMemGraph
{
static_assert(__cpp_lib_atomic_wait, "C++ must suppoet atomic wait/notify");


    class AdaptivePackedCellContainer : public RegionViewConstructor
    {
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
    };


}  