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

    };


}  