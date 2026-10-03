#pragma once 
#include "CoreOfFabricCoordinator.hpp"

namespace BidirectionalInMemGraph
{
    struct EdgeBuilder : public DescriptionOfAPC
    {
        enum class EdgeStatus : uint8_t
        {
            FREE = 0,
            RESERVED = 1,
            LIVE = 2
        };

        enum class StructureOperation : uint8_t
        {
            ADD_PARENT = 0,
            REMOVE_PARENT = 1,
            REPLACE_PARENT = 2
        };

        enum class EdgeDomain : uint8_t
        {
            PARENT_RELATIONS = 0u,
            CHILD_LIST = 1u
        };
        
        using DirtyRelationMask = uint64_t;

        struct alignas(uint64_t) ParentMaskBlock
        {
            uint64_t Block = UNSIGNED_ZERO;
        };

        struct ParentIDGeneration
        {
            uint32_t Generation = RELATION_NULL;
            uint32_t Slot = RELATION_NULL;
            friend constexpr bool operator==(
                const ParentIDGeneration&,
                const ParentIDGeneration&
            ) noexcept = default;
        };

        struct SiblingLinks
        {
            uint32_t Previous = RELATION_NULL;
            uint32_t Next = RELATION_NULL;
            friend constexpr bool operator==(
                const SiblingLinks&,
                const SiblingLinks&
            ) noexcept = default;
        };

        struct alignas(uint64_t) ParentRelation final
        {
            ParentIDGeneration Parent{};
            SiblingLinks Siblings{};
            friend constexpr bool operator==(
                const ParentRelation&,
                const ParentRelation&
            ) noexcept = default;
        };

        struct EdgeData final
        {
            uint32_t TailLocator = RELATION_NULL;
            uint32_t SeqLock = 0u;
            EdgeStatus Status = EdgeStatus::FREE;
            bool IsValid = false;
        };

        static constexpr uint8_t EDGE_TAIL_BITS = 32u;
        static constexpr uint8_t EDGE_SEQUENCE_BITS = 30u;
        static constexpr uint8_t EDGE_STATUS_BITS = 2u;


        static constexpr uint32_t RELATION_NULL = UINT32_MAX;
        static constexpr uint32_t EDGE_SEQUENCE_MASK = MaskLowBitsForU32(EDGE_SEQUENCE_BITS);
        static constexpr uint64_t EDGE_STATUS_MASK = MaskLowBitsForU64(EDGE_STATUS_BITS);

        static constexpr uint8_t PARENT_MASK_BITS_PER_BLOCK = sizeof(uint64_t) * LEN_OF_BYTE_IN_BITS;

        static constexpr uint32_t ParentMaskWordCount(uint32_t k) noexcept
        {
            return (k + PARENT_MASK_BITS_PER_BLOCK - 1u) / PARENT_MASK_BITS_PER_BLOCK;
        }
        
        static constexpr uint32_t RawEdgeTableRecordWidth(uint32_t max_direct_parents) noexcept
        {
            return ParentMaskWordCount(max_direct_parents) + 
                max_direct_parents * (sizeof(ParentRelation) / sizeof(uint64_t));
        }

        static constexpr uint32_t EdgeTableRecordWidth(uint32_t max_direct_parents) noexcept
        {
            constexpr uint32_t cells_per_cacheline = ADS::APC_CACHELINE_SIZE / sizeof(uint64_t);
            const uint32_t raw = RawEdgeTableRecordWidth(max_direct_parents);
            return (raw + cells_per_cacheline - 1u) & ~(cells_per_cacheline - 1u);
        }

        static constexpr bool IsValidConfigurableParentCapacity(uint32_t max_direct_parents, uint32_t count_of_apc) noexcept
        {
            return max_direct_parents > UNSIGNED_ZERO && max_direct_parents <= count_of_apc;
        }

        static constexpr bool IsValidRelationOrdinal(uint32_t ordinal, uint32_t configured_capacity) noexcept
        {
            return ordinal < configured_capacity;
        }

        static constexpr bool IsEmpty(const ParentRelation& relation) noexcept
        {
            return
                relation.Parent.Generation == RELATION_NULL &&
                relation.Parent.Slot == RELATION_NULL &&
                relation.Siblings.Previous == RELATION_NULL &&
                relation.Siblings.Next == RELATION_NULL;
        }

        static constexpr bool IsPartiallyEmpty(const ParentRelation& relation) noexcept
        {
            const bool parent_empty =
                relation.Parent.Generation == RELATION_NULL &&
                relation.Parent.Slot == RELATION_NULL;

            const bool sibling_empty =
                relation.Siblings.Previous == RELATION_NULL &&
                relation.Siblings.Next == RELATION_NULL;

            return parent_empty != sibling_empty;
        }

        static constexpr bool IsParentEmpty(const ParentRelation& relation) noexcept
        {
            return
                relation.Parent.Generation == RELATION_NULL &&
                relation.Parent.Slot == RELATION_NULL;
        }

        static constexpr bool IsSiblingEmpty(
            const ParentRelation& relation
        ) noexcept
        {
            return
                relation.Siblings.Previous == RELATION_NULL &&
                relation.Siblings.Next == RELATION_NULL;
        }

        static constexpr void Clear(ParentRelation& relation) noexcept
        {
            relation = ParentRelation{};
        }

        static constexpr bool CanInsertCombinedDAGRelation(
            uint32_t parent_slot,
            uint32_t child_slot
        ) noexcept
        {
            return parent_slot < child_slot;
        }

        static constexpr uint32_t NextSequence(uint32_t current) noexcept
        {
            return (current + 1u) & EDGE_SEQUENCE_MASK;
        }

        static constexpr uint64_t PackEdgeHeader(
            const EdgeData& edge
        ) noexcept
        {
            return
                static_cast<uint64_t>(edge.TailLocator) |
                (static_cast<uint64_t>(edge.SeqLock & EDGE_SEQUENCE_MASK)
                    << EDGE_TAIL_BITS) |
                (static_cast<uint64_t>(edge.Status)
                    << (EDGE_TAIL_BITS + EDGE_SEQUENCE_BITS));
        }

        static constexpr EdgeData UnpackEdgeHeader(uint64_t raw) noexcept
        {
            EdgeData edge{};
            edge.TailLocator = static_cast<uint32_t>(raw);
            edge.SeqLock = static_cast<uint32_t>(
                (raw >> EDGE_TAIL_BITS) & EDGE_SEQUENCE_MASK
            );
            edge.Status = static_cast<EdgeStatus>(
                (raw >> (EDGE_TAIL_BITS + EDGE_SEQUENCE_BITS)) &
                EDGE_STATUS_MASK
            );

            const bool known_status =
                edge.Status == EdgeStatus::FREE ||
                edge.Status == EdgeStatus::RESERVED ||
                edge.Status == EdgeStatus::LIVE;

            const bool parity_ok =
                edge.Status == EdgeStatus::RESERVED
                    ? (edge.SeqLock & 1u) != 0u
                    : (edge.SeqLock & 1u) == 0u;

            const bool tail_state_ok =
                edge.Status != EdgeStatus::FREE ||
                edge.TailLocator == RELATION_NULL;

            edge.IsValid = known_status && parity_ok && tail_state_ok;
            return edge;
        }

        static constexpr DirtyRelationMask DirtyBit(
            uint8_t ordinal
        ) noexcept
        {
            return uint64_t{1u} << ordinal;
        }

        static constexpr uint16_t CHILD_LIST_CONTROL_OFFSET = 0u;
        static constexpr uint16_t PARENT_RELATION_CONTROL_OFFSET = 1u;
        static constexpr uint16_t PARENT_RELATION_ARRAY_OFFSET = 2u;

        static constexpr uint16_t ControlOffset(EdgeDomain domain) noexcept
        {
            return domain == EdgeDomain::PARENT_RELATIONS
                ? PARENT_RELATION_CONTROL_OFFSET
                : CHILD_LIST_CONTROL_OFFSET;
        }

    };


}