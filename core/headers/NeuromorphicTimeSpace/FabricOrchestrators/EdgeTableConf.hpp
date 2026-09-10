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

        using DirtyRelationMask = uint64_t;

        static constexpr uint8_t RELATION_SLOT_BITS = 24u;
        static constexpr uint8_t RELATION_ORDINAL_BITS = 8u;

        static constexpr uint8_t EDGE_TAIL_BITS = 32u;
        static constexpr uint8_t EDGE_SEQUENCE_BITS = 30u;
        static constexpr uint8_t EDGE_STATUS_BITS = 2u;

        static_assert(EDGE_TAIL_BITS + EDGE_SEQUENCE_BITS + EDGE_STATUS_BITS == 64u);

        static constexpr uint32_t RELATION_NULL = UINT32_MAX;
        static constexpr uint32_t RELATION_SLOT_MASK = MaskLowBitsForU32(RELATION_SLOT_BITS);
        static constexpr uint32_t EDGE_SEQUENCE_MASK = MaskLowBitsForU32(EDGE_SEQUENCE_BITS);
        static constexpr uint64_t EDGE_STATUS_MASK = MaskLowBitsForU64(EDGE_STATUS_BITS);

        struct alignas(uint64_t) ParentRelation final
        {
            uint64_t ParentHandle = FABRIC_CELL_SENTINAL;
            uint64_t SiblingLocators = FABRIC_CELL_SENTINAL;
        };

        struct EdgeData final
        {
            uint32_t TailLocator = RELATION_NULL;
            uint32_t SeqLock = 0u;
            EdgeStatus Status = EdgeStatus::FREE;
            bool IsValid = false;
        };

        static constexpr size_t RawEdgeTableRecordWidth(
            uint8_t max_direct_parents
        ) noexcept
        {
            return 1u +
                static_cast<size_t>(max_direct_parents) *
                (sizeof(ParentRelation) / sizeof(uint64_t));
        }

        static constexpr size_t EdgeTableRecordWidth(
            uint8_t max_direct_parents
        ) noexcept
        {
            constexpr size_t cells_per_cacheline =
                ADS::APC_CACHELINE_SIZE / sizeof(uint64_t);
            const size_t raw = RawEdgeTableRecordWidth(max_direct_parents);
            return (raw + cells_per_cacheline - 1u) &
                ~(cells_per_cacheline - 1u);
        }

        static constexpr bool IsValidConfigurableParentCapacity(
            uint8_t value
        ) noexcept
        {
            return value > 0u &&
                value <= ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS;
        }

        static constexpr bool IsValidRelationOrdinal(
            uint8_t ordinal,
            uint8_t configured_capacity
        ) noexcept
        {
            return ordinal < configured_capacity;
        }

        static constexpr bool IsValidRelationLocator(
            uint32_t locator,
            uint32_t slot_count,
            uint8_t configured_capacity
        ) noexcept
        {
            return locator != RELATION_NULL &&
                RelationSlot(locator) < slot_count &&
                RelationOrdinal(locator) < configured_capacity;
        }

        static constexpr uint32_t PackRelationLocator(
            uint32_t apc_slot,
            uint8_t relation_ordinal
        ) noexcept
        {
            return
                (static_cast<uint32_t>(relation_ordinal)
                    << RELATION_SLOT_BITS) |
                apc_slot;
        }

        static constexpr uint32_t RelationSlot(uint32_t locator) noexcept
        {
            return locator & RELATION_SLOT_MASK;
        }

        static constexpr uint8_t RelationOrdinal(uint32_t locator) noexcept
        {
            return static_cast<uint8_t>(locator >> RELATION_SLOT_BITS);
        }

        static constexpr uint64_t MakeParentHandle(
            uint32_t parent_slot,
            uint32_t parent_generation
        ) noexcept
        {
            return TwinU32ToU64::PackDoubleUnsigned32In64(
                parent_slot,
                parent_generation
            );
        }

        static constexpr uint32_t ParentSlot(
            const ParentRelation& relation
        ) noexcept
        {
            return TwinU32ToU64::ExtractLow32Of64(
                relation.ParentHandle
            );
        }

        static constexpr uint32_t ParentGeneration(
            const ParentRelation& relation
        ) noexcept
        {
            return TwinU32ToU64::ExtractHigh32Of64(
                relation.ParentHandle
            );
        }

        static constexpr uint32_t PreviousLocator(
            const ParentRelation& relation
        ) noexcept
        {
            return TwinU32ToU64::ExtractLow32Of64(
                relation.SiblingLocators
            );
        }

        static constexpr uint32_t NextLocator(
            const ParentRelation& relation
        ) noexcept
        {
            return TwinU32ToU64::ExtractHigh32Of64(
                relation.SiblingLocators
            );
        }

        static constexpr void SetSiblingLocators(
            ParentRelation& relation,
            uint32_t previous,
            uint32_t next
        ) noexcept
        {
            relation.SiblingLocators =
                TwinU32ToU64::PackDoubleUnsigned32In64(
                    previous,
                    next
                );
        }

        static constexpr ParentRelation MakeParentRelation(
            uint32_t parent_slot,
            uint32_t parent_generation,
            uint32_t previous,
            uint32_t next
        ) noexcept
        {
            return ParentRelation{
                MakeParentHandle(parent_slot, parent_generation),
                TwinU32ToU64::PackDoubleUnsigned32In64(previous, next)
            };
        }

        static constexpr bool IsEmpty(
            const ParentRelation& relation
        ) noexcept
        {
            return relation.ParentHandle == FABRIC_CELL_SENTINAL &&
                relation.SiblingLocators == FABRIC_CELL_SENTINAL;
        }

        static constexpr bool IsPartiallyEmpty(
            const ParentRelation& relation
        ) noexcept
        {
            return
                (relation.ParentHandle == FABRIC_CELL_SENTINAL) !=
                (relation.SiblingLocators == FABRIC_CELL_SENTINAL);
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
    };


}