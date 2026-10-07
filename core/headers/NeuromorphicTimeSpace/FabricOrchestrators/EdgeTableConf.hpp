#pragma once 
#include "CoreOfFabricCoordinator.hpp"
#include <span>

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
        
        struct alignas(uint64_t) ParentMaskBlock
        {
            uint64_t Block = UNSIGNED_ZERO;
        };

        struct alignas(uint64_t) ParentIDGeneration
        {
            uint32_t Generation = RELATION_NULL;
            uint32_t Slot = RELATION_NULL;
            friend constexpr bool operator==(
                const ParentIDGeneration&,
                const ParentIDGeneration&
            ) noexcept = default;
        };

        struct alignas(uint64_t) SiblingLinks
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
            friend constexpr bool operator==(
                const EdgeData&,
                const EdgeData&
            ) noexcept = default;
        };
        static constexpr uint32_t RELATION_NULL = UINT32_MAX;

        static constexpr uint8_t PARENT_MASK_BITS_PER_BLOCK = sizeof(uint64_t) * LEN_OF_BYTE_IN_BITS;

        static constexpr uint32_t ParentMaskWordCount(uint32_t k) noexcept
        {
            return (k + PARENT_MASK_BITS_PER_BLOCK - 1u) / PARENT_MASK_BITS_PER_BLOCK;
        }
        
        static constexpr uint64_t RawEdgeTableRecordWidth(uint32_t max_direct_parents) noexcept
        {
            return static_cast<uint64_t>(ParentMaskWordCount(max_direct_parents)) + 
                max_direct_parents * (sizeof(ParentRelation) / sizeof(uint64_t));
        }

        static constexpr uint64_t EdgeTableRecordWidth(uint32_t max_direct_parents) noexcept
        {
            constexpr uint32_t cells_per_cacheline = ADS::APC_CACHELINE_SIZE / sizeof(uint64_t);
            const uint64_t raw = RawEdgeTableRecordWidth(max_direct_parents);
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

        static constexpr bool IsBoundedRelationLoatorSize(uint32_t slot_count, uint32_t max_parent) noexcept
        {
            if (slot_count == UNSIGNED_ZERO)
            {
                return false;
            }

            const uint64_t highest_locator = static_cast<uint64_t>(slot_count - 1u) * max_parent + (max_parent - 1u);
            
            return highest_locator < ADS::APC_INDEX_BOUND_SENTINAL;
        }

        static constexpr uint32_t RelationSlotandOrdinal(uint32_t locator, uint32_t max_parent) noexcept
        {
            return locator % max_parent;
        }

        static constexpr bool IsValidRelationLocator(
            uint32_t locator, uint32_t slot_count, 
            uint32_t capacity, uint32_t max_parent
        ) noexcept
        {
            return locator != RELATION_NULL &&
                RelationSlotandOrdinal(locator, max_parent) < slot_count &&
                RelationSlotandOrdinal(locator, max_parent) < capacity;

        }

        static constexpr uint32_t NextSequence(uint32_t current) noexcept
        {
            // Unsigned overflow is the sequence wrap.
            return current + 1u;
        }

        static constexpr bool IsParentEmpty(const ParentIDGeneration& parent) noexcept
        {
            return
                parent.Generation == RELATION_NULL &&
                parent.Slot == RELATION_NULL;
        }

        static constexpr bool IsParentEmpty(const ParentRelation& relation) noexcept
        {
            return IsParentEmpty(relation.Parent);
        }

        static constexpr bool IsSiblingEmpty(const SiblingLinks& sibling) noexcept
        {
            return
                sibling.Previous == RELATION_NULL &&
                sibling.Next == RELATION_NULL;
        }

        static constexpr bool IsSiblingEmpty(const ParentRelation& relation) noexcept
        {
            return IsSiblingEmpty(relation.Siblings);
        }

        static constexpr bool IsEmpty(const ParentRelation& relation) noexcept
        {
            return IsSiblingEmpty(relation) && IsParentEmpty(relation);
        }

        static constexpr bool IsPartiallyEmpty(const ParentRelation& relation) noexcept
        {
            return IsParentEmpty(relation) != IsSiblingEmpty(relation);
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

        struct EdgeTableRowView 
        {
            std::span<ParentMaskBlock> Masks{};
            std::span<ParentRelation> Relations{};
            explicit constexpr operator bool() const noexcept
            {
                return
                    !Masks.empty() &&
                    !Relations.empty();
            }
        };

    };


}