#pragma once
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{
    bool APCFinilizer::RetireAPC_(
        uint32_t slot,
        uint32_t generation,
        uint32_t max_tries
    ) noexcept
    {
        if (
            slot >= CountOfAPC_ ||
            !HandleOfAPCStatic::IsGenerationValid(generation)
        )
        {
            return false;
        }

        EdgeBuilder::EdgeData horizontal_before{};
        EdgeBuilder::EdgeData vertical_before{};

        if (
            ReserveEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                EdgeBuilder::EdgeStatus::LIVE,
                horizontal_before,
                max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            return false;
        }

        if (
            ReserveEdgeRow_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                EdgeBuilder::EdgeStatus::LIVE,
                vertical_before,
                max_tries
            ) != SeqLockedOperation::FOUND
        )
        {
            PublishReservedEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal_before,
                horizontal_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
            return false;
        }

        auto ReleaseRows___ = [&]() noexcept
        {
            PublishReservedEdgeRow_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                vertical_before,
                vertical_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
            PublishReservedEdgeRow_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal_before,
                horizontal_before.TailLocator,
                EdgeBuilder::EdgeStatus::LIVE
            );
        };

        auto ReservedRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);

            if (relations.size() != MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0u;
                ordinal < MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(
                    relations[ordinal].ParentHandle
                ).load(std::memory_order_relaxed);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(
                    relations[ordinal].SiblingLocators
                ).load(std::memory_order_relaxed);

                if (!EdgeBuilder::IsEmpty(relation))
                {
                    return false;
                }
            }
            return true;
        };

        if (
            horizontal_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            vertical_before.TailLocator != EdgeBuilder::RELATION_NULL ||
            !ReservedRowIsEmpty___(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H
            ) ||
            !ReservedRowIsEmpty___(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
            )
        )
        {
            ReleaseRows___();
            return false;
        }


        if (
            !CloseAPCGeneration_(slot, generation)
        )
        {
            ReleaseRows___();
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RESERVED,
            StateOfAPC::LIVE,
            max_tries
        ))
        {
            OpenAPCGeneration_(slot, generation);
            ReleaseRows___();
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RETIRED,
            StateOfAPC::RESERVED,
            max_tries
        ))
        {
            SwitchDescriptionState(
                slot,
                StateOfAPC::LIVE,
                StateOfAPC::RESERVED,
                max_tries
            );
            OpenAPCGeneration_(slot, generation);
            ReleaseRows___();
            return false;
        }

        PublishReservedEdgeRow_(
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
            slot,
            vertical_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        PublishReservedEdgeRow_(
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            slot,
            horizontal_before,
            EdgeBuilder::RELATION_NULL,
            EdgeBuilder::EdgeStatus::FREE
        );
        return true;
    }

    bool APCFinilizer::ReclaimRetiredSlotTemp_(uint32_t slot) noexcept
    {
        if (slot >= CountOfAPC_)
        {
            return false;
        }

        const ADS::RangeOfAPC range = GetSegmentPoolRange(slot);
        if (!range.IsValid)
        {
            return false;
        }

        if (!SwitchDescriptionState(
            slot,
            StateOfAPC::RESERVED,
            StateOfAPC::RETIRED,
            DEFAULT_MAX_TRIES
        ))
        {
            return false;
        }

        auto RestoreRetired___ = [&]() noexcept
        {
            SwitchDescriptionState(
                slot,
                StateOfAPC::RETIRED,
                StateOfAPC::RESERVED,
                DEFAULT_MAX_TRIES
            );
        };

        EdgeBuilder::EdgeData horizontal{};
        EdgeBuilder::EdgeData vertical{};

        if (
            !ReadEdgeHeader_(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                slot,
                horizontal
            ) ||
            !ReadEdgeHeader_(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                slot,
                vertical
            ) ||
            horizontal.Status != EdgeBuilder::EdgeStatus::FREE ||
            vertical.Status != EdgeBuilder::EdgeStatus::FREE ||
            horizontal.TailLocator != EdgeBuilder::RELATION_NULL ||
            vertical.TailLocator != EdgeBuilder::RELATION_NULL
        )
        {
            RestoreRetired___();
            return false;
        }

        auto FreeRowIsEmpty___ = [&](FabricSegments table) noexcept
        {
            std::span<EdgeBuilder::ParentRelation> relations =
                ParentRelations_(table, slot);

            if (relations.size() != MaxDirectParentsPerAxis_)
            {
                return false;
            }

            for (uint8_t ordinal = 0u;
                ordinal < MaxDirectParentsPerAxis_;
                ++ordinal)
            {
                EdgeBuilder::ParentRelation relation{};
                relation.ParentHandle = std::atomic_ref<uint64_t>(
                    relations[ordinal].ParentHandle
                ).load(std::memory_order_acquire);
                relation.SiblingLocators = std::atomic_ref<uint64_t>(
                    relations[ordinal].SiblingLocators
                ).load(std::memory_order_acquire);

                if (!EdgeBuilder::IsEmpty(relation))
                {
                    return false;
                }
            }
            return true;
        };

        if (
            !FreeRowIsEmpty___(
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H
            ) ||
            !FreeRowIsEmpty___(
                FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
            )
        )
        {
            RestoreRetired___();
            return false;
        }

        uint32_t new_generation = 0u;
        if (!AdvanceClosedAPCGeneration_(slot, new_generation))
        {
            RestoreRetired___();
            return false;
        }

        const size_t lifecycle_index =
            range.BeginIndex +
            static_cast<size_t>(ADS::HeaderIdentifierOfAPC::APC_LIFE_CYCLE);

        for (size_t idx = range.BeginIndex; idx < range.EndIndex; ++idx)
        {
            if (idx != lifecycle_index)
            {
                DirectlyStoreFabricUnit64(idx, 0u);
            }
        }

        return HandleOfAPCStatic::IsGenerationValid(new_generation);
    }


    constexpr bool APCFinilizer::IsNodePolicyReConfigurable_(const SD::RegionSchemaTable& table) noexcept
    {
        if (!HasDefaultRegionTable_)
        {
            return true;
        }

        for (uint8_t i = 0; i < ADS::CountOfMacroColumn(); i++)
        {
            const SD::RegionSchemaRecord& expected = DefaultRegionTable_[i];

            const SD::RegionSchemaRecord& supplied = table[i];
            const bool expected_disabled = SD::HasSchemaFlag(expected.Flags, SD::SchemaFlags::REGION_DISABLED);
            const bool supplied_disabled = SD::HasSchemaFlag(supplied.Flags, SD::SchemaFlags::REGION_DISABLED);

            if (
                supplied.Region != expected.Region ||
                expected_disabled != supplied_disabled
            )
            {
                return false;
            }

            if (expected_disabled)
            {
                if (supplied.Flags != expected.Flags)
                {
                    return false;
                }
                continue;
            }
            
            if (
                supplied.Dtype != expected.Dtype ||
                supplied.MatrixHeight != expected.MatrixHeight ||
                supplied.MatrixWidth != expected.MatrixWidth ||
                supplied.CellCount != expected.CellCount ||
                supplied.Flags != expected.Flags
            )
            {
                return false;
            }

            // An exceptional node may change synchronization policy, but not
            // physical record count or vector geometry.
            switch (supplied.Protocol)
            {
            case SD::SchemaProtocols::PRIVATE_REGION:
            case SD::SchemaProtocols::IMMUTABLE_SNAPSHOT:
            case SD::SchemaProtocols::ATOMIC_WORD_ARRAY:
                break;

            default:
                return false;
            }

            return true;
            
        }
        
    }


}