#pragma once 
#include "EdgeTableConf.hpp"

namespace BidirectionalInMemGraph
{

    struct HandleOfAPCStatic
    {

        static constexpr uint8_t HANDLE_TABLE_WIDTH = 1u;

        static constexpr uint8_t ACTIVE_OPERATION_LEN = 32u;
        static constexpr uint8_t GENERATION_LEN = 31u;
        static constexpr uint8_t CLOSED_INDICATOR_LEN = 1u;

        static constexpr uint8_t ACTIVE_SHIFT = UNSIGNED_ZERO;
        static constexpr uint8_t GENERATION_SHIFT = ACTIVE_SHIFT + ACTIVE_OPERATION_LEN;
        static constexpr uint8_t CLOSED_INDICATOR_SHIFT = GENERATION_SHIFT + GENERATION_LEN;

        static constexpr uint64_t ACTIVE_COUNT_MASK = MaskLowBitsForU64(ACTIVE_OPERATION_LEN) << ACTIVE_SHIFT;
        static constexpr uint64_t GENERATION_MASK = MaskLowBitsForU64(GENERATION_LEN) << GENERATION_SHIFT;
        static constexpr uint64_t CLOSED_MASK = MaskLowBitsForU64(CLOSED_INDICATOR_LEN) << CLOSED_INDICATOR_SHIFT;

        static constexpr uint32_t MAX_GENERATION = static_cast<uint32_t>(GENERATION_MASK >> GENERATION_SHIFT);
        static constexpr uint8_t FIRST_GENERATION = 1u;

        struct ControlValues
        {
            uint32_t Generation = UNSIGNED_ZERO;
            uint32_t ActiveAccess = UNSIGNED_ZERO;
            bool Closed = true;
        };


        static constexpr uint64_t MakeControlCell(const ControlValues& values) noexcept
        {
            return
                (static_cast<uint64_t>(values.Generation) << GENERATION_SHIFT) |
                static_cast<uint64_t>(values.ActiveAccess) |
                (values.Closed ? CLOSED_MASK : UNSIGNED_ZERO);
        }

        static constexpr ControlValues ReadControlCell(uint64_t raw) noexcept
        {
            ControlValues values{};
            values.Generation = static_cast<uint32_t>((raw & GENERATION_MASK) >> GENERATION_SHIFT);
            values.ActiveAccess = static_cast<uint32_t>(raw & ACTIVE_COUNT_MASK);
            values.Closed = (raw & CLOSED_MASK) != UNSIGNED_ZERO;
            return values;
        }

        static constexpr bool IsGenerationValid(const uint32_t& generation) noexcept
        {
            return generation >= FIRST_GENERATION &&
                generation <= MAX_GENERATION;
        }

        static constexpr uint32_t NextGeneration(const uint32_t& generation) noexcept
        {
            return generation < MAX_GENERATION ? generation + 1u : UNSIGNED_ZERO;
        }

        static constexpr bool IsOpenGeneration(uint64_t raw, uint32_t expected_generation) noexcept
        {
            const ControlValues values = ReadControlCell(raw);
            return IsGenerationValid(expected_generation) &&
            values.Generation == expected_generation &&
            !values.Closed;
        }

        static constexpr size_t CellOffset(uint32_t slot) noexcept
        {
            return static_cast<size_t>(slot);
        }
    };
    static_assert(
        (HandleOfAPCStatic::ACTIVE_COUNT_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u
    );

    static_assert(
        (HandleOfAPCStatic::CLOSED_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u
    );

    struct APCRelocationDef : public CoreOfFabricCoordinator
    {
        static constexpr bool ValidateFabricCache(const FabricCache& cache, uint64_t supplied_cell_count) noexcept
        {
            if (
                cache.FormateVersion_ != FORMAT_VERSION ||
                cache.SlabCellCount_ != supplied_cell_count ||
                cache.SlabCellCount_ <  FABRIC_UNIT_COUNT ||
                cache.SlabCellCount_ >= FABRIC_CELL_SENTINAL ||
                cache.CountOfAPC_ == UNSIGNED_ZERO ||
                cache.CountOfAPC_ > UINT32_MAX ||
                cache.CountOfAPC_ > (uint64_t{1u} << EdgeBuilder::RELATION_SLOT_BITS) ||
                cache.PerAPCRuntimeCellCount_ > UINT32_MAX ||
                !ADS::IsCapacityOfAPCValid(cache.PerAPCRuntimeCellCount_) ||
                !EdgeBuilder::IsValidConfigurableParentCapacity(cache.MaxDirectParentsPerAxis_) ||
                cache.EdgeTableRecordWidth_ != EdgeBuilder::EdgeTableRecordWidth(cache.MaxDirectParentsPerAxis_) ||
                cache.ActiveRegionMask_ == UNSIGNED_ZERO ||
                (cache.ActiveRegionMask_ & ~ADS::ValidRegionMask()) != UNSIGNED_ZERO ||
                cache.ActiveRegionCount_ != std::popcount(cache.ActiveRegionMask_) ||
                cache.ActiveRegionCount_ == UNSIGNED_ZERO ||
                cache.MatrixBatchCapacity_ == UNSIGNED_ZERO ||
                cache.MatrixViewRowCellCount_ != cache.ActiveRegionCount_ * SchemaDefinition::RegionSchemaCellCount() ||
                cache.RecordBookBeginIndex_ >=  cache.RecordBookEndIndex_ ||
                cache.RecordBookEndIndex_ > cache.SlabCellCount_ ||
                cache.SegmentPoolBegin_ >= cache.SlabCellCount_ ||
                cache.CountOfAPC_ > ((UINT64_MAX - cache.SegmentPoolBegin_) / cache.PerAPCRuntimeCellCount_) 
            )
            {
                return false;
            }
            
            return 
                (cache.SegmentPoolBegin_ + (cache.CountOfAPC_ * cache.PerAPCRuntimeCellCount_)) == cache.SlabCellCount_;
        }
    };
    



}