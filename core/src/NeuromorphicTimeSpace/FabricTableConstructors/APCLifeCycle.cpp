#include "NeuromorphicTimeSpace/SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

    ADS::RangeOfAPC APCLifeCycle::GetSegmentPoolRange(uint64_t single_description_index) noexcept
    {
        ADS::RangeOfAPC desired_segment_pool_range{};

        if (
            single_description_index >= CountOfAPC_ ||
            PerAPCRuntimeCellCount_ == UNSIGNED_ZERO
        )
        {
            return desired_segment_pool_range;
        }

        const uint64_t apc_count_offset = single_description_index * PerAPCRuntimeCellCount_;
        desired_segment_pool_range.BeginIndex = SegmentPoolBegin_ + static_cast<size_t>(apc_count_offset);
        desired_segment_pool_range.EndIndex = desired_segment_pool_range.BeginIndex + static_cast<size_t>(PerAPCRuntimeCellCount_);
        desired_segment_pool_range.IsValid =
            desired_segment_pool_range.BeginIndex >= SegmentPoolBegin_ &&
            desired_segment_pool_range.BeginIndex < desired_segment_pool_range.EndIndex &&
            desired_segment_pool_range.EndIndex <= SlabCellCount_;

        return desired_segment_pool_range;
        
    }

    DescriptionOfAPC::SeqLockAndStateStruct APCLifeCycle::ReadAPCStateAtomically_(uint64_t apc_description_index) noexcept
    {
        DescriptionOfAPC::SeqLockAndStateStruct return_files{};
        std::optional<uint64_t> maybe_id_state_idx = GetDescriptionLockIdxInFabric_(apc_description_index);
        uint64_t state_of_apc_cell = FABRIC_CELL_SENTINAL;

        if (
            !maybe_id_state_idx.has_value() ||
            !AtomicallyLoadReadAUnit(maybe_id_state_idx.value(), state_of_apc_cell)
        )
        {
            return return_files;
        }
        DescriptionOfAPC::GetSeqLockAndLifeCycle(state_of_apc_cell, return_files);
        return return_files;
    }


    bool APCLifeCycle::SwitchDescriptionState(
        uint64_t description_idx,
        StateOfAPC updated_state,
        StateOfAPC desired_state,
        uint32_t max_tries
    ) noexcept
    {
        const std::optional<uint64_t> id_state_idx = GetDescriptionLockIdxInFabric_(description_idx);
        if (
            !id_state_idx.has_value()
        )
        {
            return false;
        }

        for (size_t i = 0; i < max_tries; i++)
        {
            DSA::SeqLockAndStateStruct current_id_st = ReadAPCStateAtomically_(description_idx);
            uint64_t current_id_state_value = DSA::ComposeSeqLockAndState(current_id_st);
            DSA::SeqLockAndStateStruct updated_files{};
            updated_files.SeqLock = current_id_st.SeqLock + 1u;
            updated_files.StateOfTheAPC = updated_state;
            uint64_t updated_id_state_value = DSA::ComposeSeqLockAndState(updated_files);

            if (
                !ADS::IsValidFabricUnit(current_id_state_value) ||
                !ADS::IsValidFabricUnit(updated_id_state_value) ||
                current_id_st.StateOfTheAPC != desired_state ||
                !DSA::IsTransitionStateLeagal(current_id_st.StateOfTheAPC, updated_state)
            )
            {
                return false;
            }

            if (
                !CompareExchangeStrongFromFabric(
                    id_state_idx.value(),
                    current_id_state_value,
                    updated_id_state_value
                )
            )
            {
                continue;
            }
            
            return true;
        }
        return false;
    }

    std::optional<uint64_t> APCLifeCycle::GetDescriptionLockIdxInFabric_(uint64_t description_idx) noexcept
    {
        const ADS::RangeOfAPC range_of_segmentpool = GetSegmentPoolRange(description_idx);

        if (!range_of_segmentpool.IsValid)
        {
            return std::nullopt;
        }
        const size_t state_cell_idx = range_of_segmentpool.BeginIndex + 
            static_cast<uint8_t>(ADS::HeaderIdentifierOfAPC::APC_LIFE_CYCLE);
        
        return state_cell_idx;
    }

    void APCLifeCycle::InitAllAPCLifeCycleState() noexcept
    {
        for (size_t i = 0; i < CountOfAPC_; i++)
        {
            std::optional<uint64_t> maybe_id_state_idx = GetDescriptionLockIdxInFabric_(i);
            if (!maybe_id_state_idx.has_value())
            {
                return;
            }

            DescriptionOfAPC::SeqLockAndStateStruct values{};
            values.StateOfTheAPC = StateOfAPC::FREE;
            values.SeqLock = 2u;

            const uint64_t raw_lifecycle = DSA::ComposeSeqLockAndState(values);
            
            if (!ADS::IsValidFabricUnit(raw_lifecycle))
            {
                return;
            }
            
            DirectlyStoreFabricUnit64(maybe_id_state_idx.value(), raw_lifecycle);
        }
        
    }


}
