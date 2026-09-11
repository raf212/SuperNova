#pragma once 
#include "FabricTableConstructors/CompleteFabric.h"

namespace BidirectionalInMemGraph
{
    
    class SlabToFabricConverterAndCordinator : public CompiledDAGTableConstructor
    {
    private:

        uint64_t* AllocatePackedCellRaw_(size_t count_of_cells) noexcept;
        
        /// @brief INITIALIZES: All FabricMetaIndicies
        /// @param table_directory_begin 
        /// @param table_directory_end 
        void InitializeCompleateFabricMetaIndices_(size_t record_book_begin, size_t record_book_end) noexcept;

    protected :
        SD::RegionSchemaTable DefaultRegionTable_{};
        bool HasDefaultRegionTable_{false};

        void FreeRawPackedCells_(uint64_t*packed_cell_memory_ptr, size_t packed_cell_count) noexcept;
        void ResetScalarsofTheFabric_() noexcept;

        template<typename T>
        T* RegionT_(uint32_t slot, uint32_t cell_offset) noexcept
        {
            return reinterpret_cast<T*>(SlabBasePtr_ + SlotBegin_(slot) + cell_offset);
        }

    public:
        SlabToFabricConverterAndCordinator(/* args */) noexcept = default;

        ~SlabToFabricConverterAndCordinator() noexcept
        {
            ShutDownFabric();
        }

        SlabToFabricConverterAndCordinator(const SlabToFabricConverterAndCordinator&) = delete;
        SlabToFabricConverterAndCordinator& operator = (const SlabToFabricConverterAndCordinator&) = delete;

        void ShutDownFabric() noexcept;

        bool IsFabricActive() noexcept
        {
            return
                FabricInitialized_.load(std::memory_order_acquire) &&
                SlabBasePtr_ &&
                ADS::IsValid32BitAPCUnit(FabCache_.PerAPCRuntimeCellCount_) &&
                ADS::IsValid32BitAPCUnit(FabCache_.CountOfAPC_);
        }

        bool InitializeFabric(
            uint32_t slot_count,
            uint32_t slot_cell_count,
            const SchemaDefinition::FabricRegionConfig& region_conf,
            uint8_t max_direct_parent_per_axis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS
        ) noexcept;
        
    };


    class DAGMutationConf : public SlabToFabricConverterAndCordinator
    {
    protected:
        static constexpr uint8_t DAG_MAX_ROW_PARTICIPANTS = 7u;
        static constexpr uint8_t DAG_MAX_RELATION_DELTAS = 5u;
        static constexpr uint8_t INVALID_RELATION_ORDINAL = UINT8_MAX;

        struct DAGRowParticipant
        {
            uint32_t Slot = ADS::APC_INDEX_BOUND_SENTINAL;
            EdgeBuilder::EdgeData Before{};
            uint32_t WorkTail = EdgeBuilder::RELATION_NULL;
            bool IsParentAnchor = false;
            bool Reserved = false;
        };

        struct DAGRelationDelta
        {
            uint32_t ChildSlot = ADS::APC_INDEX_BOUND_SENTINAL;
            uint8_t Ordinal = INVALID_RELATION_ORDINAL;
            EdgeBuilder::ParentRelation Before{};
            EdgeBuilder::ParentRelation Work{};
        };

        struct DAGMutationTransaction
        {
            FabricSegments EdgeTable{};
            std::array<DAGRowParticipant, DAG_MAX_ROW_PARTICIPANTS> Rows{};
            std::array<DAGRelationDelta, DAG_MAX_RELATION_DELTAS> Relations{};
            uint8_t RowCount = 0u;
            uint8_t RelationCount = 0u;
        };

        bool AddRowParticipant_(
            DAGMutationTransaction& transaction,
            uint32_t slot,
            bool is_parent_anchor = false
        ) noexcept;

        DAGRowParticipant* FindRowParticipant_(
            DAGMutationTransaction& transaction,
            uint32_t slot
        ) noexcept;

        DAGRelationDelta* EditReservedRelation_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            uint8_t ordinal
        ) noexcept;

        bool ReserveAllRows_(
            DAGMutationTransaction& transaction,
            EdgeBuilder::EdgeStatus required_status,
            uint32_t max_tries
        ) noexcept;

        void AbortRowTransaction_(
            DAGMutationTransaction& transaction
        ) noexcept;

        void CommitRowTransaction_(
            DAGMutationTransaction& transaction,
            EdgeBuilder::EdgeStatus final_status =
                EdgeBuilder::EdgeStatus::LIVE
        ) noexcept;
    };


    class ConstructDAGOnEachAxis : public DAGMutationConf
    {
        friend class AdaptivePackedCellContainer;
        friend class FabricToAPCLinker;

    protected:
        static constexpr bool SameHeader_(
            const EdgeBuilder::EdgeData& left,
            const EdgeBuilder::EdgeData& right
        ) noexcept
        {
            return
                left.IsValid &&
                right.IsValid &&
                left.TailLocator == right.TailLocator &&
                left.SeqLock == right.SeqLock &&
                left.Status == right.Status;
        }
        static constexpr uint8_t DEFAULT_INTERNAL_TRIES__ = 1u;
    private:

        struct ParentRowScan
        {
            EdgeBuilder::EdgeData Header{};
            uint8_t MatchOrdinal = UINT8_MAX;
            uint8_t OtherOrdinal = UINT8_MAX;
            uint8_t EmptyOrdinal = UINT8_MAX;
            EdgeBuilder::ParentRelation Match{};
        };

        static constexpr bool SameRelation_(
            const EdgeBuilder::ParentRelation& left,
            const EdgeBuilder::ParentRelation& right
        ) noexcept
        {
            return
                left.ParentHandle == right.ParentHandle &&
                left.SiblingLocators == right.SiblingLocators;
        }

        SeqLockedOperation ScanParentRow_(
            FabricSegments edge_table,
            uint32_t child_slot,
            uint64_t wanted_parent_handle,
            uint64_t other_parent_handle,
            ParentRowScan& scan,
            uint32_t max_tries
        ) noexcept;

        bool AddParentRelation_(
            uint32_t parent_slot,
            uint32_t parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool RemoveParentRelation_(
            uint32_t parent_slot,
            uint32_t parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        bool ReplaceParentRelation_(
            uint32_t old_parent_slot,
            uint32_t old_parent_generation,
            uint32_t new_parent_slot,
            uint32_t new_parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

    };



}