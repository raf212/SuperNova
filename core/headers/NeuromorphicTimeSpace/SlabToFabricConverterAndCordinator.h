#pragma once 
#include "FabricTableConstructors/CompleteFabric.h"

namespace BidirectionalInMemGraph
{
    
    class SlabToFabricConverterAndCordinator : public CompiledDAGTableConstructor
    {
        friend class AdaptivePackedCellContainer;
    private:

        uint64_t* AllocatePackedCellRaw_(size_t count_of_cells) noexcept;
        bool ValidateAttachedFabricLayout_() noexcept;
        bool QuiesceFabric_() noexcept;
        bool ReopenLiveAPCGenerations_() noexcept;

    protected :
        std::atomic<bool> TrackDAGRevision_{false};

        SD::RegionSchemaTable DefaultRegionTable_{};

        void FreeRawPackedCells_(uint64_t*packed_cell_memory_ptr, size_t packed_cell_count) noexcept;
        void ResetScalarsofTheFabric_() noexcept;

        template<typename T>
        T* RegionT_(uint32_t slot, uint32_t cell_offset) noexcept
        {
            return reinterpret_cast<T*>(SlabBasePtr_ + SlotBegin_(slot) + cell_offset);
        }

    public:
        using CFC = CoreOfFabricCoordinator;

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
                FabCache_ &&
                ADS::IsValid32BitAPCUnit(FabCache_->PerAPCRuntimeCellCount_) &&
                ADS::IsValid32BitAPCUnit(FabCache_->CountOfAPC_);
        }

        bool InitializeFabric(
            uint32_t slot_count,
            uint32_t slot_cell_count,
            const SchemaDefinition::FabricRegionConfig& region_conf,
            uint32_t max_direct_parent_per_axis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS
        ) noexcept;

        bool SaveFabric(std::span<uint64_t> destination) noexcept;

        bool AttachFabric(
            uint64_t* raw_cells,
            uint64_t cell_count,
            CFC::FabricBackigOwnership ownership = CFC::FabricBackigOwnership::BORROWED
        ) noexcept;

        CFC::DetachFabric DetachFabric() noexcept;
        
    };


    class DAGMutationConf : public SlabToFabricConverterAndCordinator
    {
    protected:
        static constexpr uint8_t DAG_MAX_ROW_PARTICIPANTS = 3u;
        static constexpr uint8_t DAG_MAX_RELATION_DELTAS = 5u;

        struct ConditionalParentPublication final
        {
            using PublishFunction = void(*) (void*, uint32_t) noexcept;
            uint32_t ExpectedRowSequence = UINT32_MAX;
            uint32_t PublishedRowSequence = UINT32_MAX;
            uint32_t PublishedOrdinal = EB::RELATION_NULL;
            bool SequenceMismatch = false;
            void* Context = nullptr;
            PublishFunction Publish = nullptr;
        };

        struct DAGRowParticipant
        {
            uint32_t Slot = ADS::APC_INDEX_BOUND_SENTINAL;
            EdgeBuilder::EdgeDomain Domain =
                EdgeBuilder::EdgeDomain::PARENT_RELATIONS;
            EdgeBuilder::EdgeData Before{};
            uint32_t WorkTail = EdgeBuilder::RELATION_NULL;
            bool Reserved = false;
        };

        struct DAGRelationDelta
        {
            uint32_t ChildSlot = EB::RELATION_NULL;
            uint32_t Ordinal = EB::RELATION_NULL;
            EdgeBuilder::ParentRelation Before{};
            EdgeBuilder::ParentRelation Work{};
            bool ParentHandleDirty = false;
            bool SiblingLocatorsDirty = false;
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
            EdgeBuilder::EdgeDomain domain
        ) noexcept;

        DAGRowParticipant* FindRowParticipant_(
            DAGMutationTransaction& transaction,
            uint32_t slot,
            EdgeBuilder::EdgeDomain domain
        ) noexcept;

        DAGRelationDelta* FindOrInsertRelationDelta_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            uint32_t ordinal
        ) noexcept;

        DAGRelationDelta* EditReservedParentHandle_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            uint32_t ordinal
        ) noexcept;

        DAGRelationDelta* EditReservedSiblingLocators_(
            DAGMutationTransaction& transaction,
            uint32_t owner_parent_slot,
            uint32_t relation_locator
        ) noexcept;

        SeqLockedOperation ReserveAllRows_(
            DAGMutationTransaction& transaction,
            uint32_t max_tries
        ) noexcept;

        void AbortRowTransaction_(
            DAGMutationTransaction& transaction
        ) noexcept;

        void CommitRowTransaction_(
            DAGMutationTransaction& transaction
        ) noexcept;

        bool ValidateConditionalParentPublication_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            ConditionalParentPublication* publication
        ) noexcept;

        void PrepareConditionalParentPublication_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            uint32_t relation_ordinal,
            ConditionalParentPublication* publication
        ) noexcept;
    };


    class GHGFStructralLearningModel;
    class ConstructDAGOnEachAxis : public DAGMutationConf
    {
        friend class AdaptivePackedCellContainer;
        friend class FabricToAPCLinker;
        friend class GHGFStructralLearningModel;
    public:
        using MutationResult = AdaptivePackedCellContainer::MutationResult;

        MutationResult ReplaceParentByIdentity(
            const EB::ParentIDGeneration& old_parent,
            const EB::ParentIDGeneration& new_parent,
            const EB::ParentIDGeneration& child,
            FabricSegments edge_table,
            uint32_t max_tries,
            uint32_t internal_max_tries
        ) noexcept
        {
            return ReplaceParentRelation_(
                old_parent.Slot,
                old_parent.Generation,
                new_parent.Slot,
                new_parent.Generation,
                child.Slot,
                child.Generation,
                edge_table,
                nullptr,
                max_tries,
                internal_max_tries
            );
        }

        MutationResult AddParentByIdentity(
            const EB::ParentIDGeneration& parent,
            const EB::ParentIDGeneration& child,
            FabricSegments edge_table,
            uint32_t max_tries,
            uint32_t internal_max_tries
        ) noexcept
        {
            return AddParentRelation_(
                parent.Slot,
                parent.Generation,
                child.Slot,
                child.Generation,
                edge_table,
                nullptr,
                max_tries,
                internal_max_tries
            );
        }

        MutationResult RemoveParentByIdentity(
            const EB::ParentIDGeneration& parent,
            const EB::ParentIDGeneration& child,
            FabricSegments edge_table,
            uint32_t max_tries,
            uint32_t internal_max_tries
        ) noexcept
        {
            return RemoveParentRelation_(
                parent.Slot,
                parent.Generation,
                child.Slot,
                child.Generation,
                edge_table,
                nullptr,
                max_tries,
                internal_max_tries
            );
        }


    protected:

        struct ParentRowScan
        {
            uint32_t MatchOrdinal = EB::RELATION_NULL;
            uint32_t OtherOrdinal = EB::RELATION_NULL;
            uint32_t EmptyOrdinal = EB::RELATION_NULL;

            uint32_t Reserved = 0u;

            EB::ParentIDGeneration MatchParent{};
        };
        
        static constexpr bool SameHeader_(const EdgeBuilder::EdgeData& left, const EdgeBuilder::EdgeData& right) noexcept
        {
            return left == right;
        }

        static constexpr bool SameRelation_(const EB::ParentRelation& left, const EB::ParentRelation& right) noexcept
        {
            return left == right;
        }

        bool ScanReservedParentRow_(
            DAGMutationTransaction& transaction,
            uint32_t child_slot,
            EB::ParentIDGeneration wanted_parent_handle,
            EB::ParentIDGeneration other_parent_handle,
            ParentRowScan& scan
        ) noexcept;

        MutationResult AddParentRelation_(
            uint32_t parent_slot,
            uint32_t parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            ConditionalParentPublication* publication = nullptr,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_max_tries = DEFAULT_INTERNAL_TRIES__
        ) noexcept;

        MutationResult RemoveParentRelation_(
            uint32_t parent_slot,
            uint32_t parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            ConditionalParentPublication* publication = nullptr,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_max_tries = DEFAULT_INTERNAL_TRIES__
        ) noexcept;

        MutationResult ReplaceParentRelation_(
            uint32_t old_parent_slot,
            uint32_t old_parent_generation,
            uint32_t new_parent_slot,
            uint32_t new_parent_generation,
            uint32_t child_slot,
            uint32_t child_generation,
            FabricSegments edge_table,
            ConditionalParentPublication* publication = nullptr,
            uint32_t max_tries = DEFAULT_MAX_TRIES,
            uint32_t internal_max_tries = DEFAULT_INTERNAL_TRIES__
        ) noexcept;

    };



}