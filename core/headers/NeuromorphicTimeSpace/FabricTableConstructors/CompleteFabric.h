#pragma once 
#include "FabricConstructor.h"
#include <new>

namespace BidirectionalInMemGraph
{
    class RecordBookConstructor : public APCHandleAndRetirement
    {
        
    protected:
        using SD = SchemaDefinition;
        using DSA = DescriptionOfAPC;
        using RBC = RecordBookConf;

        bool GetRecordMapCarrierRanges_(
            const FabricSegments table_class,
            RecordBookConf::FabricSegmentBounds& return_bounds
        ) noexcept;

        void IdleAFabricTableClassRangesMemory_(FabricSegments table_class) noexcept;

        void WriteARecordBookOfTSCEntry_(
            FabricSegments table_class, 
            size_t begin, 
            size_t end 
        ) noexcept;

        bool CheckRecordBookRange_(FabricSegments segment, uint64_t expected_begin, uint64_t expected_end) noexcept;

    };

    class EdgeTableConstructor : public RecordBookConstructor
    {
    public:
        using EdgeTableRange = ADS::RangeOfAPC;
        using EB = EdgeBuilder;
        using EdgeTableRowView = EB::EdgeTableRowView;

    private:
        SeqLockedOperation ReserveParentDomain_(
            FabricSegments edge_table,
            uint32_t slot,
            EdgeBuilder::EdgeStatus required_status,
            EdgeBuilder::EdgeData& before,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        SeqLockedOperation ReserveChildDomain_(
            FabricSegments edge_table,
            uint32_t slot,
            EdgeBuilder::EdgeStatus required_status,
            EdgeBuilder::EdgeData& before,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

    protected:

        EdgeTableRowView EdgeTableRow_(FabricSegments edge_table, uint32_t row_slot) noexcept;
        EB::PMSpan ParentMask_(FabricSegments edge_table, uint32_t row_slot) noexcept
        {
            return EdgeTableRow_(edge_table, row_slot).Masks;
        }
        EB::PRSpan EdgeRelationsPerSlot_(FabricSegments edge_table, uint32_t row_slot) noexcept
        {
            return EdgeTableRow_(edge_table, row_slot).Relations;
        }

        bool ConstructEdgeTableBySlot_(FabricSegments edge_table, uint32_t slot) noexcept;

        EdgeTableRange ReadAnEdgeTableRange_(
            FabricSegments edge_table,
            uint32_t row_slot
        ) noexcept;

        bool InitializeEdgeTable_(FabricSegments edge_table) noexcept;

        bool ReadEdgeControl_(
            FabricSegments edge_table,
            uint32_t row_slot,
            EdgeBuilder::EdgeDomain domain,
            EdgeBuilder::EdgeData& edge
        ) noexcept;

        bool ReadChildDomainControl_(
            FabricSegments edge_table,
            uint32_t row_slot,
            EdgeBuilder::EdgeData& edge
        ) noexcept;

        SeqLockedOperation ReadParentHandle_(
            FabricSegments edge_table,
            uint32_t slot,
            uint32_t relation_ordinal,
            EB::ParentIDGeneration& parent_handle,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        SeqLockedOperation ReserveEdgeDomain_(
            FabricSegments edge_table,
            uint32_t row_slot,
            EdgeBuilder::EdgeDomain domain,
            EdgeBuilder::EdgeStatus required_status,
            EdgeBuilder::EdgeData& before,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        void StoreReservedParentHandle_(
            FabricSegments edge_table,
            uint32_t slot,
            uint32_t relation_ordinal,
            const EB::ParentIDGeneration& parent
        ) noexcept;

        void StoreReservedSiblingLocators_(
            FabricSegments edge_table,
            uint32_t slot,
            uint32_t relation_ordinal,
            const EB::SiblingLinks& sibblings
        ) noexcept;

        void PublishReservedEdgeDomain_(
            FabricSegments edge_table,
            uint32_t row_slot,
            EdgeBuilder::EdgeDomain domain,
            const EdgeBuilder::EdgeData& before,
            uint32_t desired_tail,
            EdgeBuilder::EdgeStatus desired_status
        ) noexcept;

        void PublishReservedEdgeRow_(
            FabricSegments edge_table,
            uint32_t row_slot,
            const EdgeBuilder::EdgeData& before,
            uint32_t desired_tail,
            EdgeBuilder::EdgeStatus desired_status
        ) noexcept;
    };

    
    class CompiledDAGTableConstructor : public EdgeTableConstructor
    {
    protected:
        std::atomic<uint64_t> SealedDAGRevision_{UNSIGNED_ZERO};
        
        void CompiledDAGRelation_(
            FabricSegments edge_table,
            uint32_t child_slot,
            uint8_t relation_ordinal,
            const EdgeBuilder::ParentRelation& relation
        ) noexcept;

        SeqLockedOperation ReadCompiledDAGParentMask_(
            FabricSegments edge_table,
            uint32_t slot,
            uint64_t& return_mask,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

    };


    



}