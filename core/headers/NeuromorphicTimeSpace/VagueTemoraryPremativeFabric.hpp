#pragma once 
#include "SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

class VagueTemoraryPremativeFabric : public APCFinilizer
{
    friend class AdaptivePackedCellContainer;
protected:
    /// @brief IN FUTURE EITHER GET RID OF THE TABLE OR: STORE INSIDE FABRICE BY USING  WildCardOfPackedCell::RAW_30x2BIT
    std::unique_ptr<std::atomic<AdaptivePackedCellContainer*>[]> APCRuntimePtrTable_{nullptr};

    bool BuildAPCRuntimePtrTable_() noexcept;

    void ClearAPCRuntimePtrTable_() noexcept;

    bool StoreAPCRuntimePtr(size_t apc_idx, AdaptivePackedCellContainer* apc_ptr) noexcept;

    AdaptivePackedCellContainer* GetAPCRuntimePtrBySlotIndex_(size_t apc_idx) noexcept;

    std::optional<uint32_t> GetASlotForNewAPCLink() noexcept;        

    SeqLockedOperation ResolveChildLocator_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t locator,
        AdaptivePackedCellContainer*& child
    ) noexcept;

    FabricToAPCLinker::RelationOparation FindParent_(
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        uint8_t relation_ordinal,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    FabricToAPCLinker::RelationOparation FindFirstChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    FabricToAPCLinker::RelationOparation FindLastChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    FabricToAPCLinker::RelationOparation FindNextChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    FabricToAPCLinker::RelationOparation FindPreviousChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

public:

    bool InitializeFabricWithPtrTable(
        uint32_t slot_count,
        uint32_t slot_cell_count,
        const SchemaDefinition::FabricRegionConfig& region_configuration,
        uint8_t max_direct_parents_per_axis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS
    ) noexcept;

    void ShutDownFabricWithPtrTable() noexcept;

    bool CreateAPC(
        AdaptivePackedCellContainer& desired_apc,
        const SD::RegionSchemaTable& region_schemas,
        uint32_t internal_max_tries = DEFAULT_MAX_TRIES,
        bool override_table = false
    ) noexcept;
    
};


}