#pragma once 
#include "SlabToFabricConverterAndCordinator.h"

namespace BidirectionalInMemGraph
{

class VagueTemoraryPremativeFabric : public APCFinilizer
{
    friend class AdaptivePackedCellContainer;
protected:

    bool GetExistingAPC_(
        uint32_t slot,
        AdaptivePackedCellContainer& apc,
        APCUseScope& apc_use,
        std::optional<uint32_t> expected_generation = std::nullopt
    ) noexcept;

    std::optional<uint32_t> GetASlotForNewAPCLink() noexcept;        

    SeqLockedOperation ResolveChildLocator_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t locator,
        APCUseScope& use,
        AdaptivePackedCellContainer& child
    ) noexcept;

    AdaptivePackedCellContainer FindParent_(
        uint32_t child_slot,
        uint32_t child_generation,
        FabricSegments edge_table,
        uint8_t relation_ordinal,
        FabricToAPCLinker::RelationOparation* result_ptr = nullptr,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    AdaptivePackedCellContainer FindFirstChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        FabricToAPCLinker::RelationOparation* result_ptr = nullptr,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    AdaptivePackedCellContainer FindLastChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        FabricToAPCLinker::RelationOparation* result_ptr = nullptr,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    AdaptivePackedCellContainer FindNextChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        FabricToAPCLinker::RelationOparation* result_ptr = nullptr,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

    AdaptivePackedCellContainer FindPreviousChild_(
        uint32_t parent_slot,
        uint32_t parent_generation,
        FabricSegments edge_table,
        uint32_t current_relation_locator,
        FabricToAPCLinker::RelationOparation* result_ptr = nullptr,
        uint32_t max_tries = DEFAULT_MAX_TRIES
    ) noexcept;

public:


    void ShutDownFabricWithPtrTable() noexcept;

    bool CreateAPC(
        AdaptivePackedCellContainer& desired_apc,
        const SD::RegionSchemaTable& region_schemas,
        uint32_t internal_max_tries = DEFAULT_MAX_TRIES,
        bool override_table = false
    ) noexcept;
    
};


}