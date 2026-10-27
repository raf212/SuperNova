#pragma once 
#include "../headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "../headers/SuperNovaComputeManager/MemoryCheckpoint.hpp"

namespace BidirectionalInMemGraph
{

    using SO = SchemaOrchestrator;
    static_assert(sizeof(SO::RegionSchemaRecord) == 6u * sizeof(std::uint64_t));
    static_assert(alignof(SO::RegionSchemaRecord) == alignof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<SO::RegionSchemaRecord>);
    static_assert(std::is_trivially_destructible_v<SO::RegionSchemaRecord>);


    using EB = EdgeBuilder;
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::SiblingLinks>::is_always_lock_free == true);
    static_assert(sizeof(EB::ParentMaskBlock) == 8u);
    static_assert(sizeof(EB::ParentIDGeneration) == 8u);
    static_assert(sizeof(EB::SiblingLinks) == 8u);
    static_assert(sizeof(EB::ParentRelation) == 16u);

    ////
    using HAS = HandleOfAPCStatic;
    static_assert(std::atomic<HAS::LifeCycleControl>::is_always_lock_free == true);
    static_assert(std::atomic<HAS::ParentRowControl>::is_always_lock_free == true);
    static_assert(std::atomic<HAS::ChildListControl>::is_always_lock_free == true);
    static_assert(sizeof(HAS::StructuralHotRow) ==  8 * sizeof(uint64_t));
    static_assert((HandleOfAPCStatic::ACTIVE_COUNT_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u);
    static_assert((HandleOfAPCStatic::CLOSED_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u);
    static_assert(sizeof(FabricCache) == 16 * sizeof(uint64_t));

    ///
    static_assert(sizeof(APCStorageGeometry::CacheOfSNCB<uint64_t>) == 8 * sizeof(uint64_t));

    //
    static_assert(sizeof(FabricMemory::NativeABI) == sizeof(std::uint64_t));

    //
    using CFC = CoreOfFabricCoordinator;
    static_assert(sizeof(CFC::NodeModelDescription) == sizeof(uint64_t)); // Atleast for now

}
