#pragma once 
#include "../headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph
{
    using EB = EdgeBuilder;
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::SiblingLinks>::is_always_lock_free == true);
    static_assert(sizeof(EB::ParentMaskBlock) == 8u);
    static_assert(sizeof(EB::ParentIDGeneration) == 8u);
    static_assert(sizeof(EB::SiblingLinks) == 8u);
    static_assert(sizeof(EB::ParentRelation) == 16u);

    ////
    using HAC = HandleOfAPCStatic;
    static_assert(std::atomic<HAC::LifeCycleControl>::is_always_lock_free == true);
    static_assert(std::atomic<HAC::ParentRowControl>::is_always_lock_free == true);
    static_assert(std::atomic<HAC::ChildListControl>::is_always_lock_free == true);
    static_assert(sizeof(HAC::StructuralHotRow) ==  8 * sizeof(uint64_t));
    static_assert((HandleOfAPCStatic::ACTIVE_COUNT_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u);
    static_assert((HandleOfAPCStatic::CLOSED_MASK & HandleOfAPCStatic::GENERATION_MASK) == 0u);
    static_assert(sizeof(FabricCache) == 16 * sizeof(uint64_t));


}
