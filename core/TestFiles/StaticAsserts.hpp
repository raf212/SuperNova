#pragma once 
#include "../headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph
{
    using EB = EdgeBuilder;
    static_assert(EB::EDGE_TAIL_BITS + EB::EDGE_SEQUENCE_BITS + EB::EDGE_STATUS_BITS == 64u);
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::ParentIDGeneration>::is_always_lock_free == true);
    static_assert(std::atomic<EB::SiblingLinks>::is_always_lock_free == true);
    static_assert(sizeof(EB::ParentMaskBlock) == 8u);
    static_assert(sizeof(EB::ParentIDGeneration) == 8u);
    static_assert(sizeof(EB::SiblingLinks) == 8u);
    static_assert(sizeof(EB::ParentRelation) == 16u);
}
