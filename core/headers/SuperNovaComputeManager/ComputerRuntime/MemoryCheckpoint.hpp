#pragma once
#include "../ComputerDefinition/MemoryDefinition.hpp"

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    class FabricCheckpoint final : private InternalValues
    {
        CheckpointCache CCache_{};
        
    };
}