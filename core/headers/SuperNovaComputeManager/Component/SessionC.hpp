#pragma once 
#include "../Runtime/NumericalResedency.hpp"
#include "../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    class Computer 
    {
        template<class T>
        using RegionT = SNTOComputerBridge<T>;

        ConfigC Config_{};
        OptimizedPrimitivesC OptimizedPrm_{};
        Device Device_{};
        NumericResidency Residency_{};
        FabricRuntimeBinding Binding_{};
        FabricExecutionGate UnboundGate_{};
        ExclusiveLease EpochLease_{};
        bool Session_{false};
        BackendC SessionBackend_{BackendC::CPU};
        
    };
}