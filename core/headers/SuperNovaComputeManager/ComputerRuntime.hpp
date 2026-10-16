#pragma once
#include "ComputerDefAndVect.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    using CDV = ComputerDefAndVect;
    using ComputerConf = CDV::ComputerConf;
    using ComputeExecutionPolicy = CDV::ComputeExecutionPolicy;
    using ComputerBackend = CDV::ComputerBackend;
    using ComputerWorkType = CDV::ComputerWorkType;

    static constexpr uint64_t MAX_SLAB_CELLS = ADS::MAX_SLAB_CELLS;

}