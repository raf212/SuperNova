#pragma once 
#include "../../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "../../../SuperNovaComputeManager/Computer.hpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    struct DenseMutationConfig final
    {
        std::vector<DenseBlock> Blocks;
        std::vector<std::vector<uint64_t>> Layers;
        size_t MaxDeviceSegments{}; // required bound for the two descriptor banks
        size_t MaxModelBindings{size_t{1} << 20}; // total ModelIR fragments, including fixed buffers
    #if defined (SN_ACPP_ENABLED)
        std::shared_ptr<ComputerDetail::RuntimeAcpp> Runtime;
    #endif
    };

    struct BlockEdit final
    {
        uint64_t Block{};
        uint32_t Layer{INACTIVE_LAYER};
        uint32_t POsition{};
    };

    struct MutationRequest final
    {
        uint64_t ExpectedRevision{};
        uint64_t TargetPass{};
        uint64_t NextPass{};
        std::vector<BlockEdit> Edits;
    };

    enum class MutationStatus : uint8_t
    {
        Prepared = 0,
        Idle,
        Busy,
        Stale,
        Invalid,
        Capacity,
        TooEarly,
        WaitingForDevice,
        Published,
        Expired,
        DeviceFailure
    };

    
}