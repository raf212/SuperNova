#pragma once 
#include "../../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

namespace BidirectionalInMemGraph 
{
    class ModelRuntime 
    {
    protected:
        APCFinilizer* FabricPtr_;

    public:
        explicit ModelRuntime(APCFinilizer* fabric_ptr) noexcept :
            FabricPtr_(fabric_ptr)
        {}

        GMI::Dense::ModelSnapshot InspectDenseModel(uint32_t instance) noexcept;

        uint64_t DenseStructureRevision(uint32_t instance) noexcept;

        bool ReplaceModelLayerAPCs(
            uint32_t model_instance,
            uint32_t layer_position,
            std::span<const GMI::ModelAPC> ordered_assignments
        ) noexcept;
    };
}