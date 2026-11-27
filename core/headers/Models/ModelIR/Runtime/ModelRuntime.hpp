#pragma once 
#include "DenseMutation.hpp"

namespace BidirectionalInMemGraph 
{
    class ModelRuntime 
    {
    protected:
        APCFinilizer* FabricPtr_;
        // Computer Computer_; we will add it in future
        
    public:
        explicit ModelRuntime(APCFinilizer* fabric_ptr) noexcept :
            FabricPtr_(fabric_ptr)
        {}

        GMI::Dense::ModelSnapshot InspectDenseModel(uint32_t instance);

        uint64_t DenseStructureRevision(uint32_t instance);

    };
}