#pragma once 
#include "DenseMutation.hpp"

namespace BidirectionalInMemGraph 
{
    class ModelRuntime 
    {
        using ModelSnapshot = GMI::Dense::ModelSnapshot;
        using MdlSnapCPtr = GMI::Dense::MdlSnapCPtr;

        friend class GMI::Dense::DenseMutationEngine;

        ModelSnapshot InspectInstalledDenseModel_(uint32_t instance);
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