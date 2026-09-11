#pragma once 
#include "../AdaptivePackedCellContainer.hpp"

namespace BidirectionalInMemGraph
{
    class GHGFModelConstructor;

    class GHGFNode : protected AdaptivePackedCellContainer
    {
        friend class GHGFModelConstructor;
    private:
        using GM = GHGFLayerModel;


        GHGFModelConstructor* GHGFFabric_{nullptr};

        std::optional<GM::GHGFNodeRole> GHGFRole_() noexcept;

        std::optional<GM::GHGFNodeRole> IsLiveGHGFSlot_() noexcept
        {
            APCUseScope apc_use = AcquireAPCUse_();
            return apc_use ?  GHGFRole_() : std::nullopt;
        }

        bool PredictGHGFNode_(uint32_t batch) noexcept;
        bool UpdateGHGFNode_(uint32_t batch) noexcept;
        bool PropogateGHGFError_(uint32_t child, uint32_t batch) noexcept;

        void ResetAPCGHGFStateRegion_() noexcept;
    public:
        bool InitializeGHGFNode(
            GM::GHGFNodeRole role
        ) noexcept;

    };
    
}