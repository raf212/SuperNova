#pragma once 
#include "../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "../AdaptivePackedCellContainer/APCNodes/GHGFNode.hpp"


namespace BidirectionalInMemGraph
{

    class GHGFModelConstructor : private APCFinilizer
    {
        friend class GHGFNode;
    public:
        using GM = GHGFLayerModel;
        using FCSpan = std::span<const float>;
        struct GHGFModelConstructionValues
        {
            std::span<GHGFNode> APCNodes{};
            std::span<const GM::GHGFNodeRole> RoleSpan{};
            std::span<const GM::GHGFConnection> ConnectionSpan{};            
        };
    private:
        GM::GHGFCache Cache_{};
        GM::GHGFStorageProfile Profile_{};

        bool IsGHGFPlanCurrent_() noexcept;

        bool PredictGHGFBatch(uint32_t batch) noexcept;
        bool UpdateGHGFBatch(FCSpan observation, uint32_t batch) noexcept;
        bool CopyGHGFPrediction_(FCSpan prediction, uint32_t batch) noexcept;

        float* GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept;
        float* GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept;
        float* GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept;

        void InvalidateGHGFModel_() noexcept;

        uint64_t GHGFParentMask_(uint32_t slot, FabricSegments axis) noexcept;
    public :

        bool ConnectGHGFParent(const GM::GHGFConnection& connection) noexcept;
        
        bool ResetGHGFState() noexcept;

        bool CompileGHGFModel() noexcept;

        bool InitializeGHGFFabric(
            uint32_t slot_count,
            const GHGFLayerModel::GHGFStorageProfile& profile
        ) noexcept;

        bool CreateNodeOfGHGF(
            GHGFNode& desired_apc,
            GM::GHGFNodeRole role
        ) noexcept
        {
            return 
                HasDefaultRegionTable_ && 
                CreateAPC(desired_apc, DefaultRegionTable_) &&
                desired_apc.InitializeGHGFNode(role);
        }
        
        bool ConstructGHGFModel(
            GHGFModelConstructionValues& model_values,
            const GHGFLayerModel::GHGFStorageProfile& profile
        ) noexcept;
            
    };
}