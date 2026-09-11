#pragma once 
#include "../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "GHGFNode.hpp"


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
        GM::GHGFCache GHGFCache_{};
        GM::GHGFStorageProfile Profile_{};
        bool IsGHGFPlanCurrent_() noexcept;

        bool PredictGHGFBatch(uint32_t batch) noexcept;
        bool UpdateGHGFBatch(FCSpan observation, uint32_t batch) noexcept;
        bool CopyGHGFPrediction_(std::span<float> prediction, uint32_t batch) noexcept;

        float* GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept;
        float* GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept;
        float* GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept;

        void InvalidateGHGFModel_() noexcept;
        uint64_t GHGFParentMask_(uint32_t slot, FabricSegments axis) noexcept;
        bool GetGHGFNode_(uint32_t slot, GHGFNode& node, APCUseScope& use) noexcept;
    public :
        using APCFinilizer::ShutDownFabric;
        using APCFinilizer::IsFabricActive;

        bool RemoveParent(const GM::GHGFConnection& connection) noexcept;
        
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
        ) noexcept;
        
        bool ConstructGHGFModel(
            GHGFModelConstructionValues& model_values,
            const GHGFLayerModel::GHGFStorageProfile& profile
        ) noexcept;
            
    };
}