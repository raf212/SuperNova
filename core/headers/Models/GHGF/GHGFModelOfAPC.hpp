#pragma once 
#include "../../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "GHGFNode.hpp"


namespace BidirectionalInMemGraph
{

    class GHGFModel : protected APCFinilizer
    {
        friend class GHGFNode;
    public:
        using GM = GHGFLayerModel;
        using ED = EdgeBuilder;
        using FCSpan = std::span<const float>;
        struct GHGFModelConstructionValues
        {
            std::span<GHGFNode> APCNodes{};
            std::span<const GM::GHGFNodeRole> RoleSpan{};
            std::span<const GM::GHGFConnection> ConnectionSpan{};           
            bool StructuralLearningActive = false;
        };
    protected:
        GM::GHGFCache GHGFCache_{};
        GM::GHGFStorageProfile Profile_{};
        bool IsGHGFModelReady_() noexcept;
        float* GHGFRegion_(uint32_t slot, uint32_t cell_offset) noexcept;
        float* GHGFStateRow_(uint32_t slot, GM::GHGFStateRow row) noexcept;
        float* GHGFErrorRow_(uint32_t slot, GM::GHGFErrorRow row) noexcept;
        float* GHGFWeight_(uint32_t slot) noexcept;
        float* FFRowGHGF_(uint32_t slot, GM::GHGFMessageFForward row) noexcept;
        float* FBRowGHGF_(uint32_t slot, GM::GHGFMessageFBackward row) noexcept;
        bool GetGHGFNode_(uint32_t slot, GHGFNode& node, APCUseScope& use) noexcept;
        void InvalidateGHGFModel_() noexcept;
        uint64_t GHGFParentMask_(uint32_t slot, FabricSegments axis) noexcept;
        std::optional<float> GetGHGFParameter_(uint32_t slot, uint32_t index) noexcept;
        bool SetGHGFParameter_(uint32_t slot, uint32_t index, float value) noexcept;
        FabricToAPCLinker::SeqLockedOperation ReadGHGFParentExecutionSnapshot_(
            uint32_t child,
            FabricSegments edge,
            GM::GHGFParentExecutionSnapshot& snapshot,
            uint32_t max_tries
        ) noexcept;
        bool ReadGHGFNodeIdentity_(
            uint32_t slot,
            GM::GHGFNodeRole& role,
            uint32_t& generation
        ) noexcept;
    public:
        using APCFinilizer::ShutDownFabric;
        using APCFinilizer::IsFabricActive;

        bool ConnectGHGFParent(const GM::GHGFConnection& connection) noexcept;
        bool RemoveParent(const GM::GHGFConnection& connection) noexcept;
        bool InitializeGHGFFabric(
            uint32_t slot_count,
            const GHGFLayerModel::GHGFStorageProfile& profile
        ) noexcept;
        bool ResetGHGFState() noexcept;
        bool CreateNodeOfGHGF(
            GHGFNode& desired_apc,
            GM::GHGFNodeRole role
        ) noexcept;


    };

    class GHGFStructralLearningModel : public GHGFModel
    {
    private:
        struct CouplingPublicationContext final
        {
            GHGFStructralLearningModel* Model = nullptr;
            uint32_t Child = GM::StorageConst::INVALID_SLOT;
            FabricSegments Edge = FabricSegments::VALUE_PARENT_EDGE_TABLE_H;
            float Coupling = GM::StorageConst::ZERO;
        };

        static void PublishCoupling_(void* context, uint8_t ordinal) noexcept;

    public:
        GM::GHGFConcurrentOperation ReadStructureSnapshotConcurrently(
            uint32_t child,
            FabricSegments edge,
            GM::GHGFStructureSnapshot& snapshot,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;

        GM::GHGFStructureMutationResult TryApplyGHGFStructureMutation(
            const GM::GHGFStructureMutation& mutation,
            uint32_t max_tries = DEFAULT_MAX_TRIES
        ) noexcept;
    };

    class GHGFModelConstructor : public GHGFStructralLearningModel
    {
        friend class GHGFNode;
    private:
        bool PredictGHGFBatchNONVectorized_(uint32_t batch) noexcept;
        bool CopyGHGFPredictionNONVectorized_(std::span<float> prediction, uint32_t batch) noexcept;
        bool UpdateGHGFBatchNONVectorized_(FCSpan observatuins, uint32_t batch) noexcept;
        bool LearnGHGFBatchNONVectorized_(
            uint32_t batch,
            const GHGFLearningConfig& learning_conf
        ) noexcept;
        bool SealGHGFModel_() noexcept;

    public :
        bool PredictModelNONVectorized(uint32_t batch, std::span<float> prediction)noexcept;
        bool UpdateModelNONVectorized(uint32_t batch, FCSpan observations)noexcept;

        std::optional<double> RunGHGFSequence(
            FCSpan observations,
            uint32_t time_count,
            uint32_t batch_count,
            std::span<float> predictios,
            bool reset_state = true
        ) noexcept;

        bool ConstructGHGFModel(
            GHGFModelConstructionValues& model_values,
            const GHGFLayerModel::GHGFStorageProfile& profile
        ) noexcept;

        std::optional<double> FitGHGFParameters(
            FCSpan observations, 
            uint32_t time_count, 
            uint32_t batch_count,
            std::span<const GM::GHGFParameterRange> parameters, 
            uint32_t passes
        ) noexcept;

        bool TrainModelNONVectorized(
            uint32_t batch,
            FCSpan observations,
            const GHGFLearningConfig& learning_cong
        ) noexcept;
    };

}