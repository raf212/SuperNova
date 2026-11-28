#pragma once

#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense
{
    namespace DMEState
    {
        using RegionKey = std::pair<uint32_t, MacroColumnOfAPC>;

        using ColumnIndex = std::unordered_map<uint64_t, uint32_t>;

        inline void Require(bool valid, const char* reason)
        {
            if (!valid) throw std::invalid_argument(reason);
        }

        inline void Require(bool valid, const std::string reason)
        {
            if (!valid) throw std::invalid_argument(reason);
        }

        struct CapacityError {};

        struct Revision final
        {
            ModelSnapshot SnapShot;
            Vect2DU64 Blocks;
            std::vector<DenseSegment> Segments;
        };

        struct StorageRegion final
        {
            std::span<std::byte> Host;
            size_t DeviceOffset{};
        };

        struct ColumnRun final
        {
            uint32_t Logical;
            uint32_t Physical;
            uint32_t Count;
        };


        using FabricSPtr = std::shared_ptr<APCFinilizer>;

        using CRRev = std::shared_ptr<const Revision>;

        struct RevisionRootAtomic final
        {
            std::atomic<CRRev> Active;
        };

        struct StateSPtrs final
        {
            std::shared_ptr<DMEState::RevisionRootAtomic> Root{std::make_shared<DMEState::RevisionRootAtomic>()};
            FabricSPtr Fabric;
            CRRev Retired;

            StateSPtrs(FabricSPtr fabric) :
                Fabric(std::move(fabric))
            {}
        };

        struct StateVectors final
        {
            std::vector<LayerDescription> Capacities;
            std::vector<uint64_t> Input;
            std::vector<uint32_t> Claimed;
            std::vector<StorageRegion> Storage;
            std::vector<DenseResidentRegion> Regions;
        };

        struct StateMaps final
        {
            std::unordered_map<uint64_t, size_t> BlockIndex;
            std::unordered_map<const BlockPlacement*, ColumnIndex> ColumnIndices;
            std::map<RegionKey, size_t> RegionIndex;
        };

        struct StateMutexes final
        {
            std::mutex Mutex;
            std::mutex Planner;
            std::shared_mutex Numerical;
        };

        struct Pending
        {
            CRRev Next;
            uint64_t Target;
            uint64_t Deadline;
        };

        struct Cache
        {
            uint32_t Instance;
            ModelSnapshot Installed;
            DenseMutationConfig Config;
            uint64_t LayoutSignature{};
            size_t FixedBindings{};
            uint64_t Clock{};
            uint64_t ParameterEpoch{};
            bool CPUWriter{};
            bool Failed{};
            bool Registered{};
            bool ReOpening{};
            bool Restored{};
            std::optional<Pending> Plan{};

            Cache(uint32_t instance):
                Instance(instance)
            {}
        };

    #if defined (SN_ACPP_ENABLED)
        struct StateGPUCache final
        {
            std::shared_ptr<ComputerDetail::RuntimeAcpp> GPU;
            std::unique_ptr<ComputerDetail::ResidentDeviceRegion<std::byte>> DeviceStorage;
            std::unique_ptr<ComputerDetail::DevicePlanDoubleBuffer<DenseSegment>> DevicePlans;
            std::optional<sycl::event> LastUse;
        };
    #endif
    }

    using DMEState::CapacityError;
    using DMEState::ColumnRun;
    using DMEState::RegionKey;
    using DMEState::Require;
    using DMEState::Revision;
    using DMEState::StorageRegion;
    using TwinU32Umap = std::unordered_map<uint32_t, uint32_t>;

    struct DenseMutationEngine::StateDM
    {
        using CRRev = DMEState::CRRev;
    #if defined (SN_ACPP_ENABLED)
        DMEState::StateGPUCache GPUCache_{};
    #endif

        DMEState::StateSPtrs SharedPtrs_;

        DMEState::StateVectors Vectors_{};

        DMEState::StateMaps Maps_{};

        DMEState::Cache Cache_;

        DMEState::StateMutexes Mutexes_{};

        StateDM(DMEState::FabricSPtr fabric, uint32_t instance) :
            SharedPtrs_(std::move(fabric)), Cache_(instance)
        {}

        ~StateDM();

        void Release(bool finalize) noexcept;
        const DenseBlock& Block(uint64_t id) const;
        const BlockPlacement& Placement(const DenseBlock& block, uint32_t layer) const;
        
        size_t ValidateTile(
            const DenseBlock& block,
            const APCMatrixTile& tile,
            size_t columns,
            SD::DataTypeOfMacroColumn dtype
        ) const;

        void AppendTile(
            Revision& result,
            const DenseBlock& block,
            uint32_t tensor,
            size_t physical_columns,
            uint32_t row_base,
            std::span<const ColumnRun> columns,
            std::span<const APCMatrixTile> tiles,
            uint32_t width,
            const TwinU32Umap& assignments,
            size_t& remaining_bindings
        ) const;

        CRRev Build(Vect2DU64 selection, uint64_t number) const;
        void ValidateCatalog();
        uint64_t Signature() const;
        Vect2DU64 Apply(const Revision& current, std::span<const BlockEdit> edits) const;
        void RequireHealthy();
        void DropPlan();
        MutationStatus AdvanceLocked();
        void Open(DenseMutationConfig config, const DenseMutationImage* restore);
        DenseMutationImage CaptureImage(const Revision& active);
        void ValidatePass(bool has_work, PassAccess access) const;
        void RestoreHostRegions(const DenseMutationImage& image) const noexcept;

    private:
        void RegisterSession_(APCFinilizer& owner, bool restoring);
        void ResolveProvisionedRegions_(APCFinilizer& owner);
        void ResolveAPCRegions_(APCFinilizer& owner, const ModelAPC& apc, uint32_t model);
        void ClaimProvisionedSlots_(APCFinilizer& owner);
        CRRev BuildInitialRevision_(const Vect2DU64& layers, uint64_t revision) const;
        void ValidateInitialLayout_(const ModelSnapshot& original) const;
        void ClearProjectedBindings_();
        void ValidateImage_(const DenseMutationImage& image) const;
    #if defined(SN_ACPP_ENABLED)
        void SelectDevice_();
        void PrepareDeviceStorage_(const DenseMutationImage* restore);
        void PublishInitialDevicePlan_(const Revision& revision);
        void DownloadRegions_(DenseMutationImage& image);
    #endif

    };

}
