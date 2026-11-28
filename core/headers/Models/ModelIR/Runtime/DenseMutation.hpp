#pragma once 
#include "../../../NeuromorphicTimeSpace/APCFinilizer.hpp"
#include "../../../SuperNovaComputeManager/Computer.hpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    using Vect2DU64 = std::vector<std::vector<uint64_t>>;
    using MdlSnapCPtr = std::shared_ptr<const ModelSnapshot>;
    
    struct DenseMutationConfig final
    {
        std::vector<DenseBlock> Blocks;
        Vect2DU64 Layers;
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
        uint32_t Position{};
    };

    struct MutationRequest final
    {
        uint64_t ExpectedRevision{};
        uint64_t TargetPass{};
        uint64_t NextPass{}; // Inclusive publication deadline.
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

    // Process-independent data. Retain the construction model and configuration
    // with this image. Physical addresses are resolved again on restore.
    struct DenseMutationImage final
    {
        static constexpr uint32_t FORMAT_VERSION = 2u;

        struct RegionImage final
        {
            ModelAPC APC;
            MacroColumnOfAPC Region{};
            std::vector<std::byte> Bytes{};
        };

        uint32_t Version{FORMAT_VERSION};
        uint32_t Instance{};
        uint64_t LayoutSignature{};
        uint64_t StructureRevision{};
        uint64_t ParameterEpoch{};
        uint64_t LastPass{};
        Vect2DU64 Layers;
        std::vector<RegionImage> Regions;
    };

    class DenseMutationEngine final
    {
        struct StateDM;

        std::unique_ptr<StateDM> State_;

        DenseMutationImage Capture_(bool finalize);

    public:
        // Initial regions, including reserves, must be initialized before opening a session.
        // Open during Fabric maintenance. The session then
        // owns numerical access for its model.
        DenseMutationEngine(
            std::shared_ptr<APCFinilizer> fabric,
            uint32_t instance,
            DenseMutationConfig config,
            const DenseMutationImage* restore = nullptr
        );

        ~DenseMutationEngine();

        bool UseGPU() const noexcept;

        MdlSnapCPtr Model() const noexcept;

        static MdlSnapCPtr Published(APCFinilizer& fabric, uint32_t instance);

        // Pass IDs come only from successful admissions to the execution interface.
        // Advance attempts the boundary before the next admission.
        uint64_t LastPass() const;

        MutationStatus Prepare(MutationRequest request);

        MutationStatus Advance();

        bool Cancel();

        // Callbacks must use borrowed views only for the indicated pass.
        // Callbacks may not reenter the engine. CPU callbacks run outside the structural lock.
        // GPU callbacks enqueue on the supplied in-order queue and return the actual final event
        // covering every numerical and descriptor access. 
        // External work must be joined to that queue before returning.
        using CPUWork = std::function<void(const DensePass&)>;
        PassStatus ExecuteCPU(uint64_t revision, PassAccess access, const CPUWork& work);

        DenseMutationImage Capture();

        // Capture authoritative state, synchronize GPU changes to the host regions and close admission.
        // Persist the returned image with the construction config before discarding it.
        // A changed structure must be reopened from an image and cannot fall back to installed ModelIR.
        DenseMutationImage Finalize();

    #if defined (SN_ACPP_ENABLED)
        using DevicePlan = ComputerDetail::DevicePlanView<DenseSegment>;
        using GPUWork = std::function<sycl::event(
            sycl::queue&, const DensePass&, DevicePlan
        )>;

        PassStatus SubmitGPU(uint64_t revision, PassAccess access, const GPUWork& work);
    #endif
    };

}