#include "SessionC.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class CheckpointCompute final
    {
        SessionCompute& Runtime_;
    public:
        explicit CheckpointCompute(SessionCompute& runtime) noexcept :
            Runtime_(runtime)
        {}

        bool  SaveCehckpoint(
            const std::filesystem::path& path, 
            uint64_t revision,
            std::span<const std::byte> metadata = {}
        ) noexcept;

        bool LoadCheckpoint(
            APCFinilizer& owner,
            const std::filesystem::path& path,
            const CoreOfFabricCoordinator::FabricInitConf& storege = {}
        ) noexcept;
    };
}