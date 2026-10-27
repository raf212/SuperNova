#pragma once 
#include "../AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    struct alignas(uint64_t) ModelInfo final
    {
        uint32_t Instance{};
        PreBuildModels Model{PreBuildModels::NONE};
        uint64_t ConfByteOffset{};
        uint64_t ConfByteCount{};
    private:
        uint8_t Reserved[sizeof(uint64_t) - sizeof(PreBuildModels)] {};
    };

    struct alignas(uint64_t) ModelDependency final
    {
        uint32_t ProducerSlot{};
        uint32_t ConsumerSlot{};
        MacroColumnOfAPC ProducerRegion{};
        MacroColumnOfAPC ConsumerRegion{};
        uint16_t Interface{};
        uint32_t Flags{};
    };

    struct alignas(uint64_t) NodeModelDescription final
    {
        uint32_t Instance{};
        uint32_t Role{};
    };

    struct alignas(uint64_t) ModelStorageDiaphragm final
    {
        uint64_t GlobalTableBeginIdx_{UNSIGNED_ZERO};
        uint64_t LocalDirectoryBeginIdx_{UNSIGNED_ZERO};
        uint64_t LocalDirectoryCellCount_{UNSIGNED_ZERO};
        uint32_t ModelCount_{UNSIGNED_ZERO};
        uint32_t ModelDependencyCount_{UNSIGNED_ZERO};
        uint64_t ModelDynamicsByteCount_{UNSIGNED_ZERO};
    };

}