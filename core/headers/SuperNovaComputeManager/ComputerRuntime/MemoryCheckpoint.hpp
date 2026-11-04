#pragma once
#include "../ComputerDefinition/MemoryDefinition.hpp"


namespace BidirectionalInMemGraph::CheckpointFormat 
{
    class FabricCheckpoint;

    class VerifyReader final : private IV
    {
        const FabricCheckpoint& Image_;
        std::ifstream Input_;
        std::array<std::byte, PAGE_BYTES> Page_{};
        size_t PageLength_{};
        std::optional<size_t> CachedPage_{};
        ReadStatic* RStatics_;
        bool Valid_{false};

        bool PageLoaded_(size_t page) noexcept;
    };

    class FabricCheckpoint final : private InternalValues
    {
        friend class VerifyReader;

        CheckpointCache CCache_{};

        constexpr uint64_t HeaderValue_(PageHeader field) const noexcept
        {
            return CCache_ ? CCache_.Header_[PageHeaderIndex(field)] : UNSIGNED_ZERO;
        }

        constexpr uint64_t CellCountFromHeader() const noexcept
        {
            return HeaderValue_(PageHeader::CELL_COUNT);
        }
        
    };
}