#pragma once
#include "../ComputerDefinition/CheckpointAccessDefinition.hpp"


namespace BidirectionalInMemGraph::CheckpointFormat 
{
    class FabricCheckpoint;

    class VerifyReader final : private IV
    {
    private:
        const FabricCheckpoint& Image_;
        std::ifstream Input_;
        std::array<std::byte, PAGE_BYTES> Page_{};
        size_t PageLength_{};
        std::optional<size_t> CachedPage_{};
        ReadStatic* RStatics_{};
        bool Valid_{false};

        bool PageLoaded_(size_t page) noexcept;
    public:
        VerifyReader(const FabricCheckpoint& image, ReadStatic* statics = nullptr);

        bool Read(uint64_t offset, std::span<std::byte> output) noexcept;
    };

    class FabricCheckpoint final : private InternalValues
    {
        friend class VerifyReader;

        std::unique_ptr<CheckpointCache> CCache_{};

        constexpr uint64_t HeaderValue_(PageHeader field) const noexcept
        {
            return CCache_ ? CCache_->Header_[PageHeaderIndex(field)] : UNSIGNED_ZERO;
        }

    public:

        bool Open(
            const std::filesystem::path& path,
            uint64_t maximum_cells = DEFAULT_STORAGE_LIMIT
        ) noexcept;


        constexpr uint64_t CellCountFromHeader() const noexcept
        {
            return HeaderValue_(PageHeader::CELL_COUNT);
        }

        constexpr FabricId Identity() const noexcept
        {
            return {
                HeaderValue_(PageHeader::FABRICLOW),
                HeaderValue_(PageHeader::FABRICLOW)
            };
        }

        constexpr  bool FabSlabCache(CFC::FabricCache& cache) const noexcept
        {
            if (CCache_) 
            {
                cache = CCache_->SlabCache_;
                return true;
            }
            cache = CFC::FabricCache{};
            return false;
        }

        constexpr std::span<const std::byte> MetaData() const noexcept
        {
            return CCache_ ? std::span<const std::byte>{
                CCache_->MetaData_.data(),
                static_cast<size_t>(HeaderValue_(PageHeader::METADATA_SIZE))
            } : std::span<const std::byte>{};
        }

        
    };

}