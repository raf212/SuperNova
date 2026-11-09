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
        ReadStatistics* RStatics_{};
        bool Valid_{false};

        bool PageLoaded_(size_t page) noexcept;
    public:
        VerifyReader(const FabricCheckpoint& image, ReadStatistics* statics = nullptr);

        bool Read(uint64_t offset, std::span<std::byte> output) noexcept;
    };

    class FabricCheckpoint final : private InternalValues
    {
        friend class VerifyReader;

        std::unique_ptr<CheckpointCache> CCache_{};

        uint64_t HeaderValue_(PageHeader field) const noexcept
        {
            return CCache_ ? CCache_->Header_[PageHeaderIndex(field)] : UNSIGNED_ZERO;
        }

        bool ReadBytes_(uint64_t offset, std::span<std::byte> output) const noexcept;

        bool ValidateRegion_(const RegionRef& ref, VerifyReader& reader) const noexcept;

    public:
        uint64_t Revision() const noexcept {return HeaderValue_(PageHeader::REVISION);}
        uint64_t Fingerprint() const noexcept {return HeaderValue_(PageHeader::HEADER_HASH);}

        bool Open(
            const std::filesystem::path& path,
            uint64_t maximum_cells = DEFAULT_STORAGE_LIMIT
        ) noexcept;


        uint64_t CellCountFromHeader() const noexcept
        {
            return HeaderValue_(PageHeader::CELL_COUNT);
        }

        FabricId Identity() const noexcept
        {
            return {
                HeaderValue_(PageHeader::FABRIC_HIGH),
                HeaderValue_(PageHeader::FABRIC_LOW)
            };
        }

         bool FabSlabCache(CFC::FabricCache& cache) const noexcept
        {
            if (CCache_) 
            {
                cache = CCache_->SlabCache_;
                return true;
            }
            cache = CFC::FabricCache{};
            return false;
        }

        std::span<const std::byte> MetaData() const noexcept
        {
            return CCache_ ? std::span<const std::byte>{
                CCache_->MetaData_.data(),
                static_cast<size_t>(HeaderValue_(PageHeader::METADATA_SIZE))
            } : std::span<const std::byte>{};
        }

        bool ReadCells(uint64_t first, std::span<uint64_t> cells) const noexcept;

        bool ReadSlab(std::vector<uint64_t>& slab) const noexcept;

        bool Validate(const RegionRef r_ref) const noexcept;

        bool ReadExtent(
            const RegionRef& ref,
            uint64_t relative,
            std::span<std::byte> destination,
            ReadStatistics* read_static = nullptr
        ) const noexcept;
    };

}