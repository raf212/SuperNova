#include <SuperNova>

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    bool VerifyReader::PageLoaded_(size_t page) noexcept
    {
        if (CachedPage_.has_value() && CachedPage_ == page) return true;
        const uint64_t slab_bytes = Image_.CellCountFromHeader() * BYTES_PER_CELL;
        if (
            page >= MAX_PAGE_COUNT ||
            page >= PageCount(slab_bytes)
        ) 
        {
            return false;
        }
        const uint64_t begin = page * PAGE_BYTES;
        const uint64_t len = static_cast<size_t>(std::min(PAGE_BYTES, (slab_bytes - begin)));

        PageLength_ = len;
        Input_.seekg(static_cast<std::streamoff>(
            Image_.HeaderValue_(PageHeader::PAYLOAD_OFFSET) + begin
        ));
        if (!Input_.read(
            reinterpret_cast<char*>(Page_.data()), 
            static_cast<std::streamsize>(len)
        )) 
        {
            return false;
        }

        auto HashPageMatches___ = [&]() noexcept -> bool
        {
            return PageHash({Page_.data(), PageLength_}) != Image_.CCache_->Hashes_[page];
        };
        if (!HashPageMatches___()) return false;
        if (RStatics_) ++RStatics_->PageReads;
        CachedPage_ = page;
        return true;
    }

    VerifyReader::VerifyReader(const FabricCheckpoint& image, ReadStatic* statics) :
        Image_(image), 
        Input_(
            image.CCache_ ? image.CCache_->Path_ : std::filesystem::path{},
            std::ios::binary
        ),
        RStatics_(statics)
    {
        if (RStatics_) ++RStatics_->FileOpens;
        if (
            !Image_.CCache_ ||
            !Input_ ||
            Image_.CellCountFromHeader() > LIMIT_OF_CELL_COUNT ||
            PageCount(Image_.CellCountFromHeader() * BYTES_PER_CELL) > MAX_PAGE_COUNT
        ) 
        {
            return;
        }
        if (RStatics_) ++RStatics_->HeaderChecks;

        for (uint64_t expected : Image_.CCache_->Header_) 
        {
            uint64_t actual{};
            if (!ReadWord(Input_, actual) || actual != expected) 
            {
                return;
            }
        }
        Valid_ = true;
    }

    bool VerifyReader::Read(uint64_t offset, std::span<std::byte> output) noexcept
    {
        if (!Valid_) return false;
        const uint64_t slab_bytes = Image_.CellCountFromHeader() * BYTES_PER_CELL;
        if (
            offset > slab_bytes ||
            output.size() > slab_bytes - offset
        ) 
        {
            return false;
        }

        size_t copied = 0;
        while (copied < output.size()) 
        {
            const uint64_t absolute = offset + copied;
            const uint64_t page = static_cast<size_t>(absolute / PAGE_BYTES);
            if (!PageLoaded_(page)) return false;

            const size_t within = static_cast<size_t>(absolute % PAGE_BYTES);
            if (within >= PageLength_) return false;

            const size_t count = std::min(
                (PageLength_ - within), (output.size() - copied) 
            );
            std::copy_n((Page_.data() + within), count, (output.data() + copied));
        }
        return true;
    }


    bool FabricCheckpoint::Open(
        const std::filesystem::path& path,
        uint64_t maximum_cells
    ) noexcept
    {
        try 
        {
            FabricCheckpoint candidate;
            candidate.CCache_ = std::make_unique<CheckpointCache>();
            candidate.CCache_->Path_ = path;
            std::ifstream input(path, std::ios::binary);
            if (!input) return false;

            for (uint64_t& v : candidate.CCache_->Header_) 
            {
                if (!ReadWord(input, v)) return false;
            }

            const FileHeader& header = candidate.CCache_->Header_;

            const uint64_t cell_count = header[PageHeaderIndex(PageHeader::CELL_COUNT)];
            const uint64_t meta_size = header[PageHeaderIndex(PageHeader::METADATA_SIZE)];
            const uint64_t page_count = header[PageHeaderIndex(PageHeader::PAGE_COUNT)];
            
            std::optional<uint64_t> total_bytes = VerifyPageHeader(header, maximum_cells);
            if (
                !total_bytes.has_value() || 
                std::filesystem::file_size(path) != total_bytes.value()
            )
            {
                return false;
            } 

            if (!candidate.Identity()) return false;
            uint64_t directory_hash = FnvOffsetBasis;

            for (size_t p = 0; p < page_count; p++) 
            {
                uint64_t& hash = candidate.CCache_->Hashes_[p];
                if (!ReadWord(input, hash)) return false;
                directory_hash = PageHash(EncodeCell(hash), directory_hash);
            }

            auto meta_data = std::span{
                candidate.CCache_->MetaData_.data(),
                static_cast<size_t>(meta_size)
            };
            
            if (!input.read(
                reinterpret_cast<char*>(meta_data.data()),
                static_cast<std::streamsize>(meta_data.size())
            )) 
            {
                return false;
            }

            directory_hash = PageHash(meta_data, directory_hash);
            if (directory_hash != header[PageHeaderIndex(PageHeader::DIRECTORY_HASH)]) 
            {
                return false;
            }

            CheckpointCache& cache = *candidate.CCache_;
            cache.Id_ = candidate.Identity();
            cache.Cells_ = cell_count;
            cache.Revision_ = header[PageHeaderIndex(PageHeader::REVISION)];
            cache.Payload_ = header[PageHeaderIndex(PageHeader::PAYLOAD_OFFSET)];
            cache.MetaBytes_ = meta_size;
            cache.TotalBytes_ = *total_bytes;

            VerifyReader reader(candidate);
            if (!reader.Read(UNSIGNED_ZERO, std::as_writable_bytes(
                std::span{&cache.SlabCache_, size_t{1}}
            ))) 
            {
                return false;
            }

            if (
                !StorageAllocationDef::ValidateFabricCache(cache.SlabCache_) ||
                cache.SlabCache_.SlabCellCount_ != cell_count
            ) 
            {
                return false;
            }

            CCache_ = std::move(candidate.CCache_);
            return true;
        } catch (...) {
            return false;
        }
    }
}