#include <SuperNova>

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    bool VerifyReader::PageLoaded_(size_t page) noexcept
    {
        if (CachedPage_.has_value() && CachedPage_ == page) return true;

        const uint64_t begin = page * PAGE_BYTES;
        const uint64_t slab_bytes = Image_.CellCountFromHeader();
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
            return PageHash(Page_, PageLength_) != Image_.CCache_.Hashes_[page];
        };
        if (!HashPageMatches___()) return false;
        if (RStatics_) ++RStatics_->PageReads;
        CachedPage_ = page;
        return true;
    }

    VerifyReader::VerifyReader(const FabricCheckpoint& image, ReadStatic* statics) :
        Image_(image), 
        Input_(
            image.CCache_ ? image.CCache_.Path_ : std::filesystem::path{},
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

        for (uint64_t expected : Image_.CCache_.Header_) 
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
            const size_t count = std::min(
                (PageLength_ - within), (output.size() - copied) 
            );

            std::copy_n((Page_.data() + within), count, (output.data() + copied));
        }
        return true;
    }



}