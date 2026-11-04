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

}