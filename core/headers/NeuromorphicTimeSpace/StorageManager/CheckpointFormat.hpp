#include "GlobalStorage.hpp"

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    enum class PageHeader : size_t 
    {
        MAGIC,
        VERSION,
        NATIVEABI,
        FABRICHIGH,
        FABRICLOW,
        REVISION,
        CELLCOUNT,
        PAGESIZE,
        PAGECOUNT,
        METADATASIZE,
        PAYLOADOFFSET,
        DIRECTORYHASH,
        TOTALFILESIZE,
        HEADERHASH,
    };

    struct InternalValues
    {
        static constexpr uint64_t FileMagic = 0x3354504b43464e53ULL; // "SNFCKPT3" on disk
        static constexpr uint64_t FileVersion = 1;
        static constexpr uint64_t BytesPerWord = sizeof(uint64_t);
        static constexpr unsigned BitsPerByte = CHAR_BIT;
        static constexpr uint64_t PageBytes = 64u * 1024u;
        static constexpr uint64_t MaximumMetadataBytes = 64u * 1024u;
        static constexpr uint64_t DefaultMaximumCells = uint64_t{10} << 27;
        static constexpr uint64_t FnvOffsetBasis = 14695981039346656037ULL;
        static constexpr uint64_t FnvPrime = 1099511628211ULL;

        static constexpr size_t HeaderWords = static_cast<size_t>(PageHeader::HEADERHASH) + 1;
        static constexpr uint64_t HeaderBytes = HeaderWords * BytesPerWord;
        static constexpr size_t HeaderDataWords = static_cast<size_t>(PageHeader::HEADERHASH);
    };

    using IV = InternalValues;
    using FileHeader = std::array<uint64_t, IV::HeaderWords>;

    static constexpr size_t FieldIndex(PageHeader field) noexcept
    {
        return static_cast<size_t>(field);
    }

    static constexpr uint64_t PageHash(std::span<const std::byte> bytes) noexcept
    {
        uint64_t hash = IV::FnvOffsetBasis;
        for (std::byte b : bytes) 
        {
            hash ^= std::to_integer<uint8_t>(b);
            hash *= IV::FnvPrime;
        }

        return hash;
    }

    static bool WritePageHeaderWords(std::ostream& output, uint64_t value) noexcept
    {
        for (size_t byte = 0; byte < sizeof(uint64_t); byte++) 
        {
            output.put(static_cast<char>(value >> (byte * sizeof(uint64_t))));
        }
        return static_cast<bool>(output);
    }


}