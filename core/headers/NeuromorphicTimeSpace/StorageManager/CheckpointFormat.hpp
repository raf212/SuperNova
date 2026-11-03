#pragma once
#include <istream>
#include <ostream>

#include "../FabricOrchestrators/HandleAndRetirement.hpp"

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
        for (unsigned byte = 0; byte < sizeof(uint64_t); byte++) 
        {
            output.put(static_cast<char>(value >> (byte * sizeof(uint64_t))));
        }
        return static_cast<bool>(output);
    }

    static  bool ReadWord(std::istream& input, uint64_t& value) noexcept
    {
        value = UNSIGNED_ZERO;
        for (unsigned b = 0; b < sizeof(uint64_t); b++) 
        {
            const int encoded = input.get();
            if (encoded == std::char_traits<char>::eof()) 
            {
                return false;
            }

            value |= uint64_t{static_cast<unsigned char>(encoded)} << (b * sizeof(uint64_t));
        }
        return true;
    }

    static constexpr uint64_t HedaderHash(const FileHeader& header) noexcept
    {
        std::array<std::byte, IV::HeaderDataWords * sizeof(uint64_t)> encoded{};
        for (size_t word = 0; word < IV::HeaderDataWords; word++) 
        {
            for (size_t b = 0; b < sizeof(uint64_t); b++) 
            {
                encoded[word * IV::BytesPerWord + b] = static_cast<std::byte>(
                    header[word] >> (b * sizeof(uint64_t))
                );
            }
        }
        return PageHash(encoded);
    }

    static constexpr uint64_t PageCount(uint64_t bytes) noexcept
    {
        return bytes == UNSIGNED_ZERO ? UNSIGNED_ZERO : (1 + ((bytes - 1) / IV::PageBytes));
    }
    
    struct FabricId
    {
        uint64_t High{};
        uint64_t Low{};
        friend constexpr bool operator == (const FabricId&, const FabricId&) = default;
        explicit operator bool() const noexcept
        {
            return High || Low;
        }
        static FabricId New()
        {
            std::random_device random;
            FabricId id{(uint64_t{random()}<<32)^random(), (uint64_t{random()}<<32)^random()};
            if (!id)
            {
                id.Low = 1;
            }
            return id;
        }
    };

    struct RegionRef
    {
        FabricId Id{};
        uint64_t ByteOffset{};
        uint64_t ByteCount{};
        uint32_t Slot{};
        uint32_t Generation{};
        uint32_t RecordOrdinal{};
        uint32_t Height{};
        uint32_t Width{};
        MacroColumnOfAPC Column{MacroColumnOfAPC::FREE_SLOT};
        SchemaDefinition::DataTypeOfMacroColumn Type{};

        friend constexpr bool operator == (const RegionRef&, const RegionRef&) noexcept = default;  
    };
}