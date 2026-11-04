#pragma once
#include <istream>
#include <ostream>

#include "../FabricOrchestrators/HandleAndRetirement.hpp"

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    static constexpr size_t CellCountInAPage = 1024; // can be optimized
    static constexpr uint64_t MaxCheckpointMetaCellCount = 1024;

    using CFC = CoreOfFabricCoordinator;
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

    enum class PageHeader : size_t 
    {
        MAGIC,
        VERSION,
        NATIVEABI,
        FABRICHIGH,
        FABRICLOW,
        REVISION,
        CELL_COUNT,
        PAGE_SIZE,
        PAGE_COUNT,
        METADATA_SIZE,
        PAYLOAD_OFFSET,
        DIRECTORY_HASH,
        TOTAL_FILE_SIZE,
        HEADER_HASH,
    };

    struct InternalValues
    {
        static constexpr uint64_t FileMagic = 0x3354504b43464e53ULL; // "SNFCKPT3" on disk
        static constexpr uint64_t FileVersion = 1;
        static constexpr uint64_t BytesPerCell = sizeof(uint64_t);
        static constexpr unsigned BitsPerByte = CHAR_BIT;
        static constexpr uint64_t PageBytes = BitsPerByte * BytesPerCell * CellCountInAPage;
        static constexpr uint64_t MaximumMetadataBytes = BitsPerByte * BytesPerCell * MaxCheckpointMetaCellCount;
        static constexpr uint64_t DefaultMaximumCells = uint64_t{10} << 27;
        static constexpr uint64_t FnvOffsetBasis = 14695981039346656037ULL;
        static constexpr uint64_t FnvPrime = 1099511628211ULL;

        static constexpr size_t HeaderWords = static_cast<size_t>(PageHeader::HEADER_HASH) + 1;
        static constexpr uint64_t HeaderBytes = HeaderWords * BytesPerCell;
        static constexpr size_t HeaderDataWords = static_cast<size_t>(PageHeader::HEADER_HASH);
        static constexpr size_t MaxPageCount = (DefaultMaximumCells * BytesPerCell + PageBytes - 1) / PageBytes;


        using FileHeader = std::array<uint64_t, HeaderWords>;

        enum class ReferanceField : size_t
        {
            FABRIC_HIGH,
            FABRIC_LOW,
            BYTE_OFFSET,
            BYTE_COUNT,
            SLOT,
            GENERATION,
            RECORD_ORDINAL,
            COLIMN,
            TYPE,
            HEIGHT,
            WIDTH,
            VERSION,
            COUNT
        };

        struct CheckpointCache
        {
            std::filesystem::path Path_{};
            FabricId Id_{};
            uint64_t Cells_{};
            uint64_t Revision_{};
            uint64_t Payload_{};
            uint64_t MetaBytes_{};
            uint64_t TotalBytes_{};
            std::array<uint64_t, MaxPageCount> Hashes_{};
            std::array<std::byte, MaximumMetadataBytes> MetaData_{};
            FileHeader Header_{};
            CFC::FabricCache SlabCache_{};
        };

    };

    struct ReadStatic
    {
        uint64_t FileOpens{};
        uint64_t HeaderChecks{};
        uint64_t PageReads{};
        uint64_t RegionValidations{};
    };

    using IV = InternalValues;
    using FileHeader = IV::FileHeader;

    static constexpr size_t FieldIndex(PageHeader field) noexcept
    {
        return static_cast<size_t>(field);
    }

    static constexpr uint64_t PageHash(std::span<const std::byte> bytes, uint64_t hash = IV::FnvOffsetBasis) noexcept
    {
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

    static constexpr std::array<std::byte, IV::BytesPerCell> EncodeWord(uint64_t value) noexcept
    {
        std::array<std::byte, IV::BytesPerCell> bytes{};
        for (size_t i = 0; i < bytes.size(); i++) 
        {
            bytes[i] = static_cast<std::byte>(value >> (i * IV::BitsPerByte));
        }
        return bytes;
    }

    static constexpr uint64_t HedaderHash(const FileHeader& header) noexcept
    {
        std::array<std::byte, IV::HeaderDataWords * IV::BytesPerCell> encoded{};
        uint64_t hash = IV::FnvOffsetBasis;
        for (size_t word = 0; word < IV::HeaderDataWords; word++) 
        {
            hash = PageHash(EncodeWord(header[word]), hash);
        }
        return hash;
    }

    static constexpr uint64_t PageCount(uint64_t bytes) noexcept
    {
        return bytes == UNSIGNED_ZERO ? UNSIGNED_ZERO : (1 + ((bytes - 1) / IV::PageBytes));
    }
    

}