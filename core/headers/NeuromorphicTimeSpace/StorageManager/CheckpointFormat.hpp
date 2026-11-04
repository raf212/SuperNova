#pragma once

#include "../FabricOrchestrators/HandleAndRetirement.hpp"

namespace BidirectionalInMemGraph::CheckpointFormat 
{
    static constexpr size_t CellCountInAPage = 1024; // can be optimized
    static constexpr uint64_t MaxCheckpointMetaCellCount = 1024;

    using CFC = CoreOfFabricCoordinator;
    struct FabricId
    {
        uint64_t High{UNSIGNED_ZERO};
        uint64_t Low{UNSIGNED_ZERO};
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
        static constexpr uint64_t FILE_MAGIC = 0x3354504b43464e53ULL; // "SNFCKPT3" on disk
        static constexpr uint64_t FILE_VERSION = 1;
        static constexpr uint64_t BYTES_PER_CELL = sizeof(uint64_t);
        static constexpr unsigned BITS_PER_BYTES = CHAR_BIT;
        static constexpr uint64_t PAGE_BYTES = BITS_PER_BYTES * BYTES_PER_CELL * CellCountInAPage;
        static constexpr uint64_t MAX_META_BYTES = BITS_PER_BYTES * BYTES_PER_CELL * MaxCheckpointMetaCellCount;
        static constexpr uint64_t DEFAULT_STORAGE_LIMIT = uint64_t{10} << 27;
        static constexpr uint64_t FnvOffsetBasis = 14695981039346656037ULL;
        static constexpr uint64_t FnvPrime = 1099511628211ULL;

        static constexpr size_t HEADER_CELLS = static_cast<size_t>(PageHeader::HEADER_HASH) + 1;
        static constexpr uint64_t HEADER_BYTES = HEADER_CELLS * BYTES_PER_CELL;
        static constexpr size_t HEADER_DATA_CELLS = static_cast<size_t>(PageHeader::HEADER_HASH);
        static constexpr size_t MAX_PAGE_COUNT = (DEFAULT_STORAGE_LIMIT * BYTES_PER_CELL + PAGE_BYTES - 1) / PAGE_BYTES;


        using FileHeader = std::array<uint64_t, HEADER_CELLS>;

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
            VERSION
        };
        static constexpr size_t REFERANCE_CELL_COUNT = static_cast<size_t>(ReferanceField::VERSION) + 1;

        struct CheckpointCache
        {
            std::filesystem::path Path_{};
            FabricId Id_{};
            uint64_t Cells_{};
            uint64_t Revision_{};
            uint64_t Payload_{};
            uint64_t MetaBytes_{};
            uint64_t TotalBytes_{};
            std::array<uint64_t, MAX_PAGE_COUNT> Hashes_{};
            std::array<std::byte, MAX_META_BYTES> MetaData_{};
            FileHeader Header_{};
            CFC::FabricCache SlabCache_{};

            explicit constexpr operator bool() const noexcept
            {
                return Id_.High != UNSIGNED_ZERO ||
                    Id_.Low != UNSIGNED_ZERO;
            }
        };

        static constexpr size_t ReferanceIndex(ReferanceField field) noexcept
        {
            return static_cast<size_t>(field);
        }

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

    static constexpr size_t PageHeaderIndex(PageHeader field) noexcept
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

    static constexpr std::array<std::byte, IV::BYTES_PER_CELL> EncodeWord(uint64_t value) noexcept
    {
        std::array<std::byte, IV::BYTES_PER_CELL> bytes{};
        for (size_t i = 0; i < bytes.size(); i++) 
        {
            bytes[i] = static_cast<std::byte>(value >> (i * IV::BITS_PER_BYTES));
        }
        return bytes;
    }

    static constexpr uint64_t HedaderHash(const FileHeader& header) noexcept
    {
        std::array<std::byte, IV::HEADER_DATA_CELLS * IV::BYTES_PER_CELL> encoded{};
        uint64_t hash = IV::FnvOffsetBasis;
        for (size_t word = 0; word < IV::HEADER_DATA_CELLS; word++) 
        {
            hash = PageHash(EncodeWord(header[word]), hash);
        }
        return hash;
    }

    static constexpr uint64_t PageCount(uint64_t bytes) noexcept
    {
        return bytes == UNSIGNED_ZERO ? UNSIGNED_ZERO : (1 + ((bytes - 1) / IV::PAGE_BYTES));
    }

}