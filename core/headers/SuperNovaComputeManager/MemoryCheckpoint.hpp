#pragma once
#include "ComputerDefAndVect.hpp"
#include <random>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <bit>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace BidirectionalInMemGraph 
{
    struct FabricMemory
    {
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

        enum class MemoryLocation : uint8_t
        {
            HOST = 0,
            GPU = 1,
            DISK = 2
        };

        enum class OpenAccess : uint8_t
        {
            READ = 0,
            WRITE = 1,
            READ_WRITE = 2
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

        struct FabricRuntimeBinding
        {
            APCFinilizer* Owner{};
            std::span<uint64_t> Cells{};
            FabricExecutionGate* Gate{};
            FabricId Id{};
            explicit operator bool() const noexcept
            {
                return !Cells.empty() && Gate && static_cast<bool>(Id);
            }
        };

        struct ResidencyInfo
        {
            uint64_t CommittedRevision{};
            uint64_t HostRevision{};
            uint64_t DeviceRevision{};
            MemoryLocation Location{MemoryLocation::HOST};
        };

        struct TransferStatics
        {
            uint64_t Allocation{};
            uint64_t UploadedBytes{};
            uint64_t DownloadedfBytes{};
            uint64_t CacheHits{};
        };

        enum class Endianess : uint16_t
        {
            LITTLE = 0,
            BIG = 1
        };

        struct alignas(uint64_t) NativeABI
        {
            uint16_t FabCacheSize = sizeof(FabricCache);
            uint16_t RegionSchemaSize = sizeof(SchemaOrchestrator::RegionSchemaRecord);
            uint16_t Version = CoreOfFabricCoordinator::FORMAT_VERSION;
            Endianess Endian = Endianess::LITTLE;
        };


        static constexpr uint64_t FILE_MAGIC = 0x3354504b43464e53ULL;
        static constexpr uint64_t FILE_VERSION = 1u;
        static constexpr uint64_t BYTES_PER_WORD = sizeof(uint64_t);
        static constexpr uint64_t PAGE_BYTES = 64 * 1024;
        static constexpr uint64_t MAX_META_BYTES = PAGE_BYTES;
        static constexpr uint64_t DEFAULT_MAX_CELLS = uint64_t{1} << 27;
        static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
        static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

        enum class ReferenceField : uint8_t
        {
            MAGIC = 0,
            VERSION = 1,
            NATIVE_ABI = 2,
            FABRIC_HIGH = 3,
            FABRIC_LOW = 4,
            REVISION = 5,
            CELLOFFSET = 6,
            PAGE_SIZE = 7,
            PAGE_COUNT = 8,
            MATA_SIZE = 9,
            PAYLOAD_OFFSET = 10,
            DIRECTORY_HASH = 11,
            TOTAL_FILE_SIZE = 12,
            HEADER_HASH = 13
        };

        static constexpr uint8_t ReferenceToIndex(ReferenceField field) noexcept
        {
            return static_cast<uint8_t>(field);
        }

        static constexpr uint8_t HEADER_DATA_WORDS = static_cast<uint8_t>(ReferenceField::HEADER_HASH);
        static constexpr uint8_t REFERENCE_WORD_COUNT = HEADER_DATA_WORDS  + 1;
        static constexpr uint64_t HEADER_BYTES = REFERENCE_WORD_COUNT * BYTES_PER_WORD;
        using FileHeader = std::array<uint64_t, REFERENCE_WORD_COUNT>;

        static constexpr uint64_t Hash(std::span<const std::byte> bytes) noexcept
        {
            uint64_t hash = FNV_OFFSET_BASIS;
            for (std::byte b : bytes) 
            {
                hash ^= std::to_integer<uint8_t>(b);
                hash *= FNV_PRIME;
            }
            return hash;
        }

        static bool WriteWord(std::ostream& out, uint64_t value) noexcept
        {
            for (uint64_t i = 0; i < BYTES_PER_WORD; i++) 
            {
                out.put(static_cast<char>(value >> (i * LEN_OF_BYTE_IN_BITS)));
            }
            return static_cast<bool>(out);
        }

        static bool ReadWord(std::istream& input, uint64_t value) noexcept
        {
            value = UNSIGNED_ZERO;
            for (uint64_t i = 0; i < BYTES_PER_WORD; i++) 
            {
                const int encoded = input.get();
                if (encoded == std::char_traits<char>::eof()) 
                {
                    return false;
                }
                
                value |= uint64_t{static_cast<unsigned char>(encoded)} << (i * LEN_OF_BYTE_IN_BITS);
            }
            return true;
        }

        static constexpr uint64_t HeaderHash(const FileHeader& header) noexcept
        {
            std::array<std::byte, HEADER_DATA_WORDS * BYTES_PER_WORD> encoded{};
            for (uint64_t i = 0; i < HEADER_DATA_WORDS; i++) 
            {
                for (uint64_t byte = 0; byte < BYTES_PER_WORD; byte++) 
                {
                    encoded[i * BYTES_PER_WORD + byte] = static_cast<std::byte>(header[i] >> (byte * LEN_OF_BYTE_IN_BITS));
                }
            }
            return Hash(encoded);
        }

        static constexpr uint64_t PageCount(uint64_t bytes) noexcept
        {
            return bytes == UNSIGNED_ZERO ? UNSIGNED_ZERO : 1 + (bytes - 1) / PAGE_BYTES;
        }

    };

    class FabricCheckpoint : public FabricMemory
    {
        std::filesystem::path Path_{};
        FabricId Id_{};
        uint64_t Cells_{};
        uint64_t Revision_{};
        uint64_t Payload_{};
        uint64_t MetaBytes_{};
        uint64_t TotalBytes{};

        std::vector<uint64_t> Hashes_{};
        std::vector<std::byte> MetaData_{};
        FileHeader Header_{};
        CoreOfFabricCoordinator::FabricCache Cache{};

        bool ReadBytes_(
            uint64_t offset,
            std::span<std::byte> output
        ) const noexcept;
    };

}
