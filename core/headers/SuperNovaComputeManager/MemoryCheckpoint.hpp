#pragma once
#include "ComputerDefAndVect.hpp"
#include <random>
#include <chrono>

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
            static FabricId New() noexcept
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
    };

}
