#pragma once
#include "Definition.hpp"
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
namespace BidirectionalInMemGraph::ComputerDetail
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
        uint16_t FabCacheSize = sizeof(CoreOfFabricCoordinator::FabricCache);
        uint16_t RegionSchemaSize = sizeof(SchemaOrchestrator::RegionSchemaRecord);
        uint16_t Version = CoreOfFabricCoordinator::FORMAT_VERSION;
        Endianess Endian = Endianess::LITTLE;
    };

    struct FabricRuntimeBinding
    {
        APCFinilizer* Owner{};
        std::span<uint64_t> Cells{};
        FabricExecutionGate* Gate{};
        FabricId Id{};
        std::span<uint64_t> Numerics{};
        uint64_t NumericalBegainCell{};

        constexpr std::span<uint64_t> NumericalCells() const noexcept
        {
            return Numerics.empty() ? Cells : Numerics;
        }

        std::optional<uint64_t> LogicalByteOffset(const void* data, uint64_t bytes) noexcept;

        uint64_t LogicalCellCount() const noexcept
        {
            return Numerics.empty() ? Cells.size() : (NumericalBegainCell + Numerics.size());
        }

        explicit operator bool() const noexcept
        {
            return !Cells.empty() && Gate && static_cast<bool>(Id);
        }
    };
}