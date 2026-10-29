#pragma once 
#include <filesystem>
#include "../../AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "../../Models/GlobalModels.hpp"

namespace BidirectionalInMemGraph
{
    struct RawPackedCellAllocator;

    struct CoreOfFabricCoordinator
    {
        /// UNCHECKED
        static constexpr size_t MODEL_DESCRIPTION_LEN = 0u;
        static constexpr size_t WORK_RECORD_WIDTH_OF_FABRIC = 0u;
        static constexpr size_t DEFAULT_FABRIC_CONTROLIO_LENGTH = 512u;
        ///--------------------------

        static constexpr uint8_t FORMAT_VERSION = 3u;

        enum class FabricBackigOwnership : uint8_t
        {
            NONE = 0,
            OWNED = 1,
            BORROWED = 2
        };

        enum class NumericalCachePolicy
        {
            ON_DEMAND = 0,
            PRE_TOUCHED = 1
        };

        struct FabricInitConf final
        {
            std::filesystem::path SegmentPoolPath{};
            const RawPackedCellAllocator* HostAlloc{};
            NumericalCachePolicy Cache{};
            uint64_t MaxCheckPointCells{uint64_t{1} << 27}; // it should be free ram * .8;-> MAX for that system based on runtime
            std::span<const GMI::ModelInfo> Models{};
            std::span<const GMI::ModelDependency> Dependencies{};
            std::span<const std::byte> Dynamics{};
            uint64_t DynamicRevision{1};
        };

        struct alignas(64) FabricCache final
        {
            uint32_t FormateVersion_{UNSIGNED_ZERO};
            ///FABRIC CONSTRUCTION
            uint32_t PerAPCRuntimeCellCount_{UNSIGNED_ZERO};
            uint32_t CountOfAPC_{UNSIGNED_ZERO};
            uint64_t SlabCellCount_{UNSIGNED_ZERO};

            // RECORD BOOK / SEGMENT POOL
            uint64_t RecordBookBeginIndex_{UNSIGNED_ZERO};
            uint64_t RecordBookEndIndex_{UNSIGNED_ZERO};
            uint64_t SegmentPoolBegin_{UNSIGNED_ZERO};
            uint32_t FirstFreeIdx_{UNSIGNED_ZERO};

            // EDGE GEOMETRY
            uint32_t MaxDirectParentsPerAxis_{UNSIGNED_ZERO};
            uint64_t EdgeTableRecordWidth_{UNSIGNED_ZERO};

            ///MATRIX CONSTRUCTION
            uint8_t ActiveRegionCount_{UNSIGNED_ZERO};
            uint16_t ActiveRegionMask_{UNSIGNED_ZERO};
            uint32_t MatrixViewRowCellCount_{UNSIGNED_ZERO};
            uint32_t MatrixBatchCapacity_{UNSIGNED_ZERO};
            uint64_t RegionAlignmentCellCount_{UNSIGNED_ZERO};

            // HOT TABLE BEGIN INDICES
            uint64_t HorizontalEdgeBeginIdx_{UNSIGNED_ZERO};
            uint64_t VerticalEdgeBeginIdx_{UNSIGNED_ZERO};
            uint64_t HandleTableBeginIndex_{UNSIGNED_ZERO};
            uint64_t MatrixViewTableBeginIndex_{UNSIGNED_ZERO};
            bool HasDefaultRegionTable_{false};

            ///Model Info
            GMI::ModelStorageDiaphragm ModelDiaphragm_ = GMI::ModelStorageDiaphragm{};

            //DISK
            uint64_t SegmentPoolDiskByteOffset_{UNSIGNED_ZERO};

        private:
            uint64_t Reserved_[1]{};
        };

        struct DetachFabric final
        {
            uint64_t* Slab_{nullptr};
            uint64_t CellCount_{UNSIGNED_ZERO};
            FabricBackigOwnership Ownership_ = FabricBackigOwnership::NONE;

            explicit constexpr operator bool() const noexcept
            {
                return Slab_ != nullptr && CellCount_ != UNSIGNED_ZERO;
            }
        };
        
        enum class RecordBookInternalIndexing : uint8_t
        {
            BEGIN64 = 0,
            END64 = 1,
        };
        static constexpr uint8_t RECORD_BOOK_WIDTH = static_cast<uint8_t>(RecordBookInternalIndexing::END64) + 1u;


        static constexpr uint8_t FABRIC_UNIT_COUNT = sizeof(FabricCache) / sizeof(uint64_t);

        static constexpr bool IsValidEdgeTable(FabricSegments table_class) noexcept
        {
            return
                table_class == FabricSegments::VALUE_PARENT_EDGE_TABLE_H ||
                table_class == FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;
        }

        static constexpr size_t DefaultFabricAlignment16Cell(size_t value) noexcept
        {
            const uint8_t alignment_value_15 = 16 - 1;
            return (value + alignment_value_15) & ~static_cast<size_t>(alignment_value_15);
        }

        static constexpr uint64_t GetStartingOfAnyFabricTable_(
            FabricSegments table_class
        ) noexcept
        {
            const uint64_t offset = static_cast<uint64_t>(table_class) * CoreOfFabricCoordinator::RECORD_BOOK_WIDTH;
            const uint64_t entry = DefaultFabricAlignment16Cell(FABRIC_UNIT_COUNT)  + offset;
            return entry;        
        }

        static constexpr uint8_t RECORD_BOOK_INTERNAL_SEGMENT_COUNT = static_cast<uint8_t>(FabricSegments::SEGMENT_POOL) + 1;

        struct FabricSegmentBounds
        {
            uint64_t BeginIndex = UNSIGNED_ZERO;
            uint64_t EndIndex = UNSIGNED_ZERO;
            FabricSegments OwnerTableOfTheBounds{};
            bool IsValid = false;  
        };
    };

    namespace GMI
    {
        using CFC = CoreOfFabricCoordinator;

        static void GetModelCatalog(
            const CFC::FabricCache& cache, 
            const uint64_t* hot,
            ModelCatalog& catalog
        ) noexcept
        {
            const ModelInfo* mi_ptr = reinterpret_cast<const ModelInfo*>(hot + cache.ModelDiaphragm_.LocalDirectoryBeginIdx_);
            const ModelDependency* dp_ptr = reinterpret_cast<const ModelDependency*>(mi_ptr + cache.ModelDiaphragm_.ModelCount_);
            const std::byte* dynamics = reinterpret_cast<const std::byte*>(dp_ptr + cache.ModelDiaphragm_.ModelDependencyCount_);

            catalog.Models = {mi_ptr, cache.ModelDiaphragm_.ModelCount_};
            catalog.Dependencies = {dp_ptr, cache.ModelDiaphragm_.ModelDependencyCount_};
            catalog.Dynamics = {dynamics, cache.ModelDiaphragm_.ModelDynamicsByteCount_};
        }

        static ModelCatalog GetModelCatalog(
            const CFC::FabricCache& cache, 
            const uint64_t* hot
        ) noexcept
        {
            ModelCatalog catalog{};
            GetModelCatalog(cache, hot, catalog);
            return catalog;
        }
    }


}
