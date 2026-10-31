#pragma once 
#include "CompleteFabric.hpp"

namespace BidirectionalInMemGraph
{
    
    class SlabToFabricConverterAndCordinator : public CompiledDAGTableConstructor
    {
        friend class AdaptivePackedCellContainer;
    private:
        uint64_t* AllocatePackedCellRaw_(size_t count_of_cells) noexcept;
        bool ValidateAttachedFabricLayout_() noexcept;
        bool QuiesceFabric_() noexcept;
        bool ReopenLiveAPCGenerations_() noexcept;

    protected :
        std::atomic<bool> TrackDAGRevision_{false};

        SD::RegionSchemaTable DefaultRegionTable_{};

        void FreeRawPackedCells_(uint64_t*packed_cell_memory_ptr, size_t packed_cell_count) noexcept;
        void ResetScalarsofTheFabric_() noexcept;

        template<typename T>
        T* RegionT_(uint32_t slot, uint32_t cell_offset) noexcept
        {
            return reinterpret_cast<T*>(SlabBasePtr_ + SlotBegin_(slot) + cell_offset);
        }

    public:
        using CFC = CoreOfFabricCoordinator;

        SlabToFabricConverterAndCordinator(/* args */) noexcept = default;

        ~SlabToFabricConverterAndCordinator() noexcept
        {
            ShutDownFabric();
            // A rejected close must not destroy the admission gate while a
            // retained APC scope still refers to it.
            if (SlabBasePtr_) std::terminate();
        }

        SlabToFabricConverterAndCordinator(const SlabToFabricConverterAndCordinator&) = delete;
        SlabToFabricConverterAndCordinator& operator = (const SlabToFabricConverterAndCordinator&) = delete;

        void ShutDownFabric() noexcept;

        bool IsFabricActive() noexcept
        {
            return
                FabricInitialized_.load(std::memory_order_acquire) &&
                SlabBasePtr_ &&
                FabCache_ &&
                ADS::ValidateSlotIdx(FabCache_->PerAPCRuntimeCellCount_) &&
                ADS::ValidateSlotIdx(FabCache_->CountOfAPC_);
        }

        bool InitializeFabric(
            uint32_t slot_count,
            uint32_t slot_cell_count,
            const SchemaDefinition::FabricRegionConfig& region_conf,
            uint32_t max_direct_parent_per_axis = ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS,
            const CFC::FabricInitConf&  storage = {}
        ) noexcept;

        bool SaveFabric(std::span<uint64_t> destination) noexcept;

        bool AttachFabric(
            uint64_t* raw_cells,
            uint64_t cell_count,
            CFC::FabricBackigOwnership ownership = CFC::FabricBackigOwnership::BORROWED
        ) noexcept;

        CFC::DetachFabric DetachFabric() noexcept;
        
    };




}