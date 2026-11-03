#pragma once 
#include "CompleteFabric.hpp"

namespace BidirectionalInMemGraph
{
    class RelocatableSlabStorage;
    class SlabToFabricConverterAndCordinator : public CompiledDAGTableConstructor
    {
        friend class AdaptivePackedCellContainer;
        friend class RelocatableSlabStorage;
        friend class Numerics::Replacement;
    protected:
        using CFC = CoreOfFabricCoordinator;
    private:
        uint64_t* AllocatePackedCellRaw_(size_t count_of_cells) noexcept;
        bool ValidateAttachedFabricLayout_() noexcept;
        bool QuiesceFabric_() noexcept;
        bool ReopenLiveAPCGenerations_() noexcept;

        bool __InitSlabAndCache__(const CFC::FabricInitConf& storage, CFC::FabricCache& cache) noexcept;
        void __InitModelStorage__(
            const GMI::ModelCatalog& catalog, 
            std::byte* directory,
            uint64_t count_of_apc
        ) noexcept;

        bool __InitMatrixRegionTable__(const StorageAllocationDef& layout) noexcept;

        struct InitFlagReset;
    protected :
        void FreeRawPackedCells_(uint64_t*packed_cell_memory_ptr, size_t packed_cell_count) noexcept;
        void ResetScalarsofTheFabric_() noexcept;

    public:

        SlabToFabricConverterAndCordinator(/* args */) noexcept = default;

        ~SlabToFabricConverterAndCordinator() noexcept
        {
            ShutDownFabric();
            // A rejected close must not destroy the admission gate while a
            // retained APC scope still refers to it.
            if (DisposableFCache___.SlabBasePtr_) std::terminate();
        }

        bool IsDiskPaged() const noexcept
        {
            return __NumericalBacking__.DataPtr() != nullptr;
        }


        SlabToFabricConverterAndCordinator(const SlabToFabricConverterAndCordinator&) = delete;
        SlabToFabricConverterAndCordinator& operator = (const SlabToFabricConverterAndCordinator&) = delete;

        void ShutDownFabric() noexcept;

        bool IsFabricActive() noexcept
        {
            return
                DisposableFCache___.FabricInitialized_.load(std::memory_order_acquire) &&
                DisposableFCache___.SlabBasePtr_ &&
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

    };

    class RelocatableSlabStorage : public SlabToFabricConverterAndCordinator
    {
    protected:
        template<typename T>
        T* RegionT_(uint32_t slot, uint32_t cell_offset) noexcept
        {
            return reinterpret_cast<T*>(ResolveCell_(SlotBegin_(slot) + cell_offset));
        }

    public:
        bool AttachFabric(
            uint64_t* raw_cells,
            uint64_t cell_count,
            CFC::FabricBackigOwnership ownership = CFC::FabricBackigOwnership::BORROWED,
            const CFC::FabricInitConf& storage = {}
        ) noexcept;

        CFC::DetachFabric DetachFabric() noexcept;

        bool SaveFabric(std::span<uint64_t> destination) noexcept;

        bool SaveHotSubFabric(std::span<uint64_t> destination) noexcept;

        bool FlushNumericalBacking() noexcept;

        bool PrefetchNumerical(
            uint32_t slot,
            uint32_t cell_offset,
            uint32_t cells
        ) noexcept;

        std::span<uint64_t> NumericalStorage() noexcept
        {
            return FabCache_ ? 
                std::span<uint64_t>{
                    NumericalBase_(), 
                    FabCache_->SlabCellCount_ - FabCache_->SegmentPoolBegin_
                } :
                std::span<uint64_t>{};
        }

        std::span<uint64_t> HotStorageDramUsm() noexcept
        {
            return FabCache_ ?
                std::span<uint64_t>{DisposableFCache___.SlabBasePtr_, FabCache_->SegmentPoolBegin_} :
                std::span<uint64_t>{};
        }

        std::span<uint64_t> ContiguousStorage() noexcept
        {
            return FabCache_  && !IsDiskPaged() ? 
                std::span<uint64_t>{DisposableFCache___.SlabBasePtr_, FabCache_->SlabCellCount_} :
                std::span<uint64_t>{};
        }

    };



}