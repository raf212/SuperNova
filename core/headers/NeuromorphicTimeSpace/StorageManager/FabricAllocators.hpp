#pragma once 
#include "ConstructAndVerify.hpp"

namespace BidirectionalInMemGraph
{
    struct RawPackedCellAllocator final
    {
        using AllocateFunction = uint64_t* (*)(
            size_t count_of_packed_cell, size_t alignment, void* user
        );

        using FreeFunction = void (*)(
            uint64_t* packed_cell_storage_ptr, 
            size_t count_of_cell, size_t alignment, void*user
        ) noexcept;

        AllocateFunction AllocatePackedCellStorage{nullptr};
        FreeFunction FreePackedCellStorage{nullptr};
        void* User{nullptr};
        size_t Alignment{BIT_COUNT_OF_UINT64_T};

        size_t EffectiveAllignment() const noexcept
        {
            return std::max<size_t>(ADS::CACELINE_SIZE_IN_BYTES, Alignment);
        }

        uint64_t* Allocate(size_t cells) const noexcept;

        void Free(uint64_t* data, size_t cells) const noexcept;

        static size_t AlignBiteCount_(size_t bytes, size_t alignment) noexcept;

        static uint64_t* DefaultAllocateAtomicCells(
            size_t count_of_packed_cell, size_t alignment, void*
        ) noexcept;

        static void DefaultFreeAtomicCells(
            uint64_t* packed_cell_storage_ptr, 
            size_t, size_t, void*
        ) noexcept;

        bool IsValid() const noexcept
        {
            return static_cast<bool>(AllocatePackedCellStorage) == static_cast<bool>(FreePackedCellStorage) &&
                std::has_single_bit(EffectiveAllignment());
        }
    };

    class PackedCellStorage final
    {
        uint64_t* Data_{};
        size_t Count_{};
        RawPackedCellAllocator Alloc_{};
        bool Owned_{false};
    
    public:
        PackedCellStorage() = default;
        PackedCellStorage(
            uint64_t* data,
            size_t count,
            RawPackedCellAllocator alloc,
            bool owned
        ) noexcept:
            Data_(data), Count_(count), Alloc_(alloc), Owned_(owned)
        {}
        PackedCellStorage(const PackedCellStorage&) = delete;
        PackedCellStorage& operator = (const PackedCellStorage&) = delete;
        PackedCellStorage(PackedCellStorage&& other) noexcept :
            Data_(other.Relese()), Count_(other.Count_),
            Alloc_(other.Alloc_), Owned_(other.Owned_)
        {}
        ~PackedCellStorage()
        {
            if (Owned_) Alloc_.Free(Data_, Count_);
        }

        bool Allocate(size_t count, RawPackedCellAllocator allocator = {}) noexcept;

        size_t Count() const noexcept {return Count_;}

        uint64_t* Data() const noexcept {return Data_;}

        uint64_t* Relese() noexcept {return std::exchange(Data_, nullptr);}
    };

    class NumericalBacking final
    {
    private:
        uint64_t* Data_{};
        size_t Bytes_{};
        std::filesystem::path PendingPath_{};

    #if defined (_WIN32)
        HANDLE File_{INVALID_HANDLE_VALUE};
        HANDLE Mapping_{};
    #else
        int File_{-1};
    #endif
    ///RAW DATA END

    #if defined (_WIN32)
        ///@brief Before using  must fill valid Data_ and Bytes_
        bool CreateWinMap_() noexcept;

        bool WinPrefetch_(uint64_t first_cell, uint64_t cells) noexcept;
    #else
        ///@brief Before using  must fill valid Data_ and Bytes_
        bool CreateLinMap_(uint64_t cells) noexcept;

        bool LinPrefetch_(uint64_t first_cell, uint64_t cells) noexcept;
    #endif

        void Swap_(NumericalBacking& other) noexcept;
    public:
        NumericalBacking() = default;
        NumericalBacking(const NumericalBacking&) = delete;
        NumericalBacking& operator = (const NumericalBacking&) = delete;
        NumericalBacking(NumericalBacking&& other) noexcept
        {
            Swap_(other);
        }
        NumericalBacking& operator = (NumericalBacking&& other) noexcept
        {
            if (this != &other) {Close(); Swap_(other);}
            return *this;
        }
        ~NumericalBacking() {Close();}

        void Close() noexcept;

        bool Create(const std::filesystem::path& path, uint64_t cells) noexcept;

        void CommitCreation() noexcept {PendingPath_.clear();}

        uint64_t* DataPtr() const noexcept {return Data_;}

        bool Flush() noexcept;

        bool Prefetch(uint64_t first_cell, uint64_t cells) noexcept;

        void WarmAll() noexcept;
    };

}

#if defined(SN_ACPP_ENABLED)
namespace BidirectionalInMemGraph::ComputerDetail::Memory
{
    class ACppFabricAlocator final
    {
        sycl::queue* Queue_;
        RawPackedCellAllocator Allocator_{};
        sycl::usm::alloc __Loc__;

        static uint64_t* AllocateFn_(size_t cells, size_t alignment, void* user)
        {
            ACppFabricAlocator& allocator = *static_cast<ACppFabricAlocator*>(user);
            if (allocator.__Loc__ == sycl::usm::alloc::device) return nullptr;

            return allocator.AllocateUsm(cells, alignment);
        }

        static void FreeFn_(uint64_t* data, size_t, size_t, void* user) noexcept
        {
            static_cast<ACppFabricAlocator*>(user)->FreeUsm(data);
        }

    public:
        explicit ACppFabricAlocator(
            sycl::queue& queue,
            sycl::usm::alloc location = sycl::usm::alloc::host
        ) noexcept :
            Queue_(&queue), __Loc__(location)
        {
            Allocator_.AllocatePackedCellStorage = &AllocateFn_;
            Allocator_.FreePackedCellStorage = &FreeFn_;
            Allocator_.User = this;
            Allocator_.Alignment = ADS::CACELINE_SIZE_IN_BYTES;
        }
        ACppFabricAlocator(const ACppFabricAlocator&) = delete;
        ACppFabricAlocator& operator = (const ACppFabricAlocator&) = delete;

        // const RawPackedCellAllocator& Allocator_

        uint64_t* AllocateUsm(size_t cells, size_t alignment = ADS::CACELINE_SIZE_IN_BYTES);

        void FreeUsm(uint64_t* data) noexcept;

        const RawPackedCellAllocator& AllocatorRaw() const noexcept
        {
            return Allocator_;
        }
    };
}
#endif

namespace BidirectionalInMemGraph::Numerics 
{
    using CFC = CoreOfFabricCoordinator;
    class Replacement final
    {
    private:
        SlabToFabricConverterAndCordinator& Self_;
        PackedCellStorage Previous_;
        NumericalBacking Numbers_;
        CFC::FabricCache* Cache_;
        CFC::FabricBackigOwnership Ownership_;
        RawPackedCellAllocator Allocator_;
        SchemaDefinition::RegionSchemaTable Schemas_;
        bool Active_;
        bool Track_;
        uint64_t Revision_;
        bool Complete_{false};
    public:
        explicit Replacement(SlabToFabricConverterAndCordinator&) noexcept;
        Replacement(const Replacement&) = delete;
        Replacement& operator = (const Replacement&) = delete;
        ~Replacement();

        uint64_t Revision() const noexcept {return Revision_;}
        void Commit() noexcept {Complete_ = true;}
    };

    template<class T>
    struct NumericalRegionRef
    {
        uint64_t Epoch{};
        uint64_t Index{LIMIT_OF_CELL_COUNT + 1u};

        explicit operator bool() const noexcept 
        {
            return Index <= LIMIT_OF_CELL_COUNT;
        }
    };

    template<class T>
    struct NumericOperand
    {
        NumericalRegionRef<T> Region{};
        uint32_t Row{ADS::APC_INDEX_BOUND_SENTINAL};
        uint64_t Offset{};
        uint64_t Count{};

        explicit operator bool() const noexcept 
        {
            return Row <= ADS::APC_INDEX_BOUND_SENTINAL && Region;
        }
    };

    template<class T>
    struct NumericalMetrixOperand
    {
        NumericalRegionRef<T> Region{};
        uint32_t FirstRow{ADS::APC_INDEX_BOUND_SENTINAL};
        uint32_t Rows{ADS::APC_INDEX_BOUND_SENTINAL};
        uint32_t FirstColumn{ADS::APC_INDEX_BOUND_SENTINAL};
        uint32_t Columns{ADS::APC_INDEX_BOUND_SENTINAL};

        explicit operator bool() const noexcept 
        {
            return Region && 
                FirstRow != ADS::APC_INDEX_BOUND_SENTINAL &&
                Rows != ADS::APC_INDEX_BOUND_SENTINAL &&
                FirstColumn != ADS::APC_INDEX_BOUND_SENTINAL &&
                Columns != ADS::APC_INDEX_BOUND_SENTINAL;
        }
    };
}
