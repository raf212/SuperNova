#pragma once
#include "HeaderOrchestrator.hpp"
#include <span>


namespace BidirectionalInMemGraph
{

    struct ResolveRegionBiteView
    {
        std::span<std::byte> Bytes{};
        const SchemaDefinition::RegionSchemaRecord* Schema = nullptr;
        uint32_t RegionOrdinal = UNSIGNED_ZERO;

        constexpr bool IsValid() const noexcept
        {
            return 
                !Bytes.empty() &&
                Schema != nullptr &&
                Schema->MatrixHeight != UNSIGNED_ZERO &&
                Schema->MatrixWidth != UNSIGNED_ZERO;
        }

        constexpr size_t ByteCount() const noexcept
        {
            return Bytes.size_bytes();
        }
    };
    static_assert(sizeof(ResolveRegionBiteView) <= 4 * sizeof(uint64_t));
    
    struct APCStorageGeometry
    {
        using SD = SchemaDefinition;
        static constexpr size_t BytesPerLocalAddressUnit() noexcept
        {
            return sizeof(uint64_t);
        }

        static constexpr size_t ByteOffsetOfLocalIndex(uint32_t local_idx) noexcept
        {
            return static_cast<size_t>(local_idx) * BytesPerLocalAddressUnit();
        }

        static constexpr size_t ByteCountOfLOcalSpan(uint32_t local_span) noexcept
        {
            return static_cast<size_t>(local_span) * BytesPerLocalAddressUnit();
        }

        template<class DType>
        static constexpr bool CanInstallTypedSpan(
            const ResolveRegionBiteView& region,
            SD::DataTypeOfMacroColumn expected_dtype
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<DType>);

            if (
                !region.IsValid() ||
                region.Schema->Dtype != expected_dtype ||
                region.ByteCount() < sizeof(DType) ||
                (region.ByteCount() % sizeof(DType)) != UNSIGNED_ZERO
            ) 
            {
                return false;
            }

            const uintptr_t address = reinterpret_cast<uintptr_t>(region.Bytes.data());
            return (address % alignof(DType)) == UNSIGNED_ZERO;
        }

        template<class DType>
        static constexpr bool CanInstallTypedSpan(const ResolveRegionBiteView& region) noexcept
        {
            const std::optional<SD::DataTypeOfMacroColumn> expected_dtype = SD::CppTypeToRegionDType<DType>();
            return CanInstallTypedSpan<DType>(region, expected_dtype.value());
        }

        template<class DType>
        static constexpr bool CanInstallAtomicSpan(
            const ResolveRegionBiteView& region,
            SD::DataTypeOfMacroColumn expected_dtype
        ) noexcept
        {
            if (!CanInstallTypedSpan<DType>(region, expected_dtype)) 
            {
                return false;
            }
            const uintptr_t address = reinterpret_cast<uintptr_t>(region.Bytes.data());
            constexpr uint64_t required_alignment = std::atomic_ref<DType>::required_alignment;

            if ((address % required_alignment) != UNSIGNED_ZERO) 
            {
                return false;
            }

            return (sizeof(DType) % required_alignment) == UNSIGNED_ZERO;
        }

        template<class DType>
        static constexpr bool CanInstallAtomicSpan(const ResolveRegionBiteView& region) noexcept
        {
            const std::optional<SD::DataTypeOfMacroColumn> expected_dtype = SD::CppTypeToRegionDType<DType>();
            return expected_dtype.has_value() &&
                CanInstallAtomicSpan<DType>(region, expected_dtype.value());
        }

        template<class DType>
        static constexpr bool InitializeFreshRegionObject(
            const ResolveRegionBiteView& region
        )noexcept
        {
            static_assert(std::is_trivially_copyable_v<DType>);
            static_assert(std::is_default_constructible_v<DType>);

            if (!CanInstallTypedSpan<DType>(region))
            {
                return false;
            }
            
            DType* type_base = reinterpret_cast<DType*>(region.Bytes.data());

            const size_t count = region.ByteCount() / sizeof(DType);

            for (size_t i = 0; i < count; i++)
            {
                std::construct_at(type_base + i);
            }
            
            return true;
        }

        template<class DT>
        struct CacheOfSNCB
        {
            DT* Data_{nullptr};
            uint32_t Height_{UNSIGNED_ZERO};
            uint32_t CapacityWidth_{UNSIGNED_ZERO};
            uint32_t ActiveWidth_{UNSIGNED_ZERO};
            uint32_t Slot_{UNSIGNED_ZERO};
            MacroColumnOfAPC Column_{MacroColumnOfAPC::FREE_SLOT};
            SD::SchemaProtocols Protocol_{SD::SchemaProtocols::PRIVATE_REGION};
            APCUseScope Use_{};
        };
    };

    template<class DT>
    class SNTOComputerBridge final
    {
        friend class Computer;
        using CacheOfSNCB = APCStorageGeometry::CacheOfSNCB<DT>;
        using SD = SchemaDefinition;
    private:
        CacheOfSNCB SNCBCache_{};
        
        explicit SNTOComputerBridge(CacheOfSNCB&& cache) noexcept:
            SNCBCache_(std::move(cache))
        {}

        DT* DataPtr() noexcept
        {
            return static_cast<bool>(*this) ? SNCBCache_.Data_ : nullptr;
        }

        const DT* DataPtr() const noexcept
        {
            return static_cast<bool>(*this) ? SNCBCache_.Data_ : nullptr;
        }
    
    public:
        SNTOComputerBridge() noexcept = default;
        SNTOComputerBridge(const SNTOComputerBridge&) = delete;
        SNTOComputerBridge& operator=(SNTOComputerBridge&&) noexcept = default;

        explicit operator bool() const noexcept
        {
            return static_cast<bool>(SNCBCache_.Use_) &&
                SNCBCache_.Height_ != UNSIGNED_ZERO &&
                SNCBCache_.CapacityWidth_ != UNSIGNED_ZERO &&
                SNCBCache_.ActiveWidth_ != UNSIGNED_ZERO &&
                SNCBCache_.ActiveWidth_ <= SNCBCache_.CapacityWidth_;
        }


        uint32_t Height() const noexcept {return SNCBCache_.Height_; }
        uint32_t CapacityWidth() const noexcept { return SNCBCache_.CapacityWidth_; }
        uint32_t ActiveWidth() const noexcept { return SNCBCache_.ActiveWidth_; }
        uint32_t Slot() const noexcept { return SNCBCache_.Slot_; }
        MacroColumnOfAPC Column() const noexcept { return SNCBCache_.Column_; }
        SD::SchemaProtocols Protocol() const noexcept { return SNCBCache_.Protocol_; }


        uint64_t Size() const noexcept
        {
            if (
                !static_cast<bool>(*this) ||
                SNCBCache_.Height_ > (UINT64_MAX / static_cast<uint64_t>(SNCBCache_.CapacityWidth_))
            ) 
            {
                return UNSIGNED_ZERO;
            }

            return static_cast<uint64_t>(SNCBCache_.Height_) * SNCBCache_.CapacityWidth_;
        }

        uint64_t ActiveElementCount() const noexcept
        {
            if (!static_cast<bool>(*this)) {
                return UNSIGNED_ZERO;
            }

            if (SNCBCache_.Height_ > UINT64_MAX / static_cast<uint64_t>(SNCBCache_.ActiveWidth_)) {
                return UNSIGNED_ZERO;
            }

            return static_cast<uint64_t>(SNCBCache_.Height_) * static_cast<uint64_t>(SNCBCache_.ActiveWidth_);
        }

        std::span<DT> Row(uint32_t row) noexcept
        {
            if (!static_cast<bool>(*this) || row >= SNCBCache_.Height_) {
                return {};
            }

            return std::span<DT>(
                SNCBCache_.Data_ + static_cast<uint64_t>(row) * static_cast<uint64_t>(SNCBCache_.CapacityWidth_),
                SNCBCache_.ActiveWidth_
            );
        }

        std::span<const DT> Row(uint32_t row) const noexcept
        {
            if (!static_cast<bool>(*this) || row >= SNCBCache_.Height_) {
                return {};
            }

            return std::span<DT>(
                SNCBCache_.Data_ + static_cast<uint64_t>(row) * static_cast<uint64_t>(SNCBCache_.CapacityWidth_),
                SNCBCache_.ActiveWidth_
            );
        }


        std::optional<std::span<DT>> RawSpan() noexcept
        {
            if (
                !static_cast<bool>(*this) ||
                SNCBCache_.Protocol_ != SchemaDefinition::SchemaProtocols::PRIVATE_REGION
            ) 
            {
                return std::nullopt;
            }

            return std::span<DT>(SNCBCache_.Data_, Size());
        }

        
    };



    
    template<class DType>
    class RegionView
    {
    public:
        using SD = SchemaDefinition;

    private:
        SD::SchemaProtocols Protocol_;
        std::span<DType> Elements_;
        APCUseScope Use_;
    
    public:
        RegionView() noexcept = default;

        RegionView(
            std::span<DType> elements,
            SD::SchemaProtocols protocol,
            APCUseScope use
        ) noexcept:
            Elements_(elements),
            Protocol_(protocol),
            Use_(std::move(use))
        {}

        constexpr bool IsValid() const noexcept
        {
            return static_cast<bool>(Use_) && !Elements_.empty();
        }

        constexpr size_t Size() const noexcept
        {
            return Use_ ?  Elements_.size() : UNSIGNED_ZERO;
        }

        constexpr SD::SchemaProtocols GetProtocol() const noexcept
        {
            return Protocol_;
        }

        std::optional<std::span<DType>> RawSpan() noexcept
        {
            if (!Use_ || Protocol_ != SD::SchemaProtocols::PRIVATE_REGION)
            {
                return std::nullopt;
            }
            
            return Elements_;
        }

        DType AtomicLoad(size_t idx, std::memory_order mem_order = std::memory_order_acquire) const noexcept
        {
            if (
                !Use_ ||
                Protocol_ != SD::SchemaProtocols::ATOMIC_WORD_ARRAY ||
                idx >= Elements_.size()
            )
            {
                return DType{};
            }

            return std::atomic_ref<DType>(Elements_[idx]).load(mem_order);
            
        }


        bool AtomicStore(
            size_t idx,
            DType value,
            std::memory_order order = std::memory_order_release
        ) noexcept
        {
            if (
                !Use_ ||
                Protocol_ != SD::SchemaProtocols::ATOMIC_WORD_ARRAY ||
                idx >= Elements_.size()
            )
            {
                return false;
            }

            std::atomic_ref<DType>(Elements_[idx]).store(value, order);
            return true;
        }

        bool AtomicCompareExchangeStrong(
            size_t idx,
            DType& expected,
            DType desired,
            std::memory_order success = std::memory_order_acq_rel,
            std::memory_order failure = std::memory_order_acquire
        ) noexcept
        {
            if (
                !Use_ ||
                Protocol_ != SD::SchemaProtocols::ATOMIC_WORD_ARRAY ||
                idx >= Elements_.size()
            )
            {
                return false;
            }

            return std::atomic_ref<DType>(Elements_[idx]).compare_exchange_strong(
                expected,
                desired,
                success,
                failure
            );
        }

    };

    
}