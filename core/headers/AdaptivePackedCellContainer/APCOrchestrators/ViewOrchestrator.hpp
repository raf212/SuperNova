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
        static constexpr bool CanInstallTypedSpan(const ResolveRegionBiteView& region) noexcept
        {
            static_assert(std::is_trivially_copyable_v<DType>);
            const std::optional<SchemaOrchestrator::DataTypeOfMacroColumn> expected_dtype = SchemaDefinition::CppTypeToRegionDType<DType>();

            if (
                !region.IsValid() ||
                !expected_dtype.has_value() ||
                region.Schema->Dtype != expected_dtype.value() ||
                region.ByteCount() < sizeof(DType) ||
                (region.ByteCount() % sizeof(DType)) != UNSIGNED_ZERO
            )
            {
                return false;
            }

            const uintptr_t address = reinterpret_cast<uintptr_t>(region.Bytes.data());

            return (address % alignof(DType)) == UNSIGNED_ZERO;
            
        }

        template<class Dtype>
        static constexpr bool CanInstallAtomicSpan(const ResolveRegionBiteView& region) noexcept
        {
            if (!CanInstallTypedSpan<Dtype>(region))
            {
                return false;
            }
            
            const uintptr_t address = reinterpret_cast<uintptr_t>(region.Bytes.data());

            constexpr size_t required_alignment = std::atomic_ref<Dtype>::required_alignment;
            
            if ((address % required_alignment) != UNSIGNED_ZERO)
            {
                return false;
            }

            return (sizeof(Dtype) % required_alignment) == UNSIGNED_ZERO;
            
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
    };

    
    template<class DType>
    class RegionView
    {
    public:
        using SD = SchemaDefinition;

    private:
        SD::SchemaProtocols Protocol_{SD::SchemaProtocols::PRIVATE_REGION};
        std::span<DType> Elements_{};
        APCUseScope Use_{};
    
    public:
        constexpr RegionView() noexcept = default;

        constexpr RegionView(
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

        std::optional<std::span<DType>> RawMutableSpan() noexcept
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