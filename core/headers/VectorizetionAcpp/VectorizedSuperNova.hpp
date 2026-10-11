#pragma once

// ============================================================================
// VectorizedSuperNova
// ============================================================================
//
// Model-neutral heterogeneous execution layer directly above APCFinilizer.
//
// Architectural boundary:
//
//   APC / Fabric / APCFinilizer
//              |
//              v
//      VectorizedSuperNova
//       /       |        \
//      CPU      GPU     CPU+GPU
//              |
//              v
//        model consumers
//         (GHGF, ...)
// 
// Responsibilities:
//   * preserve APC/Fabric lifetime and schema semantics;
//   * optionally allocate the actual Fabric slab as SYCL shared USM;
//   * expose pinned typed numeric regions;
//   * provide synchronous vector execution primitives;
//   * choose CPU/GPU/cooperative execution without model semantics.
//
// Non-responsibilities:
//   * DAG mutation;
//   * generation/state transitions;
//   * structural seqlock algorithms;
//   * model-specific equations.
//
// Vector kernels must only touch numeric regions that remain pinned for the
// whole operation. Structural control remains host/Fabric owned.
// ============================================================================

#include "../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
    #include <sycl/sycl.hpp>
#endif

namespace BidirectionalInMemGraph
{

    class VectorizedSuperNova : protected APCFinilizer
    {
    public:
        using SD = SchemaDefinition;
        using CFC = CoreOfFabricCoordinator;

        enum class VectorBackend : uint8_t
        {
            CPU = 0,
            GPU = 1,
            COOPERATIVE = 2,
            AUTOMATIC = 3
        };

        enum class VectorRuntimeError : uint8_t
        {
            NONE = 0,
            INVALID_CONFIGURATION = 1,
            CONFIGURED_AFTER_FABRIC_INITIALIZATION = 2,
            ACCELERATOR_UNAVAILABLE = 3,
            SHARED_USM_UNAVAILABLE = 4,
            INVALID_REGION = 5,
            INVALID_RANGE = 6,
            DEVICE_EXECUTION_FAILED = 7,
            EXTERNAL_FABRIC_IS_HOST_ONLY = 8
        };

        struct VectorExecutionConfig final
        {
            VectorBackend Backend = VectorBackend::AUTOMATIC;

            // AUTOMATIC keeps small lane counts on the host. A GPU launch for
            // a tiny GHGF batch is normally more expensive than a CPU loop.
            size_t GPUElementThreshold = 4096u;

            // Used only by COOPERATIVE. CPU owns the prefix and GPU owns the
            // disjoint suffix, then the call synchronizes before returning.
            float CooperativeCPUFraction = 0.25f;

            // GPU/COOPERATIVE may degrade to CPU when no GPU is visible.
            bool AllowCPUFallback = true;

            // When a GPU is available, create the Fabric slab as shared USM so
            // host control code and device kernels address the same allocation.
            bool PreferSharedUSMSlab = true;
        };

        struct VectorRuntimeInfo final
        {
            VectorBackend ConfiguredBackend = VectorBackend::CPU;
            bool AdaptiveCppCompiled = false;
            bool GPUAvailable = false;
            bool SlabUsesSharedUSM = false;
            bool RuntimeConfigured = false;
            std::string DeviceName{};
        };

        template<class T>
        class PinnedVectorRegion final
        {
            friend class VectorizedSuperNova;

        private:
            T* Data_{nullptr};
            uint32_t Height_{UNSIGNED_ZERO};
            uint32_t CapacityWidth_{UNSIGNED_ZERO};
            uint32_t ActiveWidth_{UNSIGNED_ZERO};
            uint32_t Slot_{ADS::APC_INDEX_BOUND_SENTINAL};
            MacroColumnOfAPC Column_{MacroColumnOfAPC::FREE_SLOT};
            APCUseScope Use_{};

            PinnedVectorRegion(
                T* data,
                uint32_t height,
                uint32_t capacity_width,
                uint32_t active_width,
                uint32_t slot,
                MacroColumnOfAPC column,
                APCUseScope use
            ) noexcept
                : Data_(data),
                  Height_(height),
                  CapacityWidth_(capacity_width),
                  ActiveWidth_(active_width),
                  Slot_(slot),
                  Column_(column),
                  Use_(std::move(use))
            {}

        public:
            PinnedVectorRegion() noexcept = default;

            PinnedVectorRegion(const PinnedVectorRegion&) = delete;
            PinnedVectorRegion& operator=(const PinnedVectorRegion&) = delete;

            PinnedVectorRegion(PinnedVectorRegion&&) noexcept = default;
            PinnedVectorRegion& operator=(PinnedVectorRegion&&) noexcept = default;

            explicit operator bool() const noexcept
            {
                return static_cast<bool>(Use_) &&
                    Data_ != nullptr &&
                    Height_ != UNSIGNED_ZERO &&
                    CapacityWidth_ != UNSIGNED_ZERO &&
                    ActiveWidth_ != UNSIGNED_ZERO &&
                    ActiveWidth_ <= CapacityWidth_;
            }

            T* Data() noexcept
            {
                return static_cast<bool>(*this) ? Data_ : nullptr;
            }

            const T* Data() const noexcept
            {
                return static_cast<bool>(*this) ? Data_ : nullptr;
            }

            uint32_t Height() const noexcept
            {
                return Height_;
            }

            uint32_t CapacityWidth() const noexcept
            {
                return CapacityWidth_;
            }

            uint32_t ActiveWidth() const noexcept
            {
                return ActiveWidth_;
            }

            uint32_t Slot() const noexcept
            {
                return Slot_;
            }

            MacroColumnOfAPC Column() const noexcept
            {
                return Column_;
            }

            std::span<T> Row(uint32_t row) noexcept
            {
                if (
                    !static_cast<bool>(*this) ||
                    row >= Height_
                )
                {
                    return {};
                }

                return std::span<T>(
                    Data_ + static_cast<size_t>(row) * CapacityWidth_,
                    ActiveWidth_
                );
            }

            std::span<const T> Row(uint32_t row) const noexcept
            {
                if (
                    !static_cast<bool>(*this) ||
                    row >= Height_
                )
                {
                    return {};
                }

                return std::span<const T>(
                    Data_ + static_cast<size_t>(row) * CapacityWidth_,
                    ActiveWidth_
                );
            }
        };

    private:
        VectorExecutionConfig VectorConfig_{};
        VectorRuntimeError LastVectorError_{VectorRuntimeError::NONE};

        bool VectorRuntimeConfigured_{false};
        bool GPUAvailable_{false};
        bool SlabUsesSharedUSM_{false};

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
        std::optional<sycl::queue> GPUQueue_{}; 
        std::string GPUDeviceName_{};
#endif

        void SetVectorError_(VectorRuntimeError error) noexcept
        {
            LastVectorError_ = error;
        }

        void ResetFabricAllocator_() noexcept
        {
            AllocatorOfFabric_ = RawPackedCellAllocator{};
        }

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
        static uint64_t* AllocateSharedSlab_(
            size_t count_of_cells,
            size_t alignment,
            void* user
        ) noexcept
        {
            auto* const self = static_cast<VectorizedSuperNova*>(user);

            if (
                !self ||
                !self->GPUQueue_.has_value() ||
                count_of_cells == UNSIGNED_ZERO ||
                count_of_cells > SIZE_MAX / sizeof(uint64_t)
            )
            {
                return nullptr;
            }

            alignment = std::max<size_t>(
                alignment,
                ADS::APC_CACHELINE_SIZE
            );
            alignment = std::max<size_t>(
                alignment,
                alignof(uint64_t)
            );

            try
            {
                void* memory = sycl::aligned_alloc_shared(
                    alignment,
                    count_of_cells * sizeof(uint64_t),
                    self->GPUQueue_.value()
                );

                return static_cast<uint64_t*>(memory);
            }
            catch (...)
            {
                self->SetVectorError_(
                    VectorRuntimeError::SHARED_USM_UNAVAILABLE
                );
                return nullptr;
            }
        }

        static void FreeSharedSlab_(
            uint64_t* memory,
            size_t,
            size_t,
            void* user
        ) noexcept
        {
            auto* const self = static_cast<VectorizedSuperNova*>(user);

            if (!memory || !self || !self->GPUQueue_.has_value())
            {
                return;
            }

            try
            {
                sycl::free(
                    static_cast<void*>(memory),
                    self->GPUQueue_.value()
                );
            }
            catch (...)
            {
                // Free functions are noexcept by Fabric contract. A failed
                // backend free cannot be propagated from this callback.
            }
        }

        bool ConfigureGPUQueue_() noexcept
        {
            GPUQueue_.reset();
            GPUDeviceName_.clear();
            GPUAvailable_ = false;

            try
            {
                sycl::device device{sycl::gpu_selector_v};
                GPUQueue_.emplace(
                    device,
                    sycl::property_list{
                        sycl::property::queue::in_order{}
                    }
                );

                GPUDeviceName_ =
                    device.get_info<sycl::info::device::name>();

                GPUAvailable_ = true;
                return true;
            }
            catch (...)
            {
                GPUQueue_.reset();
                GPUDeviceName_.clear();
                GPUAvailable_ = false;
                return false;
            }
        }

        bool ProbeSharedUSM_() noexcept
        {
            if (!GPUQueue_.has_value())
            {
                return false;
            }

            try
            {
                constexpr size_t PROBE_ALIGNMENT =
                    ADS::APC_CACHELINE_SIZE;
                constexpr size_t PROBE_BYTES =
                    ADS::APC_CACHELINE_SIZE;

                void* probe = sycl::aligned_alloc_shared(
                    PROBE_ALIGNMENT,
                    PROBE_BYTES,
                    GPUQueue_.value()
                );

                if (!probe)
                {
                    return false;
                }

                sycl::free(probe, GPUQueue_.value());
                return true;
            }
            catch (...)
            {
                return false;
            }
        }
#endif

        bool ConfigureRuntimeBeforeFabric_() noexcept
        {
            if (VectorRuntimeConfigured_)
            {
                return true;
            }

            return ConfigureVectorExecution(VectorConfig_);
        }

        bool IsRangeInsideSlab_(
            const void* data,
            size_t bytes
        ) const noexcept
        {
            if (
                !data ||
                bytes == UNSIGNED_ZERO ||
                !SlabBasePtr_ ||
                !FabCache_ ||
                FabCache_->SlabCellCount_ >
                    SIZE_MAX / sizeof(uint64_t)
            )
            {
                return false;
            }

            const uintptr_t begin =
                reinterpret_cast<uintptr_t>(data);
            const uintptr_t slab =
                reinterpret_cast<uintptr_t>(SlabBasePtr_);

            const size_t slab_bytes =
                static_cast<size_t>(FabCache_->SlabCellCount_) *
                sizeof(uint64_t);

            if (begin < slab)
            {
                return false;
            }

            const uintptr_t offset = begin - slab;

            return
                offset <= slab_bytes &&
                bytes <= slab_bytes - static_cast<size_t>(offset);
        }

        template<class T>
        bool IsTypedRangeInsideSlab_(
            const T* data,
            size_t count
        ) const noexcept
        {
            if (count > SIZE_MAX / sizeof(T))
            {
                return false;
            }

            return IsRangeInsideSlab_(
                data,
                count * sizeof(T)
            );
        }

        VectorBackend ResolveBackend_(
            size_t element_count,
            VectorBackend requested
        ) const noexcept
        {
            VectorBackend backend =
                requested == VectorBackend::AUTOMATIC ?
                    VectorConfig_.Backend : requested;

            if (backend == VectorBackend::AUTOMATIC)
            {
                if (
                    GPUAvailable_ &&
                    SlabUsesSharedUSM_ &&
                    element_count >=
                        VectorConfig_.GPUElementThreshold
                )
                {
                    return VectorBackend::GPU;
                }

                return VectorBackend::CPU;
            }

            if (
                (backend == VectorBackend::GPU ||
                 backend == VectorBackend::COOPERATIVE) &&
                (!GPUAvailable_ || !SlabUsesSharedUSM_)
            )
            {
                return VectorConfig_.AllowCPUFallback ?
                    VectorBackend::CPU : backend;
            }

            return backend;
        }

        template<class Operation>
        static bool ExecuteCPU_(
            size_t begin,
            size_t end,
            Operation operation
        ) noexcept
        {
            try
            {
                for (size_t i = begin; i < end; ++i)
                {
                    operation(i);
                }
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
        template<class Operation>
        bool ExecuteGPU_(
            size_t begin,
            size_t end,
            Operation operation
        ) noexcept
        {
            static_assert(
                std::is_trivially_copyable_v<Operation>,
                "Device vector operations must be trivially copyable."
            );

            if (
                !GPUQueue_.has_value() ||
                end <= begin
            )
            {
                return end == begin;
            }

            try
            {
                const size_t count = end - begin;

                GPUQueue_.value()
                    .parallel_for(
                        sycl::range<1>{count},
                        [=](sycl::id<1> id)
                        {
                            operation(
                                begin +
                                static_cast<size_t>(id[0])
                            );
                        }
                    )
                    .wait_and_throw();

                return true;
            }
            catch (...)
            {
                SetVectorError_(
                    VectorRuntimeError::DEVICE_EXECUTION_FAILED
                );
                return false;
            }
        }

        template<class Operation>
        bool ExecuteCooperatively_(
            size_t element_count,
            Operation operation
        ) noexcept
        {
            static_assert(
                std::is_trivially_copyable_v<Operation>,
                "Cooperative vector operations must be trivially copyable."
            );

            if (
                !GPUQueue_.has_value() ||
                element_count == UNSIGNED_ZERO
            )
            {
                return ExecuteCPU_(
                    UNSIGNED_ZERO,
                    element_count,
                    operation
                );
            }

            const float fraction = std::clamp(
                VectorConfig_.CooperativeCPUFraction,
                0.0f,
                1.0f
            );

            size_t cpu_count = static_cast<size_t>(
                static_cast<double>(element_count) *
                static_cast<double>(fraction)
            );

            cpu_count = std::min(cpu_count, element_count);

            if (cpu_count == element_count)
            {
                return ExecuteCPU_(
                    UNSIGNED_ZERO,
                    element_count,
                    operation
                );
            }

            try
            {
                const size_t gpu_count =
                    element_count - cpu_count;

                sycl::event gpu_event =
                    GPUQueue_.value().parallel_for(
                        sycl::range<1>{gpu_count},
                        [=](sycl::id<1> id)
                        {
                            operation(
                                cpu_count +
                                static_cast<size_t>(id[0])
                            );
                        }
                    );

                const bool cpu_ok = ExecuteCPU_(
                    UNSIGNED_ZERO,
                    cpu_count,
                    operation
                );

                gpu_event.wait_and_throw();

                return cpu_ok;
            }
            catch (...)
            {
                SetVectorError_(
                    VectorRuntimeError::DEVICE_EXECUTION_FAILED
                );
                return false;
            }
        }
#endif

    protected:
        // --------------------------------------------------------------------
        // Fabric lifecycle boundary
        // --------------------------------------------------------------------
        //
        // Models should call this inherited name exactly as they called
        // APCFinilizer::InitializeFabric before. The vector layer configures
        // the allocator first, then delegates unchanged Fabric construction.
        bool InitializeFabric(
            uint32_t slot_count,
            uint32_t slot_cell_count,
            const SD::FabricRegionConfig& region_conf,
            uint32_t max_direct_parent_per_axis =
                ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS
        ) noexcept
        {
            if (!ConfigureRuntimeBeforeFabric_())
            {
                return false;
            }

            const bool shared_allocator =
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
                AllocatorOfFabric_.AllocatePackedCellStorage ==
                    &VectorizedSuperNova::AllocateSharedSlab_;
#else
                false;
#endif

            const bool initialized =
                APCFinilizer::InitializeFabric(
                    slot_count,
                    slot_cell_count,
                    region_conf,
                    max_direct_parent_per_axis
                );

            if (!initialized)
            {
                return false;
            }

            SlabUsesSharedUSM_ = shared_allocator;
            return true;
        }

        // External/relocated ordinary host slabs remain valid SuperNova memory,
        // but are conservatively CPU-only because this layer cannot prove that
        // an arbitrary pointer belongs to this SYCL context.
        bool AttachFabric(
            uint64_t* raw_cells,
            uint64_t cell_count,
            CFC::FabricBackigOwnership ownership =
                CFC::FabricBackigOwnership::BORROWED
        ) noexcept
        {
            if (!SynchronizeVectorExecution())
            {
                return false;
            }

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            const bool shared_allocator_configured =
                AllocatorOfFabric_.AllocatePackedCellStorage ==
                    &VectorizedSuperNova::AllocateSharedSlab_;

            // An externally-owned host allocation must never later be released
            // through sycl::free(). Borrowed snapshots are safe and remain
            // CPU-only until copied into a vector-owned Fabric.
            if (
                shared_allocator_configured &&
                ownership == CFC::FabricBackigOwnership::OWNED
            )
            {
                SetVectorError_(
                    VectorRuntimeError::EXTERNAL_FABRIC_IS_HOST_ONLY
                );
                return false;
            }
#endif

            const bool attached =
                APCFinilizer::AttachFabric(
                    raw_cells,
                    cell_count,
                    ownership
                );

            if (attached)
            {
                SlabUsesSharedUSM_ = false;
            }

            return attached;
        }

        // --------------------------------------------------------------------
        // Model-neutral region pinning
        // --------------------------------------------------------------------
        template<class T>
        std::optional<PinnedVectorRegion<T>>
        PinVectorRegion_(
            uint32_t slot,
            MacroColumnOfAPC column,
            uint32_t active_width = UNSIGNED_ZERO
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);

            if (
                !IsFabricActive() ||
                !FabCache_ ||
                slot >= FabCache_->CountOfAPC_
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_REGION
                );
                return std::nullopt;
            }

            const auto dtype =
                SD::CppTypeToRegionDType<T>();

            const auto compact =
                ADS::CompactRegionIndex(
                    FabCache_->ActiveRegionMask_,
                    column
                );

            if (
                !dtype.has_value() ||
                !compact.has_value()
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_REGION
                );
                return std::nullopt;
            }

            std::span<SD::RegionSchemaRecord> table =
                MetrixViewRow_(slot);

            if (compact.value() >= table.size())
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_REGION
                );
                return std::nullopt;
            }

            const SD::RegionSchemaRecord& schema =
                table[compact.value()];

            if (
                schema.Region != column ||
                schema.Dtype != dtype.value() ||
                schema.Protocol !=
                    SD::SchemaProtocols::PRIVATE_REGION ||
                schema.MatrixHeight == UNSIGNED_ZERO ||
                schema.MatrixWidth == UNSIGNED_ZERO
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_REGION
                );
                return std::nullopt;
            }

            const uint32_t width =
                active_width == UNSIGNED_ZERO ?
                    schema.MatrixWidth : active_width;

            if (
                width == UNSIGNED_ZERO ||
                width > schema.MatrixWidth
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return std::nullopt;
            }

            AdaptivePackedCellContainer apc{};
            APCUseScope use{};

            if (!GetExistingAPC_(slot, apc, use))
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_REGION
                );
                return std::nullopt;
            }

            T* const data =
                RegionT_<T>(
                    slot,
                    schema.CellOffset
                );

            if (!data)
            {
                return std::nullopt;
            }

            return PinnedVectorRegion<T>(
                data,
                schema.MatrixHeight,
                schema.MatrixWidth,
                width,
                slot,
                column,
                std::move(use)
            );
        }

        // --------------------------------------------------------------------
        // Device-safe math helpers for model kernels
        // --------------------------------------------------------------------
        static float VectorExpValue_(float value) noexcept
        {
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            return sycl::exp(value);
#else
            return std::exp(value);
#endif
        }

        static float VectorAbsValue_(float value) noexcept
        {
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            return sycl::fabs(value);
#else
            return std::fabs(value);
#endif
        }

        static bool VectorFiniteValue_(float value) noexcept
        {
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            return sycl::isfinite(value);
#else
            return std::isfinite(value);
#endif
        }

        template<class T>
        static constexpr T VectorClampValue_(
            T value,
            T lower,
            T upper
        ) noexcept
        {
            return
                value < lower ? lower :
                value > upper ? upper :
                value;
        }

        // --------------------------------------------------------------------
        // Generic synchronous task execution
        // --------------------------------------------------------------------
        template<class Operation>
        bool VectorFor_(
            size_t element_count,
            Operation operation,
            VectorBackend requested =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            if (element_count == UNSIGNED_ZERO)
            {
                return true;
            }

            const VectorBackend backend =
                ResolveBackend_(
                    element_count,
                    requested
                );

            switch (backend)
            {
            case VectorBackend::CPU:
                return ExecuteCPU_(
                    UNSIGNED_ZERO,
                    element_count,
                    operation
                );

            case VectorBackend::GPU:
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
                if (GPUAvailable_ && SlabUsesSharedUSM_)
                {
                    return ExecuteGPU_(
                        UNSIGNED_ZERO,
                        element_count,
                        operation
                    );
                }
#endif
                SetVectorError_(
                    VectorRuntimeError::ACCELERATOR_UNAVAILABLE
                );
                return false;

            case VectorBackend::COOPERATIVE:
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
                if (GPUAvailable_ && SlabUsesSharedUSM_)
                {
                    return ExecuteCooperatively_(
                        element_count,
                        operation
                    );
                }
#endif
                SetVectorError_(
                    VectorRuntimeError::ACCELERATOR_UNAVAILABLE
                );
                return false;

            case VectorBackend::AUTOMATIC:
            default:
                return false;
            }
        }

        // --------------------------------------------------------------------
        // Reusable vector primitives for model implementations
        // --------------------------------------------------------------------
        template<class T>
        bool VectorFill_(
            T* destination,
            size_t count,
            T value,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);

            if (
                !destination ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    destination[i] = value;
                },
                backend
            );
        }

        template<class T>
        bool VectorCopy_(
            T* destination,
            const T* source,
            size_t count,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);

            if (
                !destination ||
                !source ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    source,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    destination[i] = source[i];
                },
                backend
            );
        }

        template<class T>
        bool VectorAffine_(
            T* destination,
            const T* source,
            size_t count,
            T scale,
            T bias,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_arithmetic_v<T>);

            if (
                !destination ||
                !source ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    source,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    destination[i] =
                        scale * source[i] + bias;
                },
                backend
            );
        }

        template<class T>
        bool VectorAccumulateScaled_(
            T* destination,
            const T* source,
            size_t count,
            T scale,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_arithmetic_v<T>);

            if (
                !destination ||
                !source ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    source,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    destination[i] +=
                        scale * source[i];
                },
                backend
            );
        }

        template<class T>
        bool VectorMultiply_(
            T* destination,
            const T* left,
            const T* right,
            size_t count,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_arithmetic_v<T>);

            if (
                !destination ||
                !left ||
                !right ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    left,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    right,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    destination[i] =
                        left[i] * right[i];
                },
                backend
            );
        }

        template<class T>
        bool VectorClamp_(
            T* destination,
            const T* source,
            size_t count,
            T lower,
            T upper,
            VectorBackend backend =
                VectorBackend::AUTOMATIC
        ) noexcept
        {
            static_assert(std::is_arithmetic_v<T>);

            if (
                !destination ||
                !source ||
                count == UNSIGNED_ZERO ||
                lower > upper ||
                !IsTypedRangeInsideSlab_(
                    destination,
                    count
                ) ||
                !IsTypedRangeInsideSlab_(
                    source,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

            return VectorFor_(
                count,
                [=](size_t i) noexcept
                {
                    const T value = source[i];
                    destination[i] =
                        value < lower ? lower :
                        value > upper ? upper :
                        value;
                },
                backend
            );
        }

        // External host memory is intentionally handled through an explicit
        // transfer primitive. GPU kernels must never assume an arbitrary model
        // input pointer is device-accessible.
        template<class T>
        bool VectorCopyFromHost_(
            T* slab_destination,
            const T* host_source,
            size_t count
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);

            if (
                !slab_destination ||
                !host_source ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    slab_destination,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            if (
                GPUAvailable_ &&
                SlabUsesSharedUSM_ &&
                GPUQueue_.has_value()
            )
            {
                try
                {
                    GPUQueue_.value()
                        .memcpy(
                            slab_destination,
                            host_source,
                            count * sizeof(T)
                        )
                        .wait_and_throw();

                    return true;
                }
                catch (...)
                {
                    SetVectorError_(
                        VectorRuntimeError::DEVICE_EXECUTION_FAILED
                    );
                    return false;
                }
            }
#endif

            std::copy_n(
                host_source,
                count,
                slab_destination
            );
            return true;
        }

        template<class T>
        bool VectorCopyToHost_(
            T* host_destination,
            const T* slab_source,
            size_t count
        ) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);

            if (
                !host_destination ||
                !slab_source ||
                count == UNSIGNED_ZERO ||
                !IsTypedRangeInsideSlab_(
                    slab_source,
                    count
                )
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_RANGE
                );
                return false;
            }

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            if (
                GPUAvailable_ &&
                SlabUsesSharedUSM_ &&
                GPUQueue_.has_value()
            )
            {
                try
                {
                    GPUQueue_.value()
                        .memcpy(
                            host_destination,
                            slab_source,
                            count * sizeof(T)
                        )
                        .wait_and_throw();

                    return true;
                }
                catch (...)
                {
                    SetVectorError_(
                        VectorRuntimeError::DEVICE_EXECUTION_FAILED
                    );
                    return false;
                }
            }
#endif

            std::copy_n(
                slab_source,
                count,
                host_destination
            );
            return true;
        }

    public:
        VectorizedSuperNova() noexcept = default;

        ~VectorizedSuperNova() noexcept
        {
            // The Fabric base destructor runs after derived members have been
            // destroyed. Shared-USM deallocation therefore must happen while
            // GPUQueue_ is still alive.
            ShutDownFabric();
        }

        VectorizedSuperNova(
            const VectorizedSuperNova&
        ) = delete;

        VectorizedSuperNova& operator=(
            const VectorizedSuperNova&
        ) = delete;

        bool ConfigureVectorExecution(
            const VectorExecutionConfig& config
        ) noexcept
        {
            if (IsFabricActive())
            {
                SetVectorError_(
                    VectorRuntimeError::
                        CONFIGURED_AFTER_FABRIC_INITIALIZATION
                );
                return false;
            }

            if (
                !std::isfinite(
                    config.CooperativeCPUFraction
                ) ||
                config.CooperativeCPUFraction < 0.0f ||
                config.CooperativeCPUFraction > 1.0f
            )
            {
                SetVectorError_(
                    VectorRuntimeError::INVALID_CONFIGURATION
                );
                return false;
            }

            VectorConfig_ = config;
            VectorRuntimeConfigured_ = false;
            GPUAvailable_ = false;
            SlabUsesSharedUSM_ = false;
            ResetFabricAllocator_();

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            const bool wants_gpu =
                config.Backend != VectorBackend::CPU;

            if (wants_gpu)
            {
                ConfigureGPUQueue_();
            }
            else
            {
                GPUQueue_.reset();
                GPUDeviceName_.clear();
            }

            if (
                (config.Backend == VectorBackend::GPU ||
                 config.Backend == VectorBackend::COOPERATIVE) &&
                !GPUAvailable_ &&
                !config.AllowCPUFallback
            )
            {
                SetVectorError_(
                    VectorRuntimeError::ACCELERATOR_UNAVAILABLE
                );
                return false;
            }

            if (
                GPUAvailable_ &&
                config.PreferSharedUSMSlab
            )
            {
                if (!ProbeSharedUSM_())
                {
                    if (!config.AllowCPUFallback)
                    {
                        SetVectorError_(
                            VectorRuntimeError::
                                SHARED_USM_UNAVAILABLE
                        );
                        return false;
                    }

                    GPUQueue_.reset();
                    GPUDeviceName_.clear();
                    GPUAvailable_ = false;
                }
                else
                {
                    AllocatorOfFabric_.
                        AllocatePackedCellStorage =
                            &VectorizedSuperNova::
                                AllocateSharedSlab_;

                    AllocatorOfFabric_.
                        FreePackedCellStorage =
                            &VectorizedSuperNova::
                                FreeSharedSlab_;

                    AllocatorOfFabric_.User = this;
                    AllocatorOfFabric_.Alignment =
                        std::max<size_t>(
                            AllocatorOfFabric_.Alignment,
                            ADS::APC_CACHELINE_SIZE
                        );

                    SlabUsesSharedUSM_ = true;
                }
            }
#else
            if (
                (config.Backend == VectorBackend::GPU ||
                 config.Backend == VectorBackend::COOPERATIVE) &&
                !config.AllowCPUFallback
            )
            {
                SetVectorError_(
                    VectorRuntimeError::ACCELERATOR_UNAVAILABLE
                );
                return false;
            }
#endif

            VectorRuntimeConfigured_ = true;
            SetVectorError_(VectorRuntimeError::NONE);
            return true;
        }

        bool SynchronizeVectorExecution() noexcept
        {
#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            if (GPUQueue_.has_value())
            {
                try
                {
                    GPUQueue_.value().wait_and_throw();
                }
                catch (...)
                {
                    SetVectorError_(
                        VectorRuntimeError::DEVICE_EXECUTION_FAILED
                    );
                    return false;
                }
            }
#endif
            return true;
        }

        void ShutDownFabric() noexcept
        {
            (void)SynchronizeVectorExecution();
            APCFinilizer::ShutDownFabric();
            SlabUsesSharedUSM_ = false;
        }

        bool SaveFabric(
            std::span<uint64_t> destination
        ) noexcept
        {
            return
                SynchronizeVectorExecution() &&
                APCFinilizer::SaveFabric(destination);
        }

        bool IsFabricActive() noexcept
        {
            return APCFinilizer::IsFabricActive();
        }

        VectorRuntimeError LastVectorError() const noexcept
        {
            return LastVectorError_;
        }

        VectorExecutionConfig GetVectorExecutionConfig() const noexcept
        {
            return VectorConfig_;
        }

        VectorRuntimeInfo GetVectorRuntimeInfo() const
        {
            VectorRuntimeInfo info{};
            info.ConfiguredBackend = VectorConfig_.Backend;
            info.GPUAvailable = GPUAvailable_;
            info.SlabUsesSharedUSM = SlabUsesSharedUSM_;
            info.RuntimeConfigured = VectorRuntimeConfigured_;

#if defined(SUPERNOVA_ENABLE_ADAPTIVECPP)
            info.AdaptiveCppCompiled = true;
            info.DeviceName = GPUDeviceName_;
#else
            info.AdaptiveCppCompiled = false;
#endif

            return info;
        }
    };

}
