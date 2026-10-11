#pragma once
#include "../NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

#include <algorithm>
#include <atomic>
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


///TEMPORARY
#define SN_ACPP_ENABLED true
//just for inselisense

#if defined (SN_ACPP_ENABLED)
    #include <sycl/sycl.hpp>
#endif

namespace BidirectionalInMemGraph 
{
    class Computer;

    struct ComputerDefAndVect
    {
        static constexpr uint16_t GPU_WORK_THRESHOLD = 4096;
        static constexpr float COOP_CPU_FRACTION = 0.25f;

        using SD = SchemaDefinition;
        using CDT = SD::DataTypeOfMacroColumn;

        enum class ComputerBackend : uint8_t
        {
            AUTOMATIC = 0,
            CPU = 1,
            GPU = 2,
            COOPERATIVE = 3
        };

        enum class ComputerWorkType : uint8_t
        {
            MEMORY = 0,
            ELEMENTWISE = 1,
            TRANSCENDENTAL = 2,
            REDUCTION = 3,
            LINIAR_ALGEBRA = 4,
            SPARSE_ATOMIC = 5,
            CUSTOM = 6
        };

        enum class ComputerFutureUse : uint8_t
        {
            UNKNOWN = 0,
            HOST = 1,
            ACCELERAROR = 3
        };

        enum class ComputerError : uint8_t
        {
            NONE = 0,
            INVALID_CONF = 1,
            CONFIGIRED_AFTER_FABRIC_INIT = 2,
            ACCELERATOR_UNAVAILAVLE = 3,
            SHARED_USM_UNAVAILAVLE = 4,
            CONCURRENT_SHARED_USM_UNAVAILAVLE = 5,
            INVALID_REGION = 6,
            INVALID_RANGE = 7,
            INVALIDE_DTYPE = 8,
            DEVICE_EXECUTION_FAILED = 9,
            ATOMIC_CAPABILITY_UNAVAILAVLE = 10,
            PORTABLE_DETACH_FAILED = 11
        };

        struct ComputerConf final
        {
            ComputerBackend DefaultBackend = ComputerBackend::AUTOMATIC;
            uint64_t GPUWorkThreshold = GPU_WORK_THRESHOLD;
            float CoOpCPUFraction = COOP_CPU_FRACTION;
            bool AllowCPUFallback = true;
            bool PrefereSharedUSM = true;
            bool EnablePrefetch = true;
        };

        struct ComputerRuntimeInfo final 
        {
            ComputerBackend Backend = ComputerBackend::CPU;
            bool AdaptiveCppCompiled = false;
            bool AcceleratorAvailable = false;
            bool SharedUSMAvailable = false;
            bool Atomic64Available = false;
            bool ConcurrentSharedUSM = false;
            bool SlabUsesSharedUSM = false;
            bool Configured = false;
            std::string DeviceName{}; 
        };

        struct ComputeExecutionPolicy final
        {
            ComputerBackend Backend = ComputerBackend::AUTOMATIC;
            ComputerWorkType WorkType = ComputerWorkType::ELEMENTWISE;
            ComputerFutureUse FutureUse = ComputerFutureUse::UNKNOWN;
            bool Deterministic = false;
            bool AllowCPUFallback = true;
        };

        template<class DT>
        inline static constexpr bool ComputerSclarV = std::is_same_v<std::remove_cv_t<DT>, uint8_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint16_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint32_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint64_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int8_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int16_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int32_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int64_t> ||
            std::is_same_v<std::remove_cv_t<DT>, float> ||
            std::is_same_v<std::remove_cv_t<DT>, double> ||
            std::is_same_v<std::remove_cv_t<DT>, char> ||
        #if defined (SN_ACPP_ENABLED)
            std::is_same_v<std::remove_cv_t<DT>, sycl::half>
        #endif
        ;

        template<class DT>
        inline static constexpr bool ComputerFloatingV = std::is_same_v<std::remove_cv_t<DT>, float> ||
            std::is_same_v<std::remove_cv_t<DT>, double> ||
        #if defined (SN_ACPP_ENABLED)
            std::is_same_v<std::remove_cv_t<DT>, sycl::half>
        #endif
        ;

        template<class DT> 
        struct SmallScalarPromotion
        {
            using U = std::remove_cv_t<DT>;
            using Type = std::conditional_t<
                std::is_integral_v<U> && sizeof(U) < sizeof(int32_t),
                int32_t,
                U
            >;
        };

        #if defined (SN_ACPP_ENABLED)
        template<>
        struct SmallScalarPromotion<sycl::half>
        {
            using Type = float;
        };
        #endif


        template<class... Ts>
        using DefaultAccumulatorSN = std::common_type_t<
            typename SmallScalarPromotion<std::remove_cv_t<Ts>>::Type...
        >;

        template<class DT>
        static constexpr std::optional<CDT> DataTypeOfSN() noexcept
        {
            using U = std::remove_cv_t<DT>;
            #if defined (SN_ACPP_ENABLED)
            if constexpr (std::is_same_v<U, sycl::half>)
            {
                return CDT::FLOAT16_T;
            }
            #endif
            {
                return SD::CppTypeToRegionDType<U>();
            }
        }

        struct MatMulSNDesc final
        {
            CDT DestinationType{CDT::FLOAT32_T};
            CDT LeftType{CDT::FLOAT32_T};
            CDT RightType{CDT::FLOAT32_T};
            CDT AccumulatorType{CDT::FLOAT32_T};

            uint32_t LeftRowCount{UNSIGNED_ZERO};
            uint32_t RightColumnCount{UNSIGNED_ZERO};
            uint32_t LRCommon{UNSIGNED_ZERO};

            uint32_t LeftStride{UNSIGNED_ZERO};
            uint32_t RightStride{UNSIGNED_ZERO};
            uint32_t DestinationStride{UNSIGNED_ZERO};
        };

        struct RowsDescSN final
        {
            CDT DestinationType{CDT::FLOAT32_T};
            CDT SourceType{CDT::FLOAT32_T};
            CDT AccumulatorType{CDT::FLOAT32_T};

            uint32_t Rows {UNSIGNED_ZERO};
            uint32_t Columns {UNSIGNED_ZERO};
            uint32_t SourceStride{UNSIGNED_ZERO};
            uint32_t DestinationStride{UNSIGNED_ZERO};
        };

        struct RMSNormDscSN final
        {
            CDT DestinationType{CDT::FLOAT32_T};
            CDT SourceType{CDT::FLOAT32_T};
            CDT WeightType{CDT::FLOAT32_T};
            CDT AccumulatorType{CDT::FLOAT32_T};

            uint32_t Rows{UNSIGNED_ZERO};
            uint32_t Columns{UNSIGNED_ZERO};
            uint32_t SourceStride{UNSIGNED_ZERO};
            uint32_t DestinationStride{UNSIGNED_ZERO};

            double Epsilon{0.0};
        };


        struct ComputerOptimizedPrimitives final
        {
            using MatMaulFunctionSN = bool(*)(
                void* user,
                ComputerBackend backend,
                void* destination,
                const void* left,
                const void* right,
                const MatMulSNDesc& description
            );

            using SoftMaxRowFunction = bool(*)(
                void* user,
                ComputerBackend backend,
                void* destination,
                const void* source,
                const RowsDescSN& description
            );

            using RMSNormFunctionSN = bool(*)(
                void* user,
                ComputerBackend backend,
                void* destination,
                const void* source,
                const void* weight,
                const RMSNormDscSN& desription
            );

            void* User = nullptr;
            MatMaulFunctionSN MatMul = nullptr;
            SoftMaxRowFunction SoftMaxRow = nullptr;
            RMSNormFunctionSN RMSNormRows = nullptr;
        };

        template<class DT>
        struct CacheOfSNCB
        {
            DT* Data_{nullptr};
            uint32_t Height_{UNSIGNED_ZERO};
            uint32_t CapacityWidth_{UNSIGNED_ZERO};
            uint32_t ActiveWidth_{UNSIGNED_ZERO};
            uint32_t Slot_{UNSIGNED_ZERO};
            MacroColumnOfAPC Column_{MacroColumnOfAPC::FREE_SLOT};
            APCUseScope Use_{};
        };
    };

    template<class DT>
    class SNTOComputerBridge final
    {
        friend class Computer;
        using CacheOfSNCB = ComputerDefAndVect::CacheOfSNCB<DT>;
    private:
        ComputerDefAndVect::CacheOfSNCB<DT> SNCBCache_{};
        
        SNTOComputerBridge(const CacheOfSNCB& cache) noexcept:
            SNCBCache_(std::move(cache))
        {}
    
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

        DT* DataPtr() noexcept
        {
            return static_cast<bool>(*this) ? SNCBCache_.Data_ : nullptr;
        }

        const DT* DataPtr() const noexcept
        {
            return static_cast<bool>(*this) ? SNCBCache_.Data_ : nullptr;
        }

        uint32_t Height() const noexcept {return SNCBCache_.Height_; }
        uint32_t CapacityWidth() const noexcept { return SNCBCache_.CapacityWidth_; }
        uint32_t ActiveWidth() const noexcept { return SNCBCache_.ActiveWidth_; }
        uint32_t Slot() const noexcept { return SNCBCache_.Slot_; }
        MacroColumnOfAPC Column() const noexcept { return SNCBCache_.Column_; }
        SchemaDefinition::SchemaProtocols Protocol() const noexcept { return SNCBCache_.Protocol_; }

        size_t ActiveElementCount() const noexcept
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
        
    };


}