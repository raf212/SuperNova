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
        inline static constexpr bool ComputerScalarV = std::is_same_v<std::remove_cv_t<DT>, uint8_t> ||
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
    };

    template<class DT>
    inline constexpr bool ComputerScalarV = ComputerDefAndVect::ComputerScalarV<DT>;

    template<class DT>
    inline constexpr bool ComputerFloatingV = ComputerDefAndVect::ComputerFloatingV<DT>;

    template<class DT>
    concept Computerscalar = ComputerDefAndVect::ComputerScalarV<DT>;

    template<class DT>
    concept ComputerFloating = ComputerDefAndVect::ComputerFloatingV<DT>;

    struct ComputerMath
    {
        template<ComputerFloating DT>
        static constexpr DT Exp(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::exp(value);
        #endif
            return static_cast<DT>(std::exp(value));
        }

        template<ComputerFloating DT>
        static constexpr DT Log(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::log(value);
        #endif
            return static_cast<DT>(std::log(value));
        }

        template<ComputerFloating DT>
        static constexpr DT Sqrt(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::sqrt(value);
        #endif
            return static_cast<DT>(std::sqrt(value));
        }

        template<ComputerFloating DT>
        static constexpr DT Abs(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::abs(value);
        #endif
            return static_cast<DT>(std::abs(value));
        }

        template<ComputerFloating DT>
        static bool IsFinite(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::isfinite(value);
        #endif
            return std::isfinite(value);
        }

        template<class DT>
        static constexpr DT Clamp(DT value, DT lower, DT upper) noexcept
        {
            return std::clamp(value, lower, upper);
        }
    };


}