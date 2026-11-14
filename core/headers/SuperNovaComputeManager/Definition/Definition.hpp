#pragma once
#include "../../NeuromorphicTimeSpace/StorageManager/FabricAllocators.hpp"

namespace BidirectionalInMemGraph :: ComputerDetail
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
            GPU = 2
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
            bool AllowCPUFallback = true;
        };

        struct ComputerRuntimeInfo final
        {
            ComputerBackend DefaultBackend = ComputerBackend::AUTOMATIC;
            bool AdaptiveCppCompiled = false;
            bool AcceleratorAvailable = false;
            bool DeviceUSMAvailable = false;
            bool DualQueueAvailable = false;
            bool Configured = false;
            std::string DeviceName{};
        };

        struct ComputeExecutionPolicy final
        {
            ComputerBackend Backend = ComputerBackend::AUTOMATIC;
            ComputerWorkType WorkType = ComputerWorkType::ELEMENTWISE;
            bool Deterministic = false;
            bool AllowCPUFallback = true;
        };

        template<class DT>
        inline static constexpr bool ScalarMetaValue = std::is_same_v<std::remove_cv_t<DT>, uint8_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint16_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint32_t> ||
            std::is_same_v<std::remove_cv_t<DT>, uint64_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int8_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int16_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int32_t> ||
            std::is_same_v<std::remove_cv_t<DT>, int64_t> ||
            std::is_same_v<std::remove_cv_t<DT>, float> ||
            std::is_same_v<std::remove_cv_t<DT>, double> ||
            std::is_same_v<std::remove_cv_t<DT>, char>
        #if defined (SN_ACPP_ENABLED) 
            || std::is_same_v<std::remove_cv_t<DT>, sycl::half>
        #endif
        ;

        template<class DT>
        inline static constexpr bool FloatingMetaValue = std::is_same_v<std::remove_cv_t<DT>, float> ||
            std::is_same_v<std::remove_cv_t<DT>, double> 
        #if defined (SN_ACPP_ENABLED)
            || std::is_same_v<std::remove_cv_t<DT>, sycl::half>
        #endif
        ;

        template<class DT> 
        struct SmallScalarPromotion
        {
            using U = std::remove_cv_t<DT>;
        #if defined (SN_ACPP_ENABLED)
            using Type = std::conditional_t<
                std::is_same_v<U, sycl::half>, 
                float, 
                std::conditional_t<std::is_integral_v<U> && sizeof(U) < sizeof(int32_t), int32_t, U>
            >;
        #else
            using Type = std::conditional_t<std::is_integral_v<U> && sizeof(U) < sizeof(int32_t), int32_t, U>;    
        #endif

        };

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

        enum class PrimitiveResult : uint8_t
        {
            SUCCESS = 0,
            UNSUPPORTED = 1,
            FAILED = 3
        };

        struct ComputerOptimizedPrimitives final
        {
            using MatMaulFunctionSN = PrimitiveResult(*)(
                void* user,
                ComputerBackend backend,
                void* destination,
                const void* left,
                const void* right,
                const MatMulSNDesc& description
            );

            using SoftMaxRowFunction = PrimitiveResult(*)(
                void* user,
                ComputerBackend backend,
                void* destination,
                const void* source,
                const RowsDescSN& description
            );

            using RMSNormFunctionSN = PrimitiveResult(*)(
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

        enum class NumericalAccess : uint8_t
        {
            READ = 0,
            WRITE = 1,
            READ_WRITE = 2
        };

        template<class DT>
        struct NumericRange final
        {
            DT* Host;
            uint64_t Count;
            NumericalAccess Access = std::is_const_v<DT> ?
                NumericalAccess::READ : NumericalAccess::READ_WRITE;
        };

        template<class DT>
        static constexpr NumericRange<const DT> ReadOrWriteRange(const DT* pointer, uint64_t count) noexcept
        {
            return {pointer, count};
        }

        static constexpr bool ValidBackend(ComputerBackend backend)
        {
            return backend == ComputerBackend::CPU ||
                backend == ComputerBackend::GPU || backend == ComputerBackend::AUTOMATIC;
        }

    };
    
    using BackendC = ComputerDefAndVect::ComputerBackend;
    using WorkTypeC = ComputerDefAndVect::ComputerWorkType;
    using ErrorC = ComputerDefAndVect::ComputerError;
    using ConfigC = ComputerDefAndVect::ComputerConf;
    using ExecutionPolicyC = ComputerDefAndVect::ComputeExecutionPolicy;
    using RuntimeInfoC = ComputerDefAndVect::ComputerRuntimeInfo;
    using OptimizedPrimitivesC = ComputerDefAndVect::ComputerOptimizedPrimitives;
    using PrimitiveResultC = ComputerDefAndVect::PrimitiveResult;
    using MatMulSNDesc = ComputerDefAndVect::MatMulSNDesc;
    using RowsDescSN = ComputerDefAndVect::RowsDescSN;
    using RMSNormDescSN = ComputerDefAndVect::RMSNormDscSN;
    using RMSNormDscSN = ComputerDefAndVect::RMSNormDscSN;

    template<class T> using _Legion_ = SNTOComputerBridge<T>;

    template<class DT>
    inline constexpr bool ScalarMetaValue = ComputerDefAndVect::ScalarMetaValue<DT>;

    template<class DT>
    inline constexpr bool FloatingMetaValue = ComputerDefAndVect::FloatingMetaValue<DT>;

    template<class DT>
    concept ScalarValueConcept = ComputerDefAndVect::ScalarMetaValue<DT>;

    template<class DT>
    concept FloatingValueConcept = ComputerDefAndVect::FloatingMetaValue<DT>;

    template<class... Ts>
    using DefaultAccumulatorSN = std::common_type_t<
        typename ComputerDefAndVect::SmallScalarPromotion<std::remove_cv_t<Ts>>::Type...
    >;

    template<class... Ts> using DefaultAccumulatorC = DefaultAccumulatorSN<Ts...>;
    template<class T> using NumericRange = ComputerDefAndVect::NumericRange<T>;


    struct ComputerMath
    {
        template<FloatingValueConcept DT>
        static constexpr DT Exp(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::exp(value);
        #else
            return static_cast<DT>(std::exp(value));
        #endif
        }

        template<FloatingValueConcept DT>
        static constexpr DT Log(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::log(value);
        #else
            return static_cast<DT>(std::log(value));
        #endif
        }

        template<FloatingValueConcept DT>
        static constexpr DT Sqrt(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::sqrt(value);
        #else
            return static_cast<DT>(std::sqrt(value));
        #endif
        }

        template<FloatingValueConcept DT>
        static constexpr DT Abs(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::fabs(value);
        #else
            return static_cast<DT>(std::abs(value));
        #endif
        }

        template<FloatingValueConcept DT>
        static bool IsFinite(DT value) noexcept
        {
        #if defined (SN_ACPP_ENABLED)
            return sycl::isfinite(value);
        #else
            return std::isfinite(value);
        #endif
        }

        template<class DT>
        static constexpr DT Clamp(DT value, DT lower, DT upper) noexcept
        {
            return std::clamp(value, lower, upper);
        }
    };

    template<class T> constexpr NumericRange<const T> ReadRange(const T* p, uint64_t count) noexcept 
    {
        return {p, count};
    }

    template<class T> constexpr NumericRange<T> WriteRange(T* p, uint64_t count) noexcept 
    {
        return {p, count};
    }

    template<class T> constexpr NumericRange<T> OverWriteRange(T* p, uint64_t count) noexcept
    {
        return {
            p,
            count,
            ComputerDefAndVect::NumericalAccess::WRITE
        };
    }

    template<class DT>
    static constexpr bool AtomicTypeSupport() noexcept
    {
        using U = std::remove_cv_t<DT>;
        return (sizeof(U)== sizeof(uint32_t) || sizeof(U)== sizeof(uint64_t)) &&
            (std::is_same_v<U,int> || std::is_same_v<U,unsigned int> ||
            std::is_same_v<U,long> || std::is_same_v<U,unsigned long> ||
            std::is_same_v<U,long long> || std::is_same_v<U,unsigned long long> ||
            std::is_same_v<U,float> || std::is_same_v<U,double>);
    }

}

namespace BidirectionalInMemGraph 
{
    //POST DEFINATION OF FLOAT16 when Acpp
#if defined (SN_ACPP_ENABLED)
    template<>
    inline constexpr std::optional<SchemaValidator::DataTypeOfMacroColumn>
    SchemaValidator::CppTypeToRegionDType<sycl::half>() noexcept
    {
        return SchemaValidator::DataTypeOfMacroColumn::FLOAT16_T;
    }
#endif
    //
}
