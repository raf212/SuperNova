#pragma once

#include "../SessionC.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    struct ElementWiseArithmetic
    {
    protected:
        SessionCompute& Runtime_;
    
        template<class T>
        using __DAC__ = DefaultAccumulator<T>;
    public:
        using SD = SchemaDefinition;
        template<class T>
        using __Region__ = SNTOComputerBridge<T>;

        explicit ElementWiseArithmetic(SessionCompute& runtime) noexcept :
            Runtime_(runtime)
        {}

        template<class T>
        bool CopyFromHost(T* slab_destination, const T* hot_source, size_t count) noexcept;

        template<class T>
        bool CopyToHost(T* host_destination, const T* slab_source, size_t count) noexcept;

        template<class T, class UnaryOP>
        bool Transform(
            T* destination, const T* sorce,
            size_t count,
            UnaryOP operation,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T, class BinaryOP> 
        bool Zip(
            T* destination, const T* left, const T* right,
            size_t count,
            BinaryOP operation,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Fill(T* destination, size_t count, T* value, ExecutionPolicyC policy = {}) noexcept;

        template<class T>
        bool Copy(
            T* destination, const T* sorce,
            size_t count,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Affine(
            T* destination, const T* sorce,
            size_t count,
            T scale, T bias,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool AXPY(
            T* destination, const T* sorce,
            size_t count,
            T scale,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Multiply(
            T* destination, const T* left, const T* right,
            size_t count,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Clamp(
            T* destination, const T* source,
            size_t count,
            T lower, T upper,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<FloatingValueConcept T, FloatingValueConcept AccumT = __DAC__<T>>
        bool Exp(
            T* destination, const T* source, 
            size_t count, 
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<FloatingValueConcept T, FloatingValueConcept AccumT = __DAC__<T>>
        bool Sigmoid(T* destination, const T* source, size_t count, ExecutionPolicyC policy = {}) noexcept;

        template<FloatingValueConcept T, FloatingValueConcept AccumT = __DAC__<T>>
        bool Reciprocal(
            T* destination, const T* source, size_t count,
            AccumT epsilon = {}, ExecutionPolicyC policy = {}
        ) noexcept;

    };
}