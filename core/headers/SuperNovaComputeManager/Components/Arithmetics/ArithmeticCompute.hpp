#pragma once

#include "LinearAlgebra.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class ArithmeticCompute : public LinearAlgebra
    {
    public:
        template<class T>
        using __NO__ = Numerics::NumericOperand<T>;

        template<class T>
        using __CNO__ = Numerics::NumericOperand<const T>;

        template<class T>
        using NMO = Numerics::NumericalMetrixOperand<T>;

        explicit ArithmeticCompute(SessionCompute& runtime) noexcept :
            LinearAlgebra(runtime)
        {}

        template<class T>
        bool Fill (__NO__<T> destination, T value, ExecutionPolicyC policy = {}) noexcept;

        template<class T>
        bool Copy(
            __NO__<T> destination, __CNO__<T> source, 
            T scale, T bias,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Copy(
            __NO__<T> destination, __NO__<T> source, 
            T scale, T bias,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Affine(
            __NO__<T> destination, __NO__<T> source,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool AXPY(
            __NO__<T> destination, __NO__<T> source,
            T scale,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Multiply(
            __NO__<T> destination, __NO__<T> left, __NO__<T> right,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<FloatingValueConcept T>
        bool Exp(__NO__<T> destination, __NO__<T> source, ExecutionPolicyC policy = {}) noexcept;

        template<ScalarValueConcept T, ScalarValueConcept Accum = DefaultAccumulator<T>>
        bool ReduceSum(__NO__<T> source, Accum& result, ExecutionPolicyC policy = {}) noexcept;

        template<
            ScalarValueConcept D, ScalarValueConcept S,
            ScalarValueConcept Accum = DefaultAccumulator<D, S>
        >
        bool MatMul(NMO<D> destination, NMO<S> source, ExecutionPolicyC policy = {}) noexcept;

        template<
            FloatingValueConcept D, FloatingValueConcept L, FloatingValueConcept R,
            FloatingValueConcept Accum = DefaultAccumulator<D, L, R>
        >
        bool SoftMaxRows(NMO<D> destination, NMO<L> left, NMO<R> right, ExecutionPolicyC policy = {}) noexcept;

        template<
            FloatingValueConcept D, FloatingValueConcept S, FloatingValueConcept W,
            FloatingValueConcept Accum = DefaultAccumulator<D, S, W>
        >
        bool RMSNormRows(
            NMO<D> destination, NMO<S> source, NMO<W> weight, 
            Accum epsilon, ExecutionPolicyC policy = {}
        ) noexcept;

        template<
            ScalarValueConcept L, ScalarValueConcept R,
            ScalarValueConcept Accum = DefaultAccumulator<L, R>
        >
        bool Dot(__NO__<L> left, __NO__<R> right, Accum& result, ExecutionPolicyC policy = {}) noexcept;

        template<FloatingValueConcept T>
        bool Sigmoid(__NO__<T> destination, __NO__<T> sorce, ExecutionPolicyC policy = {}) noexcept;

        template<FloatingValueConcept T, FloatingValueConcept Accum = DefaultAccumulator<T>>
        bool Reciprocal(
            __NO__<T> destination, __NO__<T> source,
            Accum epsilon = {},
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class T>
        bool Clamp(
            __NO__<T> destination, __NO__<T> source,
            T lower, T upper,
            ExecutionPolicyC policy = {}
        ) noexcept;
        
        template<FloatingValueConcept T>
        bool Allfinite(__NO__<T> source, bool& result, ExecutionPolicyC policy = {}) noexcept;
    };
}