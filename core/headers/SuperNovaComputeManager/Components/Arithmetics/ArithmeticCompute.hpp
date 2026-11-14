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

    };
}