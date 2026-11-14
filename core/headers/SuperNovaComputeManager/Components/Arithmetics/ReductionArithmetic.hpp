#pragma once

#include "ElementWiseArithmetic.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class ReductionArithmetic : public ElementWiseArithmetic
    {
    public:
        explicit ReductionArithmetic(SessionCompute& runtime) noexcept :
            ElementWiseArithmetic(runtime)
        {}

        template<class T, class TransformOP>
        bool TransformReduceSum(
            size_t count,
            TransformOP transform,
            T& result,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<ScalarValueConcept SourceT, ScalarValueConcept AccumT = DefaultAccumulator<SourceT>>
        bool ReduceSum(
            const SourceT* source,
            size_t count,
            AccumT& result,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<
            ScalarValueConcept LT, ScalarValueConcept RT, 
            ScalarValueConcept AccumT = DefaultAccumulator<LT, RT>
        >
        bool Dot(
            const LT* left, const RT* right,
            size_t count, AccumT result,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<class Predicate>
        bool AllOf(size_t count, Predicate predicate, bool& result, ExecutionPolicyC policy = {}) noexcept;

        template<class Lane, class... Ts>
        bool EvaluteRows(size_t count, ExecutionPolicyC policy, Lane lane, NumericRange<Ts>... ranges) noexcept;

        template<class Lane, class... Ts>
        bool TransformRows(size_t count, ExecutionPolicyC policy, Lane lane, NumericRange<Ts>... ranges) noexcept;

        template<FloatingValueConcept T>
        bool AllFinite(const T* sorce, size_t count, bool& result, ExecutionPolicyC policy = {}) noexcept;

        template<class T>
        bool ScatterAdd(
            T* destination, size_t destination_count, const uint32_t indecies,
            const T* values, size_t value_count,
            T scale = static_cast<T>(1),
            ExecutionPolicyC policy = {}
        ) noexcept;
    };

}