#pragma once

#include "ElementWiseArithmetic.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    struct ReductionArithmetic : public ElementWiseArithmetic
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

        template<ScalarValueConcept SourceT, ScalarValueConcept AccumT = __DAC__<SourceT>>
        bool ReduceSum(
            const SourceT* source,
            size_t count,
            AccumT& result,
            ExecutionPolicyC policy = {}
        ) noexcept;
    };

}