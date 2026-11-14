#pragma once

#include "ReductionArithmetic.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class LinearAlgebra : public ReductionArithmetic
    {
    public:
        explicit LinearAlgebra(SessionCompute& runtime) noexcept :
            ReductionArithmetic(runtime)
        {}

        template<
            ScalarValueConcept DestinationT, ScalarValueConcept LT, ScalarValueConcept RT,
            ScalarValueConcept AccumT = DefaultAccumulator<LT, RT>
        >
        bool MatMul(
            DestinationT* destination, const LT* left, const RT* right,
            uint32_t m, uint32_t n, uint32_t k,
            uint32_t l_stride, uint32_t r_stride, uint32_t destination_stride,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<
            FloatingValueConcept DestinationT, FloatingValueConcept SourceT,
            FloatingValueConcept AccumT = DefaultAccumulator<DestinationT, SourceT>
        >
        bool SoftMaxRows(
            DestinationT destination, const SourceT* source,
            uint32_t rows, uint32_t columns,
            uint32_t sorce_stride, uint32_t destination_stride,
            ExecutionPolicyC policy = {}
        ) noexcept;

        template<
            FloatingValueConcept DestinationT, FloatingValueConcept SourceT, FloatingValueConcept WeightT,
            FloatingValueConcept AccumT = DefaultAccumulator<DestinationT, SourceT, WeightT>
        >
        bool RMSNormRows(
            DestinationT* destination, const SourceT* sorce, const WeightT* weight,
            uint32_t rows, uint32_t columns,
            uint32_t source_stride, uint32_t destination_stride, AccumT epsilon,
            ExecutionPolicyC policy = {}
        ) noexcept;

        
    };
}