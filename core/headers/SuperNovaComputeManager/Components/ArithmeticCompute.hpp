#pragma once

#include "Arithmetics/ReductionArithmetic.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class ArithmeticCompute : ReductionArithmetic
    {
    public:
        explicit ArithmeticCompute(SessionCompute& runtime) noexcept :
            ReductionArithmetic(runtime)
        {}

    };
}