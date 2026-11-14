#pragma once

#include "LinearAlgebra.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class ArithmeticCompute : public LinearAlgebra
    {
    public:
        explicit ArithmeticCompute(SessionCompute& runtime) noexcept :
            LinearAlgebra(runtime)
        {}

    };
}