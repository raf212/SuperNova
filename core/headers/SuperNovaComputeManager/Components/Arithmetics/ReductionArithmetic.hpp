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

    };

}