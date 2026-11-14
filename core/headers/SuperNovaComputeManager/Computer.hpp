#pragma once
#include "Components/SessionC.hpp"
#include "Components/CkpRegion.hpp"
#include "Components/Arithmetics/ArithmeticCompute.hpp"

namespace BidirectionalInMemGraph 
{
    class Computer
    {
        ComputerDetail::SessionCompute Session_{};
        ComputerDetail::RegionCompute Region_{Session_};
        ComputerDetail::ArithmeticCompute Arithmatic_{Session_};
        ComputerDetail::CheckpointCompute Checkpoint_{Session_};

    public:
        Computer() noexcept = default;
        Computer(const Computer&) = delete;
        Computer& operator = (const Computer&) = delete;

        ComputerDetail::SessionCompute& Session() noexcept {return Session_;}
        const ComputerDetail::SessionCompute& Session() const noexcept {return Session_;}
        ComputerDetail::RegionCompute& Region() noexcept {return Region_;}
        ComputerDetail::ArithmeticCompute& Numerics()  noexcept {return Arithmatic_;}
        ComputerDetail::CheckpointCompute& Checkpoint() noexcept {return Checkpoint_;}
    };
}
