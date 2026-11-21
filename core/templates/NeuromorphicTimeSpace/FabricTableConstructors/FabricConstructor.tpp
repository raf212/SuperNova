#pragma once 

namespace BidirectionalInMemGraph 
{
    template<typename T>
    bool FabricConstructor::IsInternalBuffer(const T* data, size_t count) noexcept
    {
        if (!FabCache_ || !DisposableFCache___.SlabBasePtr_ || count == UNSIGNED_ZERO)
        {
            return false;
        }
        if (
            count > SIZE_MAX / sizeof(T) ||
            FabCache_->SlabCellCount_ > SIZE_MAX / sizeof(uint64_t)
        )
        {
            return true;
        }

        const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
        const uintptr_t slab = reinterpret_cast<uintptr_t>(DisposableFCache___.SlabBasePtr_);
        const size_t bytes = count * sizeof(T);

        auto OverLaps___ = [&](const void* mem, uint64_t cells) noexcept -> bool
        {
            const auto base = reinterpret_cast<uintptr_t>(mem);
            return begin >= base ? (begin - base < cells * sizeof(uint64_t)) : 
                (base - begin < bytes);
        };

        return OverLaps___(
            DisposableFCache___.SlabBasePtr_, 
            __NumericalBacking__.DataPtr() ? FabCache_->SegmentPoolBegin_ : FabCache_->SlabCellCount_
        ) ||
        (
            __NumericalBacking__.DataPtr() &&
            OverLaps___(__NumericalBacking__.DataPtr(), FabCache_->SlabCellCount_ - FabCache_->SegmentPoolBegin_)
        );
    }
}