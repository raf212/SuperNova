#include <SuperNova>

namespace BidirectionalInMemGraph
{
    uint64_t* RawPackedCellAllocator::Allocate(size_t cells) const noexcept
    {
        if (!IsValid() || cells > (SIZE_MAX / sizeof(uint64_t)))
        {
            return nullptr;
        }

        const auto FN_ALLOC =  AllocatePackedCellStorage ? AllocatePackedCellStorage:DefaultAllocateAtomicCells;
        return FN_ALLOC(cells, EffectiveAllignment(), User);
    }

    void RawPackedCellAllocator::Free(uint64_t* data, size_t cells) const noexcept
    {
        if (!data) return;
        const auto FN_FREE =  FreePackedCellStorage ? FreePackedCellStorage:DefaultFreeAtomicCells;
        FN_FREE(data, cells, EffectiveAllignment(), User);
    }
}
#if defined(SN_ACPP_ENABLED)
namespace BidirectionalInMemGraph::ComputerDetail::Memory
{
    uint64_t* ACppFabricAlocator::AllocateUsm(size_t cells, size_t alignment ) 
    {
        if (
            !cells ||
            cells > LIMIT_OF_CELL_COUNT ||
            cells > SIZE_MAX / sizeof(uint64_t) ||
            !std::has_single_bit(alignment) ||
            alignment < alignof(uint64_t)
        ) 
        {
            return nullptr;
        }

        try 
        {
            return sycl::aligned_alloc<uint64_t>(
                alignment, cells, *Queue_, __Loc__
            );
        }
        catch (const std::runtime_error& e) {
            std::cerr << "sycl memory alocation error: " << e.what() << "\n";
            return nullptr;
        }
    }

    void ACppFabricAlocator::FreeUsm(uint64_t* data) noexcept
    {
        if (!data) return;
        try 
        {
            sycl::free(data, *Queue_);
        }
        catch(...)
        {
            std::terminate();
        }
    }
}
#endif