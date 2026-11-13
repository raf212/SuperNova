#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{

    std::optional<uint64_t> FabricRuntimeBinding::LogicalByteOffset(const void* data, uint64_t bytes) const noexcept
    {
        const std::span<uint64_t> storage = NumericalCells();
        const uintptr_t begin = reinterpret_cast<uintptr_t>(storage.data());
        const uintptr_t address = reinterpret_cast<uintptr_t>(data);

        if (
            !data ||
            address < begin ||
            (address - begin) > storage.size_bytes() ||
            bytes > (storage.size_bytes() - (address - begin))
        ) 
        {
            return {};
        }
        return (Numerics.empty() ? 
            UNSIGNED_ZERO : (NumericalBegainCell * sizeof(uint64_t))
        ) + (address - begin);
    }


}