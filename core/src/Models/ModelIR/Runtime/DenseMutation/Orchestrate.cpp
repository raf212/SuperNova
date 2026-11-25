#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense
{
    namespace 
    {
        using RegionKey = std::pair<uint32_t, MacroColumnOfAPC>;

        void Require(bool valid, const char* reason)
        {
            if(!valid) throw std::invalid_argument(reason);
        }

        struct CapacityError{};

        struct Revision final
        {
            ModelSnapshot SnapShot;
            std::vector<std::vector<uint64_t>> Blocks;
            std::vector<DenseSegment> Segments;
        };

        struct StorageRegion final
        {
            std::span<std::byte> Host;
            size_t DeviceOffset{};
        };

        struct ColumnRun final
        {
            uint32_t Logical;
            uint32_t Physical;
            uint32_t Count;
        };

    }
}