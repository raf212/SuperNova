#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense
{
    namespace 
    {
        using RegionKey = std::pair<uint32_t, MacroColumnOfAPC>;

        using ColumnIndex = std::unordered_map<uint64_t, uint32_t>;

        using PhyBindTuple = std::tuple<
            uint32_t,
            MacroColumnOfAPC,
            uint32_t,
            uint32_t
        >;

        constexpr void Require(bool valid, const char* reason)
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

        static std::vector<ColumnRun> ProjectColumns(
            const ColumnIndex& columns,
            std::span<const uint64_t> logical 
        )
        {
            std::vector<ColumnRun> result;
            for (uint32_t i = 0; i < logical.size(); i++) 
            {
                const auto column = columns.find(logical[i]);
                Require(column != columns.end(), "A required input connection was not provisioned.");
                if (
                    !result.empty() &&
                    (result.back().Physical + result.back().Count) == column->second
                ) 
                {
                    ++result.back().Count;
                }
                else 
                {
                    result.push_back({i, column->second, 1});
                }
            }
            return result;
        }

        // Canonicalize the layout so the installed model and its initial
        // execution view can be compared independent of binding splits.
        static constexpr std::vector<PhyBindTuple> PhysicalBindings(const ModelSnapshot& model, size_t index) 
        {
            const TensorDescription& tensor = model.DenseIR.Tensors[index];
            std::vector<PhyBindTuple> result;
            for (const TensorBindings& piece : tensor.Bindings) 
            {
                const uint32_t slot = model.Assignments[tensor.LayerIndex][piece.APCIndex].Slot;
                if (!result.empty()) 
                {
                    auto& [last_slot, region, offset, count] = result.back();
                    if (
                        last_slot == slot &&
                        region == piece.Column &&
                        (offset + count) == piece.APCOffset
                    ) 
                    {
                        count += piece.ElementCount;
                        continue;
                    }
                }
                result.emplace_back(slot, piece.Column, piece.APCOffset, piece.ElementCount);
            }
            return result;
        }
    }
}