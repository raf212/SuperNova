#include "Orchestrate.cpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    namespace 
    {
        using PhysicalBinding = std::tuple<uint32_t, MacroColumnOfAPC, uint32_t, uint32_t>;
        constexpr uint64_t MINIMUM_PLANNING_LEAD = 3;

        // Canonicalize the layout so the installed model and its initial
        // execution view can be compared independent of binding splits.
        std::vector<PhysicalBinding> PhysicalBindings(const ModelSnapshot& model, size_t index)
        {
            const TensorDescription& tensor = model.DenseIR.Tensors[index];
            std::vector<PhysicalBinding> result;
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