#include <SuperNova>

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    LayerAPCArray IndividualModelView::LayerAPCs(uint32_t local_layer) const noexcept
    {
        if (local_layer >= Layers.size()) return {};
        const LayerDescription& layer = Layers[local_layer];
        if (
            layer.APCCount > layer.APCCapacity ||
            layer.FirstAPCRecord > AssignmentDirectory.size() ||
            layer.APCCapacity > (AssignmentDirectory.size() - layer.FirstAPCRecord)
        ) 
        {
            return {};
        }
        const std::span storage = AssignmentDirectory.subspan(
            layer.FirstAPCRecord, layer.APCCapacity
        );
        return {
            storage.first(layer.APCCount),
            storage,
            true
        };
    }
}