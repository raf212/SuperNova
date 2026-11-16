#include <SuperNova>

///GMI - Refers to (global model information)
namespace BidirectionalInMemGraph::GMI
{
    LayerAPCArray ViewModelIndividually::LayerAPCs(uint32_t local_layer) const noexcept
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

    ViewModelIndividually ModelIR::ModelView(uint32_t instance) const noexcept
    {
        if (instance == UNSIGNED_ZERO || instance > Description.size()) 
        {
            return {};
        }

        const IndividualModelDescription& dsc = Description[instance - 1];
        if (
            dsc.Instances != instance ||
            dsc.FirstLayer > Layers.size() ||
            dsc.DependencyCount > (Dependencies.size() - dsc.FirstDependency)
        ) 
        {
            return {};
        }
        return {
            &dsc, 
            Layers.subspan(dsc.FirstLayer, dsc.LayerCount),
            LayerAPCs
        };
    }

    std::span<const ModelDependency> ModelIR::IncomingDependencies(uint32_t instance) const noexcept
    {
        const ViewModelIndividually model = ModelView(instance);
        if (!model) return {};

        const IndividualModelDescription& dsc = *model.Description;
        if (
            dsc.FirstDependency > Dependencies.size() ||
            dsc.DependencyCount > (Dependencies.size() - dsc.FirstDependency)
        ) 
        {
            return {};
        }
        return Dependencies.subspan(dsc.FirstDependency, dsc.DependencyCount);
    }

    std::span<const std::byte> ModelIR::Configuration(uint32_t instance) const noexcept
    {
        const ViewModelIndividually model = ModelView(instance);
        if (!model) return {};

        const IndividualModelDescription& dsc = *model.Description;
        if (
            dsc.ConfByteCount > Dynamics.size() ||
            dsc.ConfByteCount > (Dynamics.size() - dsc.ConfByteOffset)
        ) 
        {
            return {};
        }

        return Dynamics.subspan(
            static_cast<size_t>(dsc.ConfByteOffset), 
            static_cast<size_t>(dsc.ConfByteCount)
        );
    }


}