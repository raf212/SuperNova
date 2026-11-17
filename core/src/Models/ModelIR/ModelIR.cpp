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
            dsc.Instance != instance ||
            dsc.FirstLayer > Layers.size() ||
            dsc.DependencyCount > (Dependencies.size() - dsc.FirstDependency) ||
            dsc.LayerCount > (Layers.size() - dsc.FirstLayer) ||
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
            dsc.ConfByteOffset > Dynamics.size() ||
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

    const ModelAPC* ModelIR::APCAt(uint32_t layer_index, uint32_t position) const noexcept
    {
        if (layer_index >= Layers.size()) return nullptr;
        const LayerDescription& layer = Layers[layer_index];
        if (
            layer.APCCount > layer.APCCapacity ||
            layer.FirstAPCRecord > LayerAPCs.size() ||
            layer.APCCapacity > (LayerAPCs.size() - layer.FirstAPCRecord) ||
            position >= layer.APCCount
        ) 
        {
            return nullptr;
        }
        return &LayerAPCs[layer.FirstAPCRecord + position];
    }

    const IndividualModelDescription* ModelIR::ModelForRecord(size_t record_index) const noexcept
    {
        if (record_index >= LayerAPCs.size()) return nullptr;
        for (const IndividualModelDescription& model : Description) 
        {
            if (
                model.FirstLayer > Layers.size() ||
                model.LayerCount > (Layers.size() - model.FirstLayer)
            ) 
            {
                return nullptr;
            }
            //can be used better search algorithm - BTA001
            for (uint32_t i = 0; i < model.LayerCount; i++) 
            {
                const LayerDescription& layer = Layers[model.FirstLayer + i];
                if (
                    record_index >= layer.FirstAPCRecord &&
                    (record_index - layer.FirstAPCRecord) < layer.APCCount
                ) 
                {
                    return &model;
                }
            }
        }
        return nullptr;
    }

    bool ModelIR::Valid(uint32_t slot_count) const noexcept
    {
        if (
            Description.size() >= ModelDependency::INVALID_MODEL_IDX ||
            Dependencies.size() > ADS::APC_INDEX_BOUND_SENTINAL
        ) 
        {
            return false;
        }

        uint64_t layer_cursor = UNSIGNED_ZERO;
        uint64_t dependency_cursor = UNSIGNED_ZERO;
        uint64_t apc_cursor = UNSIGNED_ZERO;
        for (size_t idx = 0; idx < Description.size(); idx++) 
        {
            const IndividualModelDescription& model = Description[idx];
            if (
                model.Instance != (idx +1) ||
                model.Model == PreBuildModels::NONE ||
                model.Model > LAST_DEFINED_MODEL_IDX ||
                model.StructuralADaptationAllowed > 1 ||
                model.StructureRevision == UNSIGNED_ZERO ||
                model.ConfByteOffset > Dynamics.size() ||
                model.ConfByteCount > (Dynamics.size() - model.ConfByteOffset) ||
                model.FirstLayer != layer_cursor ||
                layer_cursor > Layers.size() ||
                model.LayerCount > (Layers.size() - layer_cursor) ||
                model.FirstDependency != dependency_cursor ||
                dependency_cursor > Dependencies.size() ||
                model.DependencyCount > (Dependencies.size() - dependency_cursor)
            ) 
            {
                return false;
            }

            for (uint32_t d = 0; d < model.DependencyCount; d++) 
            {
                const ModelDependency& dep = Dependencies[dependency_cursor + d];
                if (
                    !ADS::ValidateSlotIdx(dep.Producer) ||
                    !ADS::ValidateSlotIdx(dep.Consumer) ||
                    dep.Producer == UNSIGNED_ZERO ||
                    dep.Producer > Description.size() ||
                    dep.Consumer != model.Instance ||
                    static_cast<uint8_t>(dep.ProducerRegion) >= ADS::CountOfMacroColumn() ||
                    static_cast<uint8_t>(dep.ConsumerRegion) >= ADS::CountOfMacroColumn()
                ) 
                {
                    return false;
                }
            }
            dependency_cursor += model.DependencyCount;
            layer_cursor += model.LayerCount;
        }

        if (
            layer_cursor != Layers.size() ||
            dependency_cursor != Dependencies.size()
        ) 
        {
            return false;
        }

        for (const LayerDescription& layer : Layers) 
        {
            if (
                layer.FirstAPCRecord != apc_cursor ||
                apc_cursor > LayerAPCs.size() ||
                layer.APCCount > layer.APCCapacity ||
                layer.APCCapacity > (LayerAPCs.size() - apc_cursor)
            ) 
            {
                return false;
            }

            for (uint32_t i = 0; i < layer.APCCapacity; i++) 
            {
                const ModelAPC& apc = LayerAPCs[apc_cursor + i];
                if (i < layer.APCCount) 
                {
                    if (apc.Slot >= slot_count) return false;
                }
                else if (
                    apc.Slot != ADS::APC_INDEX_BOUND_SENTINAL ||
                    apc.Generation != UNSIGNED_ZERO ||
                    apc.Role != UNSIGNED_ZERO
                ) 
                {
                    return false;
                }
            }
            apc_cursor += layer.APCCapacity;
        }

        if (apc_cursor != LayerAPCs.size()) return false;

        if (Layers.empty()) return true;
        try 
        {
            std::vector<uint8_t> seen(slot_count, UNSIGNED_ZERO);
            for (const LayerDescription& layer : Layers) 
            {
                for (uint32_t i = 0; i < layer.APCCount; i++) 
                {
                    if (seen[LayerAPCs[layer.FirstAPCRecord + i].Slot]++)
                    {
                        return false;
                    }
                }
            }
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    bool ModelIRBuildStorage::Build(
        std::span<const BuildModelDescriptionIndivisually> models,
        uint32_t slot_count
    ) noexcept
    {
        if (models.size() >= ModelDependency::INVALID_MODEL_IDX) return false;
        try 
        {
            ModelIRBuildStorage candidate{};
            candidate.Description.reserve(models.size());
            uint64_t total_layers = UNSIGNED_ZERO;
            uint64_t total_records = UNSIGNED_ZERO;
            uint64_t total_dependencies = UNSIGNED_ZERO;
            for (const BuildModelDescriptionIndivisually& mo : models) 
            {
                if (
                    mo.Instance != candidate.Description.size() + 1 ||
                    mo.Model == PreBuildModels::NONE ||
                    mo.Model > LAST_DEFINED_MODEL_IDX ||
                    !mo.StructureRevision ||
                    mo.Layers.size() > (ADS::APC_INDEX_BOUND_SENTINAL - total_layers) ||
                    mo.Dependencies.size() > (ADS::APC_INDEX_BOUND_SENTINAL - total_dependencies) ||
                    mo.Configuration.size() > (FABRIC_CELL_SENTINAL - candidate.Dynamics.size()) ||
                    mo.Configuration.size() > (FABRIC_CELL_SENTINAL - candidate.Dynamics.size())
                ) 
                {
                    return false;
                }

                total_layers += mo.Layers.size();
                total_dependencies += mo.Dependencies.size();
                for (const LayerBuildDescription& layer : mo.Layers) 
                {
                    const uint64_t capacity = layer.MaxAllowedAPCs ? layer.MaxAllowedAPCs : layer.APCs.size();
                    if (layer.APCs.size() > capacity || capacity > (ADS::APC_INDEX_BOUND_SENTINAL - total_records)) 
                    {
                        return false;
                    }
                    total_records += capacity;
                }

                candidate.Description.push_back({
                    mo.Instance,
                    mo.Model,
                    static_cast<uint8_t>(mo.StructuralADaptationAllowed), 
                    UNSIGNED_ZERO,
                    static_cast<uint64_t>(candidate.Dynamics.size()),
                    static_cast<uint64_t>(mo.Configuration.size()),
                    static_cast<uint32_t>(total_layers - mo.Layers.size()),
                    static_cast<uint32_t>(mo.Layers.size()),
                    static_cast<uint32_t>(total_dependencies - mo.Dependencies.size()),
                    static_cast<uint32_t>(mo.Dependencies.size()),
                    mo.StructureRevision
                });
                candidate.Dynamics.insert(
                    candidate.Dynamics.end(), 
                    mo.Configuration.begin(), 
                    mo.Configuration.end()
                );
                candidate.Dependencies.insert(
                    candidate.Dependencies.end(), 
                    mo.Dependencies.begin(), 
                    mo.Dependencies.end()
                );
            }

            candidate.Layers.reserve(static_cast<size_t>(total_layers));
            candidate.APCRecords.reserve(static_cast<size_t>(total_records));
            for (const BuildModelDescriptionIndivisually& model : models) 
            {
                for (const LayerBuildDescription& layer : model.Layers) 
                {
                    const uint32_t capacity = layer.MaxAllowedAPCs ? layer.MaxAllowedAPCs :
                        static_cast<uint32_t>(layer.APCs.size());
                    candidate.Layers.push_back({
                        layer.OperatorId,
                        static_cast<uint32_t>(candidate.APCRecords.size()),
                        static_cast<uint32_t>(layer.APCs.size()),
                        capacity
                    });

                    candidate.APCRecords.insert(
                        candidate.APCRecords.end(), layer.APCs.begin(), layer.APCs.end()
                    );

                    candidate.APCRecords.resize(
                        candidate.APCRecords.size() + capacity - layer.APCs.size()
                    );
                }
            }
            if (!candidate.View().Valid(slot_count)) return false;
            *this = std::move(candidate);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
    
    //CODE:: BTA001
    const IndividualModelDescription* ModelIR::ModelForSLot(uint32_t slot) const noexcept
    {
        for (const IndividualModelDescription& model : Description) 
        {
            if (
                model.FirstLayer > Layers.size() ||
                model.LayerCount > (Layers.size() - model.FirstLayer)
            ) 
            {
                return nullptr;
            }

            for (uint32_t i = 0; i < model.LayerCount; i++) 
            {
                const LayerDescription& layer = Layers[model.FirstLayer + i];
                if (
                    layer.FirstAPCRecord > LayerAPCs.size() ||
                    layer.APCCount > (LayerAPCs.size() - layer.FirstAPCRecord)
                ) 
                {
                    return nullptr;
                }

                for (uint32_t p = 0; p < layer.APCCount; p++) 
                {
                    if (LayerAPCs[layer.FirstAPCRecord + p].Slot == slot) 
                    {
                        return &model;
                    }
                }
            }
        }
        return nullptr;
    }


}