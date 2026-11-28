#include <SuperNova>

namespace BidirectionalInMemGraph 
{
    using SD = SchemaDefinition;

    uint64_t ModelRuntime::DenseStructureRevision(uint32_t instance)
    {
        auto Active___ = [&]() -> MdlSnapCPtr
        {
            MdlSnapCPtr active = GMI::Dense::DenseMutationEngine::Published(
                *FabricPtr_, instance
            );
            return active;
        };

        if (MdlSnapCPtr active = Active___()) return active->StructureRevision; 
        ReadLease admission = FabricPtr_->ExecutionGate().TryTransientRead();
        if (!admission || !FabricPtr_->IsFabricActive()) 
        {
            throw std::runtime_error("Dense revision lookup requires an active host fabric.");
        }
        if (MdlSnapCPtr active = Active___()) return active->StructureRevision; 

        GMI::ModelIR ir = GMI::GetModelIR(*FabricPtr_->GetFabricCache(), FabricPtr_->GetDisposableCache()->SlabBasePtr_);
        const GMI::ViewModelIndividually model = ir.ModelView(instance);
        if (!model || model.Description->Model != GMI::ModelRepresentation::DENSE_TENSOR_STATE) 
        {
            throw std::invalid_argument("The requested Dense model instance does not exist.");
        }
        return model.Description->StructureRevision;
    }

    GMI::Dense::ModelSnapshot ModelRuntime::InspectDenseModel(uint32_t instance)
    {
        auto Active___ = [&]() -> MdlSnapCPtr
        {
            MdlSnapCPtr active = GMI::Dense::DenseMutationEngine::Published(
                *FabricPtr_, instance
            );
            return active;
        };

        if (MdlSnapCPtr active = Active___()) return *active; 
        ReadLease admission = FabricPtr_->ExecutionGate().TryTransientRead();
        if (!admission) throw std::runtime_error("Dense model inspection needs Fabric Admission.");
        if (MdlSnapCPtr active = Active___()) return *active; 

        return InspectInstalledDenseModel_(instance);
    }

    GMI::Dense::ModelSnapshot ModelRuntime::InspectInstalledDenseModel_(uint32_t instance)
    {
        if (!FabricPtr_->IsFabricActive()) 
        {
            throw std::runtime_error("Dense model inspection requires an active host fabric.");
        }

        GMI::ModelIR ir = GMI::GetModelIR(*FabricPtr_->GetFabricCache(), FabricPtr_->GetDisposableCache()->SlabBasePtr_);
        const GMI::ViewModelIndividually model = ir.ModelView(instance);
        if (!model || model.Description->Model != GMI::ModelRepresentation::DENSE_TENSOR_STATE) 
        {
            throw std::invalid_argument("The requested Dense model instance does not exist.");
        }

        GMI::Dense::ModelSnapshot snapshshot;
        snapshshot.DenseIR = GMI::Dense::DenseModelDynamics::Decode(ir.Configuration(instance));
        snapshshot.DenseIR.Validate(model.Layers);
        snapshshot.StructureRevision = model.Description->StructureRevision;
        snapshshot.Assignments.resize(model.Layers.size());

        for (uint32_t i = 0; i < model.Layers.size(); i++) 
        {
            std::vector<GMI::ModelAPC>& assignments = snapshshot.Assignments[i];
            for (const GMI::ModelAPC& apc : model.LayerAPCs(i).Active) 
            {
                const uint32_t generation = std::atomic_ref<const uint32_t>(
                    apc.Generation
                ).load(std::memory_order_acquire);

                AdaptivePackedCellContainer handle;
                APCUseScope use;
                if (!FabricPtr_->GetExistingAPC_(apc.Slot, handle, use, generation)) 
                {
                    throw std::runtime_error("A Dense model APC is unavailable or stale.");
                }
                assignments.push_back({
                    apc.Slot,
                    generation,
                    std::atomic_ref<const uint32_t>(apc.Role).load(std::memory_order_acquire),
                    UNSIGNED_ZERO
                });
            }
        }

        for (const GMI::Dense::TensorDescription& tensor : snapshshot.DenseIR.Tensors) 
        {
            for (const GMI::Dense::TensorBindings& binding : tensor.Bindings) 
            {
                const uint32_t slot = snapshshot.Assignments[tensor.LayerIndex][binding.APCIndex].Slot;
                const SD::ConstRSSpan row = FabricPtr_->MatrixViewRow(slot);
                const auto schema = std::ranges::find(
                    row, binding.Column, &SD::RegionSchemaRecord::Region
                );
                if (
                    schema == row.end() ||
                    SD::HasSchemaFlag(schema->Flags, SD::SchemaFlags::REGION_DISABLED) ||
                    schema->Protocol != SD::SchemaProtocols::PRIVATE_REGION ||
                    schema->ModelInstance != instance ||
                    schema->Dtype != tensor.DType ||
                    (uint64_t{binding.APCOffset} + binding.ElementCount) > (uint64_t{schema->MatrixHeight} * schema->MatrixWidth)
                ) 
                {
                    throw std::invalid_argument("Invalid physical Dense tensor binding: " + tensor.Name);
                }
            }
        }
        return snapshshot;
    }

}