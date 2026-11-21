#include <SuperNova>

namespace BidirectionalInMemGraph 
{
    uint64_t ModelRuntime::DenseStructureRevision(uint32_t instance)
    {
        ReadLease admission = FabricPtr_->ExecutionGate().TryTransientRead();
        if (!admission || !FabricPtr_->IsFabricActive()) 
        {
            throw std::runtime_error("Dense revision lookup requires an active host fabric.");
        }

        GMI::ModelIR ir = GMI::GetModelIR(*FabricPtr_->GetFabricCache(), FabricPtr_->GetDisposableCache()->SlabBasePtr_);
        const GMI::ViewModelIndividually model = ir.ModelView(instance);
        if (!model || model.Description->Model != GMI::ModelRepresentation::DENSE_TENSOR_STATE) 
        {
            throw std::invalid_argument("The requested Dense model instance does not exist.");
        }
        return model.Description->StructureRevision;
    }

    
}