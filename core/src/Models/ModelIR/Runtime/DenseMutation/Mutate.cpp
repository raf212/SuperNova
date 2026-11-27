#include "Orchestrate.cpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    DenseMutationEngine::DenseMutationEngine(
        std::shared_ptr<APCFinilizer> fabric,
        uint32_t instance,
        DenseMutationConfig config,
        const DenseMutationImage* restore
    ):
        State_(std::make_unique<StateDM>(std::move(fabric), instance))
    {
        using DMEState::StateSPtrs;
        StateDM& state = *State_;
        Require(bool(state.SharedPtrs_.Fabric) && !config.Blocks.empty(), 
            "Dense mutation requires a Fabric and APC blocks");
        
        APCFinilizer& owner = *state.SharedPtrs_.Fabric;
        ExclusiveLease pin = owner.ExecutionGate().TryMaintenance();
        if (!pin) throw std::runtime_error("Demse construction needs quiescent fabric admission");
        state.Cache_.Config = std::move(config);
        {
            std::lock_guard lock(owner.ModelCacheL_.ModelLeaseMutex);
            Require(owner.IsFabricActive(), "Dense mutation requirs active fabric");
            auto prior = owner.ModelCacheL_.LiveDenseModels.find(instance);
            if (prior != owner.ModelCacheL_.LiveDenseModels.end()) 
            {
                Require(!prior->second, "A live dense session already owns this model");
                Require(restore != nullptr, "Dense model must be reopened with authoratative image.");
                state.Cache_.ReOpening = true;
                prior->second = state.SharedPtrs_.Root;
            }
            else 
            {
                owner.ModelCacheL_.LiveDenseModels.emplace(instance, state.SharedPtrs_.Root);
            }
            state.Cache_.Registered = true;
            owner.ModelCacheL_.ModelLeasesActive.store(true, std::memory_order_release);
        }
        state.Cache_.Installed = ModelRuntime(&owner).InspectDenseModel(instance);
        {
            std::lock_guard lock(owner.ModelCacheL_.ModelLeaseMutex);
            const ModelIR ir = GetModelIR(*owner.GetFabricCache(), owner.GetDisposableCache()->SlabBasePtr_);
            const ViewModelIndividually view = ir.ModelView(instance);

            Require(view && view.Description->StructuralADaptationAllowed, 
                "The dense model's structural adaptation id disabled");

            Require(std::ranges::none_of(ir.Dependencies, [&](const auto& edge){
                return edge.Producer == instance || edge.Consumer == instance;
            }), "Dense mutation requyires an independent model instence.");

            state.Vectors_.Capacities.assign(view.Layers.begin(), view.Layers.end());
            std::unordered_map<uint32_t, uint32_t> assigned;
            for (const IndividualModelDescription& model : ir.Description) 
            {
                for (const LayerDescription& layer : ir.Layers.subspan(model.FirstLayer, model.LayerCount)) 
                {
                    for (const ModelAPC& apc : ir.LayerAPCs.subspan(layer.FirstAPCRecord, layer.APCCount)) 
                    {
                        assigned.emplace(apc.Slot, model.Instance);
                    }
                }
            }

            size_t slot_count = UNSIGNED_ZERO;
            for (const DenseBlock& block : state.Cache_.Config.Blocks) 
            {
                for (const ModelAPC& apc : block.APCs) 
                {
                    const auto assignment = assigned.find(apc.Slot);
                    const uint32_t model = assignment== assigned.end() ? 0 : assignment->second;
                    AdaptivePackedCellContainer handle;
                    APCUseScope use;
                    Require((model == UNSIGNED_ZERO || model == instance) && apc.Reserved == UNSIGNED_ZERO &&
                        (!owner.ModelCacheL_.LeaseModelSlots.contains(apc.Slot)) &&
                        owner.GetExistingAPC_(apc.Slot, handle, use, apc.Generation),
                        "Dense APC is stale unavaliavle or already owned");
                    SD::ConstRSSpan schemas = owner .MatrixViewRow(apc.Slot);
                    const size_t before = state.Vectors_.Regions.size();
                    for (const SD::RegionSchemaRecord& schema : schemas) 
                    {
                        if (
                            schema.Protocol != SD::SchemaProtocols::PRIVATE_REGION ||
                            SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::REGION_DISABLED)
                        ) 
                        {
                            continue;
                        }
                        ResolveRegionBiteView raw;
                        Require(
                            schema.ModelInstance != model ||
                            handle.ResolveBridgeRegion(schema.Region, 0, raw, true),
                            "Cannot resolve a Dense APC region"
                        );
                        state.Maps_.RegionIndex.emplace(RegionKey{
                            apc.Slot, schema.Region
                        }, state.Vectors_.Regions.size());
                        
                    }
                }
            }

        }

    }
}