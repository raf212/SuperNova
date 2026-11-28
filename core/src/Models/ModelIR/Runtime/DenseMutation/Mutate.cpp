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

    DenseMutationEngine::DenseMutationEngine(
        std::shared_ptr<APCFinilizer> fabric,
        uint32_t instance,
        DenseMutationConfig config,
        const DenseMutationImage* restore
    ) :
        State_(std::make_unique<StateDM>(std::move(fabric), instance))
    {
        State_->Open(std::move(config), restore);
    }

    DenseMutationEngine::~DenseMutationEngine() = default;

    void DenseMutationEngine::StateDM::Open(DenseMutationConfig config, const DenseMutationImage* restore)
    {
        Require(SharedPtrs_.Fabric && !config.Blocks.empty(),
            "Dense mutation requires a fabric and APC blocks");
        APCFinilizer& owner = *SharedPtrs_.Fabric;
        Cache_.Config = std::move(config);
        {
            ExclusiveLease admission = owner.ExecutionGate().TryMaintenance();
            if (!admission) throw std::runtime_error("Dense construction needs quicent fabric admission.");
            RegisterSession_(owner, restore != nullptr);
            Cache_.Installed = ModelRuntime(&owner).InspectInstalledDenseModel_(Cache_.Instance);
            ResolveProvisionedRegions_(owner);
            ClaimProvisionedSlots_(owner);
        }

        ValidateCatalog();
        Cache_.LayoutSignature = Signature();
        const uint64_t installed_revision = Cache_.Installed.StructureRevision;
        if (restore) 
        {
            ValidateImage_(*restore);
            ValidateInitialLayout_(BuildInitialRevision_(
                Cache_.Config.Layers, installed_revision)->SnapShot
            );
        }
    #if defined (SN_ACPP_ENABLED)
        SelectDevice_();
        PrepareDeviceStorage_(restore);
    #endif
        const CRRev revision = BuildInitialRevision_(
            restore ? restore->Layers : Cache_.Config.Layers,
            restore ? restore->StructureRevision : installed_revision    
        );
        if (!restore) ValidateInitialLayout_(revision->SnapShot);
        ClearProjectedBindings_();

    #if defined (SN_ACPP_ENABLED)
        PublishInitialDevicePlan_(*revision);
    #endif
        if (restore) 
        {
    #if defined (SN_ACPP_ENABLED)
            if (!GPUCache_.DeviceStorage)
    #endif
            {
                RestoreHostRegions(*restore);
            }
            Cache_.Clock = restore->LastPass;
            Cache_.ParameterEpoch = restore->ParameterEpoch;
            Cache_.Restored = true;
        }
        SharedPtrs_.Root->Active.store(revision, std::memory_order_release);
    }

    void DenseMutationEngine::StateDM::RegisterSession_(APCFinilizer& owner, bool restoring)
    {
        ModelLeaseCache& leases = owner.ModelCacheL_;
        std::lock_guard lock(leases.ModelLeaseMutex);
        Require(owner.IsFabricActive(), "Dense mutation requirs active fabric");
        const auto existing = leases.LiveDenseModels.find(Cache_.Instance);
        if (existing == leases.LiveDenseModels.end())
        {
            leases.LiveDenseModels.emplace(Cache_.Instance, SharedPtrs_.Root);
        }
        else 
        {
            Require(!existing->second, "a live dense session already owns this model");
            Require(restoring, "Dense model must reopen with authorative image");
            Cache_.ReOpening = true;
            existing->second = SharedPtrs_.Root;
        }
        Cache_.Registered = true;
        leases.ModelLeasesActive.store(true, std::memory_order_release);
    }

    void DenseMutationEngine::StateDM::ResolveProvisionedRegions_(APCFinilizer& owner)
    {
        const ModelIR ir = GetModelIR(
            *owner.GetFabricCache(), 
            owner.GetDisposableCache()->SlabBasePtr_
        );
        const ViewModelIndividually view = ir.ModelView(Cache_.Instance);
        Require(view && view.Description->StructuralADaptationAllowed,
                std::string(std::format("Structural adaptation disabled for {} this dense model", Cache_.Instance)));
        Require(std::ranges::none_of(ir.Dependencies, [&](const auto& edge){
            return edge.Producer == Cache_.Instance || edge.Consumer == Cache_.Instance;
        }), "Dense model require independent model instance");

        Vectors_.Capacities.assign(view.Layers.begin(), view.Layers.end());
        TwinU32Umap installed_owners;
        for (const IndividualModelDescription& model : ir.Description) 
        {
            for (const LayerDescription& layer : ir.Layers.subspan(model.FirstLayer, model.LayerCount)) 
            {
                for (const ModelAPC& apc : ir.LayerAPCs.subspan(layer.FirstAPCRecord, layer.APCCount)) 
                {
                    installed_owners.emplace(apc.Slot, model.Instance);
                }
            }
        }

        for (const DenseBlock& block : Cache_.Config.Blocks)
        {
            for (const ModelAPC& apc : block.APCs) 
            {
                const auto assignment = installed_owners.find(apc.Slot);
                const uint32_t model = assignment == installed_owners.end() ? UNSIGNED_ZERO : assignment->second;
                ResolveAPCRegions_(owner, apc, model);
            }
        }
        Require(!Vectors_.Regions.empty(), "Dense mutation requirs provisioned APCs");
    }

    void DenseMutationEngine::StateDM::ResolveAPCRegions_(APCFinilizer& owner, const ModelAPC& apc, uint32_t model)
    {
        AdaptivePackedCellContainer handle;
        APCUseScope  use;
        Require((model == UNSIGNED_ZERO || model == Cache_.Instance) && 
            apc.Reserved && 
            owner.GetExistingAPC_(apc.Slot, handle, use),
            "Dense APC is stale / unavailavle / assigned to another model"
        );

        const size_t first = Vectors_.Regions.size();
        for (const SD::RegionSchemaRecord& schema : owner.MatrixViewRow(apc.Slot)) 
        {
            if (schema.Protocol != SD::SchemaProtocols::PRIVATE_REGION) 
            {
                continue;
            }
            ResolveRegionBiteView raw;

            Require(schema.ModelInstance == model && handle.ResolveBridgeRegion(
                schema.Region, 0, raw, true
            ), std::string(std::format("Cannot resolve the dense APC : {} Region {}", apc.Slot, static_cast<size_t>(schema.Region))));
            Require(Maps_.RegionIndex.emplace(RegionKey{apc.Slot, schema.Region}, 
                Vectors_.Regions.size()).second, "A provisioned Dense APC may belong to only one block");
            Vectors_.Regions.push_back({
                raw.Bytes.data(), apc, raw.Bytes.size(), schema.Region, schema.Dtype
            });
            Vectors_.Storage.push_back({raw.Bytes});
        }
        Require(Vectors_.Regions.size() != first, "Dense APC has no private numerical regions");
    }

    void DenseMutationEngine::StateDM::ClaimProvisionedSlots_(APCFinilizer& owner)
    {
        size_t slot_count = 0;
        for (const DenseBlock& block : Cache_.Config.Blocks) 
        {
            Require(block.APCs.size() <= (SIZE_MAX - slot_count), "Dense APC catalogue too large");
            slot_count += block.APCs.size();
        }
        Vectors_.Claimed.reserve(slot_count);
        ModelLeaseCache& leases = owner.ModelCacheL_;
        std::lock_guard lock(leases.ModelLeaseMutex);
        for (const DenseBlock& block : Cache_.Config.Blocks) 
        {
            for (const ModelAPC& apc : block.APCs) 
            {
                Require(leases.LeaseModelSlots.insert(apc.Slot).second, 
                    std::string(std::format("Dense APC : {} is leased by another model.", apc.Slot)));
                Vectors_.Claimed.push_back(apc.Slot);
            }
        }
    }

    DMEState::CRRev DenseMutationEngine::StateDM::BuildInitialRevision_(const Vect2DU64& layers, uint64_t revision) const
    {
        try 
        {
            return Build(layers, revision);
        }
        catch (const CapacityError&)
        {
            throw std::invalid_argument("Dense configuration exceeds its capacity");
        }
    }

    void DenseMutationEngine::StateDM::ValidateInitialLayout_(const ModelSnapshot& original) const
    {
        const  ModelSnapshot& installed = Cache_.Installed;
        for (size_t layer = 0; layer < installed.Assignments.size(); layer++) 
        {
            std::vector<ModelAPC> expected = installed.Assignments[layer];
            std::vector<ModelAPC> selected = original.Assignments[layer];
            std::ranges::sort(expected, {}, &ModelAPC::Slot);
            Require(std::ranges::equal(expected, selected, [](const ModelAPC& a, const ModelAPC& b){
                return a == b;
            }), "Initial dense APC assignments do not match installed ModelIR");
        }
        for (size_t tensor = 0; tensor < installed.DenseIR.Tensors.size(); ++tensor) 
        {
            Require(original.DenseIR.Tensors[tensor].Shape == installed.DenseIR.Tensors[ tensor].Shape && 
                PhysicalBindings(original, tensor) == PhysicalBindings(installed, tensor), 
                "Initial dense placements do not reproduce installed Model IR");
        }
    }

    void DenseMutationEngine::StateDM::ClearProjectedBindings_()
    {
        DenseModelDynamics& ir = Cache_.Installed.DenseIR;
        for (const Operator& op : ir.Graph->Operators)
        {
            if (op.Kind != OperatorKind::Linear) continue;
            ir.Tensors[op.Weight].Bindings.clear();
            if (op.Bias != NO_TENSOR) ir.Tensors[op.Bias].Bindings.clear();
        }
    }

    void DenseMutationEngine::StateDM::ValidateImage_(const DenseMutationImage& image) const
    {
        Require(
            image.Version == DenseMutationImage::FORMAT_VERSION &&
            image.Instance == Cache_.Instance &&
            image.LayoutSignature == Cache_.LayoutSignature &&
            image.StructureRevision >= Cache_.Installed.StructureRevision &&
            image.ParameterEpoch <= image.LastPass &&
            image.Regions.size() == Vectors_.Regions.size() &&
            (
                image.StructureRevision != Cache_.Installed.StructureRevision ||
                image.Layers == Cache_.Config.Layers
            ),
            "Dense image dosen't match the model or format"
        );

        for (size_t i = 0; i < image.Regions.size(); i++) 
        {
            const DenseMutationImage::RegionImage& saved = image.Regions[i];
            const DenseResidentRegion& region = Vectors_.Regions[i];
            Require(saved.APC == region.APC && saved.Region == region.Region &&
                saved.Bytes.size() == region.Bytes, "Dense image region  is stale or incomplete");
        }
    }






    
}