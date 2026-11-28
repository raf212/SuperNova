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

    DenseMutationImage DenseMutationEngine::Capture_(bool finalize)
    {
        StateDM& state = *State_;
        std::unique_lock numerical(state.Mutexes_.Numerical);
        std::lock_guard lock(state.Mutexes_.Mutex);
        state.RequireHealthy();
        const DMEState::CRRev active = state.SharedPtrs_.Root->Active.load(std::memory_order_acquire);
        DenseMutationImage image = state.CaptureImage(*active);
        if (finalize) 
        {
        #if defined (SN_ACPP_ENABLED)
            if (state.GPUCache_.DeviceStorage) state.RestoreHostRegions(image);
            state.GPUCache_.DevicePlans.reset();
            state.GPUCache_.DeviceStorage.reset();
        #endif
            state.Cache_.Plan.reset();
            state.Release(true);
        }
        return image;
    }

    DenseMutationEngine::~DenseMutationEngine() = default;

    bool DenseMutationEngine::UseGPU() const noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        return bool(State_->GPUCache_.GPU);
    #else
        return false;
    #endif
    }

    MdlSnapCPtr DenseMutationEngine::Model() const noexcept
    {
        DMEState::CRRev active = State_->SharedPtrs_.Root->Active.load(std::memory_order_acquire);
        if (!active) return {};
        const ModelSnapshot* snapshot = &active->SnapShot;
        return {std::move(active), snapshot};
    }

    MdlSnapCPtr DenseMutationEngine::Published(APCFinilizer &fabric, uint32_t instance)
    {
        ModelLeaseCache& leases = fabric.ModelCacheL_;
        std::shared_ptr<DMEState::RevisionRootAtomic> root;
        {
            std::lock_guard lock(leases.ModelLeaseMutex);
            const auto found = leases.LiveDenseModels.find(instance);
            if (found == leases.LiveDenseModels.end()) return {};
            root = found->second;
        }
        if (!root) throw std::runtime_error("Dense model requires an authoritative image.");
        DMEState::CRRev active = root->Active.load(std::memory_order_acquire);
        if (!active) 
        {
            throw std::runtime_error("Dense session is opening or closing.");
        }
        const ModelSnapshot* snapshot = &active->SnapShot;
        return {std::move(active), snapshot};
    }

    uint64_t DenseMutationEngine::LastPass() const
    {
        std::lock_guard lock(State_->Mutexes_.Mutex);
        return State_->Cache_.Clock;
    }

    MutationStatus DenseMutationEngine::Prepare(MutationRequest request)
    {
        StateDM& state = *State_;
        std::unique_lock planning(state.Mutexes_.Planner, std::try_to_lock);
        if (!planning) return MutationStatus::Busy;

        DMEState::CRRev current;
        DMEState::CRRev retired;
        const uint64_t deadline = request.NextPass;
        {
            std::lock_guard lock(state.Mutexes_.Mutex);
            DMEState::Cache cache = state.Cache_;
            state.RequireHealthy();
            if (cache.Plan) return MutationStatus::Busy;
            current = state.SharedPtrs_.Root->Active.load(std::memory_order_acquire);
            if (request.ExpectedRevision != current->SnapShot.StructureRevision) 
            {
                return MutationStatus::Stale;
            }
            if (
                deadline < request.TargetPass ||
                request.TargetPass < cache.Clock ||
                (request.TargetPass - cache.Clock) < MINIMUM_PLANNING_LEAD ||
                request.ExpectedRevision == FABRIC_CELL_SENTINAL
            ) 
            {
                return MutationStatus::Invalid;
            }
            retired = std::move(state.SharedPtrs_.Retired);
        }
        retired.reset();

        DMEState::CRRev next;
        try 
        {
            next = state.Build(state.Apply(*current, request.Edits), 
                (request.ExpectedRevision + 1));
        }
        catch (const CapacityError&) {return MutationStatus::Capacity;}
        catch (const std::invalid_argument&) {return MutationStatus::Invalid;}
        catch (const std::bad_alloc&) {return MutationStatus::Capacity;}
        catch (const std::length_error&) {return MutationStatus::Capacity;}

        std::lock_guard lock(state.Mutexes_.Mutex);
        state.RequireHealthy();
        if (state.SharedPtrs_.Root->Active.load(std::memory_order_acquire) != current) 
        {
            return  MutationStatus::Stale;
        }
        if (state.Cache_.Clock >= deadline) return MutationStatus::Expired;
    #if defined(SN_ACPP_ENABLED)
        if (state.GPUCache_.DevicePlans) 
        {
            using Result = ComputerDetail::PlanUpdateResult;
            const Result staged = state.GPUCache_.DevicePlans->TryStage(next->Segments);
            if (staged == Result::BUSY) return MutationStatus::Busy;
            if (staged != Result::ACCEPTED) 
            {
                state.Cache_.Failed = true;
                return MutationStatus::DeviceFailure;
            }
        }
    #endif
        state.Cache_.Plan.emplace(DMEState::Pending{
            std::move(next), request.TargetPass, deadline
        });
        return MutationStatus::Prepared;
    }

    MutationStatus DenseMutationEngine::Advance()
    {
        std::lock_guard lock(State_->Mutexes_.Mutex);
        State_->RequireHealthy();
        return State_->AdvanceLocked();
    }

    bool DenseMutationEngine::Cancel()
    {
        std::lock_guard lock(State_->Mutexes_.Mutex);
        if (!State_->Cache_.Plan) return false;
        return true;
    }

    PassStatus DenseMutationEngine::ExecuteCPU(uint64_t revision, PassAccess access, const CPUWork &work)
    {
        StateDM& state = *State_;
        if (UseGPU()) throw std::logic_error("Dense parameters reside inside GPU");
        const bool write = access == PassAccess::Write;
        std::unique_lock exclusive(state.Mutexes_.Numerical, std::defer_lock);
        std::shared_lock shared(state.Mutexes_.Numerical, std::defer_lock);
        if (!(write ? exclusive.try_lock() : shared.try_lock())) return PassStatus::Busy;
        DMEState::CRRev active;
        uint64_t pass;
        uint64_t epoch;
        {
            std::lock_guard lock(state.Mutexes_.Mutex);
            state.RequireHealthy();
            (void) state.AdvanceLocked();
            active = state.SharedPtrs_.Root->Active.load(std::memory_order_acquire);
            if (active->SnapShot.StructureRevision != revision) 
            {
                return PassStatus::Stale;
            }
            state.ValidatePass(bool(work), access);
            state.Cache_.CPUWriter = write;
            pass = ++state.Cache_.Clock;
            epoch = state.Cache_.ParameterEpoch;
        }

        try 
        {
            work({active->SnapShot, active->Segments, state.Vectors_.Regions,
                pass, epoch, access});
        }
        catch(...)
        {
            if (write) 
            {
                std::lock_guard lock(state.Mutexes_.Mutex);
                state.Cache_.CPUWriter = false;
                state.Cache_.Failed = true;
            }
            throw;
            if (write) 
            {
                std::lock_guard lock(state.Mutexes_.Mutex);
                ++state.Cache_.ParameterEpoch;
                state.Cache_.CPUWriter = false;
            }
        }
        return PassStatus::Submitted;
    }

    DenseMutationImage DenseMutationEngine::Finalize()
    {
        std::lock_guard planning(State_->Mutexes_.Planner);
        return Capture_(true);
    }

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

    DenseMutationImage DenseMutationEngine::StateDM::CaptureImage(const Revision &active)
    {
        DenseMutationImage image{
            DenseMutationImage::FORMAT_VERSION,
            Cache_.Instance,
            Cache_.LayoutSignature,
            active.SnapShot.StructureRevision,
            Cache_.ParameterEpoch,
            Cache_.Clock,
            active.Blocks,
            {}
        };
        image.Regions.reserve(Vectors_.Regions.size());
        for (const DenseResidentRegion& region : Vectors_.Regions) 
        {
            image.Regions.push_back({region.APC, region.Region, 
                std::vector<std::byte>(region.Bytes)});
        }
    #if defined (SN_ACPP_ENABLED)
        if (GPUCache_.DeviceStorage) 
        {
            DownloadRegions_(image);
            return image;
        }
    #endif
        for (size_t i = 0; i < Vectors_.Storage.size(); i++) 
        {
            const std::span<std::byte> host = Vectors_.Storage[i].Host;
            std::memcpy(image.Regions[i].Bytes.data(), host.data(), host.size());
        }
        return image;
    }

    void DenseMutationEngine::StateDM::ValidatePass(bool has_work, PassAccess access) const
    {
        Require(
            has_work && 
            (access == PassAccess::Read || access == PassAccess::Write) && 
            Cache_.Clock != FABRIC_CELL_SENTINAL && 
            (access == PassAccess::Read || Cache_.ParameterEpoch != FABRIC_CELL_SENTINAL),
            "Invalid dense pass callback / access / counter overflow"
        );
    }

    void DenseMutationEngine::StateDM::RestoreHostRegions(const DenseMutationImage& image) const noexcept
    {
        for (size_t i = 0; i < Vectors_.Storage.size(); i++) 
        {
            const std::span<std::byte> host = Vectors_.Storage[i].Host;
            std::memcpy(host.data(), image.Regions[i].Bytes.data(), host.size());
        }
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

#if defined (SN_ACPP_ENABLED)
    PassStatus DenseMutationEngine::SubmitGPU(uint64_t revision, PassAccess access, const GPUWork &work)
    {
        StateDM& state = *State_;
        DMEState::Cache& cache = state.Cache_;
        DMEState::StateGPUCache& device = state.GPUCache_;
        std::lock_guard lock(state.Mutexes_.Mutex);
        state.RequireHealthy();
        if (!device.DevicePlans) throw std::logic_error("Dense device execution is unavailavle");
        if (state.AdvanceLocked() == MutationStatus::DeviceFailure) state.RequireHealthy();
        
        const DMEState::CRRev active = state.SharedPtrs_.Root->Active.load(std::memory_order_acquire);
        if (active->SnapShot.StructureRevision != revision) return PassStatus::Stale;
        state.ValidatePass(bool(work), access);

        try 
        {
            auto tail = work(
                device.GPU->ComputeQueue,
                {
                    active->SnapShot,
                    active->Segments,
                    state.Vectors_.Regions,
                    cache.Clock + 1,
                    cache.ParameterEpoch,
                    access
                },
                device.DevicePlans->ActiveView()
            );
            device.DevicePlans->RecordUse(tail);
            device.LastUse = std::move(tail);
            ++cache.Clock;
            if (access == PassAccess::Write) ++cache.ParameterEpoch;
        }
        catch (...)
        {
            cache.Failed = true;
            throw;
        }
        return PassStatus::Submitted;
    }

    

#endif


    
}