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

    namespace DMEState 
    {
        using FabricSPtr = std::shared_ptr<APCFinilizer>;
        
        using CRRev = std::shared_ptr<const Revision>;

        struct RevisionRootAtomic final
        {
            std::atomic<CRRev> Active;
        };

        struct StateSPtrs final
        {
            std::shared_ptr<DMEState::RevisionRootAtomic> Root{std::make_shared<DMEState::RevisionRootAtomic>()};
            FabricSPtr Fabric;
            CRRev retired;

            StateSPtrs(FabricSPtr fabric) :
                Fabric(std::move(fabric))
            {}
        };

        struct StateVectors final
        {
            std::vector<LayerDescription> Capacities;
            std::vector<uint64_t> Input;
            std::vector<uint32_t> Claimed;
            std::vector<StorageRegion> Storage;
            std::vector<DenseResidentRegion> Region;
        };

        struct StateMaps final
        {
            std::unordered_map<uint64_t, size_t> BlockIndex;
            std::unordered_map<const BlockPlacement*, ColumnIndex> ColumnIndecies;
            std::map<RegionKey, size_t> RegionIndex;
        };

        struct StateMutexes final
        {
            mutable std::mutex Mutex;
            std::mutex Planner;
            std::shared_mutex Numerical;
        };

        struct Cache
        {
            uint32_t Instance;
            ModelSnapshot Installed;
            DenseMutationConfig Config;
            uint64_t LayoutSignature{};
            size_t FixedBIndings{};
            uint64_t Clock{};
            uint64_t ParameterEpoch{};
            bool CPUWriter{};
            bool Failed{};
            bool Registered{};
            bool ReOpening{};
            bool Restored{};

            Cache(uint32_t instance):
                Instance(instance)
            {}
        };

        struct Pending
        {
            CRRev Next;
            uint64_t Target;
            uint64_t Deadline;
        };

    #if defined (SN_ACPP_ENABLED)
        struct StateGPUCache final
        {
            std::shared_ptr<ComputerDetail::RuntimeAcpp> GPU;
            std::unique_ptr<ComputerDetail::ResidentDeviceRegion<std::byte>> DeviceStorage;
            std::unique_ptr<ComputerDetail::DevicePlanDoubleBuffer<DenseSegment>> DevicePlans;
            std::optional<sycl::event> LastUse;
        };
    #endif
    }

    struct DenseMutationEngine::State
    {
    #if defined (SN_ACPP_ENABLED)
        DMEState::StateGPUCache GPUCache_{};
    #endif

        DMEState::StateSPtrs SharedPtrs_;

        DMEState::StateVectors Vectors_{};

        DMEState::StateMaps Maps_{};

        DMEState::Cache Cache_;

        DMEState::StateMutexes Mutexes_{};

        DMEState::Pending Plan_{};

        State(DMEState::FabricSPtr fabric, uint32_t instance) :
            SharedPtrs_(std::move(fabric)), Cache_(instance)
        {}

        ~State()
        {
        #if defined (SN_ACPP_ENABLED)
            if (Cache_.Registered && GPUCache_.GPU && GPUCache_.GPU->Drain())
            {
                std::terminate();
            }
        #endif
            Release(false);
        }

        void Release(bool finalize) noexcept
        {
            if (!Cache_.Registered) return;
            const DMEState::CRRev last = SharedPtrs_.Root->Active.exchange({}, std::memory_order_acq_rel);
            const bool required_image = last ? 
                (
                    last->SnapShot.StructureRevision != Cache_.Installed.StructureRevision ||
                    (!finalize && (Cache_.ParameterEpoch != UNSIGNED_ZERO || Cache_.Failed || Cache_.Restored))
                ) : Cache_.ReOpening;

            std::lock_guard lock(SharedPtrs_.Fabric->ModelCacheL_.ModelLeaseMutex);
            if (required_image) 
            {
                SharedPtrs_.Fabric->ModelCacheL_.ResetInstance(Cache_.Instance);
            }
            else 
            {
                SharedPtrs_.Fabric->ModelCacheL_.LiveDenseModels.erase(Cache_.Instance);
            }

            for (const uint32_t slot : Vectors_.Claimed) 
            {
                SharedPtrs_.Fabric->ModelCacheL_.LeaseModelSlots.erase(slot);
            }

            const bool active = std::ranges::any_of(
                SharedPtrs_.Fabric->ModelCacheL_.LiveDenseModels,
                [] (const auto& entry)
                {
                    return bool(entry.second);
                }
            );

            SharedPtrs_.Fabric->ModelCacheL_.ModelLeasesActive.store(active, std::memory_order_release);
            Cache_.Registered = false;
        }

        const DenseBlock& Block(uint64_t id) const 
        {
            const auto entry = Maps_.BlockIndex.find(id);
            Require(entry != Maps_.BlockIndex.end(), "Unknown Dense APC Block.");
            return Cache_.Config.Blocks[entry->second];
        }

        const BlockPlacement& placement(const DenseBlock& block, uint32_t layer) const
        {
            const auto found = std::ranges::find(block.Placements, layer, &BlockPlacement::Layer);
            Require(found != block.Placements.end(), "Destination APC geometry was not provisioned.");
            return *found;
        };

        size_t ValidTile(
            const DenseBlock& block,
            const APCMatrixTile& tile,
            size_t columns,
            SD::DataTypeOfMacroColumn dtype
        ) noexcept
        {
            const auto slot = std::ranges::find(block.APCs, tile.Slot, &ModelAPC::Slot);
            const auto region =  Maps_.RegionIndex.find({tile.Slot, tile.Region});
            Require(slot != block.APCs.end() && region != Maps_.RegionIndex.end(), 
                "Dense tile dosen't use a private region of one of its APC's");
            Require(
                tile.Rows && tile.Columns && 
                tile.LeadingDimension >= tile.Columns &&
                (uint64_t{tile.FirstRow} + tile.Rows) <= block.Nodes.size() &&
                (uint64_t{tile.FirstColumn} + tile.Columns) <= columns, 
                "Dense APC tile dimensions or stride are invalid.");
            
            const StorageRegion& storage = Vectors_.Storage[region->second];
            const uint8_t stride = SD::DTypeByteCount(dtype).value();
            const uint64_t elements = storage.Host.size() / stride;
            const uint64_t extent = uint64_t{tile.Rows -1} * tile.LeadingDimension + tile.Columns;
            Require(
                Vectors_.Region[region->second].DType == dtype &&
                tile.Offset <= elements &&
                (uint64_t{tile.Offset} + extent) <= NO_TENSOR, 
                "Dense tile exceeds its typed APC region.");

        }

        void AppendTile(
            Revision& result,
            const DenseBlock& block,
            uint32_t tensor,
            size_t physical_columns,
            uint32_t row_base,
            std::span<const ColumnRun> columns,
            std::span<const APCMatrixTile> tiles,
            uint32_t width,
            const std::unordered_map<uint32_t, uint32_t>& assignments,
            size_t& remaining_bindings
        ) const;
    };

}