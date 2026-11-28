#include "State.hpp"

namespace BidirectionalInMemGraph::GMI::Dense
{
    namespace
    {
        using DMEState::ColumnIndex;
        constexpr ColumnRun SINGLE_COLUMN_IDENTITY_RUN{0, 0, 1};

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

    }

    DenseMutationEngine::StateDM::~StateDM()
    {
    #if defined (SN_ACPP_ENABLED)
        if (Cache_.Registered && GPUCache_.GPU && !GPUCache_.GPU->Drain())
        {
            std::terminate();
        }
    #endif
        Release(false);
    }

    void DenseMutationEngine::StateDM::Release(bool finalize) noexcept
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

    const DenseBlock& DenseMutationEngine::StateDM::Block(uint64_t id) const
    {
        const auto entry = Maps_.BlockIndex.find(id);
        Require(entry != Maps_.BlockIndex.end(), "Unknown Dense APC Block.");
        return Cache_.Config.Blocks[entry->second];
    }

    const BlockPlacement& DenseMutationEngine::StateDM::Placement(const DenseBlock& block, uint32_t layer) const
    {
        const auto found = std::ranges::find(block.Placements, layer, &BlockPlacement::Layer);
        Require(found != block.Placements.end(), "Destination APC geometry was not provisioned.");
        return *found;
    }

    size_t DenseMutationEngine::StateDM::ValidateTile(
        const DenseBlock& block,
        const APCMatrixTile& tile,
        size_t columns,
        SD::DataTypeOfMacroColumn dtype
    ) const
    {
        const auto slot = std::ranges::find(block.APCs, tile.Slot, &ModelAPC::Slot);
        const auto region =  Maps_.RegionIndex.find({tile.Slot, tile.Region});
        Require(slot != block.APCs.end() && region != Maps_.RegionIndex.end(),
            "Dense tile does not use a private region in its block.");
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
            Vectors_.Regions[region->second].DType == dtype &&
            uint64_t{tile.Offset} + extent <= elements &&
            uint64_t{tile.Offset} + extent <= NO_TENSOR,
            "Dense tile exceeds its typed APC region.");
        return region->second;
    }

    void DenseMutationEngine::StateDM::AppendTile(
        Revision& result,
        const DenseBlock& block,
        uint32_t tensor,
        size_t physical_columns,
        uint32_t row_base,
        std::span<const ColumnRun> columns,
        std::span<const APCMatrixTile> tiles,
        uint32_t width,
        const TwinU32Umap& assignments,
        size_t& remaining_bindings
    ) const
    {
        TensorDescription& target_tensor = result.SnapShot.DenseIR.Tensors[tensor];
        const uint8_t bytes = SD::DTypeByteCount(target_tensor.DType).value();
        for (const APCMatrixTile& tile : tiles)
        {
            const size_t region = ValidateTile(block, tile, physical_columns, target_tensor.DType);
            for (const ColumnRun& run : columns)
            {
                const uint32_t first = std::max(run.Physical, tile.FirstColumn);
                const uint32_t last = std::min((run.Physical + run.Count), (tile.FirstColumn + tile.Columns));
                if (first >= last) continue;
                const uint32_t count = last - first;
                const uint32_t column = run.Logical + (first - run.Physical);
                const uint32_t offset = tile.Offset + (first - tile.FirstColumn);
                const uint32_t row = row_base + tile.FirstRow;
                const std::byte* data = static_cast<const std::byte*>(Vectors_.Regions[region].Data);
                const bool contiguous = column == UNSIGNED_ZERO &&
                    count == width && tile.LeadingDimension == width &&
                    uint64_t{tile.Rows} * width <= NO_TENSOR;

                const size_t bindings = contiguous ? 1 : tile.Rows;
                if (
                    result.Segments.size() >= Cache_.Config.MaxDeviceSegments ||
                    bindings > remaining_bindings
                )
                {
                    throw CapacityError{};
                }
                remaining_bindings -= bindings;
                result.Segments.push_back({
                    data + (size_t{offset} * bytes),
                    tensor,
                    row,
                    column,
                    tile.Rows,
                    count,
                    tile.LeadingDimension
                });
                const uint32_t apc = assignments.at(tile.Slot);
                if (contiguous)
                {
                    target_tensor.Bindings.push_back({
                        apc,
                        tile.Region,
                        uint64_t{row} * width,
                        offset,
                        tile.Rows * width
                    });
                }
                else
                {
                    for (uint32_t r = 0; r < tile.Rows; r++)
                    {
                        target_tensor.Bindings.push_back({
                            apc,
                            tile.Region,
                            (uint64_t{row + r} * width) + column,
                            offset + (r * tile.LeadingDimension),
                            count
                        });
                    }
                }

            }
        }
    }

    DMEState::CRRev DenseMutationEngine::StateDM::Build(Vect2DU64 selection, uint64_t number) const
    {
        auto result = std::make_shared<Revision>();
        result->Blocks = std::move(selection);
        result->SnapShot = Cache_.Installed;
        ModelSnapshot& snapshot = result->SnapShot;
        snapshot.StructureRevision = number;
        Require(result->Blocks.size() == snapshot.Assignments.size(),
            "Dense layer count does not match.");
        std::vector<LayerDescription> layers = Vectors_.Capacities;
        Vect2DU64 values{Vectors_.Input};
        std::unordered_set<uint64_t> used_blocks;
        std::vector<bool> described(snapshot.DenseIR.Tensors.size());
        size_t remaining_bindings = Cache_.Config.MaxModelBindings - Cache_.FixedBindings;
        for (const Operator & op  : snapshot.DenseIR.Graph->Operators)
        {
            if (op.Kind == OperatorKind::ReLU)
            {
                std::vector<uint64_t> input = values[op.Input];
                values.push_back(std::move(input));
                continue;
            }

            TensorDescription& weight_tensor = snapshot.DenseIR.Tensors[op.Weight];
            const uint32_t layer_index = weight_tensor.LayerIndex;
            const std::vector<uint64_t>& input = values[op.Input];
            std::vector<ModelAPC>& apcs = snapshot.Assignments[layer_index];
            apcs.clear();
            weight_tensor.Bindings.clear();
            described[op.Weight] = true;
            if (op.Bias != NO_TENSOR)
            {
                described[op.Bias] = true;
                snapshot.DenseIR.Tensors[op.Bias].Bindings.clear();
            }

            std::vector<uint64_t> output;
            TwinU32Umap assignments;
            const ColumnRun bias_column = SINGLE_COLUMN_IDENTITY_RUN;
            for (const uint64_t id : result->Blocks[layer_index])
            {
                Require(used_blocks.insert(id).second, "Dense block was selected twice");
                const DenseBlock& block = Block(id);
                const BlockPlacement& place = Placement(block, layer_index);
                Require(place.Initialized, "Dense placement was not certified initialized.");
                for (const ModelAPC* apc : block.Carriers(place))
                {
                    if (apcs.size() >= layers[layer_index].APCCapacity)
                    {
                        throw CapacityError{};
                    }
                    assignments.emplace(apc->Slot, static_cast<uint32_t>(apcs.size()));
                    apcs.push_back(*apc);
                }
                Require(block.Nodes.size() <= (ADS::APC_INDEX_BOUND_SENTINAL - output.size()),
                    "Dense logical output exceeds 32 bit system limit");
                const uint32_t row = static_cast<uint32_t>(output.size());
                const std::vector<ColumnRun> columns = ProjectColumns(
                    Maps_.ColumnIndices.at(&place), input
                );
                AppendTile(*result, block, op.Weight, place.InputColumns.size(),
                    row, columns, place.Weight, static_cast<uint32_t>(input.size()),
                    assignments, remaining_bindings);
                if (op.Bias != NO_TENSOR)
                {
                    AppendTile(*result, block, op.Bias, 1, row, {&bias_column, 1},
                        place.Bias, 1, assignments, remaining_bindings);
                }
                output.insert(output.end(), block.Nodes.begin(), block.Nodes.end());
            }

            Require(!output.empty(), "A Linear must retain at least one output node");
            weight_tensor.Shape = {
                static_cast<uint32_t>(output.size()),
                static_cast<uint32_t>(input.size())
            };
            if (op.Bias != NO_TENSOR)
            {
                snapshot.DenseIR.Tensors[op.Bias].Shape = {weight_tensor.Shape[0]};
            }
            layers[layer_index].APCCount = static_cast<uint32_t>(apcs.size());
            values.push_back(std::move(output));
        }

        for (size_t i = 0; i < snapshot.DenseIR.Tensors.size(); i++)
        {
            TensorDescription& tensor = snapshot.DenseIR.Tensors[i];
            if (!described[i])
            {
                Require(tensor.Role == TensorRole::Buffer, "Unmapped Dense parameter.");
                // Fixed buffers keep their physical regions.
                // Per-node optimizer storage can use the independent region views.
                for (TensorBindings& binding : tensor.Bindings)
                {
                    const uint32_t slot = Cache_.Installed.Assignments[tensor.LayerIndex][binding.APCIndex].Slot;
                    const std::vector<ModelAPC>& apcs = snapshot.Assignments[tensor.LayerIndex];
                    const auto new_index = std::ranges::find(apcs, slot, &ModelAPC::Slot);
                    Require(new_index != apcs.end(), "Mutation would unbind a fixed Dense Buffer.");
                    binding.APCIndex = static_cast<uint32_t>(new_index - apcs.begin());
                }
            }
            std::ranges::sort(tensor.Bindings, {}, &TensorBindings::TensorOffset);
        }
        snapshot.DenseIR.Validate(layers);
        return result;
    }

    void DenseMutationEngine::StateDM::ValidateCatalog()
    {
        const DenseModelDynamics& ir = Cache_.Installed.DenseIR;
        Require(ir.Graph.has_value() && Cache_.Config.MaxDeviceSegments && Cache_.Config.MaxModelBindings,
            "A dense mutation graph and separate descriptor and binding budgets are required");

        std::vector<bool> layers(Vectors_.Capacities.size());
        std::vector<bool> projected(ir.Tensors.size());
        std::vector<uint32_t> origins{0};
        std::vector<const Operator*> owners(layers.size());
        for (const Operator& op : ir.Graph->Operators)
        {
            if (op.Kind == OperatorKind::ReLU)
            {
                origins.push_back(origins[op.Input]);
                continue;
            }

            const TensorDescription& tensor_weight = ir.Tensors[op.Weight];
            Require(
                !layers[tensor_weight.LayerIndex] &&
                tensor_weight.Shape[0] && tensor_weight.Shape[1],
                "Dense mutation requires one nonempty Linear per layer.");
            Require(op.Bias == NO_TENSOR ||
                ir.Tensors[op.Bias].LayerIndex == tensor_weight.LayerIndex,
                "Linear bias must belong to its weight layer");
            layers[tensor_weight.LayerIndex] = true;
            projected[op.Weight] = true;
            if (op.Bias != NO_TENSOR) projected[op.Bias] = true;
            owners[tensor_weight.LayerIndex] = &op;
            if (origins[op.Input] == UNSIGNED_ZERO && Vectors_.Input.empty())
            {
                Vectors_.Input.resize(tensor_weight.Shape[1]);
                for (size_t i = 0; i < Vectors_.Input.size(); i++)
                {
                    Vectors_.Input[i] = i + 1;
                }
            }
            origins.push_back(1);
        }

        Require(!Vectors_.Input.empty() &&
            std::ranges::all_of(layers, [](bool present){
                return present;
            }), "Each mutable layer must correspond to a Linear operator");
        for (size_t tensor = 0; tensor < ir.Tensors.size(); tensor++)
        {
            if (projected[tensor]) continue;
            const size_t count = ir.Tensors[tensor].Bindings.size();
            Require(count <= (Cache_.Config.MaxModelBindings - Cache_.FixedBindings),
                "Fixed dense bindings exceed their budget");
            Cache_.FixedBindings += count;
        }

        std::unordered_set<uint64_t> nodes;
        for (size_t i = 0; i < Cache_.Config.Blocks.size(); i++)
        {
            const DenseBlock& block = Cache_.Config.Blocks[i];
            Require(block.Id && !block.Nodes.empty() && !block.APCs.empty() &&
                !block.Placements.empty() && Maps_.BlockIndex.emplace(block.Id, i).second,
                "Dense blocks require unique IDs, nodes, APCs and placements");
            for (const uint64_t node : block.Nodes)
            {
                Require(node > Vectors_.Input.size() && nodes.insert(node).second,
                    "Dense node ids must be unique and distinct from model inputs.");
            }

            std::unordered_set<uint32_t> placements;
            for (const BlockPlacement& place : block.Placements)
            {
                Require(place.Layer < layers.size() && placements.insert(place.Layer).second &&
                    place.InputColumns.size() <= ADS::APC_INDEX_BOUND_SENTINAL &&
                    !place.Weight.empty(), "Invalid or duplicate dense placement");
                auto& columns = Maps_.ColumnIndices[&place];
                columns.reserve(place.InputColumns.size());
                for (uint32_t clm = 0; clm < place.InputColumns.size(); clm++)
                {
                    Require(place.InputColumns[clm] && columns.emplace(
                        place.InputColumns[clm], clm
                    ).second, "Dense physical columns require unique non zero ids");
                }
                const Operator& op = *owners[place.Layer];
                Require((op.Bias == NO_TENSOR) == place.Bias.empty(),
                    "Dense bias placement is missing or unexpected.");
                for (const APCMatrixTile& tile : place.Weight)
                {
                    (void)ValidateTile(block, tile, place.InputColumns.size(), ir.Tensors[op.Weight].DType);
                }
                for (const APCMatrixTile& tile : place.Bias)
                {
                    (void)ValidateTile(block, tile, 1, ir.Tensors[op.Bias].DType);
                }
            }
        }
    }

    uint64_t DenseMutationEngine::StateDM::Signature() const
    {
        using CheckpointFormat::InternalValues;

        uint64_t hash = InternalValues::FnvOffsetBasis;
        const auto append_byte = [&](uint8_t value)
        {
            hash = (hash ^ value) * InternalValues::FnvPrime;
        };
        const auto append_word = [&](uint64_t value)
        {
            for (unsigned shift = 0; shift < sizeof(uint64_t) * LEN_OF_BYTE_IN_BITS; shift += LEN_OF_BYTE_IN_BITS)
            {
                append_byte(static_cast<uint8_t>(value >> shift));
            }
        };

        const auto append_ids = [&](std::span<const uint64_t> values)
        {
            append_word(values.size());
            for (uint64_t id : values) append_word(id);
        };

        const auto append_tiles = [&](std::span<const APCMatrixTile> values)
        {
            append_word(values.size());
            for (const APCMatrixTile& tile : values)
            {
                for (uint32_t field : {tile.Slot, static_cast<uint32_t>(tile.Region),
                    tile.Offset, tile.FirstRow, tile.FirstColumn, tile.Rows,
                    tile.Columns, tile.LeadingDimension})
                {
                    append_word(field);
                }
            }
        };

        const std::vector<std::byte> encoded = Cache_.Installed.DenseIR.Encode();
        append_word(encoded.size());
        for (std::byte v : encoded) append_byte(std::to_integer<uint8_t>(v));
        append_word(Cache_.Config.Layers.size());
        for (const std::vector<uint64_t>& s : Cache_.Config.Layers) append_ids(s);
        for (const LayerDescription& l : Vectors_.Capacities) append_word(l.APCCapacity);
        append_word(Cache_.Config.Blocks.size());
        for (const DenseBlock& block : Cache_.Config.Blocks)
        {
            append_word(block.Id);
            append_ids(block.Nodes);
            append_word(block.APCs.size());
            for (const ModelAPC& apc : block.APCs)
            {
                for (auto field : {apc.Slot, apc.Generation, apc.Role})
                {
                    append_word(field);
                }
            }
            append_word(block.Placements.size());
            for (const BlockPlacement& placement : block.Placements)
            {
                append_word(placement.Layer);
                append_ids(placement.InputColumns);
                append_tiles(placement.Weight);
                append_tiles(placement.Bias);
                append_word(placement.Initialized);
            }
        }
        for (const DenseResidentRegion& region : Vectors_.Regions)
        {
            append_word(static_cast<uint8_t>(region.DType));
        }
        return hash;
    }

    Vect2DU64 DenseMutationEngine::StateDM::Apply(const Revision& current, std::span<const BlockEdit> edits) const
    {
        Vect2DU64 result = current.Blocks;
        std::unordered_set<uint64_t> edited;
        for (const BlockEdit& edit : edits)
        {
            const DenseBlock& block = Block(edit.Block);
            Require(edited.insert(edit.Block).second, "Edit each dense block only once");
            uint32_t from = INACTIVE_LAYER;
            for (uint32_t layer = 0; layer < result.size(); layer++)
            {
                const auto found = std::ranges::find(result[layer], edit.Block);
                if (found == result[layer].end()) continue;
                from = layer;
                result[layer].erase(found);
                break;
            }
            if (edit.Layer == INACTIVE_LAYER)
            {
                Require(from != INACTIVE_LAYER, "Dense block is already inactive");
                continue;
            }
            Require(edit.Layer < result.size() && edit.Position <= result[edit.Layer].size(),
                "Invalid dense block destination");
            if (from != INACTIVE_LAYER && from != edit.Layer)
            {
                const std::vector<const ModelAPC*> old = block.Carriers(Placement(block, from));
                const std::vector<const ModelAPC*> next = block.Carriers(Placement(block, edit.Layer));
                const auto common = std::ranges::count_if(next, [&](const auto* apc)
                {
                    return std::ranges::find(old, apc) != old.end();
                });
                Require(static_cast<size_t>(common) == std::min(old.size(), next.size()),
                    "A moved dense block must retain all APCs it can carry to its destination");
            }
            result[edit.Layer].insert((result[edit.Layer].begin() + edit.Position), edit.Block);
        }
        Require(result != current.Blocks, "Dense mutation has no structural effect");
        return result;
    }

    void DenseMutationEngine::StateDM::RequireHealthy()
    {
        if (!Cache_.Registered) throw std::logic_error("Dense session is finalized");
    #if defined (SN_ACPP_ENABLED)
        if (GPUCache_.GPU && !GPUCache_.GPU->Healthy()) throw std::runtime_error("Dense device execution failed");
    #endif
        if (Cache_.Failed) throw std::runtime_error("Dense numerical execution failed");
    }

    void DenseMutationEngine::StateDM::DropPlan()
    {
    #if defined (SN_ACPP_ENABLED)
        if (GPUCache_.DevicePlans) GPUCache_.DevicePlans->CancelStage();
    #endif
        SharedPtrs_.Retired = std::move(Cache_.Plan->Next);
        Cache_.Plan.reset();
    }

    MutationStatus DenseMutationEngine::StateDM::AdvanceLocked()
    {
        if (!Cache_.Plan.has_value()) return MutationStatus::Idle;
        if (Cache_.Clock >= Cache_.Plan->Deadline)
        {
            DropPlan();
            return MutationStatus::Expired;
        }
        if (Cache_.Clock + 1 < Cache_.Plan->Target) return MutationStatus::TooEarly;
        if (Cache_.CPUWriter) return MutationStatus::Busy;

    #if defined (SN_ACPP_ENABLED)
        if (GPUCache_.DevicePlans)
        {
            using R = ComputerDetail::PlanUpdateResult;
            const R status = GPUCache_.DevicePlans->TryActive(
                Cache_.Plan->Next->SnapShot.StructureRevision
            );
            if (status == R::BUSY) return MutationStatus::WaitingForDevice;
            if (status != R::ACCEPTED)
            {
                Cache_.Failed = true;
                DropPlan();
                return MutationStatus::DeviceFailure;
            }
        }
    #endif
        SharedPtrs_.Retired = SharedPtrs_.Root->Active.exchange(
            std::move(Cache_.Plan->Next), std::memory_order_acq_rel
        );
        Cache_.Plan.reset();
        return MutationStatus::Published;
    }
}
