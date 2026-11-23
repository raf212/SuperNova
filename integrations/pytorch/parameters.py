"""Describe and allocate the APC regions for PyTorch model state."""

from itertools import zip_longest
import re
import torch
from typing import Final

import SuperNovaBind as sn


__LSTM_GROUP__: Final = 4
__GRU_GROUP__ : Final = 3
__VANILLA_RNN_GROUP__ : Final = 1

_DTYPES = {
    torch.float32: sn.apc.DType.FLOAT32,
    torch.float64: sn.apc.DType.FLOAT64,
    torch.int8: sn.apc.DType.INT8,
    torch.int16: sn.apc.DType.INT16,
    torch.int32: sn.apc.DType.INT32,
    torch.int64: sn.apc.DType.INT64,
    torch.uint8: sn.apc.DType.UINT8,
    torch.uint16: sn.apc.DType.UINT16,
    torch.uint32: sn.apc.DType.UINT32,
    torch.uint64: sn.apc.DType.UINT64,
    ### Float 16 is still not supported
    torch.float16: sn.apc.DType.FLOAT16
}

def _module_state(module):
    parameters = sn.models.TensorRole.PARAMETER
    buffers = sn.models.TensorRole.BUFFER
    return {
        **{name: (parameters, value) for name, value in module.named_parameters()},
        **{name: (buffers, value) for name, value in module.named_buffers()},
    }

def _check_tensor(name, value):
    if (
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.dtype not in _DTYPES
    ):
        raise ValueError(f"{name}: expected an initialized CPU dense tensor with a supported dtype")

def _column(name, role):
    if role == sn.models.TensorRole.BUFFER:
        return sn.apc.Column.EXTRA
    if name.rsplit(".", 1)[-1].startswith("bias"):
        return sn.apc.Column.BIAS
    return sn.apc.Column.WEIGHT

def _feature_width(module, name, value):
    owner, _, local_name = name.rpartition(".")
    feature_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    if local_name == "weight" and value.numel() and isinstance(module.get_submodule(owner), feature_modules):
        return value.numel() // value.shape[0]
    return None

def _bias_name(state, weight_name, feature_count):
    name = weight_name[:-len("weight")] + "bias"
    if name in state:
        role, value = state[name]
        if role == sn.models.TensorRole.PARAMETER and tuple(value.shape) == (feature_count,):
            return name
    return None

def _feature_blocks(name, bias, feature_count, width, budget):
    if width <= budget:
        feature_per_sec = budget // width
        for first in range(0, feature_count, feature_per_sec):
            rows = min(feature_per_sec, (feature_count - first))
            regions = [(name, (first * width), rows, width)]
            if bias is not None:
                regions.append((bias, first, 1, rows))
            yield regions
        return

    for feature in  range(feature_count):
        for offset in range(0, width, budget):
            count = min(budget, (width - offset))
            regions = [(name, ((feature * width) + offset), 1, count)]
            if offset == 0 and bias is not None:
                regions.append((bias, feature, 1, 1))
            yield regions

def _recurrent_group(module, state, name):
    owner, _, local_name = name.rpartition(".")
    if not re.fullmatch(r"weight_ih(?:_l[0-9]+(?:_reverse)?)?", local_name):
        return None

    recurrent = module.get_submodule(owner)
    if isinstance(recurrent, (torch.nn.LSTM, torch.nn.LSTMCell)):
        gates = __LSTM_GROUP__
    elif isinstance(recurrent, (torch.nn.GRU, torch.nn.GRUCell)):
        gates = __GRU_GROUP__
    elif isinstance(recurrent, (torch.nn.RNN, torch.nn.RNNCell)):
        gates = __VANILLA_RNN_GROUP__
    else:
        return None

    hidden = recurrent.hidden_size
    prefix = name[:-len(local_name)]
    suffix = local_name[len("weight_ih"):]
    weight_names = [prefix + kind + suffix for kind in("weight_ih", "weight_hh")]
    bias_names = [prefix + kind + suffix for kind in ("bias_ih", "bias_hh")] if recurrent.bias else []
    if any(tensor not in state or state[tensor][0] != sn.models.TensorRole.PARAMETER
           for tensor in weight_names + bias_names):
        return None
    weights = [state[tensor][1] for tensor in weight_names]
    biases = [state[tensor][1] for tensor in bias_names]
    rows = hidden * gates

    if any(
        value.ndim != 2 or
        value.shape[0] != rows or
        not value.shape[1] or
        value.dtype != weights[0].dtype
        for value in weights
    ):
        return None

    if any(
        tuple(value.shape) != (rows,) or
        value.dtype != biases[0].dtype
        for value in biases
    ) :
        return None

    weight_rows = [(tensor, value.shape[1]) for tensor, value in zip(weight_names, weights)]
    bias_row = [(tensor, 1) for tensor in bias_names]
    return hidden, gates, weight_rows, bias_row


def _recurrent_ranges(tensors, hidden, gates, first, features):
    for gate in range(gates):
        row = (gate * hidden) + first
        for name, width in tensors:
            yield name, (row * width), (features * width)


def _fragments(ranges, budget):
    fragment = []
    remaining = budget
    for name, offset, count in ranges:
        while count:
            size = min(count, remaining)
            fragment.append((name, offset, 1, size))
            remaining -= size
            offset += size
            count -= size
            if not remaining:
                yield fragment
                fragment = []
                remaining = budget
    if fragment:
        yield fragment

def _recurrent_blocks(hidden, gates, weights, biases, budget):
    # Each column has its own budget. Concatenate its disjoint source ranges.
    region_width = gates * max(sum(width for _, width in weights), len(biases))
    features_per_apc = budget // region_width
    if features_per_apc:
        for first in range(0, hidden, features_per_apc):
            features = min(features_per_apc, (hidden - first))
            ranges = _recurrent_ranges((weights + biases), hidden, gates, first, features)
            yield[(name, offset, 1, count) for name, offset, count in ranges]
        return

    for feature in range(hidden):
        weights_for_feature = _recurrent_ranges(weights, hidden, gates, feature, 1)
        biases_for_feature = _recurrent_ranges(biases, hidden, gates, feature, 1)
        for weight, bias in zip_longest(
            _fragments(weights_for_feature, budget),
            _fragments(biases_for_feature, budget),
            fillvalue=()
        ):
            yield [*weight, *bias]

def _inspect_model_state(module, state):
    """Validate tensors, discover layers, and classify feature/recurrent groups."""
    layer_indices = {}
    assignments = []
    features = {}
    paired_biases = set()
    recurrent_groups = {}
    recurrent_tensors = set()
    seen_storage = set()

    for name, (role, value) in state.items():
        _check_tensor(name, value)
        if value.numel():
            address = value.untyped_storage().data_ptr()
            if address in seen_storage:
                raise ValueError(f"{name}: distinct tensors sharing storage are unsupported")
            seen_storage.add(address)

        owner = name.rpartition(".")[0]
        if owner not in layer_indices:
            layer_indices[owner] = len(assignments)
            assignments.append([])

        width = _feature_width(module, name, value) if role == sn.models.TensorRole.PARAMETER else None
        if width is not None:
            bias = _bias_name(state, name, value.shape[0])
            features[name] = (width, bias)
            if bias is not None:
                paired_biases.add(bias)

        group = _recurrent_group(module, state, name)
        if group is not None:
            recurrent_groups[name] = group
            _, _, weights, biases = group
            recurrent_tensors.update(tensor for tensor, _ in weights + biases)

    return layer_indices, assignments, features, paired_biases, recurrent_groups, recurrent_tensors


def _add_apc(layer, regions, state, assignments, schemas, columns, bindings):
    """Register one APC's model assignment, tensor bindings, and region schemas.

    `layer` is a layer index. Assignment.slot uses the global APC index
    (`len(schemas)`); TensorBinding uses the APC's index within its layer.
    """
    apc_index = len(assignments[layer])
    assignments[layer].append(sn.models.Assignment(len(schemas)))

    layout = {}
    for name, offset, rows, cols in regions:
        role, value = state[name]
        column = _column(name, role)
        dtype = _DTYPES[value.dtype]
        apc_offset = 0

        if column in layout:
            _, previous_rows, previous_cols = layout[column]
            apc_offset = previous_rows * previous_cols
            layout[column] = dtype, 1, apc_offset + rows * cols
        else:
            layout[column] = dtype, rows, cols

        bindings[name].append(
            sn.models.TensorBinding(apc_index, column, offset, apc_offset, rows * cols)
        )

    row = {}
    for column, (dtype, rows, cols) in layout.items():
        schema = sn.apc.RegionSchema(column, dtype, rows, cols, model_instance=1)
        row[column] = schema
        columns.setdefault(column, schema)
    schemas.append(row)


def _allocate_fabric(model, schemas, columns, segment_pool_path):
    """Create the native Fabric and install all APC region schemas in slot order."""
    if not columns:
        # Fabric requires at least one enabled column, even with no parameters.
        columns[sn.apc.Column.WEIGHT] = sn.apc.RegionSchema(
            sn.apc.Column.WEIGHT, sn.apc.DType.FLOAT32, 1, 1
        )

    # Each APC must provide a schema for every enabled Fabric column.
    placeholders = {
        column: sn.apc.RegionSchema(column, sn.apc.DType.UINT8, 1, 1, model_instance=1)
        for column in columns
    }
    slot_schemas = [
        [row.get(column, placeholder) for column, placeholder in placeholders.items()]
        for row in schemas
    ]
    slot_cells = max(
        sn.fabric.MINIMUM_APC_CELL_COUNT, max((sum(record.cell_count for record in row) for row in slot_schemas), default=0)
    )

    fabric = sn.fabric.Fabric(
        max(1, len(schemas)), list(columns.values()), slot_cells=slot_cells,
        segment_pool_path=segment_pool_path, models=[model], max_parents=1,
    )
    for slot, row in enumerate(slot_schemas):
        if fabric.create_apc(row).slot != slot:
            raise RuntimeError("New APC slot differs from the Dense ModelIR assignment")
    return fabric


def _describe_model(state, layer_indices, assignments, bindings, fingerprint, graph):
    """Build the native ModelDescription from the collected APC assignments."""
    # Preserve the tensor order expected by operator references in the Dense IR.
    tensor = [
        sn.models.TensorDescription(
            name, role, _DTYPES[value.dtype], layer_indices[name.rpartition(".")[0]],
            list(value.shape), bindings[name]
        )
        for name, (role, value) in state.items()
    ]

    layers = [sn.models.LayerDescription(0, len(layer), layer) for layer in assignments]
    return sn.models.ModelDescription(
        1,
        sn.models.Family.torch_model,
        layers,
        configuration= sn.models.encode_dense_ir(tensor, fingerprint, graph)
    )

def _build_fabric(module, state, chunk_elements, segment_pool_path, fingerprint, graph):
    """Convert PyTorch state into APC-backed native Fabric storage."""
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= (2**32 - 8):
        raise ValueError("chunk_elements must be a positive integer within the APC size range")

    (
        layer_indices, assignments, features, paired_biases,
        recurrent_groups, recurrent_tensors,
    ) = _inspect_model_state(module, state)

    schemas = []
    columns = {}
    bindings = {name: [] for name in state}

    for name, (_, value) in state.items():
        layer = layer_indices[name.rpartition(".")[0]]

        if name in recurrent_groups:
            blocks = _recurrent_blocks(*recurrent_groups[name], chunk_elements)
        elif name in features:
            width, bias = features[name]
            blocks = _feature_blocks(name, bias, value.shape[0], width, chunk_elements)
        elif name not in paired_biases and name not in recurrent_tensors:
            blocks = (
                [(name, offset, 1, min(chunk_elements, value.numel() - offset))]
                for offset in range(0, value.numel(), chunk_elements)
            )
        else:
            continue  # Handled together with its feature or recurrent group.

        for regions in blocks:
            _add_apc(layer, regions, state, assignments, schemas, columns, bindings)

    # APCs are allocated by feature, but recurrent tensors are ordered by gate.
    for name in recurrent_tensors:
        bindings[name].sort(key=lambda piece: piece.tensor_offset)

    model = _describe_model(state, layer_indices, assignments, bindings, fingerprint, graph)
    return _allocate_fabric(model, schemas, columns, segment_pool_path)
