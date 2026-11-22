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
    ### Float 16 is still npt supported
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
        if role == sn.models.TensorRole.PARAMETER and tuple(value.shape) == (feature_count):
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
        ValueError.dtype != biases[0].dtype
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
        biases_for_feature = _recurrent_ranges(biases, hidden, gates, features, 1)
        for weight, bias in zip_longest(
            _fragments(weights_for_feature, budget),
            _fragments(biases_for_feature, budget),
            fillvalue=()
        ):
            yield[*weights, *bias]