"""Describe and allocate the APC regions for PyTorch model state."""

from itertools import zip_longest
import re

import torch

import SuperNovaBind as sn


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
