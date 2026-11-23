"""Capture PyTorch topology independently of APC-backed parameter storage.

The exported graph is PyTorch-owned. Only a precisely representable subset
(Linear/ReLU) is additionally lowered into native SuperNova DenseOperatorGraph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from importlib import import_module
import json
from os import PathLike
from typing import Callable, Protocol, TypeAlias, cast

import torch
from torch import nn
from torch.export import ExportedProgram
from torch.export.graph_signature import InputSpec, OutputSpec, TensorArgument
from torch.fx import Node

import SuperNovaBind as sn

from .parameters import build_fabric, module_state

__all__ = ["capture", "capture_graph"]

# The current reference.py is empty. This protocol describes the *expected*
# future interface without falsely claiming that the class already exists.
class DenseModelLike(Protocol):
    def store_state(self, state: Mapping[str, torch.Tensor]) -> None: ...


# A constructor callback is appropriate here because the reference adapter
# itself is not yet implemented. No arbitrary native types leak as Any.
DenseModelConstructor: TypeAlias = Callable[
    [sn.fabric.Fabric, nn.Module, ExportedProgram | None, str], DenseModelLike
]


class _SliceValues(Protocol):
    # Python/PyTorch FX slices can contain Nodes, not just integer indices.
    @property
    def start(self) -> object: ...

    @property
    def stop(self) -> object: ...

    @property
    def step(self) -> object: ...


GraphArgument: TypeAlias = object  # PyTorch FX args are recursive/heterogeneous.
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)

def _validate_module(module: object) -> None:
    """Runtime validation at the API boundary (even for untyped callers)."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")


def _validate_export_inputs(example_args: object, example_kwargs: object) -> None:
    """Keep runtime checks while avoiding redundant static isinstance tests."""
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple; use (x,) for one input")
    if example_kwargs is not None and not isinstance(example_kwargs, dict):
        raise TypeError("example_kwargs must be a dict or None")


def capture_graph(
    module: nn.Module,
    example_args: tuple[object, ...],
    example_kwargs: dict[str, object] | None = None,
) -> ExportedProgram:
    """Return a strict ATen ExportedProgram from a PyTorch module."""
    _validate_module(module)
    _validate_export_inputs(example_args, example_kwargs)
    return torch.export.export(module, example_args, kwargs=example_kwargs, strict=True)


def _export_optional(
    module: nn.Module,
    example_args: tuple[object, ...] | None,
    example_kwargs: dict[str, object] | None,
) -> ExportedProgram | None:
    if example_args is None:
        if example_kwargs is not None:
            raise ValueError("Provide example_args alongside example_kwargs")
        return None
    return capture_graph(module, example_args, example_kwargs)

def _tensor_constant(value: torch.tensor) -> JSONValue:
    """Digest tensor contents without requiring NumPy to support its dtype."""
    if value.layout != torch.strided:
        raise ValueError(f"Unsupported exported constant tensor layout: {value.layout}")
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return ["tensor", str(value.dtype), list(value.shape), hashlib.sha256(raw.numpy().tobytes()).hexdigest()]

def _argument(value: GraphArgument, indices: Mapping[Node, int]) -> JSONValue:
    """Serialize a graph argument deterministically without using ``repr`` on tensors."""
    if isinstance(value, Node):
        return ["node", indices[value]]
    if isinstance(value, torch.Tensor):
        return _tensor_constant(value)
    if isinstance(value, tuple):
        return ["tuple", [_argument(item, indices) for item in cast(tuple[object, ...], value)]]
    if isinstance(value, list):
        return ["list", [_argument(item, indices) for item in cast(list[object], value)]]
    if isinstance(value, dict):
        pairs = cast(dict[object, object], value)
        return ["dict", [[_argument(key, indices), _argument(item, indices)] for key, item in pairs.items()]]
    if isinstance(value, slice):
        parts = cast(_SliceValues, value)
        return [
            "slice", _argument(parts.start, indices),
            _argument(parts.stop, indices), _argument(parts.step, indices),
        ]
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return [type(value).__name__, str(value)]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    if type(value) is float:
        return ["float", value.hex()]
    if value is None or type(value) in (bool, int, str):
        return ["literal", cast(None | bool | int | str, value)]
    raise ValueError(f"Unsupported exported graph constant: {type(value).__name__}")

def _signature(spec: InputSpec | OutputSpec, indices: Mapping[Node, int]) -> JSONValue:
    arg = spec.arg
    return [
        spec.kind.name,
        spec.target,
        type(arg).__name__,
        getattr(arg, "name", None),
        _argument(getattr(arg, "value", None), indices),
        getattr(spec, "persistent", None)
    ]