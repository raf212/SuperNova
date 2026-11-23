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

def _call_target(target: object) -> JSONValue:
    # OpOverload.__str__ is stable within the chosen PyTorch version and
    # contains the ATen namespace and overload (e.g. aten.relu.default).
    # PyTorch exposes an ATen overload's `_schema`, but its `torch._ops`
    # implementation module is intentionally private. Avoid importing that
    # private type merely for a runtime check.
    schema: object = getattr(target, "_schema", None)
    if schema is not None:
        return str(target)
    module = getattr(target, "__module__", None)
    qualified_name = getattr(target, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualified_name, str):
        return [module, qualified_name]
    raise ValueError(f"Unsupported call_function target: {target!r}")


def _fingerprint(program: ExportedProgram) -> str:
    """Hash exported dataflow, signatures and constants, excluding tensor *state*.

    NOT a compiled-executable compatibility hash: check device, shape and
    range constraints, tensor metadata and PyTorch version separately.
    """
    nodes: list[Node] = list(program.graph.nodes)
    indices : dict[Node, int] = {node: index for index, node in enumerate(nodes)}
    records : list[JSONValue] = []

    for node in nodes:
        if node.op in ("placeholder", "output"):
            target : JSONValue = None
        elif node.op == "call_function":
            target = _call_target(node.target)
        else:
            raise ValueError(f"Unsupported exported graph op: {node.op!r}")
        records.append([
            node.op, target, 
            _argument(node.args, indices),
            _argument(node.kwargs, indices)
        ])

    constants = cast(Mapping[str, object], program.constants)
    constant_records : dict[str, JSONValue] = {
        name : _argument(value, indices) for name, value in constants.items()
    }
    payload : JSONValue = [
        "supernova-export-fingerprint-v1",
        records,
        [_signature(spec, indices) for spec in program.graph_signature.input_specs],
        [_signature(spec, indices) for spec in program.graph_signature.output_specs],
        str(program.call_spec.in_spec),
        str(program.call_spec.out_spec),
        constant_records
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _lower_graph(
    program: ExportedProgram, tensor_names: Sequence[str] | Mapping[str, object],
) -> sn.models.DenseOperatorGraph | None:
    """Lower exactly supported one-input/one-output Linear/ReLU graphs.

    An unsupported pattern returns None; the ExportedProgram is still kept.
    """
    names: dict[str, int] = {name: i for i, name in enumerate(tensor_names)}
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    input_specs = program.graph_signature.input_specs
    if len(placeholders) != len(input_specs):
        return None

    inputs: list[Node] = []
    parameters: dict[Node, int] = {}
    for node, spec in zip(placeholders, input_specs, strict=True):
        if spec.kind.name == "USER_INPUT" and isinstance(spec.arg, TensorArgument):
            inputs.append(node)
        elif spec.kind.name == "PARAMETER" and spec.target in names:
            assert spec.target is not None
            parameters[node] = names[spec.target]
        else:
            return None

    outputs = program.graph_signature.output_specs
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or outputs[0].kind.name != "USER_OUTPUT"
        or not isinstance(outputs[0].arg, TensorArgument)
    ):
        return None

    values: dict[Node, int] = {inputs[0]: 0}
    operators: list[sn.models.DenseOperator] = []
    for node in program.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            if len(node.args) != 1 or not isinstance(node.args[0], (tuple, list)):
                return None
            result = cast(Sequence[object], node.args[0])
            if len(result) == 1 and isinstance(result[0], Node) and result[0] in values:
                return sn.models.DenseOperatorGraph(operators, values[result[0]])
            return None
        if (
            node.op != "call_function"
            or bool(node.kwargs)
            or not node.args
            or not isinstance(node.args[0], Node)
            or node.args[0] not in values
        ):
            return None
        operand = values[node.args[0]]
        if node.target == torch.ops.aten.linear.default and len(node.args) in (2, 3):
            weight = node.args[1]
            bias = node.args[2] if len(node.args) == 3 else None
            if not isinstance(weight, Node) or weight not in parameters:
                return None
            if bias is not None and (not isinstance(bias, Node) or bias not in parameters):
                return None
            if isinstance(bias, Node):
                op = sn.models.DenseOperator(
                    sn.models.OperatorKind.LINEAR,
                    operand,
                    parameters[weight],
                    parameters[bias],
                )
            else:
                op = sn.models.DenseOperator(
                    sn.models.OperatorKind.LINEAR, operand, parameters[weight]
                )
        elif node.target == torch.ops.aten.relu.default and len(node.args) == 1:
            op = sn.models.DenseOperator(sn.models.OperatorKind.RELU, operand)
        else:
            return None
        operators.append(op)
        values[node] = len(operators)
    return None


def _dense_model_constructor() -> DenseModelConstructor:
    """Retrieve the not-yet-implemented reference adapter with a clear error."""
    try:
        reference = import_module(".reference", package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.reference":
            raise RuntimeError(
                "integrations.pytorch.reference is missing; the DenseModel "
                "adapter must be implemented before capture() can run"
            ) from exc
        raise
    factory: object = getattr(reference, "DenseModel", None)
    candidate: object = getattr(factory, "_from_capture", None)
    if not callable(candidate):
        raise RuntimeError(
            "integrations.pytorch.reference.DenseModel is not implemented yet. "
            "Implement DenseModel._from_capture() and store_state() to use capture(). "
            "capture_graph() and _fingerprint() can be used independently."
        )
    return cast(DenseModelConstructor, candidate)


def capture(
    module: nn.Module,
    *,
    example_args: tuple[object, ...] | None = None,
    example_kwargs: dict[str, object] | None = None,
    chunk_elements: int = 4096,
    segment_pool_path: str | PathLike[str] | None = None,
) -> DenseModelLike:
    """Capture CPU tensors into APC storage, optionally preserving ATen topology.

    Requires the separate DenseModel reference adapter, not yet present in the
    supplied repository. No APC is allocated if graph export fails.
    """
    _validate_module(module)
    template = deepcopy(module)
    state = module_state(module)
    program = _export_optional(template, example_args, example_kwargs)
    fingerprint = _fingerprint(program) if program is not None else ""
    graph = _lower_graph(program, state) if program is not None else None
    constructor = _dense_model_constructor()  # fail before allocating native APCs
    fabric = build_fabric(module, state, chunk_elements, segment_pool_path, fingerprint, graph)
    dense = constructor(fabric, template, program, fingerprint)
    dense.store_state({name: value for name, (_, value) in state.items()})
    return dense
