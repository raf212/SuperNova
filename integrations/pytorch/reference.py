"""PyTorch reference execution with APC-backed dense model state.

The native Fabric owns the persistent tensors; this adapter reads independent
CPU tensors for PyTorch autograd and explicitly copies updated tensors back.
No native execution or automatic parameter synchronization is implied.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Self, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.export import ExportedProgram

import SuperNovaBind as sn

from .capture import export_optional, fingerprint, lower_graph
from .parameters import DTYPES, TensorState, check_tensor, module_state

__all__ = ["DenseModel"]

# An APC region can have several independent tensor bindings. We stage all
# writes to a region in one ndarray before writing the region exactly once.
RegionPiece: TypeAlias = tuple[sn.models.TensorBinding, sn.apc.APC]
TensorRegions: TypeAlias = tuple[sn.models.TensorDescription, torch.tensor, list[RegionPiece]]
ModuleStructure: TypeAlias = tuple[tuple[str, int, type[nn.Module], object], ...]
RegionKey: TypeAlias = tuple[int, sn.apc.Column]
StagedRegion: TypeAlias = tuple[sn.apc.APC, sn.apc.DType, NDArray[np.generic]]

def _graph_key(
    graph: sn.models.DenseOperatorGraph | None
) -> tuple[int, tuple[tuple[sn.models.OperatorKind, int, int, int], ...]] | None:
    """Structural comparison only; native opcode and tensor-reference order matter."""
    if graph is None:
        return None
    return (
        graph.output,
        tuple((op.kind, op.input, op.weight, op.bias) for op in graph.operators)
    )

class DenseModel:
    """Attach an ``nn.Module`` topology to a native APC-backed Dense ModelIR.

    ``load_state`` returns standalone CPU tensors for autograd;
    ``store_state`` explicitly commits a full state mapping to APC storage.
    ``forward_with_state`` permits optimizer-managed tensors to be supplied.
    """
    _module: nn.Module
    fabric: sn.fabric.Fabric
    instance: int
    exported_program: ExportedProgram | None
    _tensors: list[sn.models.TensorDescription]
    _assignments: list[list[sn.models.Assignment]]
    _structure_revision: int
    _attached_structure: ModuleStructure

    def __init__(
        self,
        fabric: sn.fabric.Fabric,
        module: nn.Module,
        *,
        instance: int = 1,
        example_args: tuple[object, ...] | None = None,
        example_kwargs: dict[str, object] | None = None
    ) -> None:
        self._module = deepcopy(module)
        program = export_optional(self._module, example_args, example_kwargs)
        graph_fingerprint = fingerprint(program) if program is not None else ""
        self._bind(fabric, 1, program, fingerprint)

    @classmethod
    def _from_capture(
        cls,
        fabric: sn.fabric.Fabric,
        template: nn.Module,
        program: ExportedProgram | None,
        fingerprint: str
    ) -> Self:
        """Construct after ``capture()`` has allocated the native Fabric.

        This internal factory name is required by the current capture adapter.
        """
        dense = cls.__new__(cls)
        dense._module = template
        dense._bind(fabric, 1, program, fingerprint)
        return dense

    @classmethod
    def module(self) -> nn.Module:
        """Owned execution template; call ``train()`` or ``eval()`` as needed."""
        return self._module

    def _module_structure(self) -> ModuleStructure:
        return tuple(
            (name, id(child), type(child), cast(object, child.forward))
            for name, child in self.module.named_moduls(remove_duplicate= False)
        )

    def _bind(
        self,
        fabric: sn.fabric.Fabric,
        instance: int,
        program: ExportedProgram | None,
        graph_fingerprint: str
    ) -> None:
        self.fabric = fabric
        self.instance = instance
        snapshot = fabric.inspect_dense_model(instance)
        self._tensors = snapshot.tensors
        if snapshot.topology_fingerprint:
            if program is None:
                raise ValueError(
                    "Attaching a graph-backed Dense model requires export example_args"
                )
            if snapshot.topology_fingerprint != graph_fingerprint:
                raise ValueError(
                    "The exported PyTorch topology differs from the Dense ModelIR"
                )
            expected = lower_graph(program, [tensor.name for tensor in self._tensors])
            if _graph_key(expected) != _graph_key(snapshot.graph):
                raise ValueError(
                    "The native Dense graph differs from the lowered PyTorch graph"
                )
        self.exported_program = program
        self._attached_structure = self._module_structure()
        self._assignments = self._assignments
        self._structure_revision = snapshot.structure_revision
        self._metadata()

    def _check_revision(self) -> None:
        if self.fabric.dense_structure_revision(self.instance) != self._structure_revision:
            raise RuntimeError(
                "Dense structure changed. Reattach and reload the working state."
            )

    def _metadata(self) -> TensorState:
        state = module_state(self.module)
        if set(state) != {tensor.name for tensor in self._tensors}:
            raise ValueError("PyTorch module state names differ from the Dense ModelIR")

        for tensor in self._tensors:
            role, value = state[tensor.name]
            check_tensor(tensor.name, value)
            if (role, DTYPES[value.dtype], list(value.shape)) != (
                tensor.role, tensor.dtype, tensor.shape,
            ):
                raise ValueError(
                    f"{tensor.name}: role, dtype or shape differs from the Dense ModelIR"
                )
            # Avoid reading an incompletely covered tensor into uninitialized
            # torch.empty storage, or silently writing overlapping ranges.
            offset = 0
            for binding in sorted(tensor.bindings, key=lambda b: b.tensor_offset):
                if binding.tensor_offset != offset or binding.element_count <= 0:
                    raise ValueError(
                        f"{tensor.name}: missing or overlapping tensor binding ranges"
                    )
                offset += binding.element_count
            if offset != value.numel():
                raise ValueError(f"{tensor.name}: APC bindings do not cover the tensor")
        return state

    def _regions(self) -> Iterator[TensorRegions]:
        self._check_revision()
        state = self._metadata()
        for tensor in self._tensors:
            pieces: list[RegionPiece] = []
            for binding in tensor.bindings:
                assignment = self._assignments[tensor.layer_index][binding.apc_index]
                apc = self.fabric.open_apc(assignment.slot, assignment.generation)
                pieces.append((binding, apc))
            yield tensor, state[tensor.name][1], pieces

    def load_state(self) -> dict[str, torch.Tensor]:
        """Return independent CPU tensors with original parameter grad flags."""
        state: dict[str, torch.Tensor] = {}
        for tensor, prototype, pieces in self._regions():
            value = torch.empty(tensor.shape, dtype=prototype.dtype, device="cpu")
            flat: NDArray[np.generic] = cast(NDArray[np.generic], value.numpy().reshape(-1))
            for binding, apc in pieces:
                region = apc.read_region(binding.column, tensor.dtype)
                end = binding.apc_offset + binding.element_count
                if end > region.size:
                    raise ValueError(
                        f"{tensor.name}: Dense binding exceeds its stored region"
                    )
                flat[binding.tensor_offset:binding.tensor_offset + binding.element_count] = (
                    region.flat[binding.apc_offset:end]
                )
            state[tensor.name] = value.requires_grad_(prototype.requires_grad)
        return state
    
    def store_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Stage and explicitly copy the entire optimizer-updated state to APCs."""
        regions = list(self._regions())
        if set(state) != {tensor.name for tensor, _, _ in regions}:
            raise ValueError("Expected a complete Dense model state")

        pending: dict[RegionKey, StagedRegion] = {}
        for tensor, prototype, pieces in regions:
            value = state[tensor.name]
            check_tensor(tensor.name, value)
            if (value.dtype, value.shape) != (prototype.dtype, prototype.shape):
                raise ValueError(f"{tensor.name}: state dtype or shape changed")
            flat = cast(NDArray[np.generic], value.detach().contiguous().numpy().reshape(-1))
            for binding, apc in pieces:
                key: RegionKey = (apc.slot, binding.column)
                if key not in pending:
                    # A detached copy avoids mutating pinned read-only APC views.
                    view = apc.read_region(binding.column, tensor.dtype)
                    pending[key] = (
                        apc, tensor.dtype,
                        cast(NDArray[np.generic], np.array(view, order="C", copy=True)),
                    )
                    del view
                _, dtype, region = pending[key]
                end = binding.apc_offset + binding.element_count
                if dtype != tensor.dtype or end > region.size:
                    raise ValueError(
                        f"{tensor.name}: Dense binding disagrees with the stored region"
                    )
                region.flat[binding.apc_offset:end] = flat[
                    binding.tensor_offset:binding.tensor_offset + binding.element_count
                ]

        for (_, column), (apc, dtype, region) in pending.items():
            apc.write_region(column, dtype, region)
