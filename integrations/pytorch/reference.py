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
TensorRegion: TypeAlias = tuple[sn.models.TensorDescription, torch.tensor, list[RegionPiece]]
ModuleStructure: TypeAlias = tuple[tuple[str, int, type[nn.Module], object], ...]
RegionKey: TypeAlias = tuple[int, sn.apc.Column]
StagedOrigin: TypeAlias = tuple[sn.apc.APC, sn.apc.DType, NDArray[np.generic]]

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