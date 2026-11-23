"""Partial declarations from core/bindings/src/NeuromorphicTimeSpace/Fabric.cpp."""
from os import PathLike
from typing import Final
from .apc import RegionSchema, APC
from .models import ModelDescription, DenseModelSnapshot

MINIMUM_APC_CELL_COUNT: Final[int]

class Fabric:
    def __init__(
        self,
        slots: int,
        schemas: list[RegionSchema],
        slot_cells: int = ...,
        batch_capacity: int = 1,
        max_parents: int | None = None,
        segment_pool_path: str | PathLike[str] | None = None,
        models: list[ModelDescription] = ...,
    ) -> None: ...

    def create_apc(self, schemas: list[RegionSchema]) -> APC: ...

    def inspect_dense_model(self, instance : int) -> DenseModelSnapshot: ...

    def dense_structure_revision(self, instance: int) -> int: ...

    def open_apc(self, slot : int, generation : int) -> APC: ...