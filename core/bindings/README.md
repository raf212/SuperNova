# SuperNova Python bindings

This source overlay targets the attached SuperNova snapshot. Extract the ZIP at the repository root, where `core/headers`, `core/src`, and `core/templates` already exist. It adds the binding files and replaces `core/CMakeLists.txt` so CMake builds every binding translation unit. The two commented placeholders from the snapshot, `core/bindings/ArchitectureBindings.hpp` and `core/bindings/py_packed_bind.cpp`, can be removed.

## Build and run

Use the same Python interpreter for installation, CMake, and import. The extension is written beside `core/`.

**Linux or WSL**

```sh
python3 -m pip install "pybind11==3.0.1" numpy
cmake -S core -B build-python -DCMAKE_BUILD_TYPE=Release \
  -DSUPERNOVA_BUILD_PYTHON=ON -DSUPERNOVA_BUILD_CLI=OFF \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python3)"
cmake --build build-python --config Release --target build_python_module
PYTHONPATH="$PWD" python3 core/bindings/smoke_test.py
```

**Windows PowerShell**

```powershell
python -m pip install "pybind11==3.0.1" numpy
cmake -S core -B build-python -DCMAKE_BUILD_TYPE=Release `
  -DSUPERNOVA_BUILD_PYTHON=ON -DSUPERNOVA_BUILD_CLI=OFF `
  "-Dpybind11_DIR=$(python -m pybind11 --cmakedir)" `
  "-DPython_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')"
cmake --build build-python --config Release --target build_python_module
$env:PYTHONPATH = (Get-Location).Path
python core/bindings/smoke_test.py
```

For your AdaptiveCpp build, add the same generator, compiler and package paths you use for the native core, plus `-DSUPERNOVA_ENABLE_ADAPTIVECPP=ON -DACPP_TARGETS=generic`. The smoke test exercises the CPU path. Windows and AdaptiveCpp require verification on the target toolchain.

## First call

```python
import numpy as np
import SuperNovaBind as sn

column = sn.apc.Column.WEIGHT
dtype = sn.apc.DType.FLOAT32
schema = sn.apc.RegionSchema(column, dtype, rows=2, columns=4)
fabric = sn.fabric.Fabric(slots=2, schemas=[schema])
cell = fabric.create_apc([schema])

weights = np.arange(8, dtype=np.float32).reshape(2, 4)
cell.write_region(column, dtype, weights)
print(cell.read_region(column, dtype))

ghgf = sn.models.ghgf
model = ghgf.Model([ghgf.Role.VALUE, ghgf.Role.OBSERVATION],
                   [ghgf.Connection(parent=0, child=1)], batch_capacity=2)
print(model.predict(2))
model.train(np.array([[1, 0]], dtype=np.float32))
```

## Where to make changes

| C++ binding file | Python surface |
| --- | --- |
| `src/Module.cpp` and `headers/Bindings.hpp` | Module registration |
| `src/AdaptivePackedCellContainer/APC.cpp` | Region schemas, APC handles, and parent operations |
| `headers/AdaptivePackedCellContainer/APCHandle.hpp` | Fabric ownership for borrowed native APC handles |
| `templates/AdaptivePackedCellContainer/RegionCopies.tpp` | Typed copies between APC regions and NumPy |
| `src/NeuromorphicTimeSpace/Fabric.cpp` | Fabric creation, optional disk paging, and APC allocation |
| `src/Models/ModelIR.cpp` | Model descriptions, ordered layers, APC assignments, and dependencies |
| `src/Models/GHGF.cpp` | GHGF construction, inference, training, and structural mutation |
| `src/SuperNovaComputeManager/Computer.cpp` | Backend configuration, runtime information, and checkpoint output |

Change the registration in the relevant source file when its C++ API changes. Add translation units to `bindings/CMakeLists.txt`. Keep data conversion near its caller and use a template when it avoids repeating the same operation across dtypes.

APC reads produce owned NumPy arrays. Writes require exact shape and dtype, native byte order, and C-contiguous storage. Region copies read private or immutable regions and write private regions. `CHAR` uses byte-valued `uint8` arrays. Float16 region copies require the SYCL half type from AdaptiveCpp.

Fabric schemas define the active column set. Each APC creation supplies a schema for those columns. Set `slot_cells` in 64-bit cells for larger layouts. When `max_parents` is omitted, the binding caps the core default by the slot count. The current DAG requires a parent's slot to be lower than its child's slot. An APC handle keeps its native fabric alive and continues to obey generation checks after retirement or slot reuse.

GHGF accepts binary `float32` observations in `(observation, batch)` or `(time, observation, batch)` order. Observation positions follow their slot order in the supplied roles. `run_sequence` returns `(mean_log_loss, predictions)`. When structural learning is enabled, pass a learning configuration with both coupling rates set to zero for parameter training. For structural mutations, populate `expected_row_sequence` from `structure_snapshot`.

The compute API exposes the working runtime controls and checkpoint output. Checkpoint output uses a full native fabric snapshot, so saving allocates a slab-sized buffer. Checkpoint restore, registered-region arithmetic, and live layer reassignment are pending native API work in this snapshot. ModelIR entries describe model layouts. The working model execution binding in this snapshot is GHGF.
