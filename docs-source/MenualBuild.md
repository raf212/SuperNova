# SuperNova

SuperNova is a C++20 project with an optional AdaptiveCpp/SYCL execution path for heterogeneous CPU/GPU computation.

Repository:

```text
https://github.com/raf212/SuperNova.git
```

This guide covers the complete setup from a fresh clone:

1. clone SuperNova;
2. initialize the AdaptiveCpp submodule;
3. install platform prerequisites;
4. build and install AdaptiveCpp;
5. verify CPU/GPU backends;
6. build SuperNova with AdaptiveCpp on Windows or Linux;
7. optionally build the normal non-AdaptiveCpp version and Python bindings.

> **Hardware portability**
>
> SuperNova uses AdaptiveCpp's `generic` compilation flow by default when
> `SUPERNOVA_ENABLE_ADAPTIVECPP=ON`. A generic build can execute on the runtime
> backends that were successfully built into AdaptiveCpp and are available on
> the current machine.
>
> This does **not** mean that a bare machine automatically supports every GPU.
> The appropriate vendor driver/runtime must be installed:
>
> - CPU: OpenMP backend
> - NVIDIA GPU: CUDA backend/runtime
> - AMD GPU: ROCm/HIP backend/runtime
> - Intel GPU: Level Zero and/or OpenCL backend/runtime
>
> Always run `acpp-info -l` after installing AdaptiveCpp. It is the authoritative
> check for which devices the current installation can actually see.

---

## 1. Requirements

SuperNova itself requires:

- C++20 or newer
- CMake
- Ninja for the commands in this guide
- Git
- a C++20 standard library with `std::atomic_ref` and atomic wait/notify
- Python 3 only when building Python bindings

AdaptiveCpp's full compiler profile requires a supported released LLVM toolchain.
This guide uses a reproducible LLVM/Clang configuration known to work with the
current SuperNova setup.

### Supported AdaptiveCpp execution families

| Hardware | AdaptiveCpp path | Extra software |
|---|---|---|
| CPU | OpenMP / `generic` | OpenMP-capable toolchain |
| NVIDIA GPU | CUDA / `generic` | NVIDIA driver + CUDA toolkit |
| AMD GPU | HIP / `generic` | AMD driver + ROCm |
| Intel GPU | Level Zero or OpenCL / `generic` | Intel GPU driver + Level Zero/OpenCL runtime |
| Other SPIR-V/OpenCL devices | OpenCL / `generic` | compatible OpenCL runtime |
| Apple GPU | Metal / `generic` | macOS + metal-cpp; experimental upstream |
| Vulkan devices | Vulkan / `generic` | Vulkan SDK + clspv; experimental upstream |

AdaptiveCpp currently documents Linux as its strongest-supported OS. Native
Windows supports the integrated-LLVM approach used below.

Upstream documentation:

- AdaptiveCpp installation: https://adaptivecpp.github.io/AdaptiveCpp/installing/
- AdaptiveCpp compilation flows: https://adaptivecpp.github.io/AdaptiveCpp/compilation/
- AdaptiveCpp project usage: https://adaptivecpp.github.io/AdaptiveCpp/using-acpp/
- AdaptiveCpp environment variables: https://adaptivecpp.github.io/AdaptiveCpp/env_variables/

---

# 2. Clone SuperNova and initialize submodules

The preferred fresh clone is:

```bash
git clone --recurse-submodules https://github.com/raf212/SuperNova.git
cd SuperNova
```

If SuperNova was cloned without `--recurse-submodules`:

```bash
git clone https://github.com/raf212/SuperNova.git
cd SuperNova

git submodule sync --recursive
git submodule update --init --recursive
```

Verify AdaptiveCpp:

```bash
git submodule status
git -C external/AdaptiveCpp rev-parse HEAD
```

The expected source directory is:

```text
external/AdaptiveCpp
```

Do not arbitrarily switch the AdaptiveCpp submodule to another branch when
trying to reproduce a known build. The parent SuperNova repository should pin
the intended submodule commit.

### Maintainer-only: adding the submodule if it is not yet registered

Ordinary users should not need this. A repository maintainer can register it with:

```bash
git submodule add https://github.com/AdaptiveCpp/AdaptiveCpp.git external/AdaptiveCpp
git submodule update --init --recursive
git add .gitmodules external/AdaptiveCpp
```

Commit the `.gitmodules` file and the submodule pointer together.

---

# 3. Choose a build mode

SuperNova supports two independent modes.

## 3.1 Normal C++ build

AdaptiveCpp is disabled:

```text
SUPERNOVA_ENABLE_ADAPTIVECPP=OFF
```

Use this for the scalar/reference build, conventional benchmarks, or systems
where heterogeneous execution is not required.

## 3.2 AdaptiveCpp build

AdaptiveCpp is enabled:

```text
SUPERNOVA_ENABLE_ADAPTIVECPP=ON
```

SuperNova defaults to:

```text
ACPP_TARGETS=generic
```

The `generic` flow is the recommended first choice because the same compiled
application can dispatch to supported runtime backends discovered on the
machine.

---

# 4. Linux / WSL: prerequisites

The commands below are written for Debian/Ubuntu-style systems.

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  python3 \
  python3-dev \
  libboost-test-dev \
  libnuma-dev \
  clang-18 \
  llvm-18 \
  llvm-18-dev \
  libclang-18-dev \
  lld-18 \
  libomp-18-dev
```

`libnuma-dev` is optional for SuperNova. If unavailable, SuperNova builds
without NUMA support.

Check the compiler:

```bash
clang-18 --version
clang++-18 --version
llvm-config-18 --version
```

> Package names differ between Linux distributions. The important requirement
> is a released LLVM/Clang version supported by AdaptiveCpp, with its CMake
> package files available.

---

# 5. Linux / WSL: GPU prerequisites

Install only the vendor stacks required by the machine.

## NVIDIA

Install:

- NVIDIA display/compute driver
- CUDA toolkit

Verify:

```bash
nvidia-smi
nvcc --version
```

AdaptiveCpp's generic compiler requires an LLVM installation with NVPTX support
to JIT for NVIDIA hardware.

## AMD

Install a ROCm release supported by the GPU.

Verify:

```bash
rocminfo
```

For the generic flow, AdaptiveCpp recommends a sufficiently recent ROCm stack.
The LLVM used by AdaptiveCpp must also be compatible with the ROCm LLVM level.

## Intel GPU

Install an Intel GPU driver and one or both of:

- Level Zero runtime
- OpenCL runtime

The exact package names depend on distribution and GPU generation.

## CPU only

No GPU SDK is required. AdaptiveCpp's OpenMP host backend is sufficient.

---

# 6. Build AdaptiveCpp on Linux / WSL

From the SuperNova repository root:

```bash
ROOT="$(git rev-parse --show-toplevel)"
ACPP_SRC="$ROOT/external/AdaptiveCpp"
ACPP_BUILD="$ACPP_SRC/build"
ACPP_ROOT="$ACPP_SRC/install"
```

Clean an old AdaptiveCpp build if necessary:

```bash
rm -rf "$ACPP_BUILD"
```

Configure:

```bash
cmake \
  -S "$ACPP_SRC" \
  -B "$ACPP_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang \
  -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
  -DACPP_COMPILER_FEATURE_PROFILE=full \
  -DCMAKE_INSTALL_PREFIX="$ACPP_ROOT"
```

AdaptiveCpp automatically enables runtime backends that it finds. Therefore,
install CUDA, ROCm, Level Zero or OpenCL **before** running the configure step
when those devices are wanted.

Build and install:

```bash
cmake --build "$ACPP_BUILD" \
  --target install \
  --parallel 8
```

Do not stop at a normal build. AdaptiveCpp expects to be installed.

Expose the installation:

```bash
export PATH="$ACPP_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"
```

Verify:

```bash
acpp --acpp-version
acpp-info -l
```

A CPU-only installation should at least show an OpenMP host device. GPU
backends appear only when the corresponding runtime was successfully detected
and can load on the system.

---

# 7. Build SuperNova with AdaptiveCpp on Linux / WSL

From the repository root:

```bash
ROOT="$(git rev-parse --show-toplevel)"
ACPP_ROOT="$ROOT/external/AdaptiveCpp/install"

export PATH="$ACPP_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

rm -rf "$ROOT/build-acpp-linux"
```

Configure a portable Release build:

```bash
cmake \
  -S "$ROOT/core" \
  -B "$ROOT/build-acpp-linux" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DAdaptiveCpp_DIR="$ACPP_ROOT/lib/cmake/AdaptiveCpp" \
  -DACPP_TARGETS=generic \
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON \
  -DSUPERNOVA_BUILD_CLI=ON \
  -DSUPERNOVA_BUILD_PYTHON=OFF \
  -DSUPERNOVA_ENABLE_IPO=OFF \
  -DSUPERNOVA_NATIVE_CPU=OFF \
  -DSUPERNOVA_FAST_FP=OFF
```

A successful configuration should contain:

```text
SuperNova: AdaptiveCpp enabled
SuperNova: ACPP_TARGETS = generic
...
AdaptiveCpp            : ON
AdaptiveCpp targets    : generic
```

Build:

```bash
cmake --build "$ROOT/build-acpp-linux" \
  --target SuperNova \
  --parallel 8 \
  --verbose
```

Run:

```bash
"$ROOT/build-acpp-linux/SuperNova"
```

---

# 8. Windows: prerequisites

Use a 64-bit Windows development environment.

Install:

- Git
- CMake
- Ninja
- Python 3
- Visual Studio / Build Tools with **Desktop development with C++**
- Windows SDK

For NVIDIA GPU support also install the CUDA toolkit before building the final
AdaptiveCpp toolchain.

For AMD GPU support install a compatible ROCm/HIP stack supported by the
machine and AdaptiveCpp.

For Intel GPU support install the Intel GPU driver and an OpenCL runtime.

Open a **Developer PowerShell for Visual Studio**.

Verify:

```powershell
where.exe cl
where.exe link
where.exe cmake
where.exe ninja
```

`cl.exe` and the Windows SDK environment must be available.

---

# 9. Windows: initialize paths and locate mt.exe

From the SuperNova repository root:

```powershell
$Root = (git rev-parse --show-toplevel).Trim().Replace('\', '/')
$AcppSrc = "$Root/external/AdaptiveCpp"
$AcppRoot = "$Root/external/AdaptiveCpp/install-windows"

$SdkBinRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"

$Mt = Get-ChildItem `
    -Path "$SdkBinRoot\*\x64\mt.exe" `
    -ErrorAction Stop |
    Sort-Object {
        [version]$_.Directory.Parent.Name
    } -Descending |
    Select-Object -First 1 -ExpandProperty FullName

$Mt = $Mt.Replace('\', '/')

Write-Host "Repository : $Root"
Write-Host "AdaptiveCpp: $AcppSrc"
Write-Host "Install    : $AcppRoot"
Write-Host "MT         : $Mt"
```

Do not use only:

```text
-DCMAKE_MT=mt
```

unless `mt.exe` is already on `PATH`. Passing the resolved full path is more
robust.

---

# 10. Windows: build LLVM bootstrap compiler

The tested Windows path builds AdaptiveCpp as part of LLVM. This avoids the
plugin limitations of a standalone Windows AdaptiveCpp installation.

This guide pins LLVM 20.1.8 for reproducibility.

Clone it once:

```powershell
if (-not (Test-Path "$Root/external/llvm-project")) {
    git clone `
      --depth 1 `
      --branch llvmorg-20.1.8 `
      https://github.com/llvm/llvm-project.git `
      "$Root/external/llvm-project"
}
```

Clean a previous bootstrap only when rebuilding it:

```powershell
Remove-Item `
  -Recurse `
  -Force `
  "$Root/external/llvm-project/build-bootstrap" `
  -ErrorAction SilentlyContinue
```

Configure the bootstrap compiler:

```powershell
cmake `
  -S "$Root/external/llvm-project/llvm" `
  -B "$Root/external/llvm-project/build-bootstrap" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=cl `
  -DCMAKE_CXX_COMPILER=cl `
  -DCMAKE_MT="$Mt" `
  -DCMAKE_INSTALL_PREFIX="$Root/external/llvm-bootstrap" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_ENABLE_PROJECTS="clang;lld" `
  -DLLVM_PARALLEL_LINK_JOBS=2 `
  -DLLVM_BUILD_LLVM_DYLIB=OFF `
  -DLLVM_LINK_LLVM_DYLIB=OFF
```

Build and install:

```powershell
cmake --build "$Root/external/llvm-project/build-bootstrap" `
  --target install `
  --parallel 8
```

Verify:

```powershell
& "$Root/external/llvm-bootstrap/bin/clang-cl.exe" --version
```

It should report Clang 20.1.8.

---

# 11. Windows: build LLVM + AdaptiveCpp

Set the bootstrap compiler:

```powershell
$BootstrapClang = "$Root/external/llvm-bootstrap/bin/clang-cl.exe"
```

For a normal 64-bit Intel/AMD PC, use the portable `x86-64` host target rather
than hard-coding a specific Intel or AMD microarchitecture:

```powershell
$AcppHostCpu = "x86-64"
```

The stage-2 LLVM target list below includes:

- `X86` for the host CPU;
- `NVPTX` for NVIDIA GPU code generation;
- `AMDGPU` for AMD GPU code generation.

If only CPU execution is wanted, `X86` alone is sufficient.

Clean a previous failed stage-2 build:

```powershell
Remove-Item `
  -Recurse `
  -Force `
  "$Root/external/llvm-project/build-acpp" `
  -ErrorAction SilentlyContinue
```

Configure:

```powershell
cmake `
  -S "$Root/external/llvm-project/llvm" `
  -B "$Root/external/llvm-project/build-acpp" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$BootstrapClang" `
  -DCMAKE_CXX_COMPILER="$BootstrapClang" `
  -DCMAKE_MT="$Mt" `
  -DCMAKE_INSTALL_PREFIX="$AcppRoot" `
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" `
  -DLLVM_ENABLE_PROJECTS="clang;openmp;lld" `
  -DLLVM_PARALLEL_LINK_JOBS=2 `
  -DLLVM_BUILD_LLVM_DYLIB=OFF `
  -DLLVM_LINK_LLVM_DYLIB=OFF `
  -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp `
  -DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR="$AcppSrc" `
  -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
  -DACPP_COMPILER_FEATURE_PROFILE=full `
  -DACPP_HOST_FORCE_MCPU_TARGET="$AcppHostCpu"
```

Build and install:

```powershell
cmake --build "$Root/external/llvm-project/build-acpp" `
  --target install `
  --parallel 8
```

If the machine has limited RAM, reduce parallelism:

```powershell
cmake --build "$Root/external/llvm-project/build-acpp" `
  --target install `
  --parallel 4
```

or:

```powershell
cmake --build "$Root/external/llvm-project/build-acpp" `
  --target install `
  --parallel 2
```

---

# 12. Windows: verify AdaptiveCpp

Expose the installed tools and runtime:

```powershell
$env:Path = "$AcppRoot/bin;$AcppRoot/bin/hipSYCL;$env:Path"
```

Verify the compiler:

```powershell
& "$AcppRoot/bin/clang-cl.exe" --version
```

Verify AdaptiveCpp:

```powershell
& "$AcppRoot/bin/acpp-info.exe" -l
```

A CPU-only configuration should at least contain something similar to:

```text
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
```

If CUDA, ROCm/HIP, Level Zero or OpenCL was installed and successfully detected,
the corresponding GPU backend/device should also appear.

---

# 13. Build SuperNova with AdaptiveCpp on Windows

> **Windows compiler frontend**
>
> Use the AdaptiveCpp installation's **`clang++.exe`** as SuperNova's CMake C++
> compiler for the AdaptiveCpp `generic` flow. Do not use `clang-cl.exe` for
> this SuperNova AdaptiveCpp build. `clang++.exe` targets the Windows/MSVC ABI
> but accepts GNU-style options, which is the frontend expected by AdaptiveCpp's
> CMake launcher. SuperNova detects this using
> `CMAKE_CXX_COMPILER_FRONTEND_VARIANT`.
>
> Verify:
>
> ```powershell
> & "$AcppRoot/bin/clang++.exe" --version
> ```


Keep the AdaptiveCpp installation visible:

```powershell
$Root = (git rev-parse --show-toplevel).Trim().Replace('\', '/')
$AcppRoot = "$Root/external/AdaptiveCpp/install-windows"

$env:Path = "$AcppRoot/bin;$AcppRoot/bin/hipSYCL;$env:Path"
```

Resolve `mt.exe` again if this is a new shell:

```powershell
$SdkBinRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"

$Mt = Get-ChildItem `
    -Path "$SdkBinRoot\*\x64\mt.exe" `
    -ErrorAction Stop |
    Sort-Object {
        [version]$_.Directory.Parent.Name
    } -Descending |
    Select-Object -First 1 -ExpandProperty FullName

$Mt = $Mt.Replace('\', '/')
```

Clean only the SuperNova AdaptiveCpp build directory:

```powershell
Remove-Item `
  -Recurse `
  -Force `
  "$Root/build-acpp-windows" `
  -ErrorAction SilentlyContinue
```

Configure:

```powershell
cmake `
  -S "$Root/core" `
  -B "$Root/build-acpp-windows" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER="$AcppRoot/bin/clang++.exe" `
  -DCMAKE_MT="$Mt" `
  -DAdaptiveCpp_DIR="$AcppRoot/lib/cmake/AdaptiveCpp" `
  -DACPP_TARGETS=generic `
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON `
  -DSUPERNOVA_BUILD_CLI=ON `
  -DSUPERNOVA_BUILD_PYTHON=OFF `
  -DSUPERNOVA_ENABLE_IPO=OFF `
  -DSUPERNOVA_NATIVE_CPU=OFF `
  -DSUPERNOVA_MSVC_AVX2=OFF `
  -DSUPERNOVA_FAST_FP=OFF
```

A successful configuration should report:

```text
SuperNova: AdaptiveCpp enabled
SuperNova: ACPP_TARGETS = generic
...
AdaptiveCpp            : ON
AdaptiveCpp targets    : generic
```

Build:

```powershell
cmake --build "$Root/build-acpp-windows" `
  --target SuperNova `
  --parallel 8 `
  --verbose
```

Run:

```powershell
& "$Root/build-acpp-windows/SuperNova.exe"
```

---

# 14. Selecting CPU or GPU at runtime

First inspect all visible devices:

## Linux

```bash
acpp-info -l
```

## Windows

```powershell
& "$AcppRoot/bin/acpp-info.exe" -l
```

The generic compilation flow is intended to allow the same application to use
the runtime backends available on the current system.

AdaptiveCpp also provides `ACPP_VISIBILITY_MASK` to restrict which backends are
visible.

## Linux examples

CPU/OpenMP only:

```bash
export ACPP_VISIBILITY_MASK=omp
./build-acpp-linux/SuperNova
```

NVIDIA CUDA plus CPU:

```bash
export ACPP_VISIBILITY_MASK="omp;cuda"
./build-acpp-linux/SuperNova
```

AMD HIP plus CPU:

```bash
export ACPP_VISIBILITY_MASK="omp;hip"
./build-acpp-linux/SuperNova
```

Intel Level Zero plus CPU:

```bash
export ACPP_VISIBILITY_MASK="omp;ze"
./build-acpp-linux/SuperNova
```

Intel/OpenCL plus CPU:

```bash
export ACPP_VISIBILITY_MASK="omp;ocl"
./build-acpp-linux/SuperNova
```

Return to normal automatic visibility:

```bash
unset ACPP_VISIBILITY_MASK
```

## Windows examples

CPU/OpenMP only:

```powershell
$env:ACPP_VISIBILITY_MASK = "omp"
& "$Root/build-acpp-windows/SuperNova.exe"
```

NVIDIA CUDA plus CPU:

```powershell
$env:ACPP_VISIBILITY_MASK = "omp;cuda"
& "$Root/build-acpp-windows/SuperNova.exe"
```

AMD HIP plus CPU:

```powershell
$env:ACPP_VISIBILITY_MASK = "omp;hip"
& "$Root/build-acpp-windows/SuperNova.exe"
```

Intel/OpenCL plus CPU:

```powershell
$env:ACPP_VISIBILITY_MASK = "omp;ocl"
& "$Root/build-acpp-windows/SuperNova.exe"
```

Remove the override:

```powershell
Remove-Item Env:ACPP_VISIBILITY_MASK -ErrorAction SilentlyContinue
```

> A visibility mask cannot create a backend that was not built or whose vendor
> driver/runtime is missing. Always confirm with `acpp-info -l`.

---

# 15. Normal SuperNova build without AdaptiveCpp

AdaptiveCpp is optional.

## Windows

```powershell
Remove-Item -Recurse -Force .\build-release -ErrorAction SilentlyContinue

cmake `
  -S .\core `
  -B .\build-release `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=OFF `
  -DSUPERNOVA_BUILD_CLI=ON `
  -DSUPERNOVA_BUILD_PYTHON=OFF `
  -DSUPERNOVA_ENABLE_IPO=ON `
  -DSUPERNOVA_NATIVE_CPU=OFF `
  -DSUPERNOVA_MSVC_AVX2=OFF `
  -DSUPERNOVA_FAST_FP=OFF

cmake --build .\build-release `
  --target SuperNova `
  --parallel `
  --verbose

.\build-release\SuperNova.exe
```

## Linux

```bash
rm -rf ./build-release

cmake \
  -S ./core \
  -B ./build-release \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=OFF \
  -DSUPERNOVA_BUILD_CLI=ON \
  -DSUPERNOVA_BUILD_PYTHON=OFF \
  -DSUPERNOVA_ENABLE_IPO=ON \
  -DSUPERNOVA_NATIVE_CPU=OFF \
  -DSUPERNOVA_FAST_FP=OFF

cmake --build ./build-release \
  --target SuperNova \
  --parallel \
  --verbose

./build-release/SuperNova
```

Set:

```text
SUPERNOVA_NATIVE_CPU=ON
```

only for a machine-specific Linux build where `-march=native` and
`-mtune=native` are desired.

---

# 16. Optional Python bindings

SuperNova's CMake can also build `SuperNovaBind`.

Start with the CLI-only AdaptiveCpp build first. Once that is known to work,
enable Python in a separate build directory.

## Linux

```bash
ROOT="$(git rev-parse --show-toplevel)"
ACPP_ROOT="$ROOT/external/AdaptiveCpp/install"

rm -rf "$ROOT/build-acpp-linux-python"

cmake \
  -S "$ROOT/core" \
  -B "$ROOT/build-acpp-linux-python" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DAdaptiveCpp_DIR="$ACPP_ROOT/lib/cmake/AdaptiveCpp" \
  -DACPP_TARGETS=generic \
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON \
  -DSUPERNOVA_BUILD_CLI=ON \
  -DSUPERNOVA_BUILD_PYTHON=ON \
  -DSUPERNOVA_FETCH_PYBIND11=ON \
  -DSUPERNOVA_ENABLE_IPO=OFF \
  -DSUPERNOVA_NATIVE_CPU=OFF \
  -DSUPERNOVA_FAST_FP=OFF \
  -DPython_EXECUTABLE="$(command -v python3)"

cmake --build "$ROOT/build-acpp-linux-python" \
  --target SuperNova SuperNovaBind \
  --parallel 8 \
  --verbose
```

## Windows

Use the same `$Root`, `$AcppRoot`, `$Mt` and PATH setup from the AdaptiveCpp
Windows section:

```powershell
Remove-Item `
  -Recurse `
  -Force `
  "$Root/build-acpp-windows-python" `
  -ErrorAction SilentlyContinue

cmake `
  -S "$Root/core" `
  -B "$Root/build-acpp-windows-python" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER="$AcppRoot/bin/clang++.exe" `
  -DCMAKE_MT="$Mt" `
  -DAdaptiveCpp_DIR="$AcppRoot/lib/cmake/AdaptiveCpp" `
  -DACPP_TARGETS=generic `
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON `
  -DSUPERNOVA_BUILD_CLI=ON `
  -DSUPERNOVA_BUILD_PYTHON=ON `
  -DSUPERNOVA_FETCH_PYBIND11=ON `
  -DSUPERNOVA_ENABLE_IPO=OFF `
  -DSUPERNOVA_NATIVE_CPU=OFF `
  -DSUPERNOVA_MSVC_AVX2=OFF `
  -DSUPERNOVA_FAST_FP=OFF `
  -DPython_EXECUTABLE="$((Get-Command python).Source)"

cmake --build "$Root/build-acpp-windows-python" `
  --target SuperNova SuperNovaBind `
  --parallel 8 `
  --verbose
```

---

# 17. Important SuperNova CMake options

| Option | Default | Purpose |
|---|---:|---|
| `SUPERNOVA_ENABLE_ADAPTIVECPP` | `OFF` | Enable AdaptiveCpp/SYCL |
| `ACPP_TARGETS` | `generic` when AdaptiveCpp is enabled | AdaptiveCpp compilation flow |
| `SUPERNOVA_BUILD_CLI` | `ON` | Build the native CLI |
| `SUPERNOVA_BUILD_PYTHON` | `OFF` | Build `SuperNovaBind` |
| `SUPERNOVA_FETCH_PYBIND11` | `OFF` | Allow CMake to download pybind11 |
| `SUPERNOVA_ENABLE_NUMA` | `ON` | Use libnuma when available |
| `SUPERNOVA_ENABLE_IPO` | `ON` | Enable IPO/LTO when supported |
| `SUPERNOVA_NATIVE_CPU` | `OFF` | Enable native CPU tuning on GNU-style compilers |
| `SUPERNOVA_MSVC_AVX2` | `OFF` | Explicit `/arch:AVX2` on MSVC-style frontends |
| `SUPERNOVA_FAST_FP` | `OFF` | Permit relaxed floating-point semantics |
| `SUPERNOVA_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

For first-time AdaptiveCpp validation, keep IPO/LTO and fast-math disabled.
Add them back only after correctness has been established.

---

# 18. Troubleshooting

## `The source directory "/core" does not exist`

`$ROOT` was not initialized.

Linux/WSL:

```bash
ROOT="$(git rev-parse --show-toplevel)"
```

PowerShell:

```powershell
$Root = (git rev-parse --show-toplevel).Trim().Replace('\', '/')
```

Then rerun CMake.

---

## `AdaptiveCpp_DIR`, `ACPP_TARGETS`, or `SUPERNOVA_ENABLE_ADAPTIVECPP` are "not used by the project"

The checkout is using an older SuperNova `core/CMakeLists.txt`.

The current CMake must contain:

```text
SUPERNOVA_ENABLE_ADAPTIVECPP
find_package(AdaptiveCpp CONFIG REQUIRED)
add_sycl_to_target(...)
```

Update the SuperNova checkout and reconfigure from a fresh build directory.

---

## `mt.exe` is not found on Windows

Use a Visual Studio Developer PowerShell and resolve the Windows SDK tool
explicitly:

```powershell
$SdkBinRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"

$Mt = Get-ChildItem `
    -Path "$SdkBinRoot\*\x64\mt.exe" `
    -ErrorAction Stop |
    Sort-Object {
        [version]$_.Directory.Parent.Name
    } -Descending |
    Select-Object -First 1 -ExpandProperty FullName

$Mt = $Mt.Replace('\', '/')
```

Pass:

```text
-DCMAKE_MT="$Mt"
```

---

## `llvm-mt: error: no libxml2`

CMake selected LLVM's `llvm-mt` from the bootstrap toolchain. Use the Microsoft
Windows SDK `mt.exe` resolved above and reconfigure the affected build directory
with:

```text
-DCMAKE_MT="$Mt"
```

---

## Windows link errors involving `llvm::PassBuilder`, `PipelineTuningOptions` or `OptimizationLevel`

A native-Windows static integrated LLVM build may expose a dependency-propagation
issue where `AdaptiveCpp.lib` uses LLVM's new pass manager but an LLVM tool is
not linked with `LLVMPasses`.

If the pinned AdaptiveCpp revision used by SuperNova already contains the
corresponding fix, update/reinitialize the submodule first:

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

If the issue remains, the relevant AdaptiveCpp integrated-Windows target must
propagate `LLVMPasses`. A minimal CMake workaround in AdaptiveCpp is:

```cmake
if(WIN32 AND ACPP_LLVM_COMPONENT AND LLVM_ADAPTIVECPP_LINK_INTO_TOOLS)
  if(POLICY CMP0079)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0079 NEW)
  endif()

  if(TARGET AdaptiveCpp AND TARGET LLVMPasses)
    target_link_libraries(AdaptiveCpp PUBLIC LLVMPasses)
  endif()

  if(POLICY CMP0079)
    cmake_policy(POP)
  endif()
endif()
```

Place the workaround after AdaptiveCpp's `add_subdirectory(src)` has created
the `AdaptiveCpp` target. Re-run the existing Ninja build; deleting thousands
of already-built LLVM objects is normally unnecessary.

This is a workaround for a Windows static-link integration edge case, not a
SuperNova source-code requirement.

---

## AdaptiveCpp does not see a GPU

Check:

```bash
acpp-info -l
```

or on Windows:

```powershell
& "$AcppRoot/bin/acpp-info.exe" -l
```

If only OpenMP appears, the CPU backend works but no GPU runtime backend loaded.

Check the vendor stack independently:

### NVIDIA

```bash
nvidia-smi
nvcc --version
```

### AMD

```bash
rocminfo
```

### Intel

Verify the vendor's Level Zero/OpenCL tools and driver installation.

Then reconfigure and rebuild AdaptiveCpp after the required GPU development
stack is installed.

---

## CMake configuration succeeded but `build.ninja` does not exist

The earlier configure actually failed. Read the first CMake error and fix it
before invoking:

```text
cmake --build ...
```

---

# 19. Updating the repository

Update SuperNova:

```bash
git pull
```

Then synchronize pinned submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

To deliberately update AdaptiveCpp to a newer revision, do so as a separate
tested change and commit the new submodule pointer in SuperNova.

---

# 20. Recommended validation order

For architecture development, validate in this order:

1. normal scalar SuperNova build;
2. AdaptiveCpp CPU/OpenMP build;
3. scalar-vs-AdaptiveCpp numerical equivalence;
4. NVIDIA/AMD/Intel GPU backend, if present;
5. Python bindings;
6. IPO/LTO;
7. machine-specific CPU tuning;
8. performance benchmarking.

Do not combine backend bring-up, aggressive compiler optimization, Python
binding changes and numerical algorithm changes into one debugging step.

---

# 21. Quick reference

## Linux / WSL

```bash
git clone --recurse-submodules https://github.com/raf212/SuperNova.git
cd SuperNova

ROOT="$(git rev-parse --show-toplevel)"
ACPP_ROOT="$ROOT/external/AdaptiveCpp/install"

# Build AdaptiveCpp first using section 6.

export PATH="$ACPP_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

acpp-info -l

cmake \
  -S "$ROOT/core" \
  -B "$ROOT/build-acpp-linux" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DAdaptiveCpp_DIR="$ACPP_ROOT/lib/cmake/AdaptiveCpp" \
  -DACPP_TARGETS=generic \
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON \
  -DSUPERNOVA_BUILD_CLI=ON \
  -DSUPERNOVA_BUILD_PYTHON=OFF \
  -DSUPERNOVA_ENABLE_IPO=OFF \
  -DSUPERNOVA_NATIVE_CPU=OFF \
  -DSUPERNOVA_FAST_FP=OFF

cmake --build "$ROOT/build-acpp-linux" \
  --target SuperNova \
  --parallel 8

"$ROOT/build-acpp-linux/SuperNova"
```

## Windows

```powershell
git clone --recurse-submodules https://github.com/raf212/SuperNova.git
Set-Location SuperNova

$Root = (git rev-parse --show-toplevel).Trim().Replace('\', '/')
$AcppRoot = "$Root/external/AdaptiveCpp/install-windows"

# Build integrated LLVM + AdaptiveCpp first using sections 9-11.

$env:Path = "$AcppRoot/bin;$AcppRoot/bin/hipSYCL;$env:Path"

& "$AcppRoot/bin/acpp-info.exe" -l

# Resolve $Mt using section 9.

cmake `
  -S "$Root/core" `
  -B "$Root/build-acpp-windows" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER="$AcppRoot/bin/clang++.exe" `
  -DCMAKE_MT="$Mt" `
  -DAdaptiveCpp_DIR="$AcppRoot/lib/cmake/AdaptiveCpp" `
  -DACPP_TARGETS=generic `
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON `
  -DSUPERNOVA_BUILD_CLI=ON `
  -DSUPERNOVA_BUILD_PYTHON=OFF `
  -DSUPERNOVA_ENABLE_IPO=OFF `
  -DSUPERNOVA_NATIVE_CPU=OFF `
  -DSUPERNOVA_MSVC_AVX2=OFF `
  -DSUPERNOVA_FAST_FP=OFF

cmake --build "$Root/build-acpp-windows" `
  --target SuperNova `
  --parallel 8

& "$Root/build-acpp-windows/SuperNova.exe"
```

---

## License

See `LICENSE.txt` in the SuperNova repository.
