<p align="center">
  <img
    src="./docs-source/SuperNova.png"
    alt="SuperNova Logo"
    width="650"
  />
</p>


<h4 align="center"><i>
SuperNova is being designed as a model-agnostic, learning-rule-agnostic, structurally mutable computational library.Built upon a concurrently mutable directed acyclic graph (DAG) with CSR-like sparse computation, originally designed for Generalized Hierarchical Gaussian Filter's (GHGF).The ultimate objective is to unify PyTorch compatible dense modeling with fast memory local predictive coding, Bayesian inference, and continual structural learning through vendor agnostic heterogeneous device execution using AdaptiveCpp.
</i></h4>

### Before you start

Install these **before running the automatic build scripts**.

### Windows

Required:

- Git
- CMake
- Ninja
- Visual Studio 2022 or Visual Studio Build Tools
  - **Desktop development with C++**
  - MSVC x64 compiler/toolchain
  - Windows 10/11 SDK
- PowerShell

Optional:

- Python 3 — only if building `SuperNovaBind`
- NVIDIA driver + CUDA Toolkit — for NVIDIA GPU support
- AMD ROCm/HIP — for AMD GPU support
- Intel GPU driver + OpenCL runtime — for Intel GPU support

### Linux / WSL

Required:

- Git
- CMake
- Ninja
- GCC/build tools
- Clang 18
- LLVM 18 development packages
- OpenMP development package

Ubuntu/Debian:

```bash
sudo apt update && sudo apt install -y \
  build-essential git cmake ninja-build \
  clang-18 llvm-18 llvm-18-dev libclang-18-dev lld-18 libomp-18-dev \
  libboost-test-dev libnuma-dev
```

Optional:

- Python 3 + development headers — only for `SuperNovaBind`
- NVIDIA driver + CUDA Toolkit
- AMD ROCm/HIP
- Intel Level Zero/OpenCL runtime

> GPU drivers and vendor SDKs are not installed automatically. Install them before building if GPU execution is required.

## Quick Start

Clone:

```bash
git clone https://github.com/raf212/SuperNova.git
cd SuperNova
```

The scripts initialize the AdaptiveCpp submodule automatically.

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1
```

Run:

```powershell
.\build-acpp-windows\SuperNova.exe
```

### Linux / WSL

```bash
chmod +x SuperNova-Linux-Install.sh && ./SuperNova-Linux-Install.sh
```

Run:

```bash
./build-acpp-linux/SuperNova
```

## CPU and GPU support

SuperNova uses AdaptiveCpp with the `generic` compilation flow.

The scripts automatically run:

```text
acpp-info -l
```

to show the devices available on the machine.

| Hardware | Backend |
|---|---|
| CPU | OpenMP |
| NVIDIA GPU | CUDA |
| AMD GPU | HIP / ROCm |
| Intel GPU | Level Zero / OpenCL |

CPU execution does not require a GPU SDK.

## Useful options

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1 -CpuOnly
powershell -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1 -BuildPython
powershell -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1 -RebuildAdaptiveCpp
```

Linux:

```bash
./SuperNova-Linux-Install.sh --python
./SuperNova-Linux-Install.sh --native-cpu
./SuperNova-Linux-Install.sh --rebuild-acpp
```

## Manual installation

For full manual build instructions and troubleshooting, see:

[`MenualBuild.md`](docs-source/MenualBuild.md)

## Build scripts

- `SuperNova-Windows-Install.ps1` — automatic Windows build
- `SuperNova-Linux-Install.sh` — automatic Linux / WSL build
- `MenualBuild.md` — complete manual installation documentation