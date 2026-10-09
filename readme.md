# SuperNova

SuperNova is a C++20 architecture with optional AdaptiveCpp/SYCL execution for CPU and GPU workloads.

Repository: https://github.com/raf212/SuperNova

## Before you start

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
powershell -ExecutionPolicy Bypass -File .\build-windows.ps1
```

Run:

```powershell
.\build-acpp-windows\SuperNova.exe
```

### Linux / WSL

```bash
chmod +x build-linux.sh && ./build-linux.sh
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
.\build-windows.ps1 -CpuOnly
.\build-windows.ps1 -BuildPython
.\build-windows.ps1 -RebuildAdaptiveCpp
```

Linux:

```bash
./build-linux.sh --python
./build-linux.sh --native-cpu
./build-linux.sh --rebuild-acpp
```

## Manual installation

For full manual build instructions and troubleshooting, see:

[`MenualBuild.md`](docs-source/MenualBuild.md)

## Build scripts

- `build-windows.ps1` — automatic Windows build
- `build-linux.sh` — automatic Linux / WSL build
- `MenualBuild.md` — complete manual installation documentation