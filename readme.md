# SuperNova Build Requirements

SuperNova requires a **C++20-or-newer toolchain** and CMake.

The commands below build:

- `SuperNova` — native CLI executable
- `SuperNovaBind` — Python extension module
- Release configuration with IPO/LTO enabled when supported
- Python bindings through pybind11

> **Important:** These example commands explicitly use `-G Ninja`, so **Ninja must be installed for these exact commands**. SuperNova itself is not tied to Ninja; another CMake generator may be used by removing `-G Ninja` and using the appropriate generator/build configuration.

---

## 1. Windows Requirements

Install:

- **CMake**
- **Ninja**
- **Visual Studio 2022 / Build Tools for Visual Studio** with the **Desktop development with C++** workload
- **Python 3**
- **Git** — required when CMake fetches pybind11 from GitHub

The compiler must support **C++20** and the standard-library features used by SuperNova, including `std::atomic_ref` and C++20 atomic wait/notify.

### Release Build — Windows PowerShell

> The commands below are written for **PowerShell**. The backtick `` ` `` is PowerShell's line-continuation character and must be the final character on the line.

```powershell
Remove-Item -Recurse -Force .\build-release -ErrorAction SilentlyContinue

cmake -S .\core -B .\build-release `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DSUPERNOVA_BUILD_CLI=ON `
    -DSUPERNOVA_BUILD_PYTHON=ON `
    -DSUPERNOVA_FETCH_PYBIND11=ON `
    -DSUPERNOVA_ENABLE_IPO=ON `
    -DSUPERNOVA_NATIVE_CPU=OFF `
    -DSUPERNOVA_MSVC_AVX2=OFF `
    -DSUPERNOVA_FAST_FP=OFF `
    -DPython_EXECUTABLE="$((Get-Command python).Source)"

cmake --build .\build-release `
    --target SuperNova SuperNovaBind `
    --parallel `
    --verbose

.\build-release\SuperNova.exe
```

### Optional MSVC AVX2 Build

For a machine-local build that may use AVX2 instructions, change:

```text
-DSUPERNOVA_MSVC_AVX2=OFF
```

to:

```text
-DSUPERNOVA_MSVC_AVX2=ON
```

Keep it `OFF` for the more portable Windows binary.

---

## 2. Linux Requirements

Install:

- **CMake**
- **Ninja**
- **GCC or Clang** with C++20 support
- **Python 3**
- **Git** — required when CMake fetches pybind11
- POSIX threading support — normally provided by the system toolchain

Optional:

- `libnuma` development package — SuperNova builds without it when unavailable

Typical Debian/Ubuntu packages:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3 python3-dev git libnuma-dev
```

`libnuma-dev` is optional.

---

## 3. macOS Requirements

Install:

- **Xcode Command Line Tools / Apple Clang** with C++20 support
- **CMake**
- **Ninja**
- **Python 3**
- **Git**

Example with Homebrew:

```bash
xcode-select --install
brew install cmake ninja python git
```

`libnuma` is not required on macOS.

---

## 4. Release Build — Linux and macOS

Linux and macOS use the same CMake command below when running from **Bash or Zsh**.

```bash
rm -rf ./build-release

cmake -S ./core -B ./build-release \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUPERNOVA_BUILD_CLI=ON \
    -DSUPERNOVA_BUILD_PYTHON=ON \
    -DSUPERNOVA_FETCH_PYBIND11=ON \
    -DSUPERNOVA_ENABLE_IPO=ON \
    -DSUPERNOVA_NATIVE_CPU=ON \
    -DSUPERNOVA_FAST_FP=OFF \
    -DPython_EXECUTABLE="$(command -v python3)"

cmake --build ./build-release \
    --target SuperNova SuperNovaBind \
    --parallel \
    --verbose

./build-release/SuperNova
```
---

## 5. Notes

### C++ Standard

SuperNova requires **C++20 or newer**. A newer standard may be selected, for example:

```bash
-DCMAKE_CXX_STANDARD=23
```

### Ninja

The commands in this document use:

```text
-G Ninja
```

Therefore Ninja is required **for these commands**.

If you want CMake to choose another available generator, remove:

```text
-G Ninja
```

### Python / pybind11

These examples enable the Python extension:

```text
-DSUPERNOVA_BUILD_PYTHON=ON
-DSUPERNOVA_FETCH_PYBIND11=ON
```

Because fetching is enabled, an internet connection and Git are normally required during first configuration.

For a C++-only build, use:

```text
-DSUPERNOVA_BUILD_PYTHON=OFF
```

and build only:

```bash
cmake --build ./build-release --target SuperNova --parallel
```

### Native CPU Optimization

Linux/macOS examples use:

```text
-DSUPERNOVA_NATIVE_CPU=ON
```

which allows supported GCC/Clang-style compilers to use native CPU tuning such as `-march=native` and `-mtune=native`.

This produces a machine-specific binary. Set it to `OFF` when building a binary intended for other machines.

### Fast Floating Point

The documented release build keeps:

```text
-DSUPERNOVA_FAST_FP=OFF
```

so the compiler does not intentionally relax normal floating-point semantics.

