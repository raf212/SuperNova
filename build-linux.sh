#!/usr/bin/env bash
#
# Bootstrap and build SuperNova + AdaptiveCpp on Linux / WSL.
#
# Usage:
#   ./build-linux.sh
#   ./build-linux.sh --rebuild-acpp
#   ./build-linux.sh --clean-supernova --jobs 4
#   ./build-linux.sh --python
#
# GPU vendor drivers/SDKs are intentionally NOT installed automatically.
# Install CUDA / ROCm-HIP / Intel Level Zero/OpenCL before running this script
# if GPU execution is required.

set -Eeuo pipefail

JOBS="$(nproc 2>/dev/null || echo 4)"
CLEAN_SUPERNOVA=0
REBUILD_ACPP=0
BUILD_PYTHON=0
NATIVE_CPU=0
CLANG_VERSION="${CLANG_VERSION:-18}"

usage() {
  cat <<'EOF'
Usage: build-linux.sh [options]

Options:
  --jobs N              Parallel build jobs (default: detected CPU count)
  --clean-supernova     Delete the SuperNova AdaptiveCpp build directory first
  --rebuild-acpp        Delete/rebuild the AdaptiveCpp build directory
  --python              Also build SuperNovaBind
  --native-cpu          Enable -march=native / -mtune=native for SuperNova
  --clang-version N     LLVM/Clang major version (default: 18)
  -h, --help            Show this help

Examples:
  ./build-linux.sh
  ./build-linux.sh --rebuild-acpp --jobs 8
  ./build-linux.sh --python
EOF
}

while (($#)); do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --clean-supernova)
      CLEAN_SUPERNOVA=1
      shift
      ;;
    --rebuild-acpp)
      REBUILD_ACPP=1
      shift
      ;;
    --python)
      BUILD_PYTHON=1
      shift
      ;;
    --native-cpu)
      NATIVE_CPU=1
      shift
      ;;
    --clang-version)
      CLANG_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

step() {
  printf '\n\033[1;36m==> %s\033[0m\n' "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' was not found on PATH."
}

need git
need cmake
need ninja

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" ||
  die "This script must be run from inside a SuperNova git checkout."

ACPP_SRC="$ROOT/external/AdaptiveCpp"
ACPP_BUILD="$ACPP_SRC/build"
ACPP_ROOT="$ACPP_SRC/install"
SUPERNOVA_BUILD="$ROOT/build-acpp-linux"

CLANG_C="clang-${CLANG_VERSION}"
CLANG_CXX="clang++-${CLANG_VERSION}"
LLVM_CONFIG="llvm-config-${CLANG_VERSION}"

need "$CLANG_C"
need "$CLANG_CXX"

step "SuperNova repository"
printf 'Repository         : %s\n' "$ROOT"
printf 'AdaptiveCpp source : %s\n' "$ACPP_SRC"
printf 'AdaptiveCpp install: %s\n' "$ACPP_ROOT"
printf 'Clang major        : %s\n' "$CLANG_VERSION"
printf 'Jobs               : %s\n' "$JOBS"

step "Initializing git submodules"
git -C "$ROOT" submodule sync --recursive
git -C "$ROOT" submodule update --init --recursive

[[ -f "$ACPP_SRC/CMakeLists.txt" ]] ||
  die "AdaptiveCpp submodule is missing after initialization."

printf 'AdaptiveCpp commit : %s\n' "$(git -C "$ACPP_SRC" rev-parse HEAD)"

LLVM_DIR=""
CLANG_DIR=""

if command -v "$LLVM_CONFIG" >/dev/null 2>&1; then
  LLVM_PREFIX="$("$LLVM_CONFIG" --prefix)"
  LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
  CLANG_DIR="$LLVM_PREFIX/lib/cmake/clang"
else
  # Common Debian/Ubuntu package layout.
  LLVM_PREFIX="/usr/lib/llvm-${CLANG_VERSION}"
  LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
  CLANG_DIR="$LLVM_PREFIX/lib/cmake/clang"
fi

[[ -d "$LLVM_DIR" ]] ||
  die "LLVM CMake package directory not found: $LLVM_DIR"

[[ -d "$CLANG_DIR" ]] ||
  die "Clang CMake package directory not found: $CLANG_DIR"

ACPP_INFO="$ACPP_ROOT/bin/acpp-info"

if ((REBUILD_ACPP)) || [[ ! -x "$ACPP_INFO" ]]; then
  step "Configuring AdaptiveCpp"

  if ((REBUILD_ACPP)) && [[ -d "$ACPP_BUILD" ]]; then
    rm -rf "$ACPP_BUILD"
  fi

  cmake \
    -S "$ACPP_SRC" \
    -B "$ACPP_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v "$CLANG_C")" \
    -DCMAKE_CXX_COMPILER="$(command -v "$CLANG_CXX")" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DClang_DIR="$CLANG_DIR" \
    -DCLANG_EXECUTABLE_PATH="$(command -v "$CLANG_CXX")" \
    -DACPP_COMPILER_FEATURE_PROFILE=full \
    -DCMAKE_INSTALL_PREFIX="$ACPP_ROOT"

  step "Building and installing AdaptiveCpp"
  cmake --build "$ACPP_BUILD" \
    --target install \
    --parallel "$JOBS"
else
  step "Reusing existing AdaptiveCpp installation"
fi

[[ -x "$ACPP_INFO" ]] ||
  die "AdaptiveCpp installation did not produce $ACPP_INFO"

export PATH="$ACPP_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

step "AdaptiveCpp devices/backends"
"$ACPP_INFO" -l

printf '\nGPU note:\n'
printf '  CUDA / ROCm-HIP / Intel Level Zero/OpenCL devices appear only when the\n'
printf '  matching vendor driver/runtime was installed before AdaptiveCpp configure.\n'

if ((CLEAN_SUPERNOVA)) && [[ -d "$SUPERNOVA_BUILD" ]]; then
  step "Cleaning SuperNova AdaptiveCpp build directory"
  rm -rf "$SUPERNOVA_BUILD"
fi

NATIVE_FLAG=OFF
if ((NATIVE_CPU)); then
  NATIVE_FLAG=ON
fi

PYTHON_FLAG=OFF
FETCH_PYBIND=OFF
if ((BUILD_PYTHON)); then
  need python3
  PYTHON_FLAG=ON
  FETCH_PYBIND=ON
fi

step "Configuring SuperNova with AdaptiveCpp"

CMAKE_ARGS=(
  -S "$ROOT/core"
  -B "$SUPERNOVA_BUILD"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER="$(command -v "$CLANG_CXX")"
  -DAdaptiveCpp_DIR="$ACPP_ROOT/lib/cmake/AdaptiveCpp"
  -DACPP_TARGETS=generic
  -DSUPERNOVA_ENABLE_ADAPTIVECPP=ON
  -DSUPERNOVA_BUILD_CLI=ON
  -DSUPERNOVA_BUILD_PYTHON="$PYTHON_FLAG"
  -DSUPERNOVA_FETCH_PYBIND11="$FETCH_PYBIND"
  -DSUPERNOVA_ENABLE_IPO=OFF
  -DSUPERNOVA_NATIVE_CPU="$NATIVE_FLAG"
  -DSUPERNOVA_FAST_FP=OFF
)

if ((BUILD_PYTHON)); then
  CMAKE_ARGS+=("-DPython_EXECUTABLE=$(command -v python3)")
fi

cmake "${CMAKE_ARGS[@]}"

step "Building SuperNova"

if ((BUILD_PYTHON)); then
  cmake --build "$SUPERNOVA_BUILD" \
    --target SuperNova SuperNovaBind \
    --parallel "$JOBS" \
    --verbose
else
  cmake --build "$SUPERNOVA_BUILD" \
    --target SuperNova \
    --parallel "$JOBS" \
    --verbose
fi

SUPERNOVA_EXE="$SUPERNOVA_BUILD/SuperNova"
[[ -x "$SUPERNOVA_EXE" ]] ||
  die "SuperNova executable was not found after a successful build."

step "Build complete"
printf '\033[1;32mSuperNova executable: %s\033[0m\n' "$SUPERNOVA_EXE"

cat <<EOF

Run:
  "$SUPERNOVA_EXE"

List devices:
  "$ACPP_INFO" -l

CPU-only runtime example:
  ACPP_VISIBILITY_MASK=omp "$SUPERNOVA_EXE"

The generic AdaptiveCpp flow can use any backend shown by acpp-info -l.
EOF
