#!/usr/bin/env bash
#
# Verify/build the SuperNova AdaptiveCpp toolchain and build SuperNova on Linux/WSL.
#
# NORMAL MODE:
#   - NEVER rebuilds LLVM or AdaptiveCpp.
#   - Verifies an existing LLVM + AdaptiveCpp installation.
#   - Builds only SuperNova.
#
# EXPLICIT TOOLCHAIN REBUILD MODE:
#   ./build-linux.sh -accp_rebuild
#
#   - Rebuilds a private LLVM toolchain from source.
#   - Rebuilds/reinstalls AdaptiveCpp against that LLVM.
#   - Rebuilds SuperNova afterward.
#
# Compatibility aliases are also accepted:
#   --rebuild-acpp
#   --accp-rebuild
#
# GPU vendor drivers/SDKs are intentionally NOT installed automatically.

set -Eeuo pipefail

JOBS="$(nproc 2>/dev/null || echo 4)"
CLEAN_SUPERNOVA=0
REBUILD_TOOLCHAIN=0
BUILD_PYTHON=0
NATIVE_CPU=0
CPU_ONLY=0

CLANG_VERSION="${CLANG_VERSION:-18}"
LLVM_TAG="${LLVM_TAG:-llvmorg-18.1.8}"

usage() {
  cat <<'EOF'
Usage: build-linux.sh [options]

Default behavior:
  Verify the existing LLVM + AdaptiveCpp installation and build ONLY SuperNova.
  Missing/invalid LLVM or AdaptiveCpp is an error.

Options:
  -accp_rebuild         Force rebuild of LLVM + AdaptiveCpp, then build SuperNova
  --rebuild-acpp        Alias for -accp_rebuild
  --accp-rebuild        Alias for -accp_rebuild
  --jobs N              Parallel build jobs (default: detected CPU count)
  --clean-supernova     Delete only the SuperNova AdaptiveCpp build directory
  --python              Also build SuperNovaBind
  --native-cpu          Enable -march=native / -mtune=native for SuperNova
  --cpu-only            When rebuilding LLVM, build only the X86 LLVM backend
  --clang-version N     System bootstrap LLVM/Clang major (default: 18)
  -h, --help            Show this help

Examples:
  ./build-linux.sh
  ./build-linux.sh -accp_rebuild --jobs 8
  ./build-linux.sh --clean-supernova
  ./build-linux.sh --python
EOF
}

while (($#)); do
  case "$1" in
    -accp_rebuild|--rebuild-acpp|--accp-rebuild)
      REBUILD_TOOLCHAIN=1
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 2; }
      JOBS="$2"
      shift 2
      ;;
    --clean-supernova)
      CLEAN_SUPERNOVA=1
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
    --cpu-only)
      CPU_ONLY=1
      shift
      ;;
    --clang-version)
      [[ $# -ge 2 ]] || { echo "--clang-version requires a value" >&2; exit 2; }
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

ok() {
  printf '\033[1;32m[OK]\033[0m %s\n' "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Required command '$1' was not found on PATH."
}

need git
need cmake
need ninja
need python3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" ||
  die "This script must be run from inside a SuperNova git checkout."

ACPP_SRC="$ROOT/external/AdaptiveCpp"
ACPP_BUILD="$ACPP_SRC/build"
ACPP_ROOT="$ACPP_SRC/install"

# Keep the source/build/install separate from the native-Windows LLVM checkout
# that may exist in the same repository when using WSL.
LLVM_SRC="$ROOT/external/llvm-project-linux"
LLVM_BUILD="$ROOT/external/llvm-build-linux"
LLVM_LOCAL_ROOT="$ROOT/external/llvm-linux"

SUPERNOVA_BUILD="$ROOT/build-acpp-linux"

SYSTEM_CLANG_C="clang-${CLANG_VERSION}"
SYSTEM_CLANG_CXX="clang++-${CLANG_VERSION}"
SYSTEM_LLVM_CONFIG="llvm-config-${CLANG_VERSION}"

LLVM_PREFIX=""
LLVM_DIR=""
CLANG_DIR=""
LLVM_CONFIG_PATH=""
CLANG_C_PATH=""
CLANG_CXX_PATH=""

resolve_existing_llvm() {
  # Prefer the private LLVM installation created by -accp_rebuild.
  if [[ -x "$LLVM_LOCAL_ROOT/bin/llvm-config" &&
        -x "$LLVM_LOCAL_ROOT/bin/clang" &&
        -x "$LLVM_LOCAL_ROOT/bin/clang++" ]]; then
    LLVM_CONFIG_PATH="$LLVM_LOCAL_ROOT/bin/llvm-config"
    CLANG_C_PATH="$LLVM_LOCAL_ROOT/bin/clang"
    CLANG_CXX_PATH="$LLVM_LOCAL_ROOT/bin/clang++"
  elif command -v "$SYSTEM_LLVM_CONFIG" >/dev/null 2>&1 &&
       command -v "$SYSTEM_CLANG_C" >/dev/null 2>&1 &&
       command -v "$SYSTEM_CLANG_CXX" >/dev/null 2>&1; then
    LLVM_CONFIG_PATH="$(command -v "$SYSTEM_LLVM_CONFIG")"
    CLANG_C_PATH="$(command -v "$SYSTEM_CLANG_C")"
    CLANG_CXX_PATH="$(command -v "$SYSTEM_CLANG_CXX")"
  elif command -v llvm-config >/dev/null 2>&1 &&
       command -v clang >/dev/null 2>&1 &&
       command -v clang++ >/dev/null 2>&1; then
    LLVM_CONFIG_PATH="$(command -v llvm-config)"
    CLANG_C_PATH="$(command -v clang)"
    CLANG_CXX_PATH="$(command -v clang++)"
  else
    die "No verified LLVM/Clang installation was found. Install LLVM/Clang ${CLANG_VERSION}, or run './build-linux.sh -accp_rebuild'."
  fi

  LLVM_PREFIX="$("$LLVM_CONFIG_PATH" --prefix)"
  LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
  CLANG_DIR="$LLVM_PREFIX/lib/cmake/clang"

  [[ -x "$CLANG_C_PATH" ]] ||
    die "Clang C compiler is missing: $CLANG_C_PATH"
  [[ -x "$CLANG_CXX_PATH" ]] ||
    die "Clang C++ compiler is missing: $CLANG_CXX_PATH"
  [[ -x "$LLVM_CONFIG_PATH" ]] ||
    die "llvm-config is missing: $LLVM_CONFIG_PATH"
  [[ -d "$LLVM_DIR" ]] ||
    die "LLVM CMake package directory not found: $LLVM_DIR"
  [[ -d "$CLANG_DIR" ]] ||
    die "Clang CMake package directory not found: $CLANG_DIR"
}

verify_llvm() {
  step "Verifying LLVM/Clang installation"

  "$CLANG_CXX_PATH" --version | head -n 1
  "$LLVM_CONFIG_PATH" --version

  local llvm_major
  llvm_major="$("$LLVM_CONFIG_PATH" --version | cut -d. -f1)"

  [[ "$llvm_major" =~ ^[0-9]+$ ]] ||
    die "Could not determine LLVM major version from $LLVM_CONFIG_PATH."

  ok "LLVM/Clang verified at $LLVM_PREFIX"
}

normalize_adaptivecpp_launchers() {
  # AdaptiveCpp's CMake integration invokes Python launcher scripts directly.
  # When the repository/install lives on a Windows-mounted filesystem under WSL,
  # Git/Windows tooling may leave those scripts with CRLF line endings. Linux then
  # reads "#!/usr/bin/env python3\r" and fails with:
  #   /usr/bin/env: 'python3\r': No such file or directory
  #
  # Repair only installed text launchers; this does NOT rebuild LLVM/AdaptiveCpp.
  local candidates=(
    "$ACPP_ROOT/lib/cmake/AdaptiveCpp/syclcc-launcher"
    "$ACPP_ROOT/bin/acpp"
  )

  local file
  local repaired=0

  for file in "${candidates[@]}"; do
    [[ -f "$file" ]] || continue

    if python3 - "$file" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
sys.exit(0 if b"\r\n" in data else 1)
PY
    then
      step "Normalizing CRLF line endings in AdaptiveCpp launcher"
      printf 'Launcher           : %s\n' "$file"

      python3 - "$file" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
path.write_bytes(data.replace(b"\r\n", b"\n"))
PY

      chmod +x "$file" 2>/dev/null || true
      repaired=1
    fi
  done

  if ((repaired)); then
    ok "AdaptiveCpp launcher line endings normalized for Linux/WSL"
  fi
}

verify_adaptivecpp() {
  local acpp="$ACPP_ROOT/bin/acpp"
  local acpp_info="$ACPP_ROOT/bin/acpp-info"
  local acpp_cmake_dir="$ACPP_ROOT/lib/cmake/AdaptiveCpp"
  local acpp_cmake=""

  # CMake accepts either <Package>Config.cmake or the lowercase
  # <package>-config.cmake spelling. Current AdaptiveCpp installs:
  #   lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake
  if [[ -f "$acpp_cmake_dir/adaptivecpp-config.cmake" ]]; then
    acpp_cmake="$acpp_cmake_dir/adaptivecpp-config.cmake"
  elif [[ -f "$acpp_cmake_dir/AdaptiveCppConfig.cmake" ]]; then
    acpp_cmake="$acpp_cmake_dir/AdaptiveCppConfig.cmake"
  fi

  [[ -f "$acpp" ]] ||
    die "AdaptiveCpp compiler driver not found at '$acpp'. Run './build-linux.sh -accp_rebuild'."
  [[ -x "$acpp_info" ]] ||
    die "AdaptiveCpp device tool not found at '$acpp_info'. Run './build-linux.sh -accp_rebuild'."
  [[ -n "$acpp_cmake" ]] ||
    die "AdaptiveCpp CMake package not found under '$acpp_cmake_dir'. Checked adaptivecpp-config.cmake and AdaptiveCppConfig.cmake. Run './build-linux.sh -accp_rebuild'."

  export PATH="$ACPP_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

  normalize_adaptivecpp_launchers

  step "Verifying AdaptiveCpp installation"
  "$acpp" --acpp-version
  "$acpp_info" -l
  ok "AdaptiveCpp verified at $ACPP_ROOT"
}

ensure_linux_llvm_source() {
  if [[ ! -d "$LLVM_SRC/.git" ]]; then
    step "Cloning LLVM source $LLVM_TAG"
    git clone \
      --depth 1 \
      --branch "$LLVM_TAG" \
      https://github.com/llvm/llvm-project.git \
      "$LLVM_SRC"
    return
  fi

  step "Verifying Linux LLVM source checkout"

  if [[ -n "$(git -C "$LLVM_SRC" status --porcelain)" ]]; then
    die "LLVM source checkout '$LLVM_SRC' has local changes. Clean/stash them before -accp_rebuild."
  fi

  git -C "$LLVM_SRC" fetch --depth 1 origin "refs/tags/$LLVM_TAG:refs/tags/$LLVM_TAG"

  local desired current
  desired="$(git -C "$LLVM_SRC" rev-list -n 1 "$LLVM_TAG")"
  current="$(git -C "$LLVM_SRC" rev-parse HEAD)"

  if [[ "$current" != "$desired" ]]; then
    git -C "$LLVM_SRC" checkout --detach "$LLVM_TAG"
  fi
}

rebuild_llvm_and_adaptivecpp() {
  step "Explicit toolchain rebuild requested (-accp_rebuild)"
  printf 'LLVM tag           : %s\n' "$LLVM_TAG"
  printf 'LLVM install       : %s\n' "$LLVM_LOCAL_ROOT"
  printf 'AdaptiveCpp install: %s\n' "$ACPP_ROOT"

  # Source is only required/updated in the explicit rebuild path.
  step "Initializing AdaptiveCpp submodule"
  git -C "$ROOT" submodule sync --recursive
  git -C "$ROOT" submodule update --init --recursive

  [[ -f "$ACPP_SRC/CMakeLists.txt" ]] ||
    die "AdaptiveCpp submodule is missing after initialization."

  printf 'AdaptiveCpp commit : %s\n' "$(git -C "$ACPP_SRC" rev-parse HEAD)"

  ensure_linux_llvm_source

  # A host/bootstrap compiler is still required to compile LLVM itself.
  need "$SYSTEM_CLANG_C"
  need "$SYSTEM_CLANG_CXX"

  step "Cleaning previous Linux LLVM + AdaptiveCpp build/install state"
  rm -rf \
    "$LLVM_BUILD" \
    "$LLVM_LOCAL_ROOT" \
    "$ACPP_BUILD" \
    "$ACPP_ROOT" \
    "$SUPERNOVA_BUILD"

  local llvm_targets="X86;NVPTX;AMDGPU"
  if ((CPU_ONLY)); then
    llvm_targets="X86"
  fi

  step "Configuring LLVM"
  cmake \
    -S "$LLVM_SRC/llvm" \
    -B "$LLVM_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v "$SYSTEM_CLANG_C")" \
    -DCMAKE_CXX_COMPILER="$(command -v "$SYSTEM_CLANG_CXX")" \
    -DCMAKE_INSTALL_PREFIX="$LLVM_LOCAL_ROOT" \
    -DLLVM_TARGETS_TO_BUILD="$llvm_targets" \
    -DLLVM_ENABLE_PROJECTS="clang;lld;openmp" \
    -DLLVM_PARALLEL_LINK_JOBS=2 \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF

  step "Building and installing LLVM"
  cmake --build "$LLVM_BUILD" \
    --target install \
    --parallel "$JOBS"

  LLVM_CONFIG_PATH="$LLVM_LOCAL_ROOT/bin/llvm-config"
  CLANG_C_PATH="$LLVM_LOCAL_ROOT/bin/clang"
  CLANG_CXX_PATH="$LLVM_LOCAL_ROOT/bin/clang++"
  LLVM_PREFIX="$LLVM_LOCAL_ROOT"
  LLVM_DIR="$LLVM_LOCAL_ROOT/lib/cmake/llvm"
  CLANG_DIR="$LLVM_LOCAL_ROOT/lib/cmake/clang"

  verify_llvm

  step "Configuring AdaptiveCpp against rebuilt LLVM"
  cmake \
    -S "$ACPP_SRC" \
    -B "$ACPP_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CLANG_C_PATH" \
    -DCMAKE_CXX_COMPILER="$CLANG_CXX_PATH" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DClang_DIR="$CLANG_DIR" \
    -DCLANG_EXECUTABLE_PATH="$CLANG_CXX_PATH" \
    -DACPP_COMPILER_FEATURE_PROFILE=full \
    -DCMAKE_INSTALL_PREFIX="$ACPP_ROOT"

  step "Building and installing AdaptiveCpp"
  cmake --build "$ACPP_BUILD" \
    --target install \
    --parallel "$JOBS"

  verify_adaptivecpp
}

step "SuperNova repository"
printf 'Repository         : %s\n' "$ROOT"
printf 'AdaptiveCpp install: %s\n' "$ACPP_ROOT"
printf 'Jobs               : %s\n' "$JOBS"
printf 'Toolchain rebuild  : %s\n' "$REBUILD_TOOLCHAIN"

if ((REBUILD_TOOLCHAIN)); then
  rebuild_llvm_and_adaptivecpp
else
  # Normal runs must never build/install LLVM or AdaptiveCpp.
  resolve_existing_llvm
  verify_llvm
  verify_adaptivecpp
  step "Toolchain reuse policy"
  printf 'Existing LLVM and AdaptiveCpp are valid.\n'
  printf 'Skipping ALL LLVM/AdaptiveCpp configure/build/install steps.\n'
fi

ACPP_INFO="$ACPP_ROOT/bin/acpp-info"

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

step "Configuring SuperNova with existing verified AdaptiveCpp"

CMAKE_ARGS=(
  -S "$ROOT/core"
  -B "$SUPERNOVA_BUILD"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER="$CLANG_CXX_PATH"
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

Normal mode never rebuilds LLVM/AdaptiveCpp.

Force toolchain rebuild:
  ./build-linux.sh -accp_rebuild

Run:
  "$SUPERNOVA_EXE"

List devices:
  "$ACPP_INFO" -l

CPU-only runtime example:
  ACPP_VISIBILITY_MASK=omp "$SUPERNOVA_EXE"
EOF
