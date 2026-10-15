#!/usr/bin/env bash
#
# Verify/build SuperNova + AdaptiveCpp on Linux / WSL.
#
# IMPORTANT LINUX POLICY:
#   - This script NEVER clones, builds, or installs LLVM from source.
#   - LLVM/Clang must come from the Linux system/package manager.
#   - Supported AdaptiveCpp full/generic LLVM releases: 15..21.
#
# NORMAL MODE:
#   ./build-linux.sh
#     1. Verify system LLVM/Clang and required development components.
#     2. Verify the existing AdaptiveCpp installation.
#     3. Build only SuperNova.
#
# ADAPTIVECPP REBUILD MODE:
#   ./build-linux.sh -accp_rebuild
#     1. Verify system LLVM/Clang and required development components.
#     2. Initialize/update the AdaptiveCpp git submodule.
#     3. Rebuild/reinstall AdaptiveCpp ONLY against the system LLVM.
#     4. Build SuperNova.
#
# The Windows script is the only automation that bootstraps/builds LLVM.
#
# GPU vendor drivers/SDKs are intentionally NOT installed automatically.

set -Eeuo pipefail

HARDWARE_THREADS="$(nproc 2>/dev/null || echo 4)"
if ((HARDWARE_THREADS >= 4)); then
  JOBS=$((HARDWARE_THREADS - 2))
else
  JOBS="$HARDWARE_THREADS"
fi
CLEAN_SUPERNOVA=0
REBUILD_ACPP=0
BUILD_PYTHON=0
NATIVE_CPU=0

# AdaptiveCpp currently supports official LLVM releases 15 through 21 for the
# full/generic compiler flow. 18 is a conservative default on Ubuntu 24.04/WSL.
LLVM_MIN_MAJOR=15
LLVM_MAX_MAJOR=21
CLANG_VERSION="${CLANG_VERSION:-18}"

usage() {
  cat <<'EOF'
Usage: build-linux.sh [options]

Linux policy:
  LLVM is NEVER cloned or built by this script.
  A system/package-manager LLVM/Clang release is required.

Default behavior:
  Verify system LLVM/Clang + existing AdaptiveCpp, then build ONLY SuperNova.

Options:
  -accp_rebuild         Rebuild AdaptiveCpp ONLY against system LLVM, then build SuperNova
  --rebuild-acpp        Alias for -accp_rebuild
  --accp-rebuild        Alias for -accp_rebuild
  --jobs N              Parallel build jobs (default: hardware threads - 2 when >= 4)
  --clean-supernova     Delete only the SuperNova AdaptiveCpp build directory
  --python              Also build SuperNovaBind
  --native-cpu          Enable -march=native / -mtune=native for SuperNova
  --clang-version N     System LLVM/Clang major to use (supported: 15..21; default: 18)
  -h, --help            Show this help

Examples:
  ./build-linux.sh
  ./build-linux.sh -accp_rebuild
  ./build-linux.sh -accp_rebuild --jobs 8
  ./build-linux.sh --clang-version 20
  ./build-linux.sh --clean-supernova
EOF
}

while (($#)); do
  case "$1" in
    -accp_rebuild|--rebuild-acpp|--accp-rebuild)
      REBUILD_ACPP=1
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
    --clang-version)
      [[ $# -ge 2 ]] || { echo "--clang-version requires a value" >&2; exit 2; }
      CLANG_VERSION="$2"
      shift 2
      ;;
    --cpu-only)
      printf 'NOTE: --cpu-only is no longer needed on Linux; system LLVM is reused unchanged.\n'
      shift
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

warn() {
  printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

print_base_install_help() {
  cat >&2 <<'EOF'

Install the Linux build prerequisites on Ubuntu/Debian with:

  sudo apt update
  sudo apt install -y \
    git cmake ninja-build python3 \
    build-essential \
    libboost-dev libboost-test-dev

Then rerun this script.
EOF
}

print_llvm_install_help() {
  local version="${1:-18}"

  cat >&2 <<EOF

Install/repair the required LLVM ${version} components on Ubuntu/Debian with:

  sudo apt update
  sudo apt install -y \\
    clang-${version} \\
    llvm-${version} \\
    llvm-${version}-dev \\
    libclang-${version}-dev \\
    libomp-${version}-dev \\
    lld-${version}

For SuperNova's Linux link/runtime dependencies, also ensure:

  sudo apt install -y libatomic1

AdaptiveCpp full/generic supports official LLVM releases ${LLVM_MIN_MAJOR}..${LLVM_MAX_MAJOR}.
If LLVM ${version} packages are unavailable in your distribution repositories,
install an official supported LLVM release from your distro or apt.llvm.org,
then rerun with:

  ./build-linux.sh --clang-version <major>

This script will NOT clone or build LLVM from source.
EOF
}

require_base_tools() {
  local missing=()

  local tool
  for tool in git cmake ninja python3; do
    if ! command_exists "$tool"; then
      missing+=("$tool")
    fi
  done

  if ((${#missing[@]})); then
    printf 'Missing required commands:' >&2
    printf ' %s' "${missing[@]}" >&2
    printf '\n' >&2
    print_base_install_help
    exit 1
  fi
}

require_supported_requested_version() {
  [[ "$CLANG_VERSION" =~ ^[0-9]+$ ]] ||
    die "--clang-version must be an integer major version."

  if ((CLANG_VERSION < LLVM_MIN_MAJOR || CLANG_VERSION > LLVM_MAX_MAJOR)); then
    die "LLVM/Clang ${CLANG_VERSION} is outside AdaptiveCpp's supported full/generic range ${LLVM_MIN_MAJOR}..${LLVM_MAX_MAJOR}."
  fi
}

require_base_tools
require_supported_requested_version

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" ||
  die "This script must be run from inside a SuperNova git checkout."

ACPP_SRC="$ROOT/external/AdaptiveCpp"
ACPP_BUILD="$ACPP_SRC/build"
ACPP_ROOT="$ACPP_SRC/install"
SUPERNOVA_BUILD="$ROOT/build-acpp-linux"

CLANG_C_PATH=""
CLANG_CXX_PATH=""
LLVM_CONFIG_PATH=""
LLVM_PREFIX=""
LLVM_DIR=""
CLANG_DIR=""

resolve_system_llvm() {
  local version="$CLANG_VERSION"

  local clang_c="clang-${version}"
  local clang_cxx="clang++-${version}"
  local llvm_config="llvm-config-${version}"

  local missing=()

  command_exists "$clang_c" || missing+=("$clang_c")
  command_exists "$clang_cxx" || missing+=("$clang_cxx")
  command_exists "$llvm_config" || missing+=("$llvm_config")

  if ((${#missing[@]})); then
    printf '\nERROR: Required LLVM/Clang %s commands are missing:' "$version" >&2
    printf ' %s' "${missing[@]}" >&2
    printf '\n' >&2
    print_llvm_install_help "$version"
    exit 1
  fi

  CLANG_C_PATH="$(command -v "$clang_c")"
  CLANG_CXX_PATH="$(command -v "$clang_cxx")"
  LLVM_CONFIG_PATH="$(command -v "$llvm_config")"

  LLVM_PREFIX="$("$LLVM_CONFIG_PATH" --prefix)"
  LLVM_DIR="$("$LLVM_CONFIG_PATH" --cmakedir 2>/dev/null || true)"

  if [[ -z "$LLVM_DIR" || ! -d "$LLVM_DIR" ]]; then
    LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
  fi

  CLANG_DIR="$LLVM_PREFIX/lib/cmake/clang"
}

verify_llvm_release() {
  step "Verifying system LLVM/Clang installation"

  local llvm_version
  local llvm_major
  local llvm_minor
  local clang_major

  llvm_version="$("$LLVM_CONFIG_PATH" --version)"
  llvm_major="${llvm_version%%.*}"

  local remainder="${llvm_version#*.}"
  llvm_minor="${remainder%%.*}"

  clang_major="$(
    "$CLANG_CXX_PATH" --version |
      sed -nE 's/.*clang version ([0-9]+)(\.[0-9]+)?.*/\1/p' |
      head -n 1
  )"

  printf 'Clang              : '
  "$CLANG_CXX_PATH" --version | head -n 1
  printf 'LLVM               : %s\n' "$llvm_version"
  printf 'LLVM prefix        : %s\n' "$LLVM_PREFIX"
  printf 'LLVM CMake package : %s\n' "$LLVM_DIR"
  printf 'Clang CMake package: %s\n' "$CLANG_DIR"

  [[ "$llvm_major" =~ ^[0-9]+$ ]] ||
    die "Could not determine LLVM major version from '$llvm_version'."

  if ((llvm_major < LLVM_MIN_MAJOR || llvm_major > LLVM_MAX_MAJOR)); then
    print_llvm_install_help "$CLANG_VERSION"
    die "LLVM ${llvm_version} is outside AdaptiveCpp's supported full/generic range ${LLVM_MIN_MAJOR}..${LLVM_MAX_MAJOR}."
  fi

  if [[ -z "$clang_major" || ! "$clang_major" =~ ^[0-9]+$ ]]; then
    die "Could not determine Clang major version from $CLANG_CXX_PATH."
  fi

  if ((clang_major != llvm_major)); then
    die "LLVM/Clang version mismatch: llvm-config reports ${llvm_major}, clang++ reports ${clang_major}. Install matching packages."
  fi

  # AdaptiveCpp rejects LLVM >=18 development snapshots where the release minor
  # is 0. Official LLVM >=18 releases use a nonzero minor component.
  if ((llvm_major > 17)) && [[ "$llvm_minor" =~ ^[0-9]+$ ]] && ((llvm_minor == 0)); then
    die "LLVM ${llvm_version} looks like a development snapshot. AdaptiveCpp requires an official LLVM release."
  fi

  ok "Supported LLVM ${llvm_version} / Clang ${clang_major} detected"
}

verify_llvm_development_components() {
  step "Verifying LLVM/Clang development components"

  local missing=()

  [[ -d "$LLVM_DIR" ]] ||
    missing+=("LLVM CMake package: $LLVM_DIR")

  [[ -d "$CLANG_DIR" ]] ||
    missing+=("Clang CMake package: $CLANG_DIR")

  [[ -f "$LLVM_PREFIX/include/llvm/IR/Module.h" ]] ||
    missing+=("LLVM development headers (llvm-${CLANG_VERSION}-dev)")

  [[ -f "$LLVM_PREFIX/include/clang/AST/ASTContext.h" ]] ||
    missing+=("Clang development headers (libclang-${CLANG_VERSION}-dev)")

  # Verify the compiler can actually compile/link an OpenMP program. This is
  # more reliable than assuming the package from a file-system location.
  local omp_test
  omp_test="$(mktemp --suffix=.cpp)"
  local omp_exe="${omp_test%.cpp}"

  cat >"$omp_test" <<'CPP'
#include <omp.h>
int main() {
    int n = 0;
#pragma omp parallel reduction(+:n)
    n += 1;
    return n > 0 ? 0 : 1;
}
CPP

  if ! "$CLANG_CXX_PATH" -std=c++20 -fopenmp "$omp_test" -o "$omp_exe" \
      >/dev/null 2>&1; then
    missing+=("LLVM OpenMP development runtime (libomp-${CLANG_VERSION}-dev)")
  fi

  rm -f "$omp_test" "$omp_exe"

  if ((${#missing[@]})); then
    printf '\nERROR: LLVM/Clang is installed, but required development components are missing:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    print_llvm_install_help "$CLANG_VERSION"
    exit 1
  fi

  ok "LLVM/Clang development components verified"
}

normalize_adaptivecpp_launchers() {
  # WSL repositories commonly live on /mnt/c. If Windows tooling introduced
  # CRLF into installed Python launchers, Linux sees "#!/usr/bin/env python3\r".
  # Normalize only those installed text launchers; no toolchain rebuild occurs.
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

data = pathlib.Path(sys.argv[1]).read_bytes()
sys.exit(0 if b"\r\n" in data else 1)
PY
    then
      step "Normalizing CRLF line endings in AdaptiveCpp launcher"
      printf 'Launcher           : %s\n' "$file"

      python3 - "$file" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
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

  if [[ -f "$acpp_cmake_dir/adaptivecpp-config.cmake" ]]; then
    acpp_cmake="$acpp_cmake_dir/adaptivecpp-config.cmake"
  elif [[ -f "$acpp_cmake_dir/AdaptiveCppConfig.cmake" ]]; then
    acpp_cmake="$acpp_cmake_dir/AdaptiveCppConfig.cmake"
  fi

  if [[ ! -f "$acpp" || ! -x "$acpp_info" || -z "$acpp_cmake" ]]; then
    cat >&2 <<EOF

ERROR: Existing AdaptiveCpp installation is missing or incomplete:
  expected install prefix: $ACPP_ROOT

LLVM/Clang is NOT the problem and will NOT be rebuilt.

Initialize/rebuild AdaptiveCpp against the verified system LLVM with:

  ./build-linux.sh -accp_rebuild

This rebuilds only:
  $ACPP_BUILD
  $ACPP_ROOT

It never clones or builds LLVM.
EOF
    exit 1
  fi

  export PATH="$ACPP_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

  normalize_adaptivecpp_launchers

  step "Verifying AdaptiveCpp installation"
  "$acpp" --acpp-version
  "$acpp_info" -l

  ok "AdaptiveCpp verified at $ACPP_ROOT"
}

initialize_adaptivecpp_submodule() {
  step "Initializing AdaptiveCpp submodule"

  git -C "$ROOT" submodule sync -- external/AdaptiveCpp
  git -C "$ROOT" submodule update --init --recursive external/AdaptiveCpp

  [[ -f "$ACPP_SRC/CMakeLists.txt" ]] ||
    die "AdaptiveCpp submodule is missing after initialization."

  printf 'AdaptiveCpp commit : %s\n' "$(git -C "$ACPP_SRC" rev-parse HEAD)"
}

rebuild_adaptivecpp_only() {
  step "Explicit AdaptiveCpp rebuild requested (-accp_rebuild)"
  printf 'System LLVM        : %s\n' "$("$LLVM_CONFIG_PATH" --version)"
  printf 'System LLVM prefix : %s\n' "$LLVM_PREFIX"
  printf 'AdaptiveCpp install: %s\n' "$ACPP_ROOT"
  printf 'LLVM source build  : DISABLED on Linux\n'

  initialize_adaptivecpp_submodule

  step "Cleaning previous AdaptiveCpp build/install"
  rm -rf "$ACPP_BUILD" "$ACPP_ROOT"

  # The SuperNova CMake cache may contain paths into the previous AdaptiveCpp
  # install, so force its regeneration after an explicit AdaptiveCpp rebuild.
  rm -rf "$SUPERNOVA_BUILD"

  step "Configuring AdaptiveCpp against system LLVM"
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
    -DWITH_CUDA_BACKEND=ON \
    -DCMAKE_INSTALL_PREFIX="$ACPP_ROOT"

  step "Building and installing AdaptiveCpp"
  cmake --build "$ACPP_BUILD" \
    --target install \
    --parallel "$JOBS"

  verify_adaptivecpp
}

warn_about_legacy_linux_llvm_clone() {
  local found=0
  local legacy

  for legacy in \
    "$ROOT/external/llvm-project-linux" \
    "$ROOT/external/llvm-build-linux" \
    "$ROOT/external/llvm-linux"; do
    if [[ -e "$legacy" ]]; then
      if ((found == 0)); then
        printf '\n'
        warn "Legacy LLVM directories from the old Linux script were found."
        printf 'They are ignored by this script; Linux now uses system LLVM only.\n'
        printf 'If you created them only through the old script, you may remove them manually:\n\n'
        printf '  rm -rf \\\n'
      fi
      printf '    "%s" \\\n' "$legacy"
      found=1
    fi
  done

  if ((found)); then
    printf '\n'
  fi
}

step "SuperNova repository"
printf 'Repository         : %s\n' "$ROOT"
printf 'AdaptiveCpp install: %s\n' "$ACPP_ROOT"
printf 'Jobs               : %s\n' "$JOBS"
printf 'AdaptiveCpp rebuild: %s\n' "$REBUILD_ACPP"
printf 'LLVM source build  : DISABLED on Linux\n'

warn_about_legacy_linux_llvm_clone

resolve_system_llvm
verify_llvm_release
verify_llvm_development_components

if ((REBUILD_ACPP)); then
  rebuild_adaptivecpp_only
else
  verify_adaptivecpp

  step "Toolchain reuse policy"
  printf 'System LLVM/Clang and existing AdaptiveCpp are valid.\n'
  printf 'Skipping all LLVM and AdaptiveCpp build/install steps.\n'
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
  PYTHON_FLAG=ON
  FETCH_PYBIND=ON
fi

step "Configuring SuperNova with verified system LLVM + AdaptiveCpp"

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

Linux never clones/builds LLVM.

Normal build:
  ./build-linux.sh

Rebuild AdaptiveCpp only:
  ./build-linux.sh -accp_rebuild

Run:
  "$SUPERNOVA_EXE"

List devices:
  "$ACPP_INFO" -l

CPU-only runtime example:
  ACPP_VISIBILITY_MASK=omp "$SUPERNOVA_EXE"
EOF
