#!/usr/bin/env python3
"""Build and install SuperNovaBind into the Python environment running this script.

Usage:
    python InstallSuperNovaBindings.py /path/to/build-acpp-linux
    python InstallSuperNovaBindings.py "A:\\repo\\build-acpp-windows"

Run again after changing C++ code: CMake incrementally rebuilds SuperNovaBind,
and this script installs the extension only when its binary content changes.

On Windows, installs the AdaptiveCpp runtime DLLs staged by the Windows
build script beside the extension. A .pyd cannot load if its native DLLs are
missing from its load-search directories.

Options:
    --no-build       Install an already-compiled extension without invoking CMake.
    --config NAME    CMake configuration for multi-config generators (default: Release).
    --no-verify      Skip the independent post-install import test.
    --module PATH    Explicit extension location, if using a custom output directory.
"""

import argparse
import hashlib
import json
import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

MODULE = "SuperNovaBind"


def fail(message):
    raise RuntimeError(message)


def cmake_cache(build_dir):
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        fail(f"Not a configured CMake build directory (missing {cache_file}).")

    values = {}
    for line in cache_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        if ":" in key_type:
            key, _type = key_type.split(":", 1)
            values[key] = value
    return values


def normalized_executable(path):
    # Avoid resolving symlinks: two virtual environments may share one underlying
    # Python binary but have distinct site-packages directories.
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def validate_cache(cache, no_build):
    if cache.get("SUPERNOVA_BUILD_PYTHON", "").upper() not in ("ON", "TRUE", "1", "YES"):
        fail("This build has SUPERNOVA_BUILD_PYTHON=OFF (or unset). Reconfigure "
             "SuperNova with -DSUPERNOVA_BUILD_PYTHON=ON, then try again.")

    # CMake FindPython stores this when Python was selected during configuration.
    # Using the wrong env may load an incompatible extension or DLLs.
    configured_python = (cache.get("Python_EXECUTABLE") or
                         cache.get("Python3_EXECUTABLE"))
    if configured_python and normalized_executable(configured_python) != normalized_executable(sys.executable):
        fail("The build was configured for a different Python:\n"
             f"  CMake:   {configured_python}\n"
             f"  Current: {sys.executable}\n"
             "Reconfigure this build with -DPython_EXECUTABLE=<this environment's Python>. "
             "Do not just copy a different environment's extension.")

    if not no_build and not shutil.which("cmake"):
        fail("CMake is not on PATH. Install CMake, or use --no-build to install "
             "a previously compiled extension.")


def choose_module(build_dir, cache, explicit_path):
    valid_names = {MODULE + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES}
    preferred_name = MODULE + (sysconfig.get_config_var("EXT_SUFFIX") or "")

    if explicit_path is not None:
        candidate = explicit_path.expanduser().absolute()
        if not candidate.is_file():
            fail(f"Extension file does not exist: {candidate}")
        if candidate.name not in valid_names:
            fail(f"'{candidate.name}' is not a compatible extension filename for "
                 f"Python {sys.version_info.major}.{sys.version_info.minor}.\n"
                 f"Expected one of: {', '.join(sorted(valid_names))}")
        return candidate

    cmake_source = cache.get("CMAKE_HOME_DIRECTORY")
    if not cmake_source:
        fail("CMAKE_HOME_DIRECTORY is missing from CMakeCache.txt. "
             "Use --module with the full path to the compiled extension.")

    # SuperNova's core/CMakeLists.txt explicitly sets the Python module's output
    # directory to the parent of its CMake source directory (repository root).
    repository = Path(cmake_source).expanduser().absolute().parent
    matches = ([p for p in repository.iterdir() if p.is_file() and p.name in valid_names]
               if repository.is_dir() else [])

    if not matches:
        # Fallback for a user-customized build or CMake output directory.
        matches = [p for p in build_dir.rglob(MODULE + "*")
                   if p.is_file() and p.name in valid_names]

    if not matches:
        fail(f"No {MODULE} extension matching this Python ABI was found.\n"
             f"Searched: {repository} and {build_dir}\n"
             f"Expected names: {', '.join(sorted(valid_names))}\n"
             "Build with SUPERNOVA_BUILD_PYTHON=ON, or use --module PATH.")

    # Prefer the fully ABI-tagged extension, and do not guess between ambiguous
    # outputs such as multiple configuration directories.
    matches.sort(key=lambda p: (p.name != preferred_name, str(p)))
    if len(matches) > 1 and (matches[0].name != preferred_name or
                             matches[1].name == preferred_name):
        fail("Multiple eligible extensions found; choose one using --module:\n  "
             + "\n  ".join(str(p) for p in matches))
    return matches[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_site_packages():
    destination = Path(sysconfig.get_path("platlib")).absolute()
    prefix = Path(sys.prefix).absolute()
    try:
        within_environment = os.path.commonpath((str(destination), str(prefix))) == str(prefix)
    except ValueError:
        within_environment = False
    if not within_environment:
        fail(f"Refusing to install outside the active interpreter prefix:\n"
             f"  Interpreter: {sys.executable}\n"
             f"  Prefix:      {prefix}\n"
             f"  Destination: {destination}")
    if not destination.is_dir():
        fail(f"This Python environment's site-packages directory does not exist: {destination}")
    return destination


def install_if_changed(source, destination):
    target = destination / source.name
    if target.is_file() and sha256(source) == sha256(target):
        print(f"[UNCHANGED] Already up to date: {target}")
        return target

    # Copy into the destination directory first, then replace atomically.
    # This avoids leaving a partially written extension after an interruption.
    fd, temporary_name = tempfile.mkstemp(prefix=".supernova-", suffix=source.suffix,
                                          dir=destination)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if sha256(source) != sha256(temporary):
            fail("The extension changed while it was being copied. Rebuild and try again.")
        os.replace(temporary, target)
    except OSError as error:
        fail(f"Could not replace {target}: {error}\n"
             "On Windows, close any Python processes that have SuperNovaBind loaded "
             "before updating a .pyd file.")
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[INSTALLED] {target}")
    return target


def windows_runtime_dlls(build_dir, cache):
    """Find the same AdaptiveCpp DLLs the Windows build deploys to build_dir.

    The Windows builder copies DLLs from AdaptiveCpp's bin and bin/hipSYCL
    directories into build_dir. Prefer that staged result when present; if
    someone built through CMake directly, fall back to the original directories.
    Never include unrelated DLLs when we can identify the AdaptiveCpp install.
    """
    if os.name != "nt" or cache.get("SUPERNOVA_ENABLE_ADAPTIVECPP", "").upper() not in (
            "ON", "1", "YES", "TRUE"):
        return []

    dlls = {}
    acpp_dir = cache.get("AdaptiveCpp_DIR", "")
    prefix = None
    if acpp_dir:
        path = Path(acpp_dir)
        # Usually <install>/lib/cmake/AdaptiveCpp.
        for candidate in (path, *path.parents):
            if (candidate / "bin").is_dir():
                prefix = candidate
                break

    source_dirs = []
    if prefix is not None:
        source_dirs += [prefix / "bin", prefix / "bin" / "hipSYCL"]
    # The Windows build script stages the runtime here. Prefer its exact
    # output for duplicate filenames, but use source metadata to avoid copying
    # unrelated application DLLs when the AdaptiveCpp prefix was found.
    source_names = set()
    for folder in source_dirs:
        if folder.is_dir():
            for path in folder.glob("*.dll"):
                source_names.add(path.name.lower())
                dlls[path.name.lower()] = path

    if build_dir.is_dir():
        for path in build_dir.glob("*.dll"):
            if prefix is None or path.name.lower() in source_names:
                dlls[path.name.lower()] = path

    return [dlls[key] for key in sorted(dlls)]


def install_windows_runtime(build_dir, cache, destination):
    """Install native DLLs next to the extension, tracking files we own.

    Avoid silently overwriting DLLs owned by another installed Python package.
    """
    sources = windows_runtime_dlls(build_dir, cache)
    if not sources and cache.get("SUPERNOVA_ENABLE_ADAPTIVECPP", "").upper() in (
            "ON", "1", "YES", "TRUE"):
        print("[WARN] No AdaptiveCpp runtime DLLs found. The extension may fail "
              "to import unless all native dependencies are installed elsewhere.",
              flush=True)
        return

    manifest_path = destination / ".SuperNovaBind-runtime.json"
    if manifest_path.is_file():
        try:
            tracked = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(tracked, dict):
                fail(f"Malformed SuperNova DLL ownership manifest: {manifest_path}")
        except (ValueError, OSError) as error:
            fail(f"Cannot read SuperNova DLL ownership manifest: {error}")
    else:
        tracked = {}

    # Validate every collision before copying anything.
    for source in sources:
        name = source.name.lower()
        target = destination / source.name
        if target.is_file() and sha256(source) != sha256(target):
            if name not in tracked or sha256(target) != tracked[name]:
                fail(f"Refusing to replace an untracked native DLL: {target}\n"
                     "Another package may own it. Install SuperNova into a fresh "
                     "Conda/venv environment rather than overwriting it.")

    updated = dict(tracked)
    for source in sources:
        install_if_changed(source, destination)
        updated[source.name.lower()] = sha256(source)

    # Atomic manifest replacement keeps repeat runs safe.
    fd, temporary_name = tempfile.mkstemp(prefix=".supernova-manifest-", suffix=".json",
                                          dir=destination)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    if sources:
        print(f"[RUNTIME] Checked {len(sources)} AdaptiveCpp DLL(s).", flush=True)


def verify_import(target):
    # -I excludes the current working directory and PYTHONPATH, making sure
    # imports really resolve from this environment rather than the repo root.
    check = (
        "import importlib, pathlib, sys; "
        f"m = importlib.import_module('{MODULE}'); "
        "actual = pathlib.Path(m.__file__).resolve(); "
        "expected = pathlib.Path(sys.argv[1]).resolve(); "
        "assert actual == expected, f'Loaded {actual}, expected {expected}'; "
        "print('[VERIFIED] import SuperNovaBind ->', actual)"
    )
    try:
        subprocess.run([sys.executable, "-I", "-c", check, str(target)], check=True)
    except subprocess.CalledProcessError as error:
        fail("The extension was copied, but import verification failed. This may "
             "indicate an ABI mismatch, a missing transitive DLL, or an unavailable "
             "native runtime (for example AdaptiveCpp, LLVM, OpenMP, or the "
             "Microsoft C++ runtime). On Windows, inspect the .pyd using "
             "dumpbin /DEPENDENTS or llvm-objdump -p to identify its imports.\n"
             f"Installed path: {target}\n"
             f"Verification exit code: {error.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build_dir", type=Path, help="Configured SuperNova CMake build directory")
    parser.add_argument("--no-build", action="store_true", help="Skip incremental CMake build")
    parser.add_argument("--config", default="Release", help="CMake build configuration (default: Release)")
    parser.add_argument("--module", type=Path, help="Path to an already-built extension (optional)")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-install import test")
    args = parser.parse_args()

    if sys.implementation.name != "cpython":
        fail("SuperNova's pybind11 extension currently requires CPython.")

    build_dir = args.build_dir.expanduser().absolute()
    cache = cmake_cache(build_dir)
    validate_cache(cache, args.no_build)
    destination = environment_site_packages()
    print(f"Python:        {sys.executable}")
    print(f"Build dir:     {build_dir}")
    print(f"Environment:   {sys.prefix}")
    print(f"Install dir:   {destination}", flush=True)

    if not args.no_build:
        command = ["cmake", "--build", str(build_dir), "--target", MODULE,
                   "--config", args.config]
        print("[BUILD] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    source = choose_module(build_dir, cache, args.module)
    print(f"Extension:     {source}", flush=True)
    if os.name == "nt":
        install_windows_runtime(build_dir, cache, destination)
    target = install_if_changed(source, destination)
    if not args.no_verify:
        verify_import(target)
    print("[OK] SuperNovaBind is available to this Python environment.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
