<#
.SYNOPSIS
  Bootstrap and build SuperNova + AdaptiveCpp on native Windows.

.DESCRIPTION
  Run this from a fresh SuperNova clone (or from anywhere inside the repository).
  The script:
    1. initializes git submodules;
    2. imports a Visual Studio x64 developer environment if needed;
    3. locates the Windows SDK manifest tool (mt.exe);
    4. clones/builds a pinned LLVM bootstrap compiler;
    5. builds AdaptiveCpp as an LLVM external project;
    6. verifies AdaptiveCpp devices with acpp-info;
    7. configures and builds SuperNova with ACPP_TARGETS=generic.

  GPU vendor drivers/SDKs are intentionally NOT installed automatically.
  Install CUDA / ROCm-HIP / Intel OpenCL as appropriate before running this
  script if GPU execution is required.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -RebuildAdaptiveCpp -Jobs 4

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -BuildPython
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 128)]
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount),

    [switch]$CleanSuperNova,
    [switch]$RebuildAdaptiveCpp,
    [switch]$BuildPython,
    [switch]$CpuOnly,

    [string]$LlvmTag = "llvmorg-20.1.8",
    [string]$HostCpu = "x86-64"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required command '$Name' was not found on PATH."
    }
    return $cmd
}

function Import-VsDevEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    Write-Step "Importing Visual Studio x64 C++ developer environment"

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw @"
cl.exe is not on PATH and vswhere.exe was not found.
Install Visual Studio / Build Tools with the 'Desktop development with C++' workload,
or run this script from a Developer PowerShell for Visual Studio.
"@
    }

    $installPath = (& $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath).Trim()

    if (-not $installPath) {
        throw "No Visual Studio installation with the x64 C++ toolchain was found."
    }

    $vsDevCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat was not found at '$vsDevCmd'."
    }

    $lines = & cmd.exe /s /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $lines) {
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Visual Studio environment was imported, but cl.exe is still unavailable."
    }
}

function Find-WindowsMt {
    $sdkRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (-not (Test-Path $sdkRoot)) {
        throw "Windows SDK bin directory was not found at '$sdkRoot'."
    }

    $mt = Get-ChildItem `
        -Path "$sdkRoot\*\x64\mt.exe" `
        -ErrorAction SilentlyContinue |
        Sort-Object {
            try { [version]$_.Directory.Parent.Name }
            catch { [version]"0.0" }
        } -Descending |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $mt) {
        throw "Could not locate an x64 Windows SDK mt.exe."
    }

    return $mt.Replace('\', '/')
}

function Ensure-AdaptiveCppWindowsLinkFix([string]$AdaptiveCppSource) {
    $cmakeFile = Join-Path $AdaptiveCppSource "CMakeLists.txt"
    if (-not (Test-Path $cmakeFile)) {
        throw "AdaptiveCpp CMakeLists.txt was not found at '$cmakeFile'."
    }

    $content = Get-Content -Raw -Path $cmakeFile

    if (
        $content.Contains("# SUPERNOVA_WINDOWS_LLVM_PASSES_FIX") -or
        $content.Contains("target_link_libraries(AdaptiveCpp PUBLIC LLVMPasses)")
    ) {
        return
    }

    $needle = "add_subdirectory(src)"
    $index = $content.IndexOf($needle)
    if ($index -lt 0) {
        throw "Could not locate '$needle' in AdaptiveCpp/CMakeLists.txt."
    }

    Write-Step "Applying Windows static LLVM LLVMPasses integration workaround"

    $patch = @'

# SUPERNOVA_WINDOWS_LLVM_PASSES_FIX
# Native Windows uses a static LLVM build. AdaptiveCpp's integrated compiler
# references LLVM new-pass-manager symbols, so propagate LLVMPasses to LLVM
# tools that consume AdaptiveCpp.lib.
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
'@

    $insertAt = $index + $needle.Length
    $updated = $content.Substring(0, $insertAt) + $patch + $content.Substring($insertAt)
    Set-Content -Path $cmakeFile -Value $updated -Encoding utf8 -NoNewline
}

Write-Step "Locating SuperNova repository"

$startDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Root = (& git -C $startDir rev-parse --show-toplevel).Trim().Replace('\', '/')

if (-not $Root) {
    throw "This script must be run from inside a SuperNova git checkout."
}

$AdaptiveCppSource = "$Root/external/AdaptiveCpp"
$AdaptiveCppInstall = "$AdaptiveCppSource/install-windows"
$LlvmSourceRoot = "$Root/external/llvm-project"
$LlvmBootstrapBuild = "$LlvmSourceRoot/build-bootstrap"
$LlvmAdaptiveCppBuild = "$LlvmSourceRoot/build-acpp"
$LlvmBootstrapInstall = "$Root/external/llvm-bootstrap"
$SuperNovaBuild = "$Root/build-acpp-windows"

Write-Host "Repository         : $Root"
Write-Host "AdaptiveCpp source : $AdaptiveCppSource"
Write-Host "AdaptiveCpp install: $AdaptiveCppInstall"
Write-Host "Jobs               : $Jobs"

Write-Step "Checking base tools"
Require-Command git | Out-Null
Require-Command cmake | Out-Null
Require-Command ninja | Out-Null

Import-VsDevEnvironment
Require-Command cl.exe | Out-Null
Require-Command link.exe | Out-Null

$Mt = Find-WindowsMt
Write-Host "Windows mt.exe     : $Mt"

Write-Step "Initializing git submodules"
& git -C $Root submodule sync --recursive
if ($LASTEXITCODE -ne 0) { throw "git submodule sync failed." }

& git -C $Root submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "git submodule update failed." }

if (-not (Test-Path "$AdaptiveCppSource/CMakeLists.txt")) {
    throw "AdaptiveCpp submodule is missing after initialization."
}

$AcppCommit = (& git -C $AdaptiveCppSource rev-parse HEAD).Trim()
Write-Host "AdaptiveCpp commit : $AcppCommit"

# Windows currently requires the integrated-LLVM AdaptiveCpp route for generic
# and accelerated CPU compilation.
if (-not (Test-Path "$LlvmSourceRoot/.git")) {
    Write-Step "Cloning LLVM $LlvmTag"
    & git clone `
        --depth 1 `
        --branch $LlvmTag `
        https://github.com/llvm/llvm-project.git `
        $LlvmSourceRoot

    if ($LASTEXITCODE -ne 0) { throw "LLVM clone failed." }
}
else {
    Write-Step "Reusing existing LLVM checkout"
}

$BootstrapClang = "$LlvmBootstrapInstall/bin/clang-cl.exe"

if ($RebuildAdaptiveCpp -or -not (Test-Path $BootstrapClang)) {
    Write-Step "Building LLVM bootstrap compiler"

    if ($RebuildAdaptiveCpp -and (Test-Path $LlvmBootstrapBuild)) {
        Remove-Item -Recurse -Force $LlvmBootstrapBuild
    }

    & cmake `
        -S "$LlvmSourceRoot/llvm" `
        -B $LlvmBootstrapBuild `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_C_COMPILER=cl `
        -DCMAKE_CXX_COMPILER=cl `
        -DCMAKE_MT="$Mt" `
        -DCMAKE_INSTALL_PREFIX="$LlvmBootstrapInstall" `
        -DLLVM_TARGETS_TO_BUILD="X86" `
        -DLLVM_ENABLE_PROJECTS="clang;lld" `
        -DLLVM_PARALLEL_LINK_JOBS=2 `
        -DLLVM_BUILD_LLVM_DYLIB=OFF `
        -DLLVM_LINK_LLVM_DYLIB=OFF

    if ($LASTEXITCODE -ne 0) { throw "LLVM bootstrap configure failed." }

    & cmake --build $LlvmBootstrapBuild --target install --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "LLVM bootstrap build/install failed." }
}

if (-not (Test-Path $BootstrapClang)) {
    throw "Bootstrap clang-cl.exe was not produced."
}

Write-Step "Bootstrap compiler"
& $BootstrapClang --version

Ensure-AdaptiveCppWindowsLinkFix $AdaptiveCppSource

$AcppInfo = "$AdaptiveCppInstall/bin/acpp-info.exe"

if ($RebuildAdaptiveCpp -or -not (Test-Path $AcppInfo)) {
    Write-Step "Building integrated LLVM + AdaptiveCpp"

    if ($RebuildAdaptiveCpp -and (Test-Path $LlvmAdaptiveCppBuild)) {
        Remove-Item -Recurse -Force $LlvmAdaptiveCppBuild
    }

    $llvmTargets = if ($CpuOnly) { "X86" } else { "X86;NVPTX;AMDGPU" }

    & cmake `
        -S "$LlvmSourceRoot/llvm" `
        -B $LlvmAdaptiveCppBuild `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_C_COMPILER="$BootstrapClang" `
        -DCMAKE_CXX_COMPILER="$BootstrapClang" `
        -DCMAKE_MT="$Mt" `
        -DCMAKE_INSTALL_PREFIX="$AdaptiveCppInstall" `
        -DLLVM_TARGETS_TO_BUILD="$llvmTargets" `
        -DLLVM_ENABLE_PROJECTS="clang;openmp;lld" `
        -DLLVM_PARALLEL_LINK_JOBS=2 `
        -DLLVM_BUILD_LLVM_DYLIB=OFF `
        -DLLVM_LINK_LLVM_DYLIB=OFF `
        -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp `
        -DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR="$AdaptiveCppSource" `
        -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
        -DACPP_COMPILER_FEATURE_PROFILE=full `
        -DACPP_HOST_FORCE_MCPU_TARGET="$HostCpu"

    if ($LASTEXITCODE -ne 0) { throw "Integrated LLVM + AdaptiveCpp configure failed." }

    & cmake --build $LlvmAdaptiveCppBuild --target install --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "Integrated LLVM + AdaptiveCpp build/install failed." }
}

if (-not (Test-Path $AcppInfo)) {
    throw "AdaptiveCpp installation did not produce acpp-info.exe."
}

$env:Path = "$AdaptiveCppInstall/bin;$AdaptiveCppInstall/bin/hipSYCL;$env:Path"

Write-Step "AdaptiveCpp devices/backends"
& $AcppInfo -l
if ($LASTEXITCODE -ne 0) {
    throw "acpp-info failed."
}

Write-Host ""
Write-Host "GPU note:" -ForegroundColor Yellow
Write-Host "  CUDA / ROCm-HIP / Intel OpenCL devices appear only when their vendor"
Write-Host "  driver/runtime was installed before AdaptiveCpp was configured."

if ($CleanSuperNova -and (Test-Path $SuperNovaBuild)) {
    Write-Step "Cleaning SuperNova AdaptiveCpp build directory"
    Remove-Item -Recurse -Force $SuperNovaBuild
}

Write-Step "Configuring SuperNova with AdaptiveCpp"

$pythonFlag = if ($BuildPython) { "ON" } else { "OFF" }
$fetchPybind = if ($BuildPython) { "ON" } else { "OFF" }

$configureArgs = @(
    "-S", "$Root/core",
    "-B", $SuperNovaBuild,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_COMPILER=$AdaptiveCppInstall/bin/clang-cl.exe",
    "-DCMAKE_MT=$Mt",
    "-DAdaptiveCpp_DIR=$AdaptiveCppInstall/lib/cmake/AdaptiveCpp",
    "-DACPP_TARGETS=generic",
    "-DSUPERNOVA_ENABLE_ADAPTIVECPP=ON",
    "-DSUPERNOVA_BUILD_CLI=ON",
    "-DSUPERNOVA_BUILD_PYTHON=$pythonFlag",
    "-DSUPERNOVA_FETCH_PYBIND11=$fetchPybind",
    "-DSUPERNOVA_ENABLE_IPO=OFF",
    "-DSUPERNOVA_NATIVE_CPU=OFF",
    "-DSUPERNOVA_MSVC_AVX2=OFF",
    "-DSUPERNOVA_FAST_FP=OFF"
)

if ($BuildPython) {
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python bindings were requested but python.exe was not found."
    }
    $configureArgs += "-DPython_EXECUTABLE=$($python.Source)"
}

& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "SuperNova configure failed." }

Write-Step "Building SuperNova"

if ($BuildPython) {
    & cmake --build $SuperNovaBuild --target SuperNova SuperNovaBind --parallel $Jobs --verbose
}
else {
    & cmake --build $SuperNovaBuild --target SuperNova --parallel $Jobs --verbose
}

if ($LASTEXITCODE -ne 0) { throw "SuperNova build failed." }

$SuperNovaExe = "$SuperNovaBuild/SuperNova.exe"
if (-not (Test-Path $SuperNovaExe)) {
    throw "SuperNova.exe was not found after a successful build."
}

Write-Step "Build complete"
Write-Host "SuperNova executable: $SuperNovaExe" -ForegroundColor Green
Write-Host ""
Write-Host "Run:"
Write-Host "  & `"$SuperNovaExe`""
Write-Host ""
Write-Host "List devices:"
Write-Host "  & `"$AcppInfo`" -l"
Write-Host ""
Write-Host "CPU-only runtime example:"
Write-Host '  $env:ACPP_VISIBILITY_MASK = "omp"'
Write-Host "  & `"$SuperNovaExe`""
