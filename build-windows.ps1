<#
.SYNOPSIS
  Verify the existing SuperNova LLVM + AdaptiveCpp toolchain and build SuperNova.

.DESCRIPTION
  NORMAL MODE:
    - NEVER rebuilds LLVM or AdaptiveCpp.
    - Verifies the existing integrated LLVM + AdaptiveCpp installation.
    - Builds only SuperNova.

  EXPLICIT TOOLCHAIN REBUILD MODE:
    .\build-windows.ps1 -accp_rebuild

    - Rebuilds the LLVM bootstrap compiler.
    - Rebuilds/reinstalls integrated LLVM + AdaptiveCpp.
    - Builds SuperNova afterward.

  Compatibility aliases:
    -RebuildAdaptiveCpp
    -RebuildAcpp

  GPU vendor drivers/SDKs are intentionally NOT installed automatically.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -accp_rebuild -Jobs 4

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -BuildPython
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 128)]
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount),

    [switch]$CleanSuperNova,

    [Alias("RebuildAdaptiveCpp", "RebuildAcpp")]
    [switch]$accp_rebuild,

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

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
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

function Ensure-LlvmCheckout(
    [string]$LlvmSourceRoot,
    [string]$LlvmTag
) {
    if (-not (Test-Path "$LlvmSourceRoot/.git")) {
        Write-Step "Cloning LLVM $LlvmTag"
        & git clone `
            --depth 1 `
            --branch $LlvmTag `
            https://github.com/llvm/llvm-project.git `
            $LlvmSourceRoot

        if ($LASTEXITCODE -ne 0) {
            throw "LLVM clone failed."
        }
        return
    }

    Write-Step "Verifying LLVM source checkout"

    $dirty = (& git -C $LlvmSourceRoot status --porcelain)
    if ($dirty) {
        throw "LLVM source checkout '$LlvmSourceRoot' has local changes. Clean/stash them before -accp_rebuild."
    }

    & git -C $LlvmSourceRoot fetch `
        --depth 1 `
        origin `
        "refs/tags/${LlvmTag}:refs/tags/${LlvmTag}"

    if ($LASTEXITCODE -ne 0) {
        throw "Could not fetch LLVM tag '$LlvmTag'."
    }

    $desired = (& git -C $LlvmSourceRoot rev-list -n 1 $LlvmTag).Trim()
    $current = (& git -C $LlvmSourceRoot rev-parse HEAD).Trim()

    if ($current -ne $desired) {
        & git -C $LlvmSourceRoot checkout --detach $LlvmTag
        if ($LASTEXITCODE -ne 0) {
            throw "Could not checkout LLVM tag '$LlvmTag'."
        }
    }
}

function Verify-InstalledToolchain(
    [string]$AdaptiveCppInstall
) {
    $clang = "$AdaptiveCppInstall/bin/clang++.exe"
    $llvmConfig = "$AdaptiveCppInstall/bin/llvm-config.exe"
    $acppInfo = "$AdaptiveCppInstall/bin/acpp-info.exe"

    # CMake accepts either <Package>Config.cmake or the lowercase
    # <package>-config.cmake spelling. Current AdaptiveCpp installs the latter:
    #   lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake
    $acppCmakeDir = "$AdaptiveCppInstall/lib/cmake/AdaptiveCpp"
    $acppCmakeCandidates = @(
        "$acppCmakeDir/adaptivecpp-config.cmake",
        "$acppCmakeDir/AdaptiveCppConfig.cmake"
    )

    $acppCmake = $acppCmakeCandidates |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1

    $required = @(
        $clang,
        $llvmConfig,
        $acppInfo
    )

    foreach ($path in $required) {
        if (-not (Test-Path $path)) {
            throw @"
Existing LLVM + AdaptiveCpp installation is incomplete.
Missing:
  $path

Normal mode never rebuilds the toolchain.
Run:
  .\build-windows.ps1 -accp_rebuild
"@
        }
    }

    if (-not $acppCmake) {
        $expected = $acppCmakeCandidates -join "`n  "
        throw @"
Existing LLVM + AdaptiveCpp installation is incomplete.
No AdaptiveCpp CMake package was found. Checked:
  $expected

Normal mode never rebuilds the toolchain.
Run:
  .\build-windows.ps1 -accp_rebuild
"@
    }

    Write-Step "Verifying installed LLVM"

    # IMPORTANT:
    # Native-command stdout written directly inside a PowerShell function becomes
    # part of that function's return stream. Capture it first, then print it via
    # Write-Host so this function returns exactly one structured object.
    $clangOutput = & $clang --version 2>&1
    $clangExit = $LASTEXITCODE
    foreach ($line in $clangOutput) {
        Write-Host $line
    }
    if ($clangExit -ne 0) {
        throw "Installed clang++.exe failed verification."
    }

    $llvmConfigOutput = & $llvmConfig --version 2>&1
    $llvmConfigExit = $LASTEXITCODE
    foreach ($line in $llvmConfigOutput) {
        Write-Host $line
    }
    if ($llvmConfigExit -ne 0) {
        throw "Installed llvm-config.exe failed verification."
    }

    Write-Step "Verifying installed AdaptiveCpp"

    $acppInfoOutput = & $acppInfo -l 2>&1
    $acppInfoExit = $LASTEXITCODE
    foreach ($line in $acppInfoOutput) {
        Write-Host $line
    }
    if ($acppInfoExit -ne 0) {
        throw "Installed acpp-info.exe failed verification."
    }

    Write-Ok "Existing LLVM + AdaptiveCpp installation is valid."

    return [pscustomobject]@{
        Clang      = $clang
        LlvmConfig = $llvmConfig
        AcppInfo   = $acppInfo
        AcppCmake  = $acppCmake
    }
}

function Remove-DirectoryIfPresent([string]$Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
}

Write-Step "Locating SuperNova repository"

Require-Command git | Out-Null

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
Write-Host "AdaptiveCpp install: $AdaptiveCppInstall"
Write-Host "Jobs               : $Jobs"
Write-Host "Toolchain rebuild  : $accp_rebuild"

Write-Step "Checking build tools"
Require-Command cmake | Out-Null
Require-Command ninja | Out-Null

# SuperNova still targets the Windows/MSVC ABI, so keep the MSVC/SDK development
# environment available even though downstream AdaptiveCpp compilation uses
# clang++.exe's GNU-style frontend.
Import-VsDevEnvironment
Require-Command cl.exe | Out-Null
Require-Command link.exe | Out-Null

$Mt = Find-WindowsMt
Write-Host "Windows mt.exe     : $Mt"

if ($accp_rebuild) {
    Write-Step "Explicit toolchain rebuild requested (-accp_rebuild)"

    Write-Step "Initializing AdaptiveCpp submodule"
    & git -C $Root submodule sync --recursive
    if ($LASTEXITCODE -ne 0) {
        throw "git submodule sync failed."
    }

    & git -C $Root submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        throw "git submodule update failed."
    }

    if (-not (Test-Path "$AdaptiveCppSource/CMakeLists.txt")) {
        throw "AdaptiveCpp submodule is missing after initialization."
    }

    $AcppCommit = (& git -C $AdaptiveCppSource rev-parse HEAD).Trim()
    Write-Host "AdaptiveCpp commit : $AcppCommit"

    Ensure-LlvmCheckout `
        -LlvmSourceRoot $LlvmSourceRoot `
        -LlvmTag $LlvmTag

    Write-Step "Cleaning previous LLVM + AdaptiveCpp build/install state"
    Remove-DirectoryIfPresent $LlvmBootstrapBuild
    Remove-DirectoryIfPresent $LlvmAdaptiveCppBuild
    Remove-DirectoryIfPresent $LlvmBootstrapInstall
    Remove-DirectoryIfPresent $AdaptiveCppInstall
    Remove-DirectoryIfPresent $SuperNovaBuild

    Write-Step "Building LLVM bootstrap compiler"

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

    if ($LASTEXITCODE -ne 0) {
        throw "LLVM bootstrap configure failed."
    }

    & cmake --build $LlvmBootstrapBuild --target install --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "LLVM bootstrap build/install failed."
    }

    $BootstrapClang = "$LlvmBootstrapInstall/bin/clang-cl.exe"
    if (-not (Test-Path $BootstrapClang)) {
        throw "Bootstrap clang-cl.exe was not produced."
    }

    & $BootstrapClang --version
    if ($LASTEXITCODE -ne 0) {
        throw "Bootstrap clang-cl.exe failed verification."
    }

    Ensure-AdaptiveCppWindowsLinkFix $AdaptiveCppSource

    Write-Step "Building integrated LLVM + AdaptiveCpp"

    $llvmTargets = if ($CpuOnly) {
        "X86"
    }
    else {
        "X86;NVPTX;AMDGPU"
    }

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

    if ($LASTEXITCODE -ne 0) {
        throw "Integrated LLVM + AdaptiveCpp configure failed."
    }

    & cmake --build $LlvmAdaptiveCppBuild --target install --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "Integrated LLVM + AdaptiveCpp build/install failed."
    }
}
else {
    Write-Step "Toolchain reuse policy"
    Write-Host "Normal mode: LLVM/AdaptiveCpp rebuild is disabled."
    Write-Host "Only SuperNova will be configured/built after verification."
}

$Toolchain = Verify-InstalledToolchain $AdaptiveCppInstall

$SuperNovaClang = $Toolchain.Clang
$AcppInfo = $Toolchain.AcppInfo

$env:Path = "$AdaptiveCppInstall/bin;$AdaptiveCppInstall/bin/hipSYCL;$env:Path"

Write-Host ""
Write-Host "GPU note:" -ForegroundColor Yellow
Write-Host "  CUDA / ROCm-HIP / Intel OpenCL devices appear only when their vendor"
Write-Host "  driver/runtime was installed before AdaptiveCpp was configured."

if ($CleanSuperNova -and (Test-Path $SuperNovaBuild)) {
    Write-Step "Cleaning SuperNova AdaptiveCpp build directory"
    Remove-Item -Recurse -Force $SuperNovaBuild
}

# If an old cache used clang-cl.exe, CMake must not reuse it now that the
# AdaptiveCpp downstream frontend is clang++.exe.
$cacheFile = "$SuperNovaBuild/CMakeCache.txt"
if (Test-Path $cacheFile) {
    $compilerLine = Select-String `
        -Path $cacheFile `
        -Pattern '^CMAKE_CXX_COMPILER:FILEPATH=' `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($compilerLine) {
        $cachedCompiler = ($compilerLine.Line -split '=', 2)[1].Replace('\', '/')
        $desiredCompiler = $SuperNovaClang.Replace('\', '/')

        if ($cachedCompiler.ToLowerInvariant() -ne $desiredCompiler.ToLowerInvariant()) {
            Write-Step "Compiler changed; refreshing SuperNova build directory"
            Write-Host "Cached : $cachedCompiler"
            Write-Host "Wanted : $desiredCompiler"
            Remove-Item -Recurse -Force $SuperNovaBuild
        }
    }
}

Write-Step "Configuring SuperNova with verified AdaptiveCpp"

$pythonFlag = if ($BuildPython) { "ON" } else { "OFF" }
$fetchPybind = if ($BuildPython) { "ON" } else { "OFF" }

$configureArgs = @(
    "-S", "$Root/core",
    "-B", $SuperNovaBuild,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_COMPILER=$SuperNovaClang",
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
if ($LASTEXITCODE -ne 0) {
    throw "SuperNova configure failed."
}

Write-Step "Building SuperNova"

if ($BuildPython) {
    & cmake --build $SuperNovaBuild `
        --target SuperNova SuperNovaBind `
        --parallel $Jobs `
        --verbose
}
else {
    & cmake --build $SuperNovaBuild `
        --target SuperNova `
        --parallel $Jobs `
        --verbose
}

if ($LASTEXITCODE -ne 0) {
    throw "SuperNova build failed."
}

$SuperNovaExe = "$SuperNovaBuild/SuperNova.exe"
if (-not (Test-Path $SuperNovaExe)) {
    throw "SuperNova.exe was not found after a successful build."
}

Write-Step "Build complete"
Write-Host "SuperNova executable: $SuperNovaExe" -ForegroundColor Green
Write-Host ""
Write-Host "Normal mode never rebuilds LLVM/AdaptiveCpp."
Write-Host ""
Write-Host "Force toolchain rebuild:"
Write-Host "  .\build-windows.ps1 -accp_rebuild"
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
