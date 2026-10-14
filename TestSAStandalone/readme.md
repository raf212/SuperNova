
### Run From Root
---

### LINUX:
```bash
ROOT="$(pwd)"
SN_ROOT="$ROOT"
ACPP_ROOT="$SN_ROOT/external/AdaptiveCpp/install"
TEST_BUILD="$ROOT/TestSAStandalone/build-acpp-linux"

rm -rf "$TEST_BUILD"

cmake \
  -S "$ROOT/TestSAStandalone" \
  -B "$TEST_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DSUPERNOVA_ROOT="$SN_ROOT" \
  -DSUPERNOVA_BUILD_DIR="$SN_ROOT/build-acpp-linux" \
  -DAdaptiveCpp_DIR="$ACPP_ROOT/lib/cmake/AdaptiveCpp" \
  -DACPP_TARGETS=generic

cmake --build "$TEST_BUILD" \
  --target TestSA \
  --parallel "$(nproc)" \
  --verbose

export LD_LIBRARY_PATH="$ACPP_ROOT/lib:$ACPP_ROOT/lib/hipSYCL:${LD_LIBRARY_PATH:-}"

"$TEST_BUILD/TestSA"
```

### Windows

```powershell
cd C:\PAPER\ExternalTestSuperNova

$Root = (Get-Location).Path.Replace('\', '/')
$SnRoot = "SuperNova"
$AcppRoot = "$SnRoot/external/AdaptiveCpp/install-windows"
$TestBuild = "$Root/TestSAStandalone/build-acpp-windows"
$Clang = "$AcppRoot/bin/clang++.exe"

$env:Path = "$AcppRoot/bin;$AcppRoot/bin/hipSYCL;$env:Path"

# Import Visual Studio x64 environment.
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

$VsInstall = & $VsWhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

$VsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"

cmd /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" |
    ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }

# Find Microsoft manifest tool.
$SdkBinRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"

$Mt = Get-ChildItem `
    -Path "$SdkBinRoot\*\x64\mt.exe" `
    -ErrorAction Stop |
    Sort-Object {
        [version]$_.Directory.Parent.Name
    } -Descending |
    Select-Object -First 1 -ExpandProperty FullName

$Mt = $Mt.Replace('\', '/')

# Clean standalone test build only.
Remove-Item `
    -Recurse `
    -Force `
    $TestBuild `
    -ErrorAction SilentlyContinue

# Configure.
cmake `
    -S "$Root/TestSAStandalone" `
    -B "$TestBuild" `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER="$Clang" `
    -DCMAKE_MT="$Mt" `
    -DSUPERNOVA_ROOT="$SnRoot" `
    -DSUPERNOVA_BUILD_DIR="$SnRoot/build-acpp-windows" `
    -DAdaptiveCpp_DIR="$AcppRoot/lib/cmake/AdaptiveCpp" `
    -DACPP_TARGETS=generic

# Build.
cmake --build "$TestBuild" `
    --target TestSA `
    --parallel 20 `
    --verbose

# Run.
& "$TestBuild/TestSA.exe"
```
