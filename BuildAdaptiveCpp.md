```bash
cmake \
  -S external/AdaptiveCpp \
  -B external/AdaptiveCpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang \
  -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
  -DCMAKE_INSTALL_PREFIX=/mnt/c/PAPER/LCIM-BitTheorium/external/AdaptiveCpp/install

cmake --build external/AdaptiveCpp/build \
  --target install \
  --parallel 8

cmake --install external/AdaptiveCpp/build
```

WINDOWS:

```powershell
git clone `
  --depth 1 `
  --branch llvmorg-20.1.8 `
  https://github.com/llvm/llvm-project.git `
  external/llvm-project

$Root = (git rev-parse --show-toplevel).Trim().Replace('\', '/')

cmake `
  -S "$Root/external/llvm-project/llvm" `
  -B "$Root/external/llvm-project/build-bootstrap" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=cl `
  -DCMAKE_CXX_COMPILER=cl `
  -DCMAKE_INSTALL_PREFIX="$Root/external/llvm-bootstrap" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_ENABLE_PROJECTS="clang;lld" `
  -DLLVM_PARALLEL_LINK_JOBS=2 `
  -DLLVM_BUILD_LLVM_DYLIB=OFF `
  -DLLVM_LINK_LLVM_DYLIB=OFF

cmake --build "$Root/external/llvm-project/build-bootstrap" `
  --target install `
  --parallel 8


$BootstrapClang = "$Root/external/llvm-bootstrap/bin/clang-cl.exe"

cmake `
  -S "$Root/external/llvm-project/llvm" `
  -B "$Root/external/llvm-project/build-acpp" `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$BootstrapClang" `
  -DCMAKE_CXX_COMPILER="$BootstrapClang" `
  -DCMAKE_MT=mt `
  -DCMAKE_INSTALL_PREFIX="$Root/external/AdaptiveCpp/install-windows" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_ENABLE_PROJECTS="clang;openmp;lld" `
  -DLLVM_PARALLEL_LINK_JOBS=2 `
  -DLLVM_BUILD_LLVM_DYLIB=OFF `
  -DLLVM_LINK_LLVM_DYLIB=OFF `
  -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp `
  -DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR="$Root/external/AdaptiveCpp" `
  -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
  -DACPP_HOST_FORCE_MCPU_TARGET="arrowlake-s"
```
