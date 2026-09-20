RELESE BUILD: 

WINDOWS :
```cpp

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

LINUX :

```cpp
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