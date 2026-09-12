RELESE BUILD: 

WINDOWS :
```cpp

Remove-Item -Recurse -Force .\build-release -ErrorAction SilentlyContinue

cmake -S .\core -B .\build-release `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DSUPERNOVA_ENABLE_IPO=ON `
    -DSUPERNOVA_NATIVE_CPU=OFF `
    -DPython_EXECUTABLE="$((Get-Command python).Source)" `
    -DSUPERNOVA_FAST_FP=OFF 
    
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
    -DSUPERNOVA_ENABLE_IPO=ON \
    -DSUPERNOVA_NATIVE_CPU=ON \
    -DPython_EXECUTABLE="$(command -v python3)" \
    -DSUPERNOVA_FAST_FP=OFF

cmake --build ./build-release \
    --target SuperNova SuperNovaBind \
    --parallel \
    --verbose

./build-release/SuperNova
```