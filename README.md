# SparkTTS inference with C++
Windows:
 - ONNX Runtime (DirectML backend) for BiCodec/Wav2Vec etc.
 - llama.cpp (Vulkan backend) for Qwen2.5-0.5B

macOS:
 - CoreML for BiCodec/Wav2Vec etc.
 - llama.cpp (Metal backend) for Qwen2.5-0.5B

## How to build

### Install Rust

[Rust](https://www.rust-lang.org/tools/install)

### Setup vcpkg

[vcpkg](https://github.com/microsoft/vcpkg)

### Build llama.cpp

#### Windows

1. Make sure you are using `x64 Native Tools Command Prompt for VS 2022`

2. Setup Vulkan dependencies, [llama.cpp build doc](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)

3. Build and install with CMake

```batch
cd third_party\llama.cpp
cmake -B build -G Ninja -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DCMAKE_INSTALL_PREFIX=..\..\lib\llama
cmake --build build --config Release
cmake --install build --config Release
cd ..\..
```

#### macOS (Apple Silicon)

```bash
pushd third_party/llama.cpp
cmake -B build -G Ninja -DLLAMA_CURL=OFF -DCMAKE_INSTALL_PREFIX=../../lib/llama
cmake --build build --config Release
cmake --install build --config Release
popd
```

### Build ONNX Runtime (Windows only with DirectML)

```batch
cd third_party\onnxruntime
python .\tools\ci_build/build.py ^
    --update ^
    --build ^
    --config Release ^
    --build_shared_lib ^
    --parallel ^
    --build_dir ./build ^
    --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5" ^
    --skip_tests ^
    --enable_lto ^
    --use_dml
cmake --install build\Release --config Release --prefix ..\..\lib\onnxruntime
cd ..\..
```

### Build with CMake and Ninja

#### Windows

```batch
cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release && copy /Y build\src\*.dll install\tools\bin
```

#### macOS

```bash
cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release
```
