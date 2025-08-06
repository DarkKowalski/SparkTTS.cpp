# SparkTTS with OpenVINO/llama.cpp

## How to build

### Install Rust

[Install Rust](https://www.rust-lang.org/tools/install)

### Setup OpenVINO GenAI

[Window](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_2_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)

[macOS](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_2_0&OP_SYSTEM=MACOS&DISTRIBUTION=ARCHIVE)

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

### Build ONNX Runtime (macOS only)

```bash
pushd third_party/onnxruntime
python ./tools/ci_build/build.py \
    --update \
    --build \
    --config Release \
    --build_shared_lib \
    --parallel \
    --build_dir ./build \
    --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5" \
    --skip_tests \
    --enable_lto
cmake --install build/Release --config Release --prefix ../../lib/onnxruntime
popd
```

### Build with CMake and Ninja

#### Windows

```batch
cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release && copy /Y build\src\*.dll install\bin
```

#### macOS

```bash
cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release
```
