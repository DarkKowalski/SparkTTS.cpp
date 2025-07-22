# SparkTTS with OpenVINO

## How to build

### Setup OpenVINO GenAI

[Window](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_GENAI&VERSION=v_2025_2_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)

[macOS](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_GENAI&VERSION=v_2025_2_0&OP_SYSTEM=MACOS&DISTRIBUTION=ARCHIVE)

### Setup vcpkg

[vcpkg](https://github.com/microsoft/vcpkg)


### Build with CMake and Ninja

```bash
cmake --preset=vcpkg
cmake --build --config Release
```
