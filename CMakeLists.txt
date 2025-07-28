cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Llama)
include(OpenVINO)

project(spark_tts
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Third-party dependencies
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenVINOGenAI REQUIRED)
find_package(SndFile CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_llama()

add_subdirectory(src)
