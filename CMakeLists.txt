cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Llama)

project(spark_tts
    VERSION 0.1.0
    LANGUAGES C CXX
)

option(ENABLE_PERFETTO "Enable Perfetto tracing" OFF)
option(ENABLE_FEATURE_EXTRACTOR "Enable feature extractor" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
    enable_language(OBJC OBJCXX)
endif()

# Third-party dependencies
if(WIN32)
    include(ONNXRuntime)
    find_onnxruntime()
endif()

if(ENABLE_PERFETTO)
    message(STATUS "Perfetto tracing is enabled.")
    find_package(Threads)
    find_package(unofficial-perfetto CONFIG REQUIRED)
endif()

find_package(SndFile CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_llama()

set(TOKENIZERS_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp")
add_subdirectory(${TOKENIZERS_CPP_DIR} tokenizers EXCLUDE_FROM_ALL)
set_target_properties(tokenizers_cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TOKENIZERS_CPP_DIR}/include")

add_subdirectory(src)
