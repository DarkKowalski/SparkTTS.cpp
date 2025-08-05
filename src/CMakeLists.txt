if (WIN32)
    add_library(tts_impl STATIC
        win/audio_tokenizer.cpp
        win/audio_detokenizer.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
    )

    target_link_libraries(tts_impl PRIVATE 
        openvino::runtime
        openvino::genai
        llama
    )
elseif(APPLE)
   add_library(tts_impl STATIC
        mac/audio_tokenizer.cpp
        mac/audio_detokenizer.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
    )

    target_link_libraries(tts_impl PRIVATE 
        openvino::runtime
        openvino::genai
        llama
        onnxruntime
    )

    # when using CoreML
    # target_link_libraries(tts_impl PRIVATE
    #     "-framework CoreML"
    #     "-framework Accelerate"
    #     "-framework Metal"
    # )
endif()

add_library(tts_api SHARED
    api.cpp
)
target_compile_definitions(tts_api PRIVATE
    -DSPARK_TTS_SHARED
    -DSPARK_TTS_BUILD
)
target_link_libraries(tts_api PRIVATE 
    tts_impl
    openvino::runtime
    llama
)
set_target_properties(tts_api PROPERTIES
    PUBLIC_HEADER "./api.h"
)

add_executable(spark_tts
    main.cpp
    utils.cpp
)
target_link_libraries(spark_tts PRIVATE 
    tts_impl
    SndFile::sndfile
    argparse::argparse
    openvino::runtime
    llama
)

add_executable(c_spark_tts
    main.c
    utils.cpp
)
target_link_libraries(c_spark_tts PRIVATE 
    tts_api
    SndFile::sndfile
    argparse::argparse
)

if (APPLE)
    # Set the install name for the shared library on macOS
    set_target_properties(tts_api PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )

    set_target_properties(spark_tts PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )

    set_target_properties(c_spark_tts PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )
endif()

install(TARGETS spark_tts
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION bin
)
install_llama("${CMAKE_INSTALL_PREFIX}/bin")
install_openvino("${CMAKE_INSTALL_PREFIX}/bin")
if(APPLE)
    install_onnxruntime("${CMAKE_INSTALL_PREFIX}/bin")
endif()

install(TARGETS tts_api
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/spark_tts
)

install(TARGETS c_spark_tts
    RUNTIME DESTINATION bin
)
