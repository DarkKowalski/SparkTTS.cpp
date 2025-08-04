add_executable(spark_tts
    audio_tokenizer.cpp
    audio_detokenizer.cpp
    prompt.cpp
    utils.cpp
    sampler.cpp
    tokenizer.cpp
    transformer.cpp
    synthesizer.cpp
    token_buffer.cpp
    main.cpp
)

add_library(tts_api SHARED
    audio_detokenizer.cpp
    audio_tokenizer.cpp
    prompt.cpp
    sampler.cpp
    tokenizer.cpp
    transformer.cpp
    synthesizer.cpp
    token_buffer.cpp
    api.cpp
)

add_executable(c_spark_tts
    main.c
    utils.cpp
)

target_compile_definitions(tts_api PRIVATE
    -DSPARK_TTS_SHARED
    -DSPARK_TTS_BUILD
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

target_link_libraries(spark_tts PRIVATE 
    openvino::runtime
    openvino::genai
    SndFile::sndfile
    argparse::argparse
    llama
)

target_link_libraries(tts_api PRIVATE 
    openvino::runtime
    openvino::genai
    llama
)

target_link_libraries(c_spark_tts PRIVATE 
    tts_api
    SndFile::sndfile
)

install(TARGETS spark_tts
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION bin
)
install_llama("${CMAKE_INSTALL_PREFIX}/bin")
install_openvino("${CMAKE_INSTALL_PREFIX}/bin")

install(TARGETS tts_api
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(TARGETS c_spark_tts
    RUNTIME DESTINATION bin
)
