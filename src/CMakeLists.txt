add_executable(spark_tts
    audio_tokenizer.cpp
    audio_detokenizer.cpp
    prompt.cpp
    utils.cpp
    sampler.cpp
    tokenizer.cpp
    transformer.cpp
    main.cpp
)

target_link_libraries(spark_tts PRIVATE 
    openvino::runtime
    openvino::genai
    SndFile::sndfile
    argparse::argparse
    llama
)

install(TARGETS spark_tts
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION bin
)
install_llama("${CMAKE_INSTALL_PREFIX}/bin")
install_openvino("${CMAKE_INSTALL_PREFIX}/bin")
