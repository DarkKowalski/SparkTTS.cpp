add_executable(spark_tts
    audio_tokenizer.cpp
    audio_detokenizer.cpp
    prompt.cpp
    utils.cpp
    llm_streamer.cpp
    main.cpp
)

target_link_libraries(spark_tts PRIVATE 
    openvino::runtime
    openvino::genai
    SndFile::sndfile
    argparse::argparse
)
