if(WIN32)
    # C-API Begin
    add_library(tts_api SHARED
        win/audio_tokenizer.cpp
        win/audio_detokenizer.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
        profiler/perfetto_categories.cpp
        profiler/profiler.cpp
        api.cpp
    )

    target_link_libraries(tts_api PRIVATE
        openvino::runtime
        llama
        tokenizers_cpp
    )

    target_compile_definitions(tts_api PRIVATE
        SPARK_TTS_SHARED
        SPARK_TTS_BUILD
    )

    set_target_properties(tts_api PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_LIST_DIR}/api.h")

    if(ENABLE_PERFETTO)
        if(MSVC)
            target_compile_options(tts_api PRIVATE "/permissive-")
        endif()

        target_compile_definitions(tts_api PRIVATE ENABLE_PERFETTO)
        target_compile_definitions(tts_api PRIVATE -DWIN32_LEAN_AND_MEAN -DNOMINMAX)
        target_link_libraries(tts_api PRIVATE unofficial::perfetto::perfetto ${CMAKE_THREAD_LIBS_INIT} ws2_32)
    endif()

    install(TARGETS tts_api
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/tts
    )

    # C-API End

    # C API Example
    add_executable(tts_api_example
        main.c
        utils.cpp
    )

    target_link_libraries(tts_api_example PRIVATE
        tts_api
        SndFile::sndfile
    )

    install(TARGETS tts_api_example
        RUNTIME DESTINATION bin
    )

    # C API Example End

    # C++ CLI Begin
    add_executable(tts_cli
        win/audio_tokenizer.cpp
        win/audio_detokenizer.cpp
        profiler/perfetto_categories.cpp
        profiler/profiler.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
        main.cpp
        utils.cpp
    )

    target_link_libraries(tts_cli PRIVATE
        openvino::runtime
        llama
        SndFile::sndfile
        argparse::argparse
        tokenizers_cpp
    )

    if(ENABLE_PERFETTO)
        if(MSVC)
            target_compile_options(tts_cli PRIVATE "/permissive-")
        endif()

        target_compile_definitions(tts_cli PRIVATE ENABLE_PERFETTO)
        target_compile_definitions(tts_cli PRIVATE -DWIN32_LEAN_AND_MEAN -DNOMINMAX)
        target_link_libraries(tts_cli PRIVATE unofficial::perfetto::perfetto ${CMAKE_THREAD_LIBS_INIT} ws2_32)
    endif()

    install(TARGETS tts_cli
        RUNTIME DESTINATION bin
    )

    # C++ CLI End
    install_llama("${CMAKE_INSTALL_PREFIX}/bin")
    install_openvino("${CMAKE_INSTALL_PREFIX}/bin")

elseif(APPLE)
    # C-API Begin
    add_library(tts_api SHARED
        mac/audio_tokenizer.cpp
        mac/audio_detokenizer.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
        profiler/perfetto_categories.cpp
        profiler/profiler.cpp
    )

    target_link_libraries(tts_api PRIVATE
        openvino::runtime
        llama
        onnxruntime
    )

    target_compile_definitions(tts_api PRIVATE
        SPARK_TTS_SHARED
        SPARK_TTS_BUILD
    )

    # when using CoreML
    # target_link_libraries(tts_api PRIVATE
    # "-framework CoreML"
    # "-framework Accelerate"
    # "-framework Metal"
    # )
    set_target_properties(tts_api PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_LIST_DIR}/api.h")
    set_target_properties(tts_api PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )

    if(ENABLE_PERFETTO)
        target_compile_definitions(tts_api PRIVATE ENABLE_PERFETTO)
        target_link_libraries(tts_api PRIVATE unofficial::perfetto::perfetto ${CMAKE_THREAD_LIBS_INIT})
    endif()

    install(TARGETS tts_api
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/tts
    )

    # C-API End

    # C API Example
    add_executable(tts_api_example
        main.c
        utils.cpp
    )

    set_target_properties(tts_api_example PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )

    target_link_libraries(tts_api_example PRIVATE
        tts_api
        SndFile::sndfile
    )

    install(TARGETS tts_api_example
        RUNTIME DESTINATION bin
    )

    # C API Example End

    # C++ CLI Begin
    add_executable(tts_cli
        mac/audio_tokenizer.cpp
        mac/audio_detokenizer.cpp
        profiler/perfetto_categories.cpp
        profiler/profiler.cpp
        prompt.cpp
        sampler.cpp
        tokenizer.cpp
        transformer.cpp
        synthesizer.cpp
        token_buffer.cpp
        main.cpp
        utils.cpp
    )

    set_target_properties(tts_cli PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "@loader_path/."
        MACOSX_RPATH TRUE
    )

    target_link_libraries(tts_cli PRIVATE
        openvino::runtime
        llama
        SndFile::sndfile
        argparse::argparse
    )

    if(ENABLE_PERFETTO)
        target_compile_definitions(tts_cli PRIVATE ENABLE_PERFETTO)
        target_link_libraries(tts_cli PRIVATE unofficial::perfetto::perfetto ${CMAKE_THREAD_LIBS_INIT})
    endif()

    install(TARGETS tts_cli
        RUNTIME DESTINATION bin
    )

    # C++ CLI End
    install_llama("${CMAKE_INSTALL_PREFIX}/bin")
    install_openvino("${CMAKE_INSTALL_PREFIX}/bin")
    install_onnxruntime("${CMAKE_INSTALL_PREFIX}/bin")
endif()
