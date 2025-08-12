if(WIN32)
    # C-API Begin
    add_library(tts_api SHARED
        win/audio_tokenizer_impl.cpp
        win/audio_detokenizer_impl.cpp
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
        llama
        tokenizers_cpp
        onnxruntime
    )

    target_compile_definitions(tts_api PRIVATE
        TTS_SHARED=1
        TTS_BUILD=1
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
        RUNTIME DESTINATION api/bin
        LIBRARY DESTINATION api/lib
        ARCHIVE DESTINATION api/lib
        PUBLIC_HEADER DESTINATION api/include/tts
    )
    install_llama("${CMAKE_INSTALL_PREFIX}/api/bin")
    install_onnxruntime("${CMAKE_INSTALL_PREFIX}/api/bin")
    install_directml("${CMAKE_INSTALL_PREFIX}/api/bin")

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
        RUNTIME DESTINATION tools/bin
    )

    # C API Example End

    # C++ CLI Begin
    add_executable(tts_cli
        win/audio_tokenizer_impl.cpp
        win/audio_detokenizer_impl.cpp
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
        llama
        SndFile::sndfile
        argparse::argparse
        tokenizers_cpp
        onnxruntime
        nlohmann_json::nlohmann_json
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
        RUNTIME DESTINATION tools/bin
    )

    # C++ CLI End
    install_llama("${CMAKE_INSTALL_PREFIX}/tools/bin")
    install_onnxruntime("${CMAKE_INSTALL_PREFIX}/tools/bin")
    install_directml("${CMAKE_INSTALL_PREFIX}/tools/bin")

elseif(APPLE)
    # C-API Begin
    add_library(tts_api SHARED
        mac/gen/AudioDetokenizer.m
        mac/gen/AudioTokenizer.m
        mac/audio_tokenizer_impl.mm
        mac/audio_detokenizer_impl.mm
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
        llama
        tokenizers_cpp
    )

    target_compile_options(tts_api PRIVATE
        -fvisibility=hidden
    )

    target_compile_definitions(tts_api PRIVATE
        TTS_SHARED=1
        TTS_BUILD=1
    )

    # when using CoreML
    target_link_libraries(tts_api PRIVATE
        "-framework CoreML"
        "-framework Accelerate"
        "-framework Metal"
        "-framework Foundation"
    )
    target_compile_options(tts_api PRIVATE
        -fobjc-arc
    )

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
        RUNTIME DESTINATION api/bin
        LIBRARY DESTINATION api/bin
        ARCHIVE DESTINATION api/lib
        PUBLIC_HEADER DESTINATION api/include/tts
    )
    install_llama("${CMAKE_INSTALL_PREFIX}/api/bin")

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
        RUNTIME DESTINATION tools/bin
    )

    # C API Example End

    # C++ CLI Begin
    add_executable(tts_cli
        mac/gen/AudioDetokenizer.m
        mac/gen/AudioTokenizer.m
        mac/audio_tokenizer_impl.mm
        mac/audio_detokenizer_impl.mm
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
        llama
        SndFile::sndfile
        argparse::argparse
        tokenizers_cpp
        nlohmann_json::nlohmann_json
    )

    target_link_libraries(tts_cli PRIVATE
        "-framework CoreML"
        "-framework Accelerate"
        "-framework Metal"
        "-framework Foundation"
    )

    target_compile_options(tts_cli PRIVATE
        -fobjc-arc
    )

    if(ENABLE_PERFETTO)
        target_compile_definitions(tts_cli PRIVATE ENABLE_PERFETTO)
        target_link_libraries(tts_cli PRIVATE unofficial::perfetto::perfetto ${CMAKE_THREAD_LIBS_INIT})
    endif()

    install(TARGETS tts_cli
        RUNTIME DESTINATION tools/bin
    )

    # C++ CLI End
    install_llama("${CMAKE_INSTALL_PREFIX}/tools/bin")
endif()
