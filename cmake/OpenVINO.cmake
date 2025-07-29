set(OPENVINO_ROOT_DIR $ENV{INTEL_OPENVINO_DIR})

function(install_openvino dest_dir)
    if (WIN32)
        # if Release
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            file(GLOB_RECURSE LIB_FILES "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Release/*.dll")
            install(FILES ${LIB_FILES} DESTINATION ${dest_dir})

            # Third-party libraries
            file(GLOB_RECURSE THIRD_PARTY_LIB_FILES "${OPENVINO_ROOT_DIR}/runtime/3rdparty/tbb/bin/*.dll")
            # exclude _debug.dll files
            list(FILTER THIRD_PARTY_LIB_FILES EXCLUDE REGEX ".*_debug.dll")
            install(FILES ${THIRD_PARTY_LIB_FILES} DESTINATION ${dest_dir})
        else()
            file(GLOB_RECURSE LIB_FILES "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Debug/*.dll")
            install(FILES ${LIB_FILES} DESTINATION ${dest_dir})

            # Third-party libraries
            file(GLOB_RECURSE THIRD_PARTY_LIB_FILES "${OPENVINO_ROOT_DIR}/runtime/3rdparty/tbb/bin/*_debug.dll")
            install(FILES ${THIRD_PARTY_LIB_FILES} DESTINATION ${dest_dir})
        endif()
    elseif (APPLE)
        # not implemented yet
        message(FATAL_ERROR "OpenVINO installation for macOS is not implemented yet.")
    endif()
endfunction(install_openvino)
