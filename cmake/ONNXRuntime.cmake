set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib/onnxruntime")

function(find_onnxruntime)
    add_library(onnxruntime INTERFACE IMPORTED)
    if(APPLE)
        set_target_properties(onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT_DIR}/include"
            INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib"
        )
    elseif(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT_DIR}/include"
            INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib"
        )
    endif()
endfunction(find_onnxruntime)

function(install_onnxruntime dest_dir)
    if (APPLE)
        file(GLOB_RECURSE LIB_FILES "${ONNXRUNTIME_ROOT_DIR}/lib/*.dylib")
        install(FILES ${LIB_FILES} DESTINATION ${dest_dir})
    elseif(WIN32)
        file(GLOB_RECURSE LIB_FILES "${ONNXRUNTIME_ROOT_DIR}/bin/*.dll")
        install(FILES ${LIB_FILES} DESTINATION ${dest_dir})
    endif()
endfunction(install_onnxruntime)
