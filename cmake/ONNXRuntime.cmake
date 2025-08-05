set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib/onnxruntime")

function(find_onnxruntime)
    add_library(onnxruntime INTERFACE IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT_DIR}/include"
        INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib"
    )
endfunction(find_onnxruntime)

function(install_onnxruntime dest_dir)
    if (APPLE)
        file(GLOB_RECURSE LIB_FILES "${ONNXRUNTIME_ROOT_DIR}/lib/*.dylib")
        install(FILES ${LIB_FILES} DESTINATION ${dest_dir})
    endif()
endfunction(install_onnxruntime)
