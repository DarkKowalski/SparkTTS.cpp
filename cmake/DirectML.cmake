set(DIRECT_ML_VERSION "1.15.4")
set(DirectML_URL "https://globalcdn.nuget.org/packages/microsoft.ai.directml.1.15.4.nupkg?packageVersion=${DIRECT_ML_VERSION}")
set(DIRECT_ML_PACKAGE "${CMAKE_BINARY_DIR}/DirectML/microsoft.ai.directml.${DIRECT_ML_VERSION}.nupkg")
set(DIRECT_ML_NUSPEC "${CMAKE_BINARY_DIR}/DirectML/Microsoft.AI.DirectML.nuspec")
set(DIRECT_ML_DLL "${CMAKE_BINARY_DIR}/DirectML/bin/x64-win/DirectML.dll")

function(download_directml)
    if (NOT WIN32)
        message(FATAL_ERROR "DirectML is only supported on Windows.")
    endif()

    # Download to build directory
    if (NOT EXISTS "${DIRECT_ML_PACKAGE}")
        message(STATUS "DirectML package not found, downloading...")
        file(DOWNLOAD "${DirectML_URL}" "${DIRECT_ML_PACKAGE}" SHOW_PROGRESS)
    else()
        message(STATUS "DirectML package already exists, skipping download.")
    endif()

    # Unzip the package
    if (NOT EXISTS "${DIRECT_ML_NUSPEC}")
        message(STATUS "Extracting DirectML package...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DIRECT_ML_PACKAGE}" 
                        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/DirectML")
    else()
        message(STATUS "DirectML package already extracted, skipping extraction.")
    endif()
endfunction()

function(install_directml dest_dir)
    if (NOT EXISTS "${DIRECT_ML_DLL}")
        message(FATAL_ERROR "DirectML DLL not found. Please ensure the package is downloaded and extracted.")
    endif()

    install(FILES "${DIRECT_ML_DLL}" DESTINATION "${dest_dir}")
endfunction()
