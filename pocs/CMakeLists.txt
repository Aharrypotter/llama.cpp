# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_executable(kv-lowrank-projection-check kv_lowrank_projection_check.cpp)
    target_link_libraries(kv-lowrank-projection-check PRIVATE llama-common)

    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
    endif()
endif()
