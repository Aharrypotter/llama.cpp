message(STATUS "Including HTP backend")

set(GGML_SOURCES_HTP
    ggml-htp.cc
    dsprpc_interface.h
    dsprpc_interface.cc
    ggml-htp-impl.h
    htp-cpu-impl.c
    htp-cpu-vec.h
    htp-cpu-vec.cpp
    htp-cpu-ops.cpp
    htp-cpu-ops.h
    htp-cpu-threadpool.c
    htp-debug.cc
    htp-debug.h
    htp-ops.cc
    htp-ops.h
    rpcmem_mapper.cc
    rpcmem_mapper.h
)

# NOTE(hzx): htp-cpu-impl.c references vec_dot functions from ggml-cpu/quants.c
# To avoid runtime dynamic linking issues (undefined symbols causing SIGSEGV),
# we compile quants.c and arch-specific quants.c directly into libggml-htp.so
if (GGML_SYSTEM_ARCH STREQUAL "ARM")
    list(APPEND GGML_SOURCES_HTP
        ../ggml-cpu/quants.c
        ../ggml-cpu/arch/arm/quants.c
    )
elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
    list(APPEND GGML_SOURCES_HTP
        ../ggml-cpu/quants.c
        ../ggml-cpu/arch/x86/quants.c
    )
else()
    # For other architectures, include the generic quants.c
    list(APPEND GGML_SOURCES_HTP ../ggml-cpu/quants.c)
endif()

ggml_add_backend_library(ggml-htp ${GGML_SOURCES_HTP})

# NOTE(hzx): Still link ggml-cpu for other dependencies, but vec_dot symbols are now internal
if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-htp PRIVATE ggml-cpu)
endif()

# Include directories for quants headers
target_include_directories(ggml-htp PRIVATE ../ggml-cpu)
