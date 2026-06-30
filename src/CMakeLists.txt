llama_add_compile_flags()

option(LLAMA_KV_BLOCKSVD "Enable EdgeKV Block SVD KV cache compression" OFF)

if (LLAMA_KV_BLOCKSVD)
    find_package(Eigen3 REQUIRED)
endif()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_SOURCES
    ../include/llama.h
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-lowrank.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
    unicode.h
    ${LLAMA_MODELS_SOURCES}
)

if (LLAMA_KV_BLOCKSVD)
    list(APPEND LLAMA_SOURCES llama-kv-blocksvd.cpp)
endif()

add_library(llama ${LLAMA_SOURCES})

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE . ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

if (LLAMA_KV_BLOCKSVD)
    target_link_libraries(llama PUBLIC ggml Eigen3::Eigen)
    target_compile_definitions(llama PRIVATE LLAMA_KV_BLOCKSVD)
else()
    target_link_libraries(llama PUBLIC ggml)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
