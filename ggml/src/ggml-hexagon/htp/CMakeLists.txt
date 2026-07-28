cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(HTP_LIB ggml-htp-${DSP_VERSION})

set(GGML_HEXAGON_EDGEKV_AOT_OBJECT "" CACHE FILEPATH "v79 EdgeKV reconstruct AOT object")
set(GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT "" CACHE FILEPATH "v79 EdgeKV direct-attention AOT object")
set(GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY "" CACHE FILEPATH "Hexagon-MLIR async runtime archive")
set(GGML_HEXAGON_EDGEKV_BLOCKGTQ_FIXTURE "" CACHE FILEPATH "Frozen B2-T4 Block-GTQ simulator fixture")
option(GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION "Build the B2-T4P attribution matrix" OFF)
option(GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY "Build the B2-T4P-H3 target observability diagnostic" OFF)
option(GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS "Build the B2-T4P-H4 target logit-path diagnostic" OFF)
option(GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS "Build the B2-T4P-H5 query-rotation diagnostic" OFF)
option(GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX "Build the B2-T4P-H7 cross-libm correctness matrix" OFF)
option(GGML_HEXAGON_EDGEKV_BUILD_TESTS "Build the v79 EdgeKV simulator tests" OFF)

if ((GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX) OR
    (GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS AND
     GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS))
    message(FATAL_ERROR "Block-GTQ attribution, H3/H4/H5 diagnostics, and H7 cross-libm builds are mutually exclusive")
endif()

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    worker-pool.c
    hex-dma.c
    matmul-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    flash-attn-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    repeat-ops.c
    argsort-ops.c
    ssm-conv.c
    cumsum-ops.c
    fill-ops.c
    diag-ops.c
    solve-tri-ops.c
    edgekv-aot-runtime.c
    edgekv-reconstruct-ops.c
    edgekv-attn-decode-ops.c
    edgekv-blockgtq-attn.c
    edgekv-blockgtq-attn-ops.c
    edgekv-blockgtq-attn-v2-ops.c
    ../../edgekv-blockgtq-runtime.c
    ../../edgekv-blockgtq-runtime.h
)

if (DSP_VERSION STREQUAL "v79")
    target_sources(${HTP_LIB} PRIVATE
        edgekv-hvx-gqa-attn.c
        edgekv-hvx-gqa-block.c)
    target_compile_definitions(${HTP_LIB} PRIVATE HTP_EDGEKV_HVX_GQA=1)
endif()

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})

if (DSP_VERSION STREQUAL "v79" AND GGML_HEXAGON_EDGEKV_BUILD_TESTS)
    if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_FIXTURE)
        if (NOT EXISTS "${GGML_HEXAGON_EDGEKV_BLOCKGTQ_FIXTURE}")
            message(FATAL_ERROR "Block-GTQ fixture does not exist: ${GGML_HEXAGON_EDGEKV_BLOCKGTQ_FIXTURE}")
        endif()
        configure_file(
            tests/edgekv-blockgtq-fixture.S.in
            ${CMAKE_CURRENT_BINARY_DIR}/edgekv-blockgtq-fixture.S
            @ONLY)
        set_source_files_properties(
            ${CMAKE_CURRENT_BINARY_DIR}/edgekv-blockgtq-fixture.S
            PROPERTIES GENERATED TRUE)
        set(EDGEKV_BLOCKGTQ_TEST_LIB test-edgekv-blockgtq-attn-${DSP_VERSION})
        add_library(${EDGEKV_BLOCKGTQ_TEST_LIB} SHARED
            tests/test-edgekv-blockgtq-attn.c
            edgekv-blockgtq-attn.c
            edgekv-blockgtq-attn-ops.c
            flash-attn-ops.c
            worker-pool.c
            hex-dma.c
            ${CMAKE_CURRENT_BINARY_DIR}/edgekv-blockgtq-fixture.S)
        set_source_files_properties(
            tests/test-edgekv-blockgtq-attn.c
            PROPERTIES COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")
        target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
            HTP_EDGEKV_BLOCKGTQ_TEST=1)
        if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_ATTRIBUTION)
            target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX=1
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_ATTRIBUTION=1)
        endif()
        if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY)
            target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX=1
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY=1)
        endif()
        if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_LOGIT_FORENSICS)
            target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX=1
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY=1
                EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS=1)
        endif()
        if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_QUERY_ROTATION_FORENSICS)
            target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX=1
                HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_ATTRIBUTION=1
                EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY=1
                EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS=1)
        endif()
        if (GGML_HEXAGON_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX)
            target_compile_definitions(${EDGEKV_BLOCKGTQ_TEST_LIB} PRIVATE
                HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX=1
                EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY=1)
        endif()
    endif()

    set(EDGEKV_HVX_GQA_TEST_LIB test-edgekv-hvx-gqa-block-${DSP_VERSION})
    add_library(${EDGEKV_HVX_GQA_TEST_LIB} SHARED
        tests/test-edgekv-hvx-gqa-block.c
        edgekv-hvx-gqa-block.c
        edgekv-hvx-gqa-scalar.c)
    # Keep fixture and scalar-reference work out of the hand-written HVX region.
    set_source_files_properties(tests/test-edgekv-hvx-gqa-block.c PROPERTIES
        COMPILE_OPTIONS "-fno-lto;-O0;-fno-vectorize;-fno-slp-vectorize")
    set_source_files_properties(edgekv-hvx-gqa-scalar.c PROPERTIES
        COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")

    set(EDGEKV_HVX_GQA_MOBILE_TEST_LIB test-edgekv-hvx-gqa-mobile-${DSP_VERSION})
    add_library(${EDGEKV_HVX_GQA_MOBILE_TEST_LIB} SHARED
        tests/test-edgekv-hvx-gqa-mobile.c
        edgekv-hvx-gqa-block.c
        edgekv-hvx-gqa-scalar.c
        worker-pool.c)
    set_source_files_properties(tests/test-edgekv-hvx-gqa-mobile.c PROPERTIES
        COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")

    set(EDGEKV_HVX_GQA_ATTN_MOBILE_TEST_LIB test-edgekv-hvx-gqa-attn-mobile-${DSP_VERSION})
    add_library(${EDGEKV_HVX_GQA_ATTN_MOBILE_TEST_LIB} SHARED
        tests/test-edgekv-hvx-gqa-attn-mobile.c
        edgekv-hvx-gqa-attn.c
        edgekv-hvx-gqa-block.c
        edgekv-hvx-gqa-scalar.c
        flash-attn-ops.c
        worker-pool.c
        hex-dma.c)
    set_source_files_properties(tests/test-edgekv-hvx-gqa-attn-mobile.c PROPERTIES
        COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")

    set(EDGEKV_HVX_GQA_ATTN_OP_TEST_LIB test-edgekv-attn-decode-hvx-${DSP_VERSION})
    add_library(${EDGEKV_HVX_GQA_ATTN_OP_TEST_LIB} SHARED
        tests/test-edgekv-attn-decode.c
        edgekv-attn-decode-ops.c
        edgekv-hvx-gqa-attn.c
        edgekv-hvx-gqa-block.c
        flash-attn-ops.c
        worker-pool.c
        hex-dma.c)
    set_source_files_properties(tests/test-edgekv-attn-decode.c PROPERTIES
        COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")
    target_compile_definitions(${EDGEKV_HVX_GQA_ATTN_OP_TEST_LIB} PRIVATE
        HTP_EDGEKV_HVX_GQA=1
        HTP_EDGEKV_PROFILE_KERNEL=1)
endif()

if (DSP_VERSION STREQUAL "v79" AND
    (GGML_HEXAGON_EDGEKV_AOT_OBJECT OR GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT))
    if (NOT EXISTS "${GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY}")
        message(FATAL_ERROR "Hexagon-MLIR runtime does not exist: ${GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY}")
    endif()

    set(EDGEKV_AOT_OBJECTS)
    if (GGML_HEXAGON_EDGEKV_AOT_OBJECT)
        if (NOT EXISTS "${GGML_HEXAGON_EDGEKV_AOT_OBJECT}")
            message(FATAL_ERROR "EdgeKV reconstruct AOT object does not exist: ${GGML_HEXAGON_EDGEKV_AOT_OBJECT}")
        endif()
        message(STATUS "ggml-htp-v79: enabling EdgeKV C3 reconstruct AOT adapter")
        target_compile_definitions(${HTP_LIB} PRIVATE HTP_EDGEKV_RECONSTRUCT_AOT=1)
        list(APPEND EDGEKV_AOT_OBJECTS "${GGML_HEXAGON_EDGEKV_AOT_OBJECT}")
    endif()
    if (GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT)
        if (NOT EXISTS "${GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT}")
            message(FATAL_ERROR "EdgeKV direct-attention AOT object does not exist: ${GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT}")
        endif()
        message(STATUS "ggml-htp-v79: enabling EdgeKV direct-attention AOT adapter")
        target_compile_definitions(${HTP_LIB} PRIVATE HTP_EDGEKV_DIRECT_AOT=1)
        list(APPEND EDGEKV_AOT_OBJECTS "${GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT}")
    endif()
    target_link_libraries(${HTP_LIB} PRIVATE ${EDGEKV_AOT_OBJECTS} "${GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY}")

    if (GGML_HEXAGON_EDGEKV_BUILD_TESTS AND GGML_HEXAGON_EDGEKV_AOT_OBJECT)
        set(EDGEKV_TEST_LIB test-edgekv-reconstruct-${DSP_VERSION})
        add_library(${EDGEKV_TEST_LIB} SHARED
            tests/test-edgekv-reconstruct.c
            edgekv-aot-runtime.c
            edgekv-reconstruct-ops.c)
        target_compile_definitions(${EDGEKV_TEST_LIB} PRIVATE HTP_EDGEKV_RECONSTRUCT_AOT=1)
        target_link_libraries(${EDGEKV_TEST_LIB} PRIVATE
            "${GGML_HEXAGON_EDGEKV_AOT_OBJECT}"
            "${GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY}")
    endif()
    if (GGML_HEXAGON_EDGEKV_BUILD_TESTS AND GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT)
        set(EDGEKV_DIRECT_TEST_LIB test-edgekv-attn-decode-${DSP_VERSION})
        add_library(${EDGEKV_DIRECT_TEST_LIB} SHARED
            tests/test-edgekv-attn-decode.c
            edgekv-aot-runtime.c
            edgekv-attn-decode-ops.c
            flash-attn-ops.c
            worker-pool.c
            hex-dma.c)
        # Fixture construction is outside every timed region; keep it out of Release LTO/vectorization.
        set_source_files_properties(tests/test-edgekv-attn-decode.c PROPERTIES
            COMPILE_OPTIONS "-fno-lto;-O3;-fno-vectorize;-fno-slp-vectorize")
        target_compile_definitions(${EDGEKV_DIRECT_TEST_LIB} PRIVATE
            HTP_EDGEKV_DIRECT_AOT=1
            HTP_EDGEKV_PROFILE_AOT=1)
        target_link_libraries(${EDGEKV_DIRECT_TEST_LIB} PRIVATE
            "${GGML_HEXAGON_EDGEKV_DIRECT_AOT_OBJECT}"
            "${GGML_HEXAGON_EDGEKV_RUNTIME_LIBRARY}")
    endif()
endif()

if (GGML_HEXAGON_FA_EXP2_HF)
    message(STATUS "ggml-htp: HMX_FA_USE_EXP2_HF=1 (use FP16 exp2 polynomial in FA softmax)")
    target_compile_definitions(${HTP_LIB} PRIVATE HMX_FA_USE_EXP2_HF=1)
endif()

# HMX acceleration: available on v73+ architectures
set(HTP_HMX_VERSIONS v73 v75 v79 v81)
list(FIND HTP_HMX_VERSIONS ${DSP_VERSION} _hmx_idx)

if (_hmx_idx GREATER_EQUAL 0)
    target_sources(${HTP_LIB} PRIVATE
        hmx-queue.c
        hmx-matmul-ops.c
        hmx-flash-attn-ops.c
    )

    # -mhmx enables HMX instruction set (needed by files that include hmx-utils.h)
    set_source_files_properties(
        hmx-matmul-ops.c
        hmx-flash-attn-ops.c
        PROPERTIES COMPILE_OPTIONS "-mhmx"
    )

    target_compile_definitions(${HTP_LIB} PRIVATE HTP_HAS_HMX=1)
endif()

build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

install(TARGETS ${HTP_LIB})
