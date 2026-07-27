// SPDX-License-Identifier: BSD-3-Clause

#ifndef EDGEKV_BLOCKGTQ_ATTN_H
#define EDGEKV_BLOCKGTQ_ATTN_H

#include <stddef.h>
#include <stdint.h>

enum {
    EDGEKV_BLOCKGTQ_ABI_VERSION = 1,
    EDGEKV_BLOCKGTQ_LAYERS = 36,
    EDGEKV_BLOCKGTQ_QUERY_HEADS = 16,
    EDGEKV_BLOCKGTQ_KV_HEADS = 2,
    EDGEKV_BLOCKGTQ_HEAD_DIM = 128,
    EDGEKV_BLOCKGTQ_HEADS = 72,
    EDGEKV_BLOCKGTQ_GQA = 8,
    EDGEKV_BLOCKGTQ_MAX_SEGMENTS = 8,
    EDGEKV_BLOCKGTQ_K_CODE_STRIDE = 76,
    EDGEKV_BLOCKGTQ_K_NORM_STRIDE = 16,
    EDGEKV_BLOCKGTQ_V_CODE_STRIDE = 64,
    EDGEKV_BLOCKGTQ_V_NORM_STRIDE = 2,
    EDGEKV_BLOCKGTQ_CAPACITY = 2048,
    EDGEKV_BLOCKGTQ_TRANSFORM_BYTES = 948800,
    EDGEKV_BLOCKGTQ_CONSUMER_BYTES = 143168,
    EDGEKV_BLOCKGTQ_SHARED_BYTES = 23616,
    EDGEKV_BLOCKGTQ_DYNAMIC_BYTES = 23298048,
    EDGEKV_BLOCKGTQ_OUTPUT_FLOATS =
        EDGEKV_BLOCKGTQ_QUERY_HEADS * EDGEKV_BLOCKGTQ_HEAD_DIM,
};

enum edgekv_blockgtq_status {
    EDGEKV_BLOCKGTQ_OK = 0,
    EDGEKV_BLOCKGTQ_INVALID_ARGUMENT = -1,
    EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH = -2,
    EDGEKV_BLOCKGTQ_DYNAMIC_LAYOUT_MISMATCH = -3,
};

struct edgekv_blockgtq_inputs {
    const float * query;
    const uint8_t * transform;
    size_t transform_bytes;
    const uint8_t * consumer;
    size_t consumer_bytes;
    const uint8_t * shared;
    size_t shared_bytes;
    const uint8_t * history;
    size_t history_bytes;
    int layer_index;
    int sequence_length;
    int capacity;
};

// Diagnostics are optional and are used only by the preregistered simulator
// correctness test. The production operation passes NULL.
struct edgekv_blockgtq_diagnostics {
    float * logits;       // [16, sequence_length]
    float * weights;      // [16, sequence_length]
    float * denominators; // [16]
    float * rotated_v;    // [16, 128], normalized weighted accumulation
};

#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
struct edgekv_blockgtq_target_observability {
    uint32_t maximum_logit_bits[EDGEKV_BLOCKGTQ_QUERY_HEADS];
    uint64_t logit_bits_fnv1a64[EDGEKV_BLOCKGTQ_QUERY_HEADS];
    uint32_t denominator_bits[EDGEKV_BLOCKGTQ_QUERY_HEADS];
    uint64_t weight_bits_fnv1a64[EDGEKV_BLOCKGTQ_QUERY_HEADS];
#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
    uint32_t rotated_query_bits[EDGEKV_BLOCKGTQ_HEAD_DIM];
    uint64_t code_fnv1a64[128][4];
    uint64_t lut_bits_fnv1a64[128][4];
    uint32_t norm_bits[128][4];
    uint32_t cumulative_dot_bits[128][4];
    uint32_t pre_scale_dot_bits[128];
    uint32_t logit_bits[128];
    uint32_t segment_count;
#endif
};
#endif

struct edgekv_blockgtq_history_addresses {
    size_t k_codes;
    size_t k_norms;
    size_t v_codes;
    size_t v_norms;
};

#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
enum edgekv_blockgtq_stage {
    EDGEKV_BLOCKGTQ_STAGE_CORE_INPUT_VALIDATION = 0,
    EDGEKV_BLOCKGTQ_STAGE_STATIC_PACKAGE_VALIDATION,
    EDGEKV_BLOCKGTQ_STAGE_CATALOG_CONSTRUCTION,
    EDGEKV_BLOCKGTQ_STAGE_QUERY_ROTATION,
    EDGEKV_BLOCKGTQ_STAGE_FIRST_K_DECODE_DOT,
    EDGEKV_BLOCKGTQ_STAGE_DIAGNOSTICS_OFF_SECOND_K_SOFTMAX_V_FUSED,
    EDGEKV_BLOCKGTQ_STAGE_DIAGNOSTICS_ON_LOGIT_REUSE_SOFTMAX_V_FUSED,
    EDGEKV_BLOCKGTQ_STAGE_NORMALIZATION,
    EDGEKV_BLOCKGTQ_STAGE_INVERSE_V_ROTATION,
    EDGEKV_BLOCKGTQ_STAGE_COUNT,
};

typedef uint64_t (*edgekv_blockgtq_clock_fn)(void * context);

struct edgekv_blockgtq_attribution {
    edgekv_blockgtq_clock_fn clock;
    void * clock_context;
    uint64_t pcycles[EDGEKV_BLOCKGTQ_STAGE_COUNT];
    uint32_t intervals[EDGEKV_BLOCKGTQ_STAGE_COUNT];
};

enum {
    EDGEKV_BLOCKGTQ_COUNTER_OVERHEAD_SAMPLES = 33,
};

struct edgekv_blockgtq_operation_attribution {
    uint64_t adapter_input_validation_pcycles;
    uint64_t operation_body_pcycles;
    uint64_t counter_overhead_pcycles[
        EDGEKV_BLOCKGTQ_COUNTER_OVERHEAD_SAMPLES];
    int diagnostics_enabled;
    struct edgekv_blockgtq_attribution core;
};

const char * edgekv_blockgtq_stage_name(enum edgekv_blockgtq_stage stage);

int edgekv_blockgtq_attn_decode_profiled(
    const struct edgekv_blockgtq_inputs * inputs,
    float * output,
    struct edgekv_blockgtq_diagnostics * diagnostics,
    struct edgekv_blockgtq_attribution * attribution
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
    , struct edgekv_blockgtq_target_observability * observability
#endif
);
#endif

int edgekv_blockgtq_history_addresses(
    int capacity, int token, int head,
    struct edgekv_blockgtq_history_addresses * addresses);

int edgekv_blockgtq_validate_static(const uint8_t * transform,
                                    size_t transform_bytes,
                                    const uint8_t * consumer,
                                    size_t consumer_bytes,
                                    const uint8_t * shared,
                                    size_t shared_bytes);

int edgekv_blockgtq_attn_decode(
    const struct edgekv_blockgtq_inputs * inputs,
    float * output,
    struct edgekv_blockgtq_diagnostics * diagnostics
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
    , struct edgekv_blockgtq_target_observability * observability
#endif
);

#endif
