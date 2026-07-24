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

struct edgekv_blockgtq_history_addresses {
    size_t k_codes;
    size_t k_norms;
    size_t v_codes;
    size_t v_norms;
};

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
    struct edgekv_blockgtq_diagnostics * diagnostics);

#endif
