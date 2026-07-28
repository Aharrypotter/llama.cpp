// SPDX-License-Identifier: MIT

#ifndef GGML_EDGEKV_BLOCKGTQ_RUNTIME_H
#define GGML_EDGEKV_BLOCKGTQ_RUNTIME_H

#include "ggml-hexagon/htp/edgekv-blockgtq-attn.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EDGEKV_BLOCKGTQ_APPEND_BYTES = 316,
};

int edgekv_blockgtq_pack_token_v2(
    const float * k,
    const float * v,
    const uint8_t * producer,
    size_t producer_bytes,
    const uint8_t * consumer,
    size_t consumer_bytes,
    const uint8_t * shared,
    size_t shared_bytes,
    int layer_index,
    uint8_t output[EDGEKV_BLOCKGTQ_APPEND_BYTES]);

int edgekv_blockgtq_commit_token_v2(
    uint8_t * history,
    size_t history_bytes,
    int capacity,
    int layer_index,
    int token_index,
    const uint8_t current[EDGEKV_BLOCKGTQ_APPEND_BYTES],
    size_t current_bytes);

#ifdef __cplusplus
}
#endif

#endif
