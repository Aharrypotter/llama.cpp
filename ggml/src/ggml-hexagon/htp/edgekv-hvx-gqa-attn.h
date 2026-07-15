// SPDX-License-Identifier: MIT

#pragma once

#include "edgekv-hvx-gqa-block.h"

#include <stddef.h>
#include <stdint.h>

enum {
    EDGEKV_HVX_GQA_ATTN_N_BLOCKS           = 16,
    EDGEKV_HVX_GQA_ATTN_GROUP_SIZE         = 4,
    EDGEKV_HVX_GQA_ATTN_N_HEAD_KV          = 8,
    EDGEKV_HVX_GQA_ATTN_N_HEAD_Q           = 16,
    EDGEKV_HVX_GQA_ATTN_MAX_WORKERS        = 4,
    EDGEKV_HVX_GQA_ATTN_RECENT_SIZE        = 128,
    EDGEKV_HVX_GQA_ATTN_U_BLOCK_BYTES      = EDGEKV_HVX_GQA_BLOCK_SIZE * EDGEKV_HVX_GQA_RANK,
    EDGEKV_HVX_GQA_ATTN_VH_LAYER_STRIDE    = EDGEKV_HVX_GQA_ATTN_N_HEAD_KV * EDGEKV_HVX_GQA_HEAD_DIM,
    EDGEKV_HVX_GQA_ATTN_VH_BLOCK_BYTES     = EDGEKV_HVX_GQA_RANK * EDGEKV_HVX_GQA_VH_RANK_STRIDE,
    EDGEKV_HVX_GQA_ATTN_ACTIVE_TOKEN_ELEMS = EDGEKV_HVX_GQA_ATTN_N_HEAD_KV * EDGEKV_HVX_GQA_HEAD_DIM,
    EDGEKV_HVX_GQA_ATTN_BLOCK_LOGITS  = EDGEKV_HVX_GQA_ATTN_N_BLOCKS * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE,
    EDGEKV_HVX_GQA_ATTN_ACTIVE_LOGITS = EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_ATTN_RECENT_SIZE,
    EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT = EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM,
    EDGEKV_HVX_GQA_ATTN_WORKSPACE_FLOATS =
        EDGEKV_HVX_GQA_ATTN_BLOCK_LOGITS + EDGEKV_HVX_GQA_ATTN_ACTIVE_LOGITS + EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT,
    EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES = EDGEKV_HVX_GQA_ATTN_WORKSPACE_FLOATS * sizeof(float),
};

struct edgekv_hvx_gqa_attn_inputs {
    const float *    query;
    const int8_t *   k_u;
    const int8_t *   k_vh;
    const float *    k_scale;
    const int8_t *   v_u;
    const int8_t *   v_vh;
    const float *    v_scale;
    const int32_t *  block_positions;
    const _Float16 * active_k;
    const _Float16 * active_v;
    const int32_t *  active_positions;
    int32_t          query_position;
};

struct edgekv_hvx_gqa_attn_profile {
    uint64_t factor_k;
    uint64_t active_k;
    uint64_t mask_softmax;
    uint64_t factor_v;
    uint64_t active_v_normalize;
    uint64_t total;
};

// Fixed C4C2 mobile consumer for one KV head and its two adjacent GQA heads.
// Input, output, and workspace bases must be 128-byte aligned. The caller owns
// worker scheduling and provides one workspace per concurrent worker.
void edgekv_hvx_gqa_attn_hkv(const struct edgekv_hvx_gqa_attn_inputs * inputs,
                             int                                       layer_index,
                             int                                       hkv,
                             float                                     attention_scale,
                             void *                                    workspace,
                             float *                                   output,
                             struct edgekv_hvx_gqa_attn_profile *      profile);
