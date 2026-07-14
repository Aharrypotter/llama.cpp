// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

enum {
    EDGEKV_HVX_GQA_HEADS          = 2,
    EDGEKV_HVX_GQA_BLOCK_SIZE     = 64,
    EDGEKV_HVX_GQA_RANK           = 32,
    EDGEKV_HVX_GQA_HEAD_DIM       = 128,
    EDGEKV_HVX_GQA_VH_RANK_STRIDE = 4096,
};

// Fixed-shape C4C0 probe for one KV head and its two GQA query heads.
// All buffers must be 128-byte aligned. U is [block_size, rank]. Vh points at
// one [block, rank=0, layer, hkv, dim=0] row in the mobile pool, where the next
// rank begins 4096 bytes later because layer and Hkv remain interleaved.
void edgekv_hvx_gqa_block_k(const float *  q,
                            const int8_t * k_u,
                            const int8_t * k_vh,
                            const float *  k_scale,
                            float          attention_scale,
                            float *        logits);

void edgekv_hvx_gqa_block_v(const float *  weights,
                            const int8_t * v_u,
                            const int8_t * v_vh,
                            const float *  v_scale,
                            float *        output);

void edgekv_hvx_gqa_block(const float *  q,
                          const int8_t * k_u,
                          const int8_t * k_vh,
                          const float *  k_scale,
                          float          attention_scale,
                          const float *  weights,
                          const int8_t * v_u,
                          const int8_t * v_vh,
                          const float *  v_scale,
                          float *        logits,
                          float *        output);

void edgekv_scalar_gqa_block_k(const float *  q,
                               const int8_t * k_u,
                               const int8_t * k_vh,
                               const float *  k_scale,
                               float          attention_scale,
                               float *        logits);

void edgekv_scalar_gqa_block_v(const float *  weights,
                               const int8_t * v_u,
                               const int8_t * v_vh,
                               const float *  v_scale,
                               float *        output);

void edgekv_scalar_gqa_block(const float *  q,
                             const int8_t * k_u,
                             const int8_t * k_vh,
                             const float *  k_scale,
                             float          attention_scale,
                             const float *  weights,
                             const int8_t * v_u,
                             const int8_t * v_vh,
                             const float *  v_scale,
                             float *        logits,
                             float *        output);
