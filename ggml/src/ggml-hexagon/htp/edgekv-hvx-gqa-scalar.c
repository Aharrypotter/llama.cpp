// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-block.h"

void edgekv_scalar_gqa_block_k(const float *  q,
                               const int8_t * k_u,
                               const int8_t * k_vh,
                               const float *  k_scale,
                               float          attention_scale,
                               float *        logits) {
    float q_rank[EDGEKV_HVX_GQA_HEADS][EDGEKV_HVX_GQA_RANK];
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
            float sum = 0.0f;
            for (int dim = 0; dim < EDGEKV_HVX_GQA_HEAD_DIM; ++dim) {
                sum +=
                    q[head * EDGEKV_HVX_GQA_HEAD_DIM + dim] * (float) k_vh[rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE + dim];
            }
            q_rank[head][rank] = sum * k_scale[rank];
        }
    }

    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
            float sum = 0.0f;
            for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
                sum += q_rank[head][rank] * (float) k_u[token * EDGEKV_HVX_GQA_RANK + rank];
            }
            logits[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] = sum * attention_scale;
        }
    }
}

void edgekv_scalar_gqa_block_v(const float *  weights,
                               const int8_t * v_u,
                               const int8_t * v_vh,
                               const float *  v_scale,
                               float *        output) {
    float rank_value[EDGEKV_HVX_GQA_HEADS][EDGEKV_HVX_GQA_RANK];
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
            float sum = 0.0f;
            for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
                sum +=
                    weights[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] * (float) v_u[token * EDGEKV_HVX_GQA_RANK + rank];
            }
            rank_value[head][rank] = sum * v_scale[rank];
        }
    }

    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int dim = 0; dim < EDGEKV_HVX_GQA_HEAD_DIM; ++dim) {
            float sum = 0.0f;
            for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
                sum += rank_value[head][rank] * (float) v_vh[rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE + dim];
            }
            output[head * EDGEKV_HVX_GQA_HEAD_DIM + dim] = sum;
        }
    }
}

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
                             float *        output) {
    edgekv_scalar_gqa_block_k(q, k_u, k_vh, k_scale, attention_scale, logits);
    edgekv_scalar_gqa_block_v(weights, v_u, v_vh, v_scale, output);
}
