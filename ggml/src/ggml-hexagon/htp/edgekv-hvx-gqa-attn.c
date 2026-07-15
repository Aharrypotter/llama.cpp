// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-attn.h"

#include "HAP_perf.h"
#include "hvx-utils.h"

#include <math.h>
#include <stdint.h>

static inline uint64_t edgekv_profile_start(const struct edgekv_hvx_gqa_attn_profile * profile) {
    return profile ? HAP_perf_get_pcycles() : 0;
}

static inline void edgekv_profile_add(struct edgekv_hvx_gqa_attn_profile * profile, uint64_t * field, uint64_t start) {
    if (profile) {
        *field += HAP_perf_get_pcycles() - start;
    }
}

static inline void edgekv_query_to_f16(const float * src, HVX_Vector dst[2]) {
    const HVX_Vector * vectors = (const HVX_Vector *) src;
    dst[0]                     = hvx_vec_f32_to_f16(vectors[0], vectors[1]);
    dst[1]                     = hvx_vec_f32_to_f16(vectors[2], vectors[3]);
}

static inline HVX_Vector edgekv_dot_128_f16(HVX_Vector q[2], HVX_Vector x0, HVX_Vector x1) {
    HVX_VectorPair acc = Q6_W_vzero();
    acc                = hvx_vec_mpyacc_f32_f16(acc, q[0], x0);
    acc                = hvx_vec_mpyacc_f32_f16(acc, q[1], x1);
    HVX_Vector sum     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc), Q6_V_hi_W(acc)));
    return hvx_vec_reduce_sum_f32(sum);
}

static void edgekv_active_k_logits(const float *    query,
                                   const _Float16 * active_k,
                                   int              hkv,
                                   float            attention_scale,
                                   float *          logits) {
    HVX_Vector q0[2];
    HVX_Vector q1[2];
    edgekv_query_to_f16(query, q0);
    edgekv_query_to_f16(query + EDGEKV_HVX_GQA_HEAD_DIM, q1);

    const HVX_Vector scale = hvx_vec_splat_f32(attention_scale);
    for (int token = 0; token < EDGEKV_HVX_GQA_ATTN_RECENT_SIZE; ++token) {
        const _Float16 * row =
            active_k + token * EDGEKV_HVX_GQA_ATTN_ACTIVE_TOKEN_ELEMS + hkv * EDGEKV_HVX_GQA_HEAD_DIM;
        const HVX_Vector * values = (const HVX_Vector *) row;
        HVX_Vector         sum0   = edgekv_dot_128_f16(q0, values[0], values[1]);
        HVX_Vector         sum1   = edgekv_dot_128_f16(q1, values[0], values[1]);
        sum0                      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sum0, scale));
        sum1                      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sum1, scale));
        hvx_vec_store_u(logits + token, sizeof(float), sum0);
        hvx_vec_store_u(logits + EDGEKV_HVX_GQA_ATTN_RECENT_SIZE + token, sizeof(float), sum1);
    }
}

static inline HVX_VectorPred edgekv_valid_positions(HVX_Vector positions, int32_t query_position) {
    const HVX_Vector     minus_one   = Q6_V_vsplat_R(-1);
    const HVX_Vector     query       = Q6_V_vsplat_R(query_position);
    const HVX_VectorPred nonnegative = Q6_Q_vcmp_gt_VwVw(positions, minus_one);
    const HVX_VectorPred future      = Q6_Q_vcmp_gt_VwVw(positions, query);
    return Q6_Q_and_QQ(nonnegative, Q6_Q_not_Q(future));
}

static void edgekv_mask_scores(float *         scores,
                               const int32_t * positions,
                               int             tokens,
                               int32_t         query_position,
                               HVX_Vector      max_acc[EDGEKV_HVX_GQA_HEADS]) {
    const HVX_Vector negative_infinity = hvx_vec_splat_f32(-INFINITY);
    for (int token = 0; token < tokens; token += VLEN_FP32) {
        const HVX_Vector     pos   = *(const HVX_Vector *) (positions + token);
        const HVX_VectorPred valid = edgekv_valid_positions(pos, query_position);
        for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
            HVX_Vector *     score_ptr = (HVX_Vector *) (scores + head * tokens + token);
            const HVX_Vector masked    = Q6_V_vmux_QVV(valid, *score_ptr, negative_infinity);
            *score_ptr                 = masked;
            max_acc[head]              = Q6_Vsf_vmax_VsfVsf(max_acc[head], masked);
        }
    }
}

static void edgekv_zero_scores(float * scores, int tokens) {
    hvx_splat_f32_a(scores, 0.0f, tokens);
}

static void edgekv_exp_scores(float * scores, int tokens, float max_logit, HVX_Vector * sum_acc) {
    const HVX_Vector max_vec = hvx_vec_splat_f32(max_logit);
    HVX_Vector *     vectors = (HVX_Vector *) scores;
    for (int token = 0; token < tokens; token += VLEN_FP32) {
        const int        vector_index = token / VLEN_FP32;
        const HVX_Vector shifted      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(vectors[vector_index], max_vec));
        const HVX_Vector weights      = hvx_vec_exp_f32(shifted);
        vectors[vector_index]         = weights;
        *sum_acc                      = Q6_Vsf_vadd_VsfVsf(*sum_acc, weights);
    }
}

static void edgekv_mask_softmax(float *         block_logits,
                                float *         active_logits,
                                const int32_t * block_positions,
                                const int32_t * active_positions,
                                int32_t         query_position,
                                float           sums[EDGEKV_HVX_GQA_HEADS]) {
    const HVX_Vector negative_infinity             = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector       max_acc[EDGEKV_HVX_GQA_HEADS] = { negative_infinity, negative_infinity };

    for (int block = 0; block < EDGEKV_HVX_GQA_ATTN_N_BLOCKS; ++block) {
        edgekv_mask_scores(block_logits + block * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE,
                           block_positions + block * EDGEKV_HVX_GQA_BLOCK_SIZE, EDGEKV_HVX_GQA_BLOCK_SIZE,
                           query_position, max_acc);
    }
    edgekv_mask_scores(active_logits, active_positions, EDGEKV_HVX_GQA_ATTN_RECENT_SIZE, query_position, max_acc);

    float max_logit[EDGEKV_HVX_GQA_HEADS];
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        max_logit[head] = hvx_vec_get_f32(hvx_vec_reduce_max_f32(max_acc[head]));
    }

    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        if (!isfinite(max_logit[head])) {
            for (int block = 0; block < EDGEKV_HVX_GQA_ATTN_N_BLOCKS; ++block) {
                float * scores = block_logits + block * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE +
                                 head * EDGEKV_HVX_GQA_BLOCK_SIZE;
                edgekv_zero_scores(scores, EDGEKV_HVX_GQA_BLOCK_SIZE);
            }
            edgekv_zero_scores(active_logits + head * EDGEKV_HVX_GQA_ATTN_RECENT_SIZE, EDGEKV_HVX_GQA_ATTN_RECENT_SIZE);
            sums[head] = 0.0f;
            continue;
        }

        HVX_Vector sum_acc = Q6_V_vzero();
        for (int block = 0; block < EDGEKV_HVX_GQA_ATTN_N_BLOCKS; ++block) {
            float * scores = block_logits + block * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE +
                             head * EDGEKV_HVX_GQA_BLOCK_SIZE;
            edgekv_exp_scores(scores, EDGEKV_HVX_GQA_BLOCK_SIZE, max_logit[head], &sum_acc);
        }
        edgekv_exp_scores(active_logits + head * EDGEKV_HVX_GQA_ATTN_RECENT_SIZE, EDGEKV_HVX_GQA_ATTN_RECENT_SIZE,
                          max_logit[head], &sum_acc);
        sums[head] = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(sum_acc));
    }
}

static void edgekv_add_output(float * output, const float * partial) {
    HVX_Vector *       dst = (HVX_Vector *) output;
    const HVX_Vector * src = (const HVX_Vector *) partial;
    for (int i = 0; i < EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT / VLEN_FP32; ++i) {
        dst[i] = Q6_Vsf_vadd_VsfVsf(dst[i], src[i]);
    }
}

static void edgekv_active_v_accumulate(const float * weights, const _Float16 * active_v, int hkv, float * output) {
    __fp16 weights_f16[EDGEKV_HVX_GQA_HEADS][EDGEKV_HVX_GQA_ATTN_RECENT_SIZE] __attribute__((aligned(128)));
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        const HVX_Vector * source = (const HVX_Vector *) (weights + head * EDGEKV_HVX_GQA_ATTN_RECENT_SIZE);
        HVX_Vector *       target = (HVX_Vector *) weights_f16[head];
        target[0]                 = hvx_vec_f32_to_f16(source[0], source[1]);
        target[1]                 = hvx_vec_f32_to_f16(source[2], source[3]);
    }

    HVX_VectorPair out00 = Q6_W_vzero();
    HVX_VectorPair out01 = Q6_W_vzero();
    HVX_VectorPair out10 = Q6_W_vzero();
    HVX_VectorPair out11 = Q6_W_vzero();
    for (int token = 0; token < EDGEKV_HVX_GQA_ATTN_RECENT_SIZE; ++token) {
        const _Float16 * row =
            active_v + token * EDGEKV_HVX_GQA_ATTN_ACTIVE_TOKEN_ELEMS + hkv * EDGEKV_HVX_GQA_HEAD_DIM;
        const HVX_Vector * values  = (const HVX_Vector *) row;
        const HVX_Vector   value0  = Q6_Vh_vshuff_Vh(values[0]);
        const HVX_Vector   value1  = Q6_Vh_vshuff_Vh(values[1]);
        const HVX_Vector   weight0 = hvx_vec_splat_f16(weights_f16[0][token]);
        const HVX_Vector   weight1 = hvx_vec_splat_f16(weights_f16[1][token]);
        out00                      = hvx_vec_mpyacc_f32_f16(out00, value0, weight0);
        out01                      = hvx_vec_mpyacc_f32_f16(out01, value1, weight0);
        out10                      = hvx_vec_mpyacc_f32_f16(out10, value0, weight1);
        out11                      = hvx_vec_mpyacc_f32_f16(out11, value1, weight1);
    }

    float        partial[EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT] __attribute__((aligned(128)));
    HVX_Vector * partial0 = (HVX_Vector *) partial;
    HVX_Vector * partial1 = (HVX_Vector *) (partial + EDGEKV_HVX_GQA_HEAD_DIM);
    partial0[0]           = Q6_V_lo_W(out00);
    partial0[1]           = Q6_V_hi_W(out00);
    partial0[2]           = Q6_V_lo_W(out01);
    partial0[3]           = Q6_V_hi_W(out01);
    partial1[0]           = Q6_V_lo_W(out10);
    partial1[1]           = Q6_V_hi_W(out10);
    partial1[2]           = Q6_V_lo_W(out11);
    partial1[3]           = Q6_V_hi_W(out11);
    edgekv_add_output(output, partial);
}

void edgekv_hvx_gqa_attn_hkv(const struct edgekv_hvx_gqa_attn_inputs * inputs,
                             int                                       layer_index,
                             int                                       hkv,
                             float                                     attention_scale,
                             void *                                    workspace,
                             float *                                   output,
                             struct edgekv_hvx_gqa_attn_profile *      profile) {
    const uint64_t total_start   = edgekv_profile_start(profile);
    float *        block_logits  = (float *) workspace;
    float *        active_logits = block_logits + EDGEKV_HVX_GQA_ATTN_BLOCK_LOGITS;
    float *        partial       = active_logits + EDGEKV_HVX_GQA_ATTN_ACTIVE_LOGITS;
    float *        hkv_output    = output + hkv * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM;
    const float *  query         = inputs->query + hkv * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM;
    const size_t   vh_head_offset =
        (size_t) layer_index * EDGEKV_HVX_GQA_ATTN_VH_LAYER_STRIDE + (size_t) hkv * EDGEKV_HVX_GQA_HEAD_DIM;

    hvx_splat_f32_a(hkv_output, 0.0f, EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT);

    uint64_t stage_start = edgekv_profile_start(profile);
    for (int block = 0; block < EDGEKV_HVX_GQA_ATTN_N_BLOCKS; ++block) {
        edgekv_hvx_gqa_block_k(query, inputs->k_u + block * EDGEKV_HVX_GQA_ATTN_U_BLOCK_BYTES,
                               inputs->k_vh + (size_t) block * EDGEKV_HVX_GQA_ATTN_VH_BLOCK_BYTES + vh_head_offset,
                               inputs->k_scale + block * EDGEKV_HVX_GQA_RANK, attention_scale,
                               block_logits + block * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE);
    }
    edgekv_profile_add(profile, profile ? &profile->factor_k : NULL, stage_start);

    stage_start = edgekv_profile_start(profile);
    edgekv_active_k_logits(query, inputs->active_k, hkv, attention_scale, active_logits);
    edgekv_profile_add(profile, profile ? &profile->active_k : NULL, stage_start);

    float sums[EDGEKV_HVX_GQA_HEADS];
    stage_start = edgekv_profile_start(profile);
    edgekv_mask_softmax(block_logits, active_logits, inputs->block_positions, inputs->active_positions,
                        inputs->query_position, sums);
    edgekv_profile_add(profile, profile ? &profile->mask_softmax : NULL, stage_start);

    stage_start = edgekv_profile_start(profile);
    for (int block = 0; block < EDGEKV_HVX_GQA_ATTN_N_BLOCKS; ++block) {
        edgekv_hvx_gqa_block_v(block_logits + block * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE,
                               inputs->v_u + block * EDGEKV_HVX_GQA_ATTN_U_BLOCK_BYTES,
                               inputs->v_vh + (size_t) block * EDGEKV_HVX_GQA_ATTN_VH_BLOCK_BYTES + vh_head_offset,
                               inputs->v_scale + block * EDGEKV_HVX_GQA_RANK, partial);
        edgekv_add_output(hkv_output, partial);
    }
    edgekv_profile_add(profile, profile ? &profile->factor_v : NULL, stage_start);

    stage_start = edgekv_profile_start(profile);
    edgekv_active_v_accumulate(active_logits, inputs->active_v, hkv, hkv_output);
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        const float inverse_sum = sums[head] > 0.0f ? 1.0f / sums[head] : 0.0f;
        hvx_scale_f32_aa((uint8_t *) (hkv_output + head * EDGEKV_HVX_GQA_HEAD_DIM),
                         (const uint8_t *) (hkv_output + head * EDGEKV_HVX_GQA_HEAD_DIM), EDGEKV_HVX_GQA_HEAD_DIM,
                         inverse_sum);
    }
    edgekv_profile_add(profile, profile ? &profile->active_v_normalize : NULL, stage_start);
    edgekv_profile_add(profile, profile ? &profile->total : NULL, total_start);
}
