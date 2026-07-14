// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-block.h"

#include "hvx-utils.h"

#include <stdint.h>

static inline HVX_Vector edgekv_i8x32_to_f16(HVX_Vector packed_rows, int row) {
    const HVX_Vector rotated = row == 0 ? packed_rows : Q6_V_vror_VR(packed_rows, row * 32);
    const HVX_Vector words   = Q6_V_lo_W(Q6_Wh_vunpack_Vb(rotated));
    const HVX_Vector values  = Q6_Vhf_equals_Vh(words);
    return Q6_V_vand_QV(Q6_Q_vsetq_R(64), values);
}

static inline void edgekv_i8x128_to_f16(const int8_t * src, HVX_Vector * lo, HVX_Vector * hi) {
    const HVX_VectorPair words = Q6_Wh_vunpack_Vb(hvx_vmemu(src));
    *lo                        = Q6_Vhf_equals_Vh(Q6_V_lo_W(words));
    *hi                        = Q6_Vhf_equals_Vh(Q6_V_hi_W(words));
}

static inline void edgekv_query_to_f16(const float * src, HVX_Vector dst[2]) {
    const HVX_Vector * vectors = (const HVX_Vector *) src;
    dst[0]                     = hvx_vec_f32_to_f16(vectors[0], vectors[1]);
    dst[1]                     = hvx_vec_f32_to_f16(vectors[2], vectors[3]);
}

static inline void edgekv_dot_2x128_f16_unreduced(HVX_Vector   q0[2],
                                                  HVX_Vector   q1[2],
                                                  HVX_Vector   x0,
                                                  HVX_Vector   x1,
                                                  HVX_Vector * sum0,
                                                  HVX_Vector * sum1) {
    HVX_VectorPair acc0 = Q6_W_vzero();
    HVX_VectorPair acc1 = Q6_W_vzero();
    acc0                = hvx_vec_mpyacc_f32_f16(acc0, q0[0], x0);
    acc0                = hvx_vec_mpyacc_f32_f16(acc0, q0[1], x1);
    acc1                = hvx_vec_mpyacc_f32_f16(acc1, q1[0], x0);
    acc1                = hvx_vec_mpyacc_f32_f16(acc1, q1[1], x1);

    *sum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0)));
    *sum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1)));
}

static inline HVX_Vector edgekv_dot_4x32_f16(HVX_Vector q, HVX_Vector x0, HVX_Vector x1, HVX_Vector x2, HVX_Vector x3) {
    HVX_VectorPair acc0 = hvx_vec_mpyacc_f32_f16(Q6_W_vzero(), q, x0);
    HVX_VectorPair acc1 = hvx_vec_mpyacc_f32_f16(Q6_W_vzero(), q, x1);
    HVX_VectorPair acc2 = hvx_vec_mpyacc_f32_f16(Q6_W_vzero(), q, x2);
    HVX_VectorPair acc3 = hvx_vec_mpyacc_f32_f16(Q6_W_vzero(), q, x3);

    HVX_Vector_x4 sums = {
        .v = {
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0))),
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1))),
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc2), Q6_V_hi_W(acc2))),
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(acc3), Q6_V_hi_W(acc3))),
        },
    };
    return hvx_vec_reduce_sum_f32x4(sums);
}

void edgekv_hvx_gqa_block_k(const float *  q,
                            const int8_t * k_u,
                            const int8_t * k_vh,
                            const float *  k_scale,
                            float          attention_scale,
                            float *        logits) {
    HVX_Vector q0[2];
    HVX_Vector q1[2];
    edgekv_query_to_f16(q, q0);
    edgekv_query_to_f16(q + EDGEKV_HVX_GQA_HEAD_DIM, q1);

    float q_rank_f32[EDGEKV_HVX_GQA_HEADS][EDGEKV_HVX_GQA_RANK] __attribute__((aligned(128)));
    for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; rank += 4) {
        HVX_Vector head0_sums[4];
        HVX_Vector head1_sums[4];
#pragma unroll(4)
        for (int row = 0; row < 4; ++row) {
            HVX_Vector vh0;
            HVX_Vector vh1;
            edgekv_i8x128_to_f16(k_vh + (rank + row) * EDGEKV_HVX_GQA_VH_RANK_STRIDE, &vh0, &vh1);
            edgekv_dot_2x128_f16_unreduced(q0, q1, vh0, vh1, &head0_sums[row], &head1_sums[row]);
        }

        const HVX_Vector_x4 head0 = {
            .v = { head0_sums[0], head0_sums[1], head0_sums[2], head0_sums[3] },
        };
        const HVX_Vector_x4 head1 = {
            .v = { head1_sums[0], head1_sums[1], head1_sums[2], head1_sums[3] },
        };
        hvx_vec_store_u(q_rank_f32[0] + rank, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(head0));
        hvx_vec_store_u(q_rank_f32[1] + rank, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(head1));
    }

    const HVX_Vector rank_scale = *(const HVX_Vector *) k_scale;
    HVX_Vector       rank0_f32  = *(const HVX_Vector *) q_rank_f32[0];
    HVX_Vector       rank1_f32  = *(const HVX_Vector *) q_rank_f32[1];
    rank0_f32                   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(rank0_f32, rank_scale));
    rank1_f32                   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(rank1_f32, rank_scale));
    const HVX_Vector rank0      = hvx_vec_f32_to_f16(rank0_f32, Q6_V_vzero());
    const HVX_Vector rank1      = hvx_vec_f32_to_f16(rank1_f32, Q6_V_vzero());
    const HVX_Vector scale      = hvx_vec_splat_f32(attention_scale);
    for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; token += 4) {
        const HVX_Vector packed = hvx_vmemu(k_u + token * EDGEKV_HVX_GQA_RANK);
        const HVX_Vector u0     = edgekv_i8x32_to_f16(packed, 0);
        const HVX_Vector u1     = edgekv_i8x32_to_f16(packed, 1);
        const HVX_Vector u2     = edgekv_i8x32_to_f16(packed, 2);
        const HVX_Vector u3     = edgekv_i8x32_to_f16(packed, 3);

        HVX_Vector sums0 = edgekv_dot_4x32_f16(rank0, u0, u1, u2, u3);
        HVX_Vector sums1 = edgekv_dot_4x32_f16(rank1, u0, u1, u2, u3);
        sums0            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sums0, scale));
        sums1            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sums1, scale));
        hvx_vec_store_u(logits + token, 4 * sizeof(float), sums0);
        hvx_vec_store_u(logits + EDGEKV_HVX_GQA_BLOCK_SIZE + token, 4 * sizeof(float), sums1);
    }
}

void edgekv_hvx_gqa_block_v(const float *  weights,
                            const int8_t * v_u,
                            const int8_t * v_vh,
                            const float *  v_scale,
                            float *        output) {
    const HVX_Vector * weight_vectors = (const HVX_Vector *) weights;
    __fp16 weights_f16[EDGEKV_HVX_GQA_HEADS][EDGEKV_HVX_GQA_BLOCK_SIZE] __attribute__((aligned(128)));
    *(HVX_Vector *) weights_f16[0] = hvx_vec_f32_to_f16(weight_vectors[0], weight_vectors[1]);
    *(HVX_Vector *) weights_f16[1] = hvx_vec_f32_to_f16(weight_vectors[2], weight_vectors[3]);

    HVX_VectorPair rank_acc0 = Q6_W_vzero();
    HVX_VectorPair rank_acc1 = Q6_W_vzero();
    for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; token += 4) {
        const HVX_Vector packed = hvx_vmemu(v_u + token * EDGEKV_HVX_GQA_RANK);
        for (int row = 0; row < 4; ++row) {
            const HVX_Vector values = Q6_Vh_vshuff_Vh(edgekv_i8x32_to_f16(packed, row));
            const HVX_Vector w0     = hvx_vec_splat_f16(weights_f16[0][token + row]);
            const HVX_Vector w1     = hvx_vec_splat_f16(weights_f16[1][token + row]);
            rank_acc0               = hvx_vec_mpyacc_f32_f16(rank_acc0, values, w0);
            rank_acc1               = hvx_vec_mpyacc_f32_f16(rank_acc1, values, w1);
        }
    }

    const HVX_Vector scales = hvx_vec_f32_to_f16(*(const HVX_Vector *) v_scale, Q6_V_vzero());
    HVX_Vector       rank0  = hvx_vec_f32_to_f16(Q6_V_lo_W(rank_acc0), Q6_V_hi_W(rank_acc0));
    HVX_Vector       rank1  = hvx_vec_f32_to_f16(Q6_V_lo_W(rank_acc1), Q6_V_hi_W(rank_acc1));
    rank0                   = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(rank0, scales));
    rank1                   = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(rank1, scales));

    __fp16 rank_scaled[EDGEKV_HVX_GQA_HEADS][64] __attribute__((aligned(128)));
    *(HVX_Vector *) rank_scaled[0] = rank0;
    *(HVX_Vector *) rank_scaled[1] = rank1;

    HVX_VectorPair out00 = Q6_W_vzero();
    HVX_VectorPair out01 = Q6_W_vzero();
    HVX_VectorPair out10 = Q6_W_vzero();
    HVX_VectorPair out11 = Q6_W_vzero();
    for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
        HVX_Vector vh0;
        HVX_Vector vh1;
        edgekv_i8x128_to_f16(v_vh + rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE, &vh0, &vh1);
        vh0 = Q6_Vh_vshuff_Vh(vh0);
        vh1 = Q6_Vh_vshuff_Vh(vh1);

        const HVX_Vector coeff0 = hvx_vec_splat_f16(rank_scaled[0][rank]);
        const HVX_Vector coeff1 = hvx_vec_splat_f16(rank_scaled[1][rank]);
        out00                   = hvx_vec_mpyacc_f32_f16(out00, vh0, coeff0);
        out01                   = hvx_vec_mpyacc_f32_f16(out01, vh1, coeff0);
        out10                   = hvx_vec_mpyacc_f32_f16(out10, vh0, coeff1);
        out11                   = hvx_vec_mpyacc_f32_f16(out11, vh1, coeff1);
    }

    HVX_Vector * out0 = (HVX_Vector *) output;
    HVX_Vector * out1 = (HVX_Vector *) (output + EDGEKV_HVX_GQA_HEAD_DIM);
    out0[0]           = Q6_V_lo_W(out00);
    out0[1]           = Q6_V_hi_W(out00);
    out0[2]           = Q6_V_lo_W(out01);
    out0[3]           = Q6_V_hi_W(out01);
    out1[0]           = Q6_V_lo_W(out10);
    out1[1]           = Q6_V_hi_W(out10);
    out1[2]           = Q6_V_lo_W(out11);
    out1[3]           = Q6_V_hi_W(out11);
}

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
                          float *        output) {
    edgekv_hvx_gqa_block_k(q, k_u, k_vh, k_scale, attention_scale, logits);
    edgekv_hvx_gqa_block_v(weights, v_u, v_vh, v_scale, output);
}
