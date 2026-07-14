// SPDX-License-Identifier: MIT

#include "HAP_perf.h"
#include "htp-ctx.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    N_BLOCKS                = 2,
    BLOCK_SIZE              = 16,
    RANK_K                  = 8,
    RANK_V                  = 4,
    GROUP_SIZE              = 2,
    LAYER_INDEX             = 1,
    N_HEAD_Q                = 4,
    N_HEAD_KV               = 2,
    HEAD_DIM_K              = 32,
    HEAD_DIM_V              = 16,
    RECENT_SIZE             = 8,
    U_STORAGE_BYTES         = 384,
    V_U_OFFSET_BYTES        = 256,
    VH_STORAGE_BYTES        = 2560,
    V_VH_OFFSET_BYTES       = 2048,
    SCALE_STORAGE_BYTES     = 256,
    V_SCALE_OFFSET_BYTES    = 128,
    RECENT_V_OFFSET_BYTES   = 1024,
    RECENT_POS_OFFSET_BYTES = 1536,
    RECENT_TOTAL_BYTES      = 1664,
};

static _Float16 q[N_HEAD_Q * HEAD_DIM_K] __attribute__((aligned(128)));
static int8_t   u_storage[U_STORAGE_BYTES] __attribute__((aligned(128)));
static int8_t   vh_storage[VH_STORAGE_BYTES] __attribute__((aligned(128)));
static uint8_t  scale_storage[SCALE_STORAGE_BYTES] __attribute__((aligned(128)));
static int32_t  block_positions[N_BLOCKS * BLOCK_SIZE] __attribute__((aligned(128)));
static uint8_t  recent_storage[RECENT_TOTAL_BYTES] __attribute__((aligned(128)));
static float    output[N_HEAD_Q * HEAD_DIM_V] __attribute__((aligned(128)));

static void initialize_inputs(void) {
    for (int i = 0; i < N_HEAD_Q * HEAD_DIM_K; ++i) {
        q[i] = (_Float16) ((float) ((i * 5 + 3) % 19 - 9) * 0.0078125f);
    }
    for (int i = 0; i < V_U_OFFSET_BYTES; ++i) {
        u_storage[i] = (int8_t) ((i * 3 + 1) % 11 - 5);
    }
    for (int i = 0; i < U_STORAGE_BYTES - V_U_OFFSET_BYTES; ++i) {
        u_storage[V_U_OFFSET_BYTES + i] = (int8_t) ((i * 5 + 2) % 9 - 4);
    }
    for (int i = 0; i < V_VH_OFFSET_BYTES; ++i) {
        vh_storage[i] = (int8_t) ((i * 7 + 3) % 13 - 6);
    }
    for (int i = 0; i < VH_STORAGE_BYTES - V_VH_OFFSET_BYTES; ++i) {
        vh_storage[V_VH_OFFSET_BYTES + i] = (int8_t) ((i * 2 + 4) % 11 - 5);
    }

    memset(scale_storage, 0, sizeof(scale_storage));
    float * k_scale = (float *) scale_storage;
    float * v_scale = (float *) (scale_storage + V_SCALE_OFFSET_BYTES);
    for (int i = 0; i < N_BLOCKS * RANK_K; ++i) {
        k_scale[i] = 0.0015f + 0.0001f * (float) (i % 5);
    }
    for (int i = 0; i < N_BLOCKS * RANK_V; ++i) {
        v_scale[i] = 0.0020f + 0.0002f * (float) (i % 3);
    }

    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; ++i) {
        block_positions[i] = i;
    }
    block_positions[N_BLOCKS * BLOCK_SIZE - 1] = -1;
    block_positions[N_BLOCKS * BLOCK_SIZE - 2] = -1;
    block_positions[N_BLOCKS * BLOCK_SIZE - 3] = -1;

    memset(recent_storage, 0, sizeof(recent_storage));
    _Float16 * recent_k         = (_Float16 *) recent_storage;
    _Float16 * recent_v         = (_Float16 *) (recent_storage + RECENT_V_OFFSET_BYTES);
    int32_t *  recent_positions = (int32_t *) (recent_storage + RECENT_POS_OFFSET_BYTES);
    for (int i = 0; i < RECENT_SIZE * N_HEAD_KV * HEAD_DIM_K; ++i) {
        recent_k[i] = (_Float16) ((float) ((i * 3 + 2) % 17 - 8) * 0.015625f);
    }
    for (int i = 0; i < RECENT_SIZE * N_HEAD_KV * HEAD_DIM_V; ++i) {
        recent_v[i] = (_Float16) ((float) ((i * 7 + 1) % 15 - 7) * 0.015625f);
    }
    for (int i = 0; i < RECENT_SIZE; ++i) {
        recent_positions[i] = 29 + i;
    }
    recent_positions[RECENT_SIZE] = 33;
    memset(output, 0, sizeof(output));
}

static void consume_value(float logit, const float * value, float * max_logit, float * sum_exp, float * accumulator) {
    const float next_max  = *max_logit > logit ? *max_logit : logit;
    const float old_scale = *sum_exp > 0.0f ? expf(*max_logit - next_max) : 0.0f;
    const float new_scale = expf(logit - next_max);
    *sum_exp              = *sum_exp * old_scale + new_scale;
    for (int d = 0; d < HEAD_DIM_V; ++d) {
        accumulator[d] = accumulator[d] * old_scale + value[d] * new_scale;
    }
    *max_logit = next_max;
}

static void reference_head(int hq, float * expected) {
    const int        hkv              = hq / (N_HEAD_Q / N_HEAD_KV);
    const int8_t *   k_u              = u_storage;
    const int8_t *   v_u              = u_storage + V_U_OFFSET_BYTES;
    const int8_t *   k_vh             = vh_storage;
    const int8_t *   v_vh             = vh_storage + V_VH_OFFSET_BYTES;
    const float *    k_scale          = (const float *) scale_storage;
    const float *    v_scale          = (const float *) (scale_storage + V_SCALE_OFFSET_BYTES);
    const _Float16 * recent_k         = (const _Float16 *) recent_storage;
    const _Float16 * recent_v         = (const _Float16 *) (recent_storage + RECENT_V_OFFSET_BYTES);
    const int32_t *  recent_positions = (const int32_t *) (recent_storage + RECENT_POS_OFFSET_BYTES);
    const int32_t    query_position   = recent_positions[RECENT_SIZE];
    const float      attention_scale  = 0.17677669529663687f;

    float max_logit               = -INFINITY;
    float sum_exp                 = 0.0f;
    float accumulator[HEAD_DIM_V] = { 0 };
    float value[HEAD_DIM_V];
    float q_rank[RANK_K];

    for (int block = 0; block < N_BLOCKS; ++block) {
        for (int rank = 0; rank < RANK_K; ++rank) {
            float     dot     = 0.0f;
            const int vh_base = ((((block * RANK_K + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + hkv) * HEAD_DIM_K);
            for (int d = 0; d < HEAD_DIM_K; ++d) {
                dot += (float) k_vh[vh_base + d] * (float) q[hq * HEAD_DIM_K + d];
            }
            q_rank[rank] = dot * k_scale[block * RANK_K + rank];
        }

        for (int token = 0; token < BLOCK_SIZE; ++token) {
            const int token_index = block * BLOCK_SIZE + token;
            if (block_positions[token_index] < 0 || block_positions[token_index] > query_position) {
                continue;
            }
            float logit = 0.0f;
            for (int rank = 0; rank < RANK_K; ++rank) {
                logit += (float) k_u[token_index * RANK_K + rank] * q_rank[rank];
            }
            logit *= attention_scale;

            for (int d = 0; d < HEAD_DIM_V; ++d) {
                float v = 0.0f;
                for (int rank = 0; rank < RANK_V; ++rank) {
                    const int vh_index =
                        ((((block * RANK_V + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + hkv) * HEAD_DIM_V + d);
                    v += (float) v_u[token_index * RANK_V + rank] * v_scale[block * RANK_V + rank] *
                         (float) v_vh[vh_index];
                }
                value[d] = v;
            }
            consume_value(logit, value, &max_logit, &sum_exp, accumulator);
        }
    }

    for (int token = 0; token < RECENT_SIZE; ++token) {
        if (recent_positions[token] < 0 || recent_positions[token] > query_position) {
            continue;
        }
        float     logit  = 0.0f;
        const int k_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
        for (int d = 0; d < HEAD_DIM_K; ++d) {
            logit += (float) recent_k[k_base + d] * (float) q[hq * HEAD_DIM_K + d];
        }
        logit *= attention_scale;
        const int v_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_V;
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            value[d] = (float) recent_v[v_base + d];
        }
        consume_value(logit, value, &max_logit, &sum_exp, accumulator);
    }

    for (int d = 0; d < HEAD_DIM_V; ++d) {
        expected[d] = sum_exp > 0.0f ? accumulator[d] / sum_exp : 0.0f;
    }
}

static float absolute_error(float actual, float expected) {
    const float error = actual - expected;
    return error < 0.0f ? -error : error;
}

static int verify_output(float * max_error) {
    *max_error = 0.0f;
    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        float expected[HEAD_DIM_V];
        reference_head(hq, expected);
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            const float error = absolute_error(output[hq * HEAD_DIM_V + d], expected[d]);
            if (error > *max_error) {
                *max_error = error;
            }
        }
    }
    return *max_error <= 0.0005f;
}

static struct htp_tensor make_tensor(void * data, uint32_t size, uint16_t type) {
    struct htp_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.data = (uint32_t) (uintptr_t) data;
    tensor.size = size;
    tensor.type = type;
    return tensor;
}

int main(void) {
    static const int32_t params[HTP_EDGEKV_ATTN_PARAM_COUNT] = {
        N_BLOCKS,
        BLOCK_SIZE,
        RANK_K,
        RANK_V,
        GROUP_SIZE,
        LAYER_INDEX,
        N_HEAD_Q,
        N_HEAD_KV,
        HEAD_DIM_K,
        HEAD_DIM_V,
        RECENT_SIZE,
        RECENT_V_OFFSET_BYTES,
        RECENT_POS_OFFSET_BYTES,
        RECENT_TOTAL_BYTES,
        0x3e3504f3,
    };

    initialize_inputs();
    struct htp_tensor q_tensor        = make_tensor(q, sizeof(q), HTP_TYPE_F16);
    struct htp_tensor u_tensor        = make_tensor(u_storage, sizeof(u_storage), HTP_TYPE_I8);
    struct htp_tensor vh_tensor       = make_tensor(vh_storage, sizeof(vh_storage), HTP_TYPE_I8);
    struct htp_tensor scale_tensor    = make_tensor(scale_storage, sizeof(scale_storage), HTP_TYPE_F32);
    struct htp_tensor position_tensor = make_tensor(block_positions, sizeof(block_positions), HTP_TYPE_I32);
    struct htp_tensor recent_tensor   = make_tensor(recent_storage, sizeof(recent_storage), HTP_TYPE_I8);
    struct htp_tensor output_tensor   = make_tensor(output, sizeof(output), HTP_TYPE_F32);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.op = HTP_OP_EDGEKV_ATTN_DECODE;
    memcpy(octx.op_params, params, sizeof(params));
    octx.src[0] = &q_tensor;
    octx.src[1] = &u_tensor;
    octx.src[2] = &vh_tensor;
    octx.src[3] = &scale_tensor;
    octx.src[4] = &position_tensor;
    octx.src[5] = &recent_tensor;
    octx.dst    = &output_tensor;

    int status = op_edgekv_attn_decode(&octx);
    if (status != HTP_STATUS_OK) {
        printf("FAIL: EdgeKV direct-attention adapter returned status %d\n", status);
        return 1;
    }

    float max_error;
    if (!verify_output(&max_error)) {
        printf("FAIL: EdgeKV direct-attention numerical mismatch max_error=%g\n", (double) max_error);
        return 2;
    }

    enum { ITERATIONS = 3 };

    const uint64_t start = HAP_perf_get_pcycles();
    for (int i = 0; i < ITERATIONS; ++i) {
        status = op_edgekv_attn_decode(&octx);
        if (status != HTP_STATUS_OK) {
            printf("FAIL: benchmark iteration %d returned status %d\n", i, status);
            return 3;
        }
    }
    const uint64_t end         = HAP_perf_get_pcycles();
    const uint64_t avg_pcycles = (end - start) / ITERATIONS;

    printf("PASS: EdgeKV direct-attention adapter max_error=%g avg_pcycles=%llu\n", (double) max_error,
           (unsigned long long) avg_pcycles);
    return 0;
}
