// SPDX-License-Identifier: MIT

#include "HAP_perf.h"
#include "htp-ctx.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    N_BLOCKS            = 2,
    BLOCK_SIZE          = 16,
    RANK_K              = 8,
    RANK_V              = 4,
    GROUP_SIZE          = 2,
    LAYER_INDEX         = 1,
    N_HEAD_KV           = 2,
    HEAD_DIM_K          = 32,
    HEAD_DIM_V          = 16,
    K_U_BYTES           = 256,
    V_U_OFFSET_BYTES    = 256,
    U_STORAGE_BYTES     = 384,
    K_VH_BYTES          = 2048,
    V_VH_OFFSET_BYTES   = 2048,
    VH_STORAGE_BYTES    = 2560,
    V_SCALE_OFFSET      = 128,
    SCALE_STORAGE_BYTES = 256,
    DENSE_V_OFFSET      = 4096,
    DENSE_TOTAL_BYTES   = 6144,
};

static int8_t  u_storage[U_STORAGE_BYTES] __attribute__((aligned(128)));
static int8_t  vh_storage[VH_STORAGE_BYTES] __attribute__((aligned(128)));
static uint8_t scale_storage[SCALE_STORAGE_BYTES] __attribute__((aligned(128)));
static int32_t block_positions[N_BLOCKS * BLOCK_SIZE] __attribute__((aligned(128)));
static uint8_t dense_storage[DENSE_TOTAL_BYTES] __attribute__((aligned(128)));

static void initialize_inputs(void) {
    for (int i = 0; i < K_U_BYTES; ++i) {
        u_storage[i] = (int8_t) ((i * 3 + 1) % 9 - 4);
    }
    for (int i = 0; i < U_STORAGE_BYTES - V_U_OFFSET_BYTES; ++i) {
        u_storage[V_U_OFFSET_BYTES + i] = (int8_t) ((i * 5 + 2) % 7 - 3);
    }
    for (int i = 0; i < K_VH_BYTES; ++i) {
        vh_storage[i] = (int8_t) ((i * 7 + 3) % 11 - 5);
    }
    for (int i = 0; i < VH_STORAGE_BYTES - V_VH_OFFSET_BYTES; ++i) {
        vh_storage[V_VH_OFFSET_BYTES + i] = (int8_t) ((i * 2 + 4) % 9 - 4);
    }

    memset(scale_storage, 0, sizeof(scale_storage));
    float * k_scale = (float *) scale_storage;
    float * v_scale = (float *) (scale_storage + V_SCALE_OFFSET);
    for (int i = 0; i < N_BLOCKS * RANK_K; ++i) {
        k_scale[i] = 0.0025f + 0.0001f * (float) (i % 5);
    }
    for (int i = 0; i < N_BLOCKS * RANK_V; ++i) {
        v_scale[i] = 0.0030f + 0.0002f * (float) (i % 3);
    }

    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; ++i) {
        block_positions[i] = i;
    }
    block_positions[N_BLOCKS * BLOCK_SIZE - 1] = -1;
    block_positions[N_BLOCKS * BLOCK_SIZE - 2] = -1;
    block_positions[N_BLOCKS * BLOCK_SIZE - 3] = -1;
    memset(dense_storage, 0xa5, sizeof(dense_storage));
}

static float reference_k(int h, int block, int token, int d) {
    const float * k_scale = (const float *) scale_storage;
    float         acc     = 0.0f;
    for (int rank = 0; rank < RANK_K; ++rank) {
        const int u_index  = (block * BLOCK_SIZE + token) * RANK_K + rank;
        const int vh_index = ((((block * RANK_K + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + h) * HEAD_DIM_K + d);
        acc += (float) u_storage[u_index] * k_scale[block * RANK_K + rank] * (float) vh_storage[vh_index];
    }
    return block_positions[block * BLOCK_SIZE + token] >= 0 ? acc : 0.0f;
}

static float reference_v(int h, int block, int token, int d) {
    const int8_t * v_u     = u_storage + V_U_OFFSET_BYTES;
    const int8_t * v_vh    = vh_storage + V_VH_OFFSET_BYTES;
    const float *  v_scale = (const float *) (scale_storage + V_SCALE_OFFSET);
    float          acc     = 0.0f;
    for (int rank = 0; rank < RANK_V; ++rank) {
        const int u_index  = (block * BLOCK_SIZE + token) * RANK_V + rank;
        const int vh_index = ((((block * RANK_V + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + h) * HEAD_DIM_V + d);
        acc += (float) v_u[u_index] * v_scale[block * RANK_V + rank] * (float) v_vh[vh_index];
    }
    return block_positions[block * BLOCK_SIZE + token] >= 0 ? acc : 0.0f;
}

static float absolute_error(_Float16 actual, float expected) {
    const _Float16 rounded_expected = (_Float16) expected;
    float          error            = (float) actual - (float) rounded_expected;
    return error < 0.0f ? -error : error;
}

static int verify_output(float * max_k_error, float * max_v_error) {
    const _Float16 * dense_k = (const _Float16 *) dense_storage;
    const _Float16 * dense_v = (const _Float16 *) (dense_storage + DENSE_V_OFFSET);
    *max_k_error             = 0.0f;
    *max_v_error             = 0.0f;

    for (int h = 0; h < N_HEAD_KV; ++h) {
        for (int block = 0; block < N_BLOCKS; ++block) {
            for (int token = 0; token < BLOCK_SIZE; ++token) {
                for (int d = 0; d < HEAD_DIM_K; ++d) {
                    const int   index = ((h * N_BLOCKS + block) * BLOCK_SIZE + token) * HEAD_DIM_K + d;
                    const float error = absolute_error(dense_k[index], reference_k(h, block, token, d));
                    if (error > *max_k_error) {
                        *max_k_error = error;
                    }
                }
                for (int d = 0; d < HEAD_DIM_V; ++d) {
                    const int   index = ((h * N_BLOCKS + block) * BLOCK_SIZE + token) * HEAD_DIM_V + d;
                    const float error = absolute_error(dense_v[index], reference_v(h, block, token, d));
                    if (error > *max_v_error) {
                        *max_v_error = error;
                    }
                }
            }
        }
    }
    return *max_k_error <= 0.0005f && *max_v_error <= 0.0005f;
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
    static const int32_t params[HTP_EDGEKV_PARAM_COUNT] = {
        N_BLOCKS,       BLOCK_SIZE,        RANK_K,     RANK_V,           GROUP_SIZE,        LAYER_INDEX,
        N_HEAD_KV,      HEAD_DIM_K,        HEAD_DIM_V, V_U_OFFSET_BYTES, V_VH_OFFSET_BYTES, V_SCALE_OFFSET,
        DENSE_V_OFFSET, DENSE_TOTAL_BYTES,
    };

    initialize_inputs();
    struct htp_tensor u         = make_tensor(u_storage, sizeof(u_storage), HTP_TYPE_I8);
    struct htp_tensor vh        = make_tensor(vh_storage, sizeof(vh_storage), HTP_TYPE_I8);
    struct htp_tensor scale     = make_tensor(scale_storage, sizeof(scale_storage), HTP_TYPE_F32);
    struct htp_tensor positions = make_tensor(block_positions, sizeof(block_positions), HTP_TYPE_I32);
    struct htp_tensor dense     = make_tensor(dense_storage, sizeof(dense_storage), HTP_TYPE_F16);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.op = HTP_OP_EDGEKV_RECONSTRUCT;
    memcpy(octx.op_params, params, sizeof(params));
    octx.src[0] = &u;
    octx.src[1] = &vh;
    octx.src[2] = &scale;
    octx.src[3] = &positions;
    octx.dst    = &dense;

    int status = op_edgekv_reconstruct(&octx);
    if (status != HTP_STATUS_OK) {
        printf("FAIL: EdgeKV HTP adapter returned status %d\n", status);
        return 1;
    }

    float max_k_error;
    float max_v_error;
    if (!verify_output(&max_k_error, &max_v_error)) {
        printf("FAIL: numerical mismatch max_k_error=%g max_v_error=%g\n", (double) max_k_error, (double) max_v_error);
        return 2;
    }

    enum { ITERATIONS = 10 };

    const uint64_t start = HAP_perf_get_pcycles();
    for (int i = 0; i < ITERATIONS; ++i) {
        status = op_edgekv_reconstruct(&octx);
        if (status != HTP_STATUS_OK) {
            printf("FAIL: benchmark iteration %d returned status %d\n", i, status);
            return 3;
        }
    }
    const uint64_t end         = HAP_perf_get_pcycles();
    const uint64_t avg_pcycles = (end - start) / ITERATIONS;

    printf("PASS: EdgeKV HTP adapter max_k_error=%g max_v_error=%g avg_pcycles=%llu\n", (double) max_k_error,
           (double) max_v_error, (unsigned long long) avg_pcycles);
    return 0;
}
