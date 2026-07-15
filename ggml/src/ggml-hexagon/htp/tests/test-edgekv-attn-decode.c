// SPDX-License-Identifier: MIT

#include "htp-ctx.h"

#ifdef HTP_EDGEKV_HVX_GQA
#include "edgekv-hvx-gqa-attn.h"
#endif

#include "HAP_compute_res.h"
#include "HAP_perf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    N_BLOCKS                         = 16,
    BLOCK_SIZE                       = 64,
    RANK_K                           = 32,
    RANK_V                           = 32,
    GROUP_SIZE                       = 4,
    N_HEAD_Q                         = 16,
    N_HEAD_KV                        = 8,
    HEAD_DIM_K                       = 128,
    HEAD_DIM_V                       = 128,
    RECENT_SIZE                      = 128,
    K_U_BYTES                        = 32768,
    V_U_OFFSET_BYTES                 = 32768,
    U_STORAGE_BYTES                  = 65536,
    K_VH_BYTES                       = 2097152,
    V_VH_OFFSET_BYTES                = 2097152,
    VH_STORAGE_BYTES                 = 4194304,
    METADATA_V_SCALE_OFFSET_BYTES    = 2048,
    METADATA_BLOCK_POS_OFFSET_BYTES  = 4096,
    METADATA_RECENT_POS_OFFSET_BYTES = 8192,
    METADATA_TOTAL_BYTES             = 8832,
    KV_SIZE                          = N_BLOCKS * BLOCK_SIZE + RECENT_SIZE,
    N_THREADS                        = 4,
    PERFORMANCE_SAMPLES              = 3,
    ACTIVE_K_BYTES                   = RECENT_SIZE * N_HEAD_KV * HEAD_DIM_K * 2,
    ACTIVE_V_BYTES                   = RECENT_SIZE * N_HEAD_KV * HEAD_DIM_V * 2,
    DENSE_K_BYTES                    = N_HEAD_KV * KV_SIZE * HEAD_DIM_K * 2,
    DENSE_V_BYTES                    = N_HEAD_KV * KV_SIZE * HEAD_DIM_V * 2,
    DENSE_MASK_BYTES                 = KV_SIZE * 2,
};

#ifdef HTP_EDGEKV_HVX_GQA
#define EDGEKV_DIRECT_KERNEL_NAME "direct_hvx"
#define EDGEKV_TEST_MILESTONE     "C4C3"
#else
#define EDGEKV_DIRECT_KERNEL_NAME "direct_aot"
#define EDGEKV_TEST_MILESTONE     "C4B0"
#endif

static float    q[N_HEAD_Q * HEAD_DIM_K] __attribute__((aligned(128)));
static float    output[N_HEAD_Q * HEAD_DIM_V] __attribute__((aligned(128)));
static float    dense_output[N_HEAD_Q * HEAD_DIM_V] __attribute__((aligned(128)));

static int8_t *   u_storage;
static int8_t *   vh_storage;
static uint8_t *  metadata;
static _Float16 * active_k;
static _Float16 * active_v;
static _Float16 * dense_k;
static _Float16 * dense_v;
static _Float16 * dense_mask;

static void release_input_buffers(void) {
    free(dense_mask);
    free(dense_v);
    free(dense_k);
    free(active_v);
    free(active_k);
    free(metadata);
    free(vh_storage);
    free(u_storage);
}

static int allocate_input_buffers(void) {
    u_storage  = (int8_t *) memalign(128, U_STORAGE_BYTES);
    vh_storage = (int8_t *) memalign(128, VH_STORAGE_BYTES);
    metadata   = (uint8_t *) memalign(128, METADATA_TOTAL_BYTES);
    active_k   = (_Float16 *) memalign(128, ACTIVE_K_BYTES);
    active_v   = (_Float16 *) memalign(128, ACTIVE_V_BYTES);
    dense_k    = (_Float16 *) memalign(128, DENSE_K_BYTES);
    dense_v    = (_Float16 *) memalign(128, DENSE_V_BYTES);
    dense_mask = (_Float16 *) memalign(128, DENSE_MASK_BYTES);
    if (!u_storage || !vh_storage || !metadata || !active_k || !active_v || !dense_k || !dense_v || !dense_mask) {
        release_input_buffers();
        return 0;
    }
    return 1;
}

static int k_vh_index(int block, int rank, int layer, int hkv, int dim) {
    return ((((block * RANK_K + rank) * GROUP_SIZE + layer) * N_HEAD_KV + hkv) * HEAD_DIM_K + dim);
}

static int v_vh_index(int block, int rank, int layer, int hkv, int dim) {
    return ((((block * RANK_V + rank) * GROUP_SIZE + layer) * N_HEAD_KV + hkv) * HEAD_DIM_V + dim);
}

static void initialize_inputs(void) {
    memset(q, 0, sizeof(q));
    memset(u_storage, 0, U_STORAGE_BYTES);
    memset(vh_storage, 0, VH_STORAGE_BYTES);
    memset(metadata, 0, METADATA_TOTAL_BYTES);
    memset(active_k, 0, ACTIVE_K_BYTES);
    memset(active_v, 0, ACTIVE_V_BYTES);
    memset(output, 0, sizeof(output));

    float *   k_scale         = (float *) metadata;
    float *   v_scale         = (float *) (metadata + METADATA_V_SCALE_OFFSET_BYTES);
    int32_t * block_positions = (int32_t *) (metadata + METADATA_BLOCK_POS_OFFSET_BYTES);
    int32_t * active_positions = (int32_t *) (metadata + METADATA_RECENT_POS_OFFSET_BYTES);

    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        q[hq * HEAD_DIM_K] = 0.5f + 0.015625f * (float) hq;
    }
    for (int block = 0; block < N_BLOCKS; ++block) {
        k_scale[block * RANK_K] = 0.01f + 0.0001f * (float) block;
        v_scale[block * RANK_V] = 0.02f + 0.0002f * (float) block;
        for (int token = 0; token < BLOCK_SIZE; ++token) {
            const int token_index = block * BLOCK_SIZE + token;
            u_storage[token_index * RANK_K] = (int8_t) ((token_index % 5) - 2);
            u_storage[V_U_OFFSET_BYTES + token_index * RANK_V] = (int8_t) ((token_index % 7) - 3);
            block_positions[token_index] = token_index;
        }
        for (int layer = 0; layer < GROUP_SIZE; ++layer) {
            for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
                vh_storage[k_vh_index(block, 0, layer, hkv, 0)] =
                    (int8_t) (1 + layer + (hkv % 3));
                for (int d = 0; d < HEAD_DIM_V; ++d) {
                    vh_storage[V_VH_OFFSET_BYTES + v_vh_index(block, 0, layer, hkv, d)] =
                        (int8_t) ((layer + 1) * (1 + d % 3) + (hkv % 2));
                }
            }
        }
    }
    for (int i = N_BLOCKS * BLOCK_SIZE - 7; i < N_BLOCKS * BLOCK_SIZE; ++i) {
        block_positions[i] = -1;
    }

    for (int token = 0; token < RECENT_SIZE; ++token) {
        active_positions[token] = N_BLOCKS * BLOCK_SIZE + token;
        for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
            const int base = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
            active_k[base] = (_Float16) (0.01f * (float) ((token % 7) - 3) + 0.001f * (float) hkv);
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                active_v[base + d] =
                    (_Float16) (0.005f * (float) ((token + d) % 11 - 5) + 0.0005f * (float) hkv);
            }
        }
    }
    active_positions[RECENT_SIZE] = N_BLOCKS * BLOCK_SIZE + RECENT_SIZE - 6;
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

static void reference_head(int hq, int layer, float * expected) {
    const int       hkv              = hq / (N_HEAD_Q / N_HEAD_KV);
    const int8_t *  k_u              = u_storage;
    const int8_t *  v_u              = u_storage + V_U_OFFSET_BYTES;
    const int8_t *  k_vh             = vh_storage;
    const int8_t *  v_vh             = vh_storage + V_VH_OFFSET_BYTES;
    const float *   k_scale          = (const float *) metadata;
    const float *   v_scale          = (const float *) (metadata + METADATA_V_SCALE_OFFSET_BYTES);
    const int32_t * block_positions  = (const int32_t *) (metadata + METADATA_BLOCK_POS_OFFSET_BYTES);
    const int32_t * active_positions = (const int32_t *) (metadata + METADATA_RECENT_POS_OFFSET_BYTES);
    const int32_t   query_position   = active_positions[RECENT_SIZE];
    const float     attention_scale  = 0.08838834764831845f;

    float max_logit               = -INFINITY;
    float sum_exp                 = 0.0f;
    float accumulator[HEAD_DIM_V] = { 0 };
    float value[HEAD_DIM_V];

    for (int block = 0; block < N_BLOCKS; ++block) {
        const float q_rank = (float) k_vh[k_vh_index(block, 0, layer, hkv, 0)] * q[hq * HEAD_DIM_K] *
                             k_scale[block * RANK_K];
        for (int token = 0; token < BLOCK_SIZE; ++token) {
            const int token_index = block * BLOCK_SIZE + token;
            if (block_positions[token_index] < 0 || block_positions[token_index] > query_position) {
                continue;
            }
            const float logit = (float) k_u[token_index * RANK_K] * q_rank * attention_scale;
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                value[d] = (float) v_u[token_index * RANK_V] * v_scale[block * RANK_V] *
                           (float) v_vh[v_vh_index(block, 0, layer, hkv, d)];
            }
            consume_value(logit, value, &max_logit, &sum_exp, accumulator);
        }
    }

    for (int token = 0; token < RECENT_SIZE; ++token) {
        if (active_positions[token] < 0 || active_positions[token] > query_position) {
            continue;
        }
        const int   base  = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
        const float logit = (float) active_k[base] * q[hq * HEAD_DIM_K] * attention_scale;
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            value[d] = (float) active_v[base + d];
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

static int verify_output(int layer, float * max_error) {
    *max_error = 0.0f;
    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        float expected[HEAD_DIM_V];
        reference_head(hq, layer, expected);
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            const float error = absolute_error(output[hq * HEAD_DIM_V + d], expected[d]);
            if (error > *max_error) {
                *max_error = error;
            }
        }
    }
    return *max_error <= 0.002f;
}

static void materialize_dense_kv(int layer) {
    const int8_t *  k_u              = u_storage;
    const int8_t *  v_u              = u_storage + V_U_OFFSET_BYTES;
    const int8_t *  k_vh             = vh_storage;
    const int8_t *  v_vh             = vh_storage + V_VH_OFFSET_BYTES;
    const float *   k_scale          = (const float *) metadata;
    const float *   v_scale          = (const float *) (metadata + METADATA_V_SCALE_OFFSET_BYTES);
    const int32_t * block_positions  = (const int32_t *) (metadata + METADATA_BLOCK_POS_OFFSET_BYTES);
    const int32_t * active_positions = (const int32_t *) (metadata + METADATA_RECENT_POS_OFFSET_BYTES);
    const int32_t   query_position   = active_positions[RECENT_SIZE];

    for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
        for (int token_index = 0; token_index < N_BLOCKS * BLOCK_SIZE; ++token_index) {
            const int block = token_index / BLOCK_SIZE;
            const int base  = (hkv * KV_SIZE + token_index) * HEAD_DIM_K;
            for (int d = 0; d < HEAD_DIM_K; ++d) {
                const float value = (float) k_u[token_index * RANK_K] * k_scale[block * RANK_K] *
                                    (float) k_vh[k_vh_index(block, 0, layer, hkv, d)];
                dense_k[base + d] = (_Float16) value;
            }
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                const float value = (float) v_u[token_index * RANK_V] * v_scale[block * RANK_V] *
                                    (float) v_vh[v_vh_index(block, 0, layer, hkv, d)];
                dense_v[base + d] = (_Float16) value;
            }
        }
        for (int token = 0; token < RECENT_SIZE; ++token) {
            const int dense_base  = (hkv * KV_SIZE + N_BLOCKS * BLOCK_SIZE + token) * HEAD_DIM_K;
            const int active_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
            memcpy(dense_k + dense_base, active_k + active_base, HEAD_DIM_K * sizeof(_Float16));
            memcpy(dense_v + dense_base, active_v + active_base, HEAD_DIM_V * sizeof(_Float16));
        }
    }

    for (int token = 0; token < N_BLOCKS * BLOCK_SIZE; ++token) {
        dense_mask[token] = (block_positions[token] >= 0 && block_positions[token] <= query_position) ?
                                (_Float16) 0.0f : (_Float16) -INFINITY;
    }
    for (int token = 0; token < RECENT_SIZE; ++token) {
        dense_mask[N_BLOCKS * BLOCK_SIZE + token] =
            (active_positions[token] >= 0 && active_positions[token] <= query_position) ?
                (_Float16) 0.0f : (_Float16) -INFINITY;
    }
}

static int verify_dense_output(int layer, float * max_error) {
    *max_error = 0.0f;
    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        float expected[HEAD_DIM_V];
        reference_head(hq, layer, expected);
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            const float actual = dense_output[hq * HEAD_DIM_V + d];
            if (!isfinite(actual)) {
                return 0;
            }
            const float error = absolute_error(actual, expected[d]);
            if (error > *max_error) {
                *max_error = error;
            }
        }
    }
    return *max_error <= 0.01f;
}

static struct htp_tensor make_tensor(void * data, uint32_t size, uint16_t type) {
    struct htp_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.data = (uint32_t) (uintptr_t) data;
    tensor.size = size;
    tensor.type = type;
    return tensor;
}

static struct htp_tensor make_tensor_4d(
        void * data, uint32_t size, uint16_t type, uint32_t element_size,
        uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3) {
    struct htp_tensor tensor = make_tensor(data, size, type);
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = ne3;
    tensor.nb[0] = element_size;
    tensor.nb[1] = tensor.nb[0] * ne0;
    tensor.nb[2] = tensor.nb[1] * ne1;
    tensor.nb[3] = tensor.nb[2] * ne2;
    return tensor;
}

static int init_dense_context(struct htp_context * ctx) {
    memset(ctx, 0, sizeof(*ctx));

    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size);

    ctx->vtcm_rctx = HAP_compute_res_acquire(&attr, 1000000u);
    if (!ctx->vtcm_rctx) {
        return 0;
    }

    void * vtcm_ptr = NULL;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_rctx = 0;
        return 0;
    }
    ctx->vtcm_base = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size = vtcm_size;
    ctx->n_threads = N_THREADS;

    for (int i = 0; i < N_THREADS; ++i) {
        ctx->dma[i] = dma_queue_create(128);
        if (!ctx->dma[i]) {
            return 0;
        }
    }
    return worker_pool_init(&ctx->worker_pool, N_THREADS) == AEE_SUCCESS;
}

static void release_dense_context(struct htp_context * ctx) {
    if (ctx->worker_pool) {
        worker_pool_release(&ctx->worker_pool);
    }
    for (int i = 0; i < N_THREADS; ++i) {
        dma_queue_delete(ctx->dma[i]);
        ctx->dma[i] = NULL;
    }
    if (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_rctx = 0;
    }
}

static int run_dense_attention(struct htp_context * ctx) {
    struct htp_tensor q_tensor =
        make_tensor_4d(q, sizeof(q), HTP_TYPE_F32, sizeof(float), HEAD_DIM_K, 1, N_HEAD_Q, 1);
    struct htp_tensor k_tensor =
        make_tensor_4d(dense_k, DENSE_K_BYTES, HTP_TYPE_F16, sizeof(_Float16), HEAD_DIM_K, KV_SIZE, N_HEAD_KV, 1);
    struct htp_tensor v_tensor =
        make_tensor_4d(dense_v, DENSE_V_BYTES, HTP_TYPE_F16, sizeof(_Float16), HEAD_DIM_V, KV_SIZE, N_HEAD_KV, 1);
    struct htp_tensor mask_tensor =
        make_tensor_4d(dense_mask, DENSE_MASK_BYTES, HTP_TYPE_F16, sizeof(_Float16), KV_SIZE, 1, 1, 1);
    struct htp_tensor output_tensor =
        make_tensor_4d(dense_output, sizeof(dense_output), HTP_TYPE_F32, sizeof(float), HEAD_DIM_V, N_HEAD_Q, 1, 1);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx       = ctx;
    octx.op        = HTP_OP_FLASH_ATTN_EXT;
    octx.src[0]    = &q_tensor;
    octx.src[1]    = &k_tensor;
    octx.src[2]    = &v_tensor;
    octx.src[3]    = &mask_tensor;
    octx.dst       = &output_tensor;
    octx.n_threads = N_THREADS;

    const float scale         = 0.08838834764831845f;
    const float max_bias      = 0.0f;
    const float logit_softcap = 0.0f;
    memcpy(octx.op_params + 0, &scale, sizeof(scale));
    memcpy(octx.op_params + 1, &max_bias, sizeof(max_bias));
    memcpy(octx.op_params + 2, &logit_softcap, sizeof(logit_softcap));
    memset(dense_output, 0, sizeof(dense_output));
    return op_flash_attn_ext(&octx);
}

static uint64_t median3(uint64_t a, uint64_t b, uint64_t c) {
    if (a > b) {
        const uint64_t tmp = a;
        a = b;
        b = tmp;
    }
    if (b > c) {
        const uint64_t tmp = b;
        b = c;
        c = tmp;
    }
    return a > b ? a : b;
}

static uint64_t min3(uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t ab = a < b ? a : b;
    return ab < c ? ab : c;
}

static uint64_t max3(uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t ab = a > b ? a : b;
    return ab > c ? ab : c;
}

int main(void) {
    int32_t params[HTP_EDGEKV_ATTN_PARAM_COUNT] = {
        N_BLOCKS,
        BLOCK_SIZE,
        RANK_K,
        RANK_V,
        GROUP_SIZE,
        0,
        N_HEAD_Q,
        N_HEAD_KV,
        HEAD_DIM_K,
        HEAD_DIM_V,
        RECENT_SIZE,
        METADATA_V_SCALE_OFFSET_BYTES,
        METADATA_BLOCK_POS_OFFSET_BYTES,
        METADATA_RECENT_POS_OFFSET_BYTES,
        METADATA_TOTAL_BYTES,
        0x3db504f3,
    };

    if (!allocate_input_buffers()) {
        printf("FAIL: could not allocate aligned input buffers\n");
        return 1;
    }
    initialize_inputs();
    struct htp_context dense_ctx;
    if (!init_dense_context(&dense_ctx)) {
        release_dense_context(&dense_ctx);
        release_input_buffers();
        printf("FAIL: could not initialize dense FlashAttention context\n");
        return 2;
    }

    struct htp_tensor q_tensor        = make_tensor(q, sizeof(q), HTP_TYPE_F32);
    struct htp_tensor u_tensor        = make_tensor(u_storage, U_STORAGE_BYTES, HTP_TYPE_I8);
    struct htp_tensor vh_tensor       = make_tensor(vh_storage, VH_STORAGE_BYTES, HTP_TYPE_I8);
    struct htp_tensor metadata_tensor = make_tensor(metadata, METADATA_TOTAL_BYTES, HTP_TYPE_I8);
    struct htp_tensor active_k_tensor = make_tensor(active_k, ACTIVE_K_BYTES, HTP_TYPE_F16);
    struct htp_tensor active_v_tensor = make_tensor(active_v, ACTIVE_V_BYTES, HTP_TYPE_F16);
    struct htp_tensor output_tensor   = make_tensor(output, sizeof(output), HTP_TYPE_F32);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx       = &dense_ctx;
    octx.op        = HTP_OP_EDGEKV_ATTN_DECODE;
    octx.src[0]    = &q_tensor;
    octx.src[1]    = &u_tensor;
    octx.src[2]    = &vh_tensor;
    octx.src[3]    = &metadata_tensor;
    octx.src[4]    = &active_k_tensor;
    octx.src[5]    = &active_v_tensor;
    octx.dst       = &output_tensor;
    octx.n_threads = N_THREADS;

#ifdef HTP_EDGEKV_HVX_GQA
    memcpy(octx.op_params, params, sizeof(params));
    struct htp_context undersized_ctx = dense_ctx;
    undersized_ctx.vtcm_size = N_THREADS * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES - 1;
    octx.ctx = &undersized_ctx;
    const int undersized_status = op_edgekv_attn_decode(&octx);
    octx.ctx = &dense_ctx;
    if (undersized_status != HTP_STATUS_VTCM_TOO_SMALL) {
        printf("FAIL: undersized VTCM returned status %d, expected %d\n", undersized_status,
               HTP_STATUS_VTCM_TOO_SMALL);
        release_dense_context(&dense_ctx);
        release_input_buffers();
        return 8;
    }
    printf("PASS: VTCM guard required_bytes=%u available_bytes=%u status=%d\n",
           N_THREADS * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES, undersized_ctx.vtcm_size,
           undersized_status);
#endif

    int result = 0;
    for (int layer = 0; layer < GROUP_SIZE; ++layer) {
        params[HTP_EDGEKV_ATTN_LAYER_INDEX] = layer;
        memcpy(octx.op_params, params, sizeof(params));
        memset(output, 0, sizeof(output));
        materialize_dense_kv(layer);

        const uint64_t direct_start = HAP_perf_get_pcycles();
        const int      status       = op_edgekv_attn_decode(&octx);
        const uint64_t direct_op_pcycles = HAP_perf_get_pcycles() - direct_start;
        if (status != HTP_STATUS_OK) {
            printf("FAIL: layer %d returned status %d\n", layer, status);
            result = 2;
            goto cleanup;
        }

        float direct_max_error;
        if (!verify_output(layer, &direct_max_error)) {
            printf("FAIL: direct layer %d numerical mismatch max_error=%g\n", layer, (double) direct_max_error);
            result = 3;
            goto cleanup;
        }

        const uint64_t dense_start = HAP_perf_get_pcycles();
        const int      dense_status = run_dense_attention(&dense_ctx);
        const uint64_t dense_op_pcycles = HAP_perf_get_pcycles() - dense_start;
        if (dense_status != HTP_STATUS_OK) {
            printf("FAIL: dense layer %d returned status %d\n", layer, dense_status);
            result = 4;
            goto cleanup;
        }
        float dense_max_error;
        if (!verify_dense_output(layer, &dense_max_error)) {
            printf("FAIL: dense layer %d numerical mismatch max_error=%g\n", layer, (double) dense_max_error);
            result = 5;
            goto cleanup;
        }

        printf("PASS: layer=%d direct_error=%g dense_error=%g direct_kernel_pcycles=%llu "
               "direct_op_pcycles=%llu dense_op_pcycles=%llu\n",
               layer, (double) direct_max_error, (double) dense_max_error,
               (unsigned long long) edgekv_attn_decode_last_aot_pcycles(),
               (unsigned long long) direct_op_pcycles, (unsigned long long) dense_op_pcycles);
    }

    params[HTP_EDGEKV_ATTN_LAYER_INDEX] = 0;
    memcpy(octx.op_params, params, sizeof(params));
    materialize_dense_kv(0);

    uint64_t direct_kernel_samples[PERFORMANCE_SAMPLES];
    uint64_t direct_op_samples[PERFORMANCE_SAMPLES];
    uint64_t dense_op_samples[PERFORMANCE_SAMPLES];
    for (int sample = 0; sample < PERFORMANCE_SAMPLES; ++sample) {
        memset(output, 0, sizeof(output));
        const uint64_t direct_start = HAP_perf_get_pcycles();
        const int      direct_status = op_edgekv_attn_decode(&octx);
        direct_op_samples[sample] = HAP_perf_get_pcycles() - direct_start;
        direct_kernel_samples[sample] = edgekv_attn_decode_last_aot_pcycles();
        if (direct_status != HTP_STATUS_OK) {
            printf("FAIL: direct performance sample %d returned status %d\n", sample, direct_status);
            result = 6;
            goto cleanup;
        }

        const uint64_t dense_start = HAP_perf_get_pcycles();
        const int      dense_status = run_dense_attention(&dense_ctx);
        dense_op_samples[sample] = HAP_perf_get_pcycles() - dense_start;
        if (dense_status != HTP_STATUS_OK) {
            printf("FAIL: dense performance sample %d returned status %d\n", sample, dense_status);
            result = 7;
            goto cleanup;
        }
    }

    const uint64_t direct_kernel_median =
        median3(direct_kernel_samples[0], direct_kernel_samples[1], direct_kernel_samples[2]);
    const uint64_t direct_op_median =
        median3(direct_op_samples[0], direct_op_samples[1], direct_op_samples[2]);
    const uint64_t dense_op_median = median3(dense_op_samples[0], dense_op_samples[1], dense_op_samples[2]);
    const double ratio = (double) direct_op_median / (double) dense_op_median;
    const char * decision = ratio <= 1.0 ? "GO" : (ratio <= 1.25 ? "OPTIMIZE" : "STOP");

    printf("PERF: %s raw=[%llu,%llu,%llu] median=%llu min=%llu max=%llu\n", EDGEKV_DIRECT_KERNEL_NAME,
           (unsigned long long) direct_kernel_samples[0], (unsigned long long) direct_kernel_samples[1],
           (unsigned long long) direct_kernel_samples[2], (unsigned long long) direct_kernel_median,
           (unsigned long long) min3(direct_kernel_samples[0], direct_kernel_samples[1], direct_kernel_samples[2]),
           (unsigned long long) max3(direct_kernel_samples[0], direct_kernel_samples[1], direct_kernel_samples[2]));
    printf("PERF: direct_op raw=[%llu,%llu,%llu] median=%llu min=%llu max=%llu\n",
           (unsigned long long) direct_op_samples[0], (unsigned long long) direct_op_samples[1],
           (unsigned long long) direct_op_samples[2], (unsigned long long) direct_op_median,
           (unsigned long long) min3(direct_op_samples[0], direct_op_samples[1], direct_op_samples[2]),
           (unsigned long long) max3(direct_op_samples[0], direct_op_samples[1], direct_op_samples[2]));
    printf("PERF: dense_op raw=[%llu,%llu,%llu] median=%llu min=%llu max=%llu path=HVX-decode threads=%d\n",
           (unsigned long long) dense_op_samples[0], (unsigned long long) dense_op_samples[1],
           (unsigned long long) dense_op_samples[2], (unsigned long long) dense_op_median,
           (unsigned long long) min3(dense_op_samples[0], dense_op_samples[1], dense_op_samples[2]),
           (unsigned long long) max3(dense_op_samples[0], dense_op_samples[1], dense_op_samples[2]), N_THREADS);
    printf("PASS: EdgeKV %s ratio=%.6f decision=%s dynamic_metadata_bytes=516\n",
           EDGEKV_TEST_MILESTONE, ratio, decision);

cleanup:
    release_dense_context(&dense_ctx);
    release_input_buffers();
    return result;
}
