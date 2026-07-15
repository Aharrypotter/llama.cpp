// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-attn.h"
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "htp-ctx.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    N_BLOCKS              = EDGEKV_HVX_GQA_ATTN_N_BLOCKS,
    BLOCK_SIZE            = EDGEKV_HVX_GQA_BLOCK_SIZE,
    RANK                  = EDGEKV_HVX_GQA_RANK,
    GROUP_SIZE            = EDGEKV_HVX_GQA_ATTN_GROUP_SIZE,
    N_HEAD_Q              = EDGEKV_HVX_GQA_ATTN_N_HEAD_Q,
    N_HEAD_KV             = EDGEKV_HVX_GQA_ATTN_N_HEAD_KV,
    HEAD_DIM              = EDGEKV_HVX_GQA_HEAD_DIM,
    RECENT_SIZE           = EDGEKV_HVX_GQA_ATTN_RECENT_SIZE,
    N_THREADS             = 4,
    PERFORMANCE_SAMPLES   = 5,
    MATERIALIZED_RANKS    = 2,
    KV_SIZE               = N_BLOCKS * BLOCK_SIZE + RECENT_SIZE,
    U_BYTES               = N_BLOCKS * BLOCK_SIZE * RANK,
    VH_BYTES              = N_BLOCKS * RANK * GROUP_SIZE * N_HEAD_KV * HEAD_DIM,
    SCALE_ELEMS           = N_BLOCKS * RANK,
    BLOCK_POSITION_ELEMS  = N_BLOCKS * BLOCK_SIZE,
    ACTIVE_POSITION_ELEMS = RECENT_SIZE + 1,
    ACTIVE_ELEMS          = RECENT_SIZE * N_HEAD_KV * HEAD_DIM,
    DENSE_ELEMS           = N_HEAD_KV * KV_SIZE * HEAD_DIM,
    DENSE_MASK_ELEMS      = KV_SIZE,
    OUTPUT_ELEMS          = N_HEAD_Q * HEAD_DIM,
};

struct fixture {
    float *    query;
    int8_t *   k_u;
    int8_t *   k_vh;
    float *    k_scale;
    int8_t *   v_u;
    int8_t *   v_vh;
    float *    v_scale;
    int32_t *  block_positions;
    _Float16 * active_k;
    _Float16 * active_v;
    int32_t *  active_positions;
    float *    output;
    float *    reference_output;
    _Float16 * dense_k;
    _Float16 * dense_v;
    _Float16 * dense_mask;
    float *    dense_output;
    float *    scalar_workspace;
};

struct consumer_run {
    const struct edgekv_hvx_gqa_attn_inputs * inputs;
    uint8_t *                                 workspace;
    float *                                   output;
    int                                       layer;
    struct edgekv_hvx_gqa_attn_profile *      profiles;
};

static volatile float perf_sink;

static size_t vh_index(int block, int rank, int layer, int hkv, int dim) {
    return ((((size_t) block * RANK + rank) * GROUP_SIZE + layer) * N_HEAD_KV + hkv) * HEAD_DIM + dim;
}

static void release_fixture(struct fixture * fixture) {
    free(fixture->scalar_workspace);
    free(fixture->dense_output);
    free(fixture->dense_mask);
    free(fixture->dense_v);
    free(fixture->dense_k);
    free(fixture->reference_output);
    free(fixture->output);
    free(fixture->active_positions);
    free(fixture->active_v);
    free(fixture->active_k);
    free(fixture->block_positions);
    free(fixture->v_scale);
    free(fixture->v_vh);
    free(fixture->v_u);
    free(fixture->k_scale);
    free(fixture->k_vh);
    free(fixture->k_u);
    free(fixture->query);
    memset(fixture, 0, sizeof(*fixture));
}

static int allocate_fixture(struct fixture * fixture) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->query            = (float *) memalign(128, OUTPUT_ELEMS * sizeof(float));
    fixture->k_u              = (int8_t *) memalign(128, U_BYTES);
    fixture->k_vh             = (int8_t *) memalign(128, VH_BYTES);
    fixture->k_scale          = (float *) memalign(128, SCALE_ELEMS * sizeof(float));
    fixture->v_u              = (int8_t *) memalign(128, U_BYTES);
    fixture->v_vh             = (int8_t *) memalign(128, VH_BYTES);
    fixture->v_scale          = (float *) memalign(128, SCALE_ELEMS * sizeof(float));
    fixture->block_positions  = (int32_t *) memalign(128, BLOCK_POSITION_ELEMS * sizeof(int32_t));
    fixture->active_k         = (_Float16 *) memalign(128, ACTIVE_ELEMS * sizeof(_Float16));
    fixture->active_v         = (_Float16 *) memalign(128, ACTIVE_ELEMS * sizeof(_Float16));
    fixture->active_positions = (int32_t *) memalign(128, ACTIVE_POSITION_ELEMS * sizeof(int32_t));
    fixture->output           = (float *) memalign(128, OUTPUT_ELEMS * sizeof(float));
    fixture->reference_output = (float *) memalign(128, OUTPUT_ELEMS * sizeof(float));
    fixture->dense_k          = (_Float16 *) memalign(128, DENSE_ELEMS * sizeof(_Float16));
    fixture->dense_v          = (_Float16 *) memalign(128, DENSE_ELEMS * sizeof(_Float16));
    fixture->dense_mask       = (_Float16 *) memalign(128, DENSE_MASK_ELEMS * sizeof(_Float16));
    fixture->dense_output     = (float *) memalign(128, OUTPUT_ELEMS * sizeof(float));
    fixture->scalar_workspace = (float *) memalign(128, EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES);

    if (!fixture->query || !fixture->k_u || !fixture->k_vh || !fixture->k_scale || !fixture->v_u || !fixture->v_vh ||
        !fixture->v_scale || !fixture->block_positions || !fixture->active_k || !fixture->active_v ||
        !fixture->active_positions || !fixture->output || !fixture->reference_output || !fixture->dense_k ||
        !fixture->dense_v || !fixture->dense_mask || !fixture->dense_output || !fixture->scalar_workspace) {
        release_fixture(fixture);
        return 0;
    }
    return 1;
}

static void initialize_fixture(struct fixture * fixture) {
    memset(fixture->k_vh, 0, VH_BYTES);
    memset(fixture->v_vh, 0, VH_BYTES);
    memset(fixture->output, 0, OUTPUT_ELEMS * sizeof(float));
    memset(fixture->reference_output, 0, OUTPUT_ELEMS * sizeof(float));
    memset(fixture->dense_output, 0, OUTPUT_ELEMS * sizeof(float));

    for (int i = 0; i < OUTPUT_ELEMS; ++i) {
        fixture->query[i] = (float) (((i * 13 + 5) % 29) - 14) * 0.03125f;
    }
    for (int block = 0; block < N_BLOCKS; ++block) {
        for (int token = 0; token < BLOCK_SIZE; ++token) {
            const int token_index                 = block * BLOCK_SIZE + token;
            fixture->block_positions[token_index] = token_index;
            for (int rank = 0; rank < RANK; ++rank) {
                const int logical     = token_index * RANK + rank;
                fixture->k_u[logical] = (int8_t) (((logical * 7 + 3) % 15) - 7);
                fixture->v_u[logical] = (int8_t) (((logical * 5 + 1) % 13) - 6);
            }
        }
        for (int rank = 0; rank < RANK; ++rank) {
            fixture->k_scale[block * RANK + rank] = 0.0025f + 0.00003125f * (float) ((block + rank) % 7);
            fixture->v_scale[block * RANK + rank] = 0.0030f + 0.00002500f * (float) ((block + rank) % 5);
            if (rank >= MATERIALIZED_RANKS) {
                continue;
            }
            for (int layer = 0; layer < GROUP_SIZE; ++layer) {
                for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
                    for (int dim = 0; dim < HEAD_DIM; ++dim) {
                        const int logical =
                            ((((block * RANK + rank) * GROUP_SIZE + layer) * N_HEAD_KV + hkv) * HEAD_DIM + dim);
                        fixture->k_vh[vh_index(block, rank, layer, hkv, dim)] =
                            (int8_t) (((logical * 11 + 2) % 13) - 6);
                        fixture->v_vh[vh_index(block, rank, layer, hkv, dim)] = (int8_t) (((logical * 3 + 4) % 11) - 5);
                    }
                }
            }
        }
    }
    for (int token = N_BLOCKS * BLOCK_SIZE - 7; token < N_BLOCKS * BLOCK_SIZE; ++token) {
        fixture->block_positions[token] = -1;
    }

    const int compressed_valid = N_BLOCKS * BLOCK_SIZE - 7;
    for (int token = 0; token < RECENT_SIZE; ++token) {
        fixture->active_positions[token] = compressed_valid + token;
        for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                const int index          = (token * N_HEAD_KV + hkv) * HEAD_DIM + dim;
                fixture->active_k[index] = (_Float16) ((float) (((token * 3 + hkv * 5 + dim * 7) % 19) - 9) * 0.0020f);
                fixture->active_v[index] =
                    (_Float16) ((float) (((token * 5 + hkv * 3 + dim * 11) % 23) - 11) * 0.0015f);
            }
        }
    }
    fixture->active_positions[RECENT_SIZE] = compressed_valid + RECENT_SIZE - 6;
}

static void make_inputs(const struct fixture * fixture, struct edgekv_hvx_gqa_attn_inputs * inputs) {
    inputs->query            = fixture->query;
    inputs->k_u              = fixture->k_u;
    inputs->k_vh             = fixture->k_vh;
    inputs->k_scale          = fixture->k_scale;
    inputs->v_u              = fixture->v_u;
    inputs->v_vh             = fixture->v_vh;
    inputs->v_scale          = fixture->v_scale;
    inputs->block_positions  = fixture->block_positions;
    inputs->active_k         = fixture->active_k;
    inputs->active_v         = fixture->active_v;
    inputs->active_positions = fixture->active_positions;
    inputs->query_position   = fixture->active_positions[RECENT_SIZE];
}

static void reference_hkv(struct fixture * fixture, int layer, int hkv) {
    const float   attention_scale = 0.08838834764831845f;
    float *       block_logits    = fixture->scalar_workspace;
    float *       active_logits   = block_logits + EDGEKV_HVX_GQA_ATTN_BLOCK_LOGITS;
    float *       partial         = active_logits + EDGEKV_HVX_GQA_ATTN_ACTIVE_LOGITS;
    float *       output          = fixture->reference_output + hkv * EDGEKV_HVX_GQA_HEADS * HEAD_DIM;
    const float * query           = fixture->query + hkv * EDGEKV_HVX_GQA_HEADS * HEAD_DIM;

    for (int block = 0; block < N_BLOCKS; ++block) {
        edgekv_scalar_gqa_block_k(query, fixture->k_u + block * BLOCK_SIZE * RANK,
                                  fixture->k_vh + vh_index(block, 0, layer, hkv, 0), fixture->k_scale + block * RANK,
                                  attention_scale, block_logits + block * EDGEKV_HVX_GQA_HEADS * BLOCK_SIZE);
    }
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int token = 0; token < RECENT_SIZE; ++token) {
            const int base = (token * N_HEAD_KV + hkv) * HEAD_DIM;
            float     sum  = 0.0f;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                sum += query[head * HEAD_DIM + dim] * (float) fixture->active_k[base + dim];
            }
            active_logits[head * RECENT_SIZE + token] = sum * attention_scale;
        }
    }

    float max_logit[EDGEKV_HVX_GQA_HEADS] = { -INFINITY, -INFINITY };
    for (int block = 0; block < N_BLOCKS; ++block) {
        for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
            float * logits = block_logits + block * EDGEKV_HVX_GQA_HEADS * BLOCK_SIZE + head * BLOCK_SIZE;
            for (int token = 0; token < BLOCK_SIZE; ++token) {
                const int position = fixture->block_positions[block * BLOCK_SIZE + token];
                if (position < 0 || position > fixture->active_positions[RECENT_SIZE]) {
                    logits[token] = -INFINITY;
                } else if (logits[token] > max_logit[head]) {
                    max_logit[head] = logits[token];
                }
            }
        }
    }
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int token = 0; token < RECENT_SIZE; ++token) {
            const int position = fixture->active_positions[token];
            float *   logit    = active_logits + head * RECENT_SIZE + token;
            if (position < 0 || position > fixture->active_positions[RECENT_SIZE]) {
                *logit = -INFINITY;
            } else if (*logit > max_logit[head]) {
                max_logit[head] = *logit;
            }
        }
    }

    float sums[EDGEKV_HVX_GQA_HEADS] = { 0.0f, 0.0f };
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int block = 0; block < N_BLOCKS; ++block) {
            float * logits = block_logits + block * EDGEKV_HVX_GQA_HEADS * BLOCK_SIZE + head * BLOCK_SIZE;
            for (int token = 0; token < BLOCK_SIZE; ++token) {
                const float weight = isfinite(logits[token]) ? expf(logits[token] - max_logit[head]) : 0.0f;
                logits[token]      = weight;
                sums[head] += weight;
            }
        }
        for (int token = 0; token < RECENT_SIZE; ++token) {
            float *     logit  = active_logits + head * RECENT_SIZE + token;
            const float weight = isfinite(*logit) ? expf(*logit - max_logit[head]) : 0.0f;
            *logit             = weight;
            sums[head] += weight;
        }
    }

    memset(output, 0, EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT * sizeof(float));
    for (int block = 0; block < N_BLOCKS; ++block) {
        edgekv_scalar_gqa_block_v(
            block_logits + block * EDGEKV_HVX_GQA_HEADS * BLOCK_SIZE, fixture->v_u + block * BLOCK_SIZE * RANK,
            fixture->v_vh + vh_index(block, 0, layer, hkv, 0), fixture->v_scale + block * RANK, partial);
        for (int i = 0; i < EDGEKV_HVX_GQA_ATTN_PARTIAL_OUTPUT; ++i) {
            output[i] += partial[i];
        }
    }
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        for (int token = 0; token < RECENT_SIZE; ++token) {
            const float weight = active_logits[head * RECENT_SIZE + token];
            const int   base   = (token * N_HEAD_KV + hkv) * HEAD_DIM;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                output[head * HEAD_DIM + dim] += weight * (float) fixture->active_v[base + dim];
            }
        }
        const float inverse_sum = sums[head] > 0.0f ? 1.0f / sums[head] : 0.0f;
        for (int dim = 0; dim < HEAD_DIM; ++dim) {
            output[head * HEAD_DIM + dim] *= inverse_sum;
        }
    }
}

static void reference_all(struct fixture * fixture, int layer) {
    for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
        reference_hkv(fixture, layer, hkv);
    }
}

static void materialize_dense(struct fixture * fixture, int layer) {
    for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
        for (int token_index = 0; token_index < N_BLOCKS * BLOCK_SIZE; ++token_index) {
            const int block = token_index / BLOCK_SIZE;
            const int base  = (hkv * KV_SIZE + token_index) * HEAD_DIM;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                float k_value = 0.0f;
                float v_value = 0.0f;
                for (int rank = 0; rank < MATERIALIZED_RANKS; ++rank) {
                    k_value += (float) fixture->k_u[token_index * RANK + rank] * fixture->k_scale[block * RANK + rank] *
                               (float) fixture->k_vh[vh_index(block, rank, layer, hkv, dim)];
                    v_value += (float) fixture->v_u[token_index * RANK + rank] * fixture->v_scale[block * RANK + rank] *
                               (float) fixture->v_vh[vh_index(block, rank, layer, hkv, dim)];
                }
                fixture->dense_k[base + dim] = (_Float16) k_value;
                fixture->dense_v[base + dim] = (_Float16) v_value;
            }
        }
        for (int token = 0; token < RECENT_SIZE; ++token) {
            const int dense_base  = (hkv * KV_SIZE + N_BLOCKS * BLOCK_SIZE + token) * HEAD_DIM;
            const int active_base = (token * N_HEAD_KV + hkv) * HEAD_DIM;
            memcpy(fixture->dense_k + dense_base, fixture->active_k + active_base, HEAD_DIM * sizeof(_Float16));
            memcpy(fixture->dense_v + dense_base, fixture->active_v + active_base, HEAD_DIM * sizeof(_Float16));
        }
    }

    const int query_position = fixture->active_positions[RECENT_SIZE];
    for (int token = 0; token < N_BLOCKS * BLOCK_SIZE; ++token) {
        const int position = fixture->block_positions[token];
        fixture->dense_mask[token] =
            (position >= 0 && position <= query_position) ? (_Float16) 0.0f : (_Float16) -INFINITY;
    }
    for (int token = 0; token < RECENT_SIZE; ++token) {
        const int position = fixture->active_positions[token];
        fixture->dense_mask[N_BLOCKS * BLOCK_SIZE + token] =
            (position >= 0 && position <= query_position) ? (_Float16) 0.0f : (_Float16) -INFINITY;
    }
}

static struct htp_tensor make_tensor_4d(void *   data,
                                        uint32_t size,
                                        uint16_t type,
                                        uint32_t element_size,
                                        uint32_t ne0,
                                        uint32_t ne1,
                                        uint32_t ne2,
                                        uint32_t ne3) {
    struct htp_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.data  = (uint32_t) (uintptr_t) data;
    tensor.size  = size;
    tensor.type  = type;
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

static int init_context(struct htp_context * ctx) {
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

static void release_context(struct htp_context * ctx) {
    if (ctx->worker_pool) {
        worker_pool_release(&ctx->worker_pool);
    }
    for (int i = 0; i < N_THREADS; ++i) {
        if (ctx->dma[i]) {
            dma_queue_delete(ctx->dma[i]);
            ctx->dma[i] = NULL;
        }
    }
    if (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_rctx = 0;
    }
}

static int run_dense_attention(struct htp_context * ctx, struct fixture * fixture) {
    struct htp_tensor q      = make_tensor_4d(fixture->query, OUTPUT_ELEMS * sizeof(float), HTP_TYPE_F32, sizeof(float),
                                              HEAD_DIM, 1, N_HEAD_Q, 1);
    struct htp_tensor k      = make_tensor_4d(fixture->dense_k, DENSE_ELEMS * sizeof(_Float16), HTP_TYPE_F16,
                                              sizeof(_Float16), HEAD_DIM, KV_SIZE, N_HEAD_KV, 1);
    struct htp_tensor v      = make_tensor_4d(fixture->dense_v, DENSE_ELEMS * sizeof(_Float16), HTP_TYPE_F16,
                                              sizeof(_Float16), HEAD_DIM, KV_SIZE, N_HEAD_KV, 1);
    struct htp_tensor mask   = make_tensor_4d(fixture->dense_mask, DENSE_MASK_ELEMS * sizeof(_Float16), HTP_TYPE_F16,
                                              sizeof(_Float16), KV_SIZE, 1, 1, 1);
    struct htp_tensor output = make_tensor_4d(fixture->dense_output, OUTPUT_ELEMS * sizeof(float), HTP_TYPE_F32,
                                              sizeof(float), HEAD_DIM, N_HEAD_Q, 1, 1);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx       = ctx;
    octx.op        = HTP_OP_FLASH_ATTN_EXT;
    octx.src[0]    = &q;
    octx.src[1]    = &k;
    octx.src[2]    = &v;
    octx.src[3]    = &mask;
    octx.dst       = &output;
    octx.n_threads = N_THREADS;

    const float scale         = 0.08838834764831845f;
    const float max_bias      = 0.0f;
    const float logit_softcap = 0.0f;
    memcpy(octx.op_params + 0, &scale, sizeof(scale));
    memcpy(octx.op_params + 1, &max_bias, sizeof(max_bias));
    memcpy(octx.op_params + 2, &logit_softcap, sizeof(logit_softcap));
    memset(fixture->dense_output, 0, OUTPUT_ELEMS * sizeof(float));
    return op_flash_attn_ext(&octx);
}

static void consumer_worker(unsigned int n, unsigned int i, void * data) {
    struct consumer_run *                run       = (struct consumer_run *) data;
    struct edgekv_hvx_gqa_attn_profile * profile   = run->profiles ? run->profiles + i : NULL;
    void *                               workspace = run->workspace + i * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES;
    for (int hkv = (int) i; hkv < N_HEAD_KV; hkv += (int) n) {
        edgekv_hvx_gqa_attn_hkv(run->inputs, run->layer, hkv, 0.08838834764831845f, workspace, run->output, profile);
    }
}

static int run_consumer(struct htp_context *                      ctx,
                        const struct edgekv_hvx_gqa_attn_inputs * inputs,
                        struct fixture *                          fixture,
                        int                                       layer,
                        struct edgekv_hvx_gqa_attn_profile *      profiles) {
    struct consumer_run run = {
        .inputs    = inputs,
        .workspace = ctx->vtcm_base,
        .output    = fixture->output,
        .layer     = layer,
        .profiles  = profiles,
    };
    if (profiles) {
        memset(profiles, 0, N_THREADS * sizeof(*profiles));
    }
    return worker_pool_run_func(ctx->worker_pool, consumer_worker, &run, N_THREADS) == AEE_SUCCESS;
}

static float max_error(const float * actual, const float * expected, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(actual[i])) {
            return INFINITY;
        }
        float error = actual[i] - expected[i];
        error       = error < 0.0f ? -error : error;
        if (error > result) {
            result = error;
        }
    }
    return result;
}

static uint64_t median5(const uint64_t samples[PERFORMANCE_SAMPLES]) {
    uint64_t sorted[PERFORMANCE_SAMPLES];
    memcpy(sorted, samples, sizeof(sorted));
    for (int i = 1; i < PERFORMANCE_SAMPLES; ++i) {
        const uint64_t value = sorted[i];
        int            j     = i - 1;
        while (j >= 0 && sorted[j] > value) {
            sorted[j + 1] = sorted[j];
            --j;
        }
        sorted[j + 1] = value;
    }
    return sorted[PERFORMANCE_SAMPLES / 2];
}

static uint64_t profile_max(const struct edgekv_hvx_gqa_attn_profile profiles[N_THREADS], size_t offset) {
    uint64_t result = 0;
    for (int i = 0; i < N_THREADS; ++i) {
        const uint64_t value = *(const uint64_t *) ((const uint8_t *) (profiles + i) + offset);
        if (value > result) {
            result = value;
        }
    }
    return result;
}

static void print_samples(const char * name, const uint64_t samples[PERFORMANCE_SAMPLES], uint64_t median) {
    printf("PERF: %s raw=[%llu,%llu,%llu,%llu,%llu] median=%llu\n", name, (unsigned long long) samples[0],
           (unsigned long long) samples[1], (unsigned long long) samples[2], (unsigned long long) samples[3],
           (unsigned long long) samples[4], (unsigned long long) median);
}

int main(void) {
    struct fixture fixture;
    if (!allocate_fixture(&fixture)) {
        printf("FAIL: could not allocate C4C2 fixture\n");
        return 1;
    }
    initialize_fixture(&fixture);

    struct htp_context ctx;
    if (!init_context(&ctx)) {
        release_context(&ctx);
        release_fixture(&fixture);
        printf("FAIL: could not initialize C4C2 HTP context\n");
        return 2;
    }
    if (ctx.vtcm_size < N_THREADS * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES) {
        release_context(&ctx);
        release_fixture(&fixture);
        printf("FAIL: C4C2 workspace exceeds VTCM\n");
        return 3;
    }

    struct edgekv_hvx_gqa_attn_inputs inputs;
    make_inputs(&fixture, &inputs);
    int result = 0;
    for (int layer = 0; layer < GROUP_SIZE; ++layer) {
        reference_all(&fixture, layer);
        if (!run_consumer(&ctx, &inputs, &fixture, layer, NULL)) {
            printf("FAIL: C4C2 consumer dispatch failed at layer %d\n", layer);
            result = 4;
            goto cleanup;
        }
        const float consumer_error = max_error(fixture.output, fixture.reference_output, OUTPUT_ELEMS);
        if (consumer_error > 0.02f) {
            printf("FAIL: C4C2 consumer mismatch layer=%d max_error=%g\n", layer, (double) consumer_error);
            result = 5;
            goto cleanup;
        }

        materialize_dense(&fixture, layer);
        if (run_dense_attention(&ctx, &fixture) != HTP_STATUS_OK) {
            printf("FAIL: C4C2 dense attention failed at layer %d\n", layer);
            result = 6;
            goto cleanup;
        }
        const float dense_error = max_error(fixture.dense_output, fixture.reference_output, OUTPUT_ELEMS);
        if (dense_error > 0.03f) {
            printf("FAIL: C4C2 dense mismatch layer=%d max_error=%g\n", layer, (double) dense_error);
            result = 7;
            goto cleanup;
        }
        printf("PASS: C4C2 layer=%d consumer_error=%g dense_error=%g\n", layer, (double) consumer_error,
               (double) dense_error);
    }

    materialize_dense(&fixture, 0);
    struct edgekv_hvx_gqa_attn_profile profiles[N_THREADS];
    if (!run_consumer(&ctx, &inputs, &fixture, 0, profiles)) {
        printf("FAIL: C4C2 profile dispatch failed\n");
        result = 8;
        goto cleanup;
    }
    printf(
        "PROFILE: critical factor_k=%llu active_k=%llu mask_softmax=%llu factor_v=%llu "
        "active_v_normalize=%llu worker_total=%llu workspace_per_worker=%u\n",
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, factor_k)),
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, active_k)),
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, mask_softmax)),
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, factor_v)),
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, active_v_normalize)),
        (unsigned long long) profile_max(profiles, offsetof(struct edgekv_hvx_gqa_attn_profile, total)),
        (unsigned int) EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES);

    run_consumer(&ctx, &inputs, &fixture, 0, NULL);
    run_dense_attention(&ctx, &fixture);
    uint64_t consumer_samples[PERFORMANCE_SAMPLES];
    uint64_t dense_samples[PERFORMANCE_SAMPLES];
    for (int sample = 0; sample < PERFORMANCE_SAMPLES; ++sample) {
        uint64_t start = HAP_perf_get_pcycles();
        if (!run_consumer(&ctx, &inputs, &fixture, 0, NULL)) {
            printf("FAIL: C4C2 performance dispatch %d failed\n", sample);
            result = 9;
            goto cleanup;
        }
        consumer_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += fixture.output[sample];

        start = HAP_perf_get_pcycles();
        if (run_dense_attention(&ctx, &fixture) != HTP_STATUS_OK) {
            printf("FAIL: dense performance dispatch %d failed\n", sample);
            result = 10;
            goto cleanup;
        }
        dense_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += fixture.dense_output[sample];
    }

    const uint64_t consumer_median = median5(consumer_samples);
    const uint64_t dense_median    = median5(dense_samples);
    const double   ratio           = (double) consumer_median / (double) dense_median;
    const char *   decision        = ratio <= 1.0 ? "GO" : ratio <= 1.25 ? "OPTIMIZE" : "STOP";
    print_samples("c4c2_full_consumer", consumer_samples, consumer_median);
    print_samples("dense_htp", dense_samples, dense_median);
    printf(
        "PASS: C4C2 ratio=%.6f decision=%s shape=B64/R32/G4/Hq16/Hkv8/D128/recent128 "
        "threads=%d sink=%g\n",
        ratio, decision, N_THREADS, (double) perf_sink);

cleanup:
    release_context(&ctx);
    release_fixture(&fixture);
    return result;
}
