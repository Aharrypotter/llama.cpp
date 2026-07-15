// SPDX-License-Identifier: BSD-3-Clause

#include "hex-utils.h"
#include "htp-ctx.h"

#ifdef HTP_EDGEKV_HVX_GQA
#include "edgekv-hvx-gqa-attn.h"
#endif

#include "HAP_perf.h"

#include <stdint.h>
#include <string.h>

struct edgekv_memref_f32_2_direct {
    float * allocated;
    float * aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

struct edgekv_memref_i8_3_direct {
    int8_t * allocated;
    int8_t * aligned;
    int64_t offset;
    int64_t sizes[3];
    int64_t strides[3];
};

struct edgekv_memref_i8_5_direct {
    int8_t * allocated;
    int8_t * aligned;
    int64_t offset;
    int64_t sizes[5];
    int64_t strides[5];
};

struct edgekv_memref_i32_2_direct {
    int32_t * allocated;
    int32_t * aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

struct edgekv_memref_f16_3_direct {
    uint16_t * allocated;
    uint16_t * aligned;
    int64_t offset;
    int64_t sizes[3];
    int64_t strides[3];
};

struct edgekv_memref_i32_1_direct {
    int32_t * allocated;
    int32_t * aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

#ifdef HTP_EDGEKV_DIRECT_AOT
extern void edgekv_attn_decode_direct_int8_kernel(int64_t,
                                                  struct edgekv_memref_f32_2_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i8_3_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i8_5_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_f32_2_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i8_3_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i8_5_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_f32_2_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i32_2_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_f16_3_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_f16_3_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i32_1_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_i32_1_direct *,
                                                  int64_t,
                                                  struct edgekv_memref_f32_2_direct *,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int);
#endif

#if defined(HTP_EDGEKV_PROFILE_AOT) || defined(HTP_EDGEKV_PROFILE_KERNEL)
#define HTP_EDGEKV_PROFILE_KERNEL_ENABLED 1
static uint64_t edgekv_last_kernel_pcycles;
#endif

uint64_t edgekv_attn_decode_last_aot_pcycles(void) {
#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    return edgekv_last_kernel_pcycles;
#else
    return 0;
#endif
}

static bool edgekv_is_direct_mobile_v79_profile(const int32_t * p) {
    static const int32_t expected[HTP_EDGEKV_ATTN_PARAM_COUNT] = {
        16, 64, 32, 32, 4, 0, 16, 8, 128, 128, 128, 2048, 4096, 8192, 8832, 0x3db504f3,
    };

    for (int i = 0; i < HTP_EDGEKV_ATTN_PARAM_COUNT; ++i) {
        if (i != HTP_EDGEKV_ATTN_LAYER_INDEX && p[i] != expected[i]) {
            return false;
        }
    }
    return p[HTP_EDGEKV_ATTN_LAYER_INDEX] >= 0 &&
           p[HTP_EDGEKV_ATTN_LAYER_INDEX] < expected[HTP_EDGEKV_ATTN_GROUP_SIZE];
}

#ifdef HTP_EDGEKV_HVX_GQA
enum {
    EDGEKV_HVX_GQA_ATTN_U_HALF_BYTES =
        EDGEKV_HVX_GQA_ATTN_N_BLOCKS * EDGEKV_HVX_GQA_ATTN_U_BLOCK_BYTES,
    EDGEKV_HVX_GQA_ATTN_VH_HALF_BYTES =
        EDGEKV_HVX_GQA_ATTN_N_BLOCKS * EDGEKV_HVX_GQA_ATTN_VH_BLOCK_BYTES,
};

struct edgekv_hvx_gqa_attn_run {
    struct edgekv_hvx_gqa_attn_inputs inputs;
    uint8_t *                         workspace;
    float *                           output;
    float                             attention_scale;
    int                               layer_index;
};

static void edgekv_hvx_gqa_attn_worker(unsigned int n, unsigned int i, void * data) {
    struct edgekv_hvx_gqa_attn_run * run = (struct edgekv_hvx_gqa_attn_run *) data;
    void * workspace = run->workspace + i * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES;
    for (int hkv = (int) i; hkv < EDGEKV_HVX_GQA_ATTN_N_HEAD_KV; hkv += (int) n) {
        edgekv_hvx_gqa_attn_hkv(&run->inputs, run->layer_index, hkv, run->attention_scale,
                                workspace, run->output, NULL);
    }
}

static int edgekv_attn_decode_hvx(struct htp_ops_context * octx) {
    struct htp_context * ctx = octx->ctx;
    if (!ctx || !ctx->worker_pool || !ctx->vtcm_base) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t n_workers = octx->n_threads ? octx->n_threads : ctx->n_threads;
    if (n_workers > EDGEKV_HVX_GQA_ATTN_MAX_WORKERS) {
        n_workers = EDGEKV_HVX_GQA_ATTN_MAX_WORKERS;
    }
    if (n_workers == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t workspace_bytes = n_workers * EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES;
    if (ctx->vtcm_size < workspace_bytes) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const int32_t * p = octx->op_params;
    const int8_t * u_base = (const int8_t *) (uintptr_t) octx->src[1]->data;
    const int8_t * vh_base = (const int8_t *) (uintptr_t) octx->src[2]->data;
    const uint8_t * metadata_base = (const uint8_t *) (uintptr_t) octx->src[3]->data;
    const int32_t * active_positions =
        (const int32_t *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_RECENT_POSITIONS_OFFSET_BYTES]);

    struct edgekv_hvx_gqa_attn_run run = {
        .inputs = {
            .query            = (const float *) (uintptr_t) octx->src[0]->data,
            .k_u              = u_base,
            .k_vh             = vh_base,
            .k_scale          = (const float *) metadata_base,
            .v_u              = u_base + EDGEKV_HVX_GQA_ATTN_U_HALF_BYTES,
            .v_vh             = vh_base + EDGEKV_HVX_GQA_ATTN_VH_HALF_BYTES,
            .v_scale          = (const float *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_V_SCALE_OFFSET_BYTES]),
            .block_positions  = (const int32_t *)
                (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_BLOCK_POSITIONS_OFFSET_BYTES]),
            .active_k         = (const _Float16 *) (uintptr_t) octx->src[4]->data,
            .active_v         = (const _Float16 *) (uintptr_t) octx->src[5]->data,
            .active_positions = active_positions,
            .query_position   = active_positions[p[HTP_EDGEKV_ATTN_RECENT_SIZE]],
        },
        .workspace   = ctx->vtcm_base,
        .output      = (float *) (uintptr_t) octx->dst->data,
        .layer_index = p[HTP_EDGEKV_ATTN_LAYER_INDEX],
    };
    memcpy(&run.attention_scale, p + HTP_EDGEKV_ATTN_SCALE_BITS, sizeof(run.attention_scale));

    octx->dst_spad.src             = NULL;
    octx->dst_spad.data            = ctx->vtcm_base;
    octx->dst_spad.size_per_thread = EDGEKV_HVX_GQA_ATTN_WORKSPACE_BYTES;
    octx->dst_spad.size            = workspace_bytes;

#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    const uint64_t start = HAP_perf_get_pcycles();
#endif
    const AEEResult result =
        worker_pool_run_func(ctx->worker_pool, edgekv_hvx_gqa_attn_worker, &run, n_workers);
#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    edgekv_last_kernel_pcycles = HAP_perf_get_pcycles() - start;
#endif
    return result == AEE_SUCCESS ? HTP_STATUS_OK : HTP_STATUS_INTERNAL_ERR;
}
#endif

int op_edgekv_attn_decode(struct htp_ops_context * octx) {
    const struct htp_tensor * q_storage        = octx->src[0];
    const struct htp_tensor * u_storage        = octx->src[1];
    const struct htp_tensor * vh_storage       = octx->src[2];
    const struct htp_tensor * metadata_storage = octx->src[3];
    const struct htp_tensor * active_k_storage = octx->src[4];
    const struct htp_tensor * active_v_storage = octx->src[5];
    const struct htp_tensor * dst              = octx->dst;
    const int32_t *           p                = octx->op_params;

#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    edgekv_last_kernel_pcycles = 0;
#endif
    if (!q_storage || !u_storage || !vh_storage || !metadata_storage || !active_k_storage ||
        !active_v_storage || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (q_storage->type != HTP_TYPE_F32 || u_storage->type != HTP_TYPE_I8 || vh_storage->type != HTP_TYPE_I8 ||
        metadata_storage->type != HTP_TYPE_I8 || active_k_storage->type != HTP_TYPE_F16 ||
        active_v_storage->type != HTP_TYPE_F16 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!edgekv_is_direct_mobile_v79_profile(p)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q_storage->size < 8192 || u_storage->size < 65536 || vh_storage->size < 4194304 ||
        metadata_storage->size < 8832 || active_k_storage->size < 262144 || active_v_storage->size < 262144 ||
        dst->size < 8192) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (!hex_is_aligned((const void *) (uintptr_t) q_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) u_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) vh_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) metadata_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) active_k_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) active_v_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) dst->data, 128)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

#ifdef HTP_EDGEKV_HVX_GQA
    const int hvx_status = edgekv_attn_decode_hvx(octx);
    if (hvx_status == HTP_STATUS_OK) {
        return hvx_status;
    }
#ifndef HTP_EDGEKV_DIRECT_AOT
    return hvx_status;
#else
    if (hvx_status != HTP_STATUS_NO_SUPPORT && hvx_status != HTP_STATUS_VTCM_TOO_SMALL) {
        return hvx_status;
    }
#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    edgekv_last_kernel_pcycles = 0;
#endif
#endif
#endif

#ifndef HTP_EDGEKV_DIRECT_AOT
    return HTP_STATUS_NO_SUPPORT;
#else
    float *    q_base        = (float *) (uintptr_t) q_storage->data;
    int8_t *   u_base        = (int8_t *) (uintptr_t) u_storage->data;
    int8_t *   vh_base       = (int8_t *) (uintptr_t) vh_storage->data;
    uint8_t *  metadata_base = (uint8_t *) (uintptr_t) metadata_storage->data;
    uint16_t * active_k_base = (uint16_t *) (uintptr_t) active_k_storage->data;
    uint16_t * active_v_base = (uint16_t *) (uintptr_t) active_v_storage->data;
    float *    output_base   = (float *) (uintptr_t) dst->data;

    struct edgekv_memref_f32_2_direct q = {
        .allocated = q_base,
        .aligned   = q_base,
        .offset    = 0,
        .sizes     = { 16, 128 },
        .strides   = { 128, 1 },
    };
    struct edgekv_memref_i8_3_direct k_u = {
        .allocated = u_base,
        .aligned   = u_base,
        .offset    = 0,
        .sizes     = { 16, 64, 32 },
        .strides   = { 2048, 32, 1 },
    };
    struct edgekv_memref_i8_5_direct k_vh = {
        .allocated = vh_base,
        .aligned   = vh_base,
        .offset    = 0,
        .sizes     = { 16, 32, 4, 8, 128 },
        .strides   = { 131072, 4096, 1024, 128, 1 },
    };
    struct edgekv_memref_f32_2_direct k_scale = {
        .allocated = (float *) metadata_base,
        .aligned   = (float *) metadata_base,
        .offset    = 0,
        .sizes     = { 16, 32 },
        .strides   = { 32, 1 },
    };

    int8_t * v_u_base  = u_base + 32768;
    int8_t * v_vh_base = vh_base + 2097152;
    struct edgekv_memref_i8_3_direct v_u = {
        .allocated = v_u_base,
        .aligned   = v_u_base,
        .offset    = 0,
        .sizes     = { 16, 64, 32 },
        .strides   = { 2048, 32, 1 },
    };
    struct edgekv_memref_i8_5_direct v_vh = {
        .allocated = v_vh_base,
        .aligned   = v_vh_base,
        .offset    = 0,
        .sizes     = { 16, 32, 4, 8, 128 },
        .strides   = { 131072, 4096, 1024, 128, 1 },
    };
    struct edgekv_memref_f32_2_direct v_scale = {
        .allocated = (float *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_V_SCALE_OFFSET_BYTES]),
        .aligned   = (float *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_V_SCALE_OFFSET_BYTES]),
        .offset    = 0,
        .sizes     = { 16, 32 },
        .strides   = { 32, 1 },
    };
    struct edgekv_memref_i32_2_direct block_positions = {
        .allocated = (int32_t *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_BLOCK_POSITIONS_OFFSET_BYTES]),
        .aligned   = (int32_t *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_BLOCK_POSITIONS_OFFSET_BYTES]),
        .offset    = 0,
        .sizes     = { 16, 64 },
        .strides   = { 64, 1 },
    };
    struct edgekv_memref_f16_3_direct active_k = {
        .allocated = active_k_base,
        .aligned   = active_k_base,
        .offset    = 0,
        .sizes     = { 128, 8, 128 },
        .strides   = { 1024, 128, 1 },
    };
    struct edgekv_memref_f16_3_direct active_v = {
        .allocated = active_v_base,
        .aligned   = active_v_base,
        .offset    = 0,
        .sizes     = { 128, 8, 128 },
        .strides   = { 1024, 128, 1 },
    };

    int32_t * active_positions_base =
        (int32_t *) (metadata_base + p[HTP_EDGEKV_ATTN_METADATA_RECENT_POSITIONS_OFFSET_BYTES]);
    int32_t * query_position_base = active_positions_base + p[HTP_EDGEKV_ATTN_RECENT_SIZE];
    struct edgekv_memref_i32_1_direct active_positions = {
        .allocated = active_positions_base,
        .aligned   = active_positions_base,
        .offset    = 0,
        .sizes     = { 128 },
        .strides   = { 1 },
    };
    struct edgekv_memref_i32_1_direct query_position = {
        .allocated = query_position_base,
        .aligned   = query_position_base,
        .offset    = 0,
        .sizes     = { 1 },
        .strides   = { 1 },
    };
    struct edgekv_memref_f32_2_direct output = {
        .allocated = output_base,
        .aligned   = output_base,
        .offset    = 0,
        .sizes     = { 16, 128 },
        .strides   = { 128, 1 },
    };

#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    const uint64_t start = HAP_perf_get_pcycles();
#endif
    // The generated symbol executes one Triton program per call; reproduce the mobile grid explicitly.
    for (int program_id = 0; program_id < 4; ++program_id) {
        edgekv_attn_decode_direct_int8_kernel(
            0, &q, 0, &k_u, 0, &k_vh, 0, &k_scale, 0, &v_u, 0, &v_vh, 0, &v_scale, 0, &block_positions, 0,
            &active_k, 0, &active_v, 0, &active_positions, 0, &query_position, 0, &output,
            p[HTP_EDGEKV_ATTN_LAYER_INDEX], 4, 1, 1, program_id, 0, 0);
    }
#ifdef HTP_EDGEKV_PROFILE_KERNEL_ENABLED
    edgekv_last_kernel_pcycles = HAP_perf_get_pcycles() - start;
#endif

    return HTP_STATUS_OK;
#endif
}
