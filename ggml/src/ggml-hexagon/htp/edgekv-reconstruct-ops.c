// SPDX-License-Identifier: BSD-3-Clause

#include "hex-utils.h"
#include "htp-ctx.h"

#include <stdint.h>
#include <string.h>

// These descriptors mirror Hexagon-MLIR's ranked memref ABI. The current AOT
// artifact is the exact C3 profile; later profiles can share this adapter and
// select a different symbol after the shape check.
struct edgekv_memref_i8_3 {
    int8_t * allocated;
    int8_t * aligned;
    int64_t  offset;
    int64_t  sizes[3];
    int64_t  strides[3];
};

struct edgekv_memref_i8_5 {
    int8_t * allocated;
    int8_t * aligned;
    int64_t  offset;
    int64_t  sizes[5];
    int64_t  strides[5];
};

struct edgekv_memref_f32_2 {
    float * allocated;
    float * aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

struct edgekv_memref_i32_2 {
    int32_t * allocated;
    int32_t * aligned;
    int64_t   offset;
    int64_t   sizes[2];
    int64_t   strides[2];
};

struct edgekv_memref_f16_4 {
    uint16_t * allocated;
    uint16_t * aligned;
    int64_t    offset;
    int64_t    sizes[4];
    int64_t    strides[4];
};

#ifdef HTP_EDGEKV_RECONSTRUCT_AOT
extern void edgekv_reconstruct_tile_int8_kernel(int64_t,
                                                struct edgekv_memref_i8_3 *,
                                                int64_t,
                                                struct edgekv_memref_i8_5 *,
                                                int64_t,
                                                struct edgekv_memref_f32_2 *,
                                                int64_t,
                                                struct edgekv_memref_i8_3 *,
                                                int64_t,
                                                struct edgekv_memref_i8_5 *,
                                                int64_t,
                                                struct edgekv_memref_f32_2 *,
                                                int64_t,
                                                struct edgekv_memref_i32_2 *,
                                                int64_t,
                                                struct edgekv_memref_f16_4 *,
                                                int64_t,
                                                struct edgekv_memref_f16_4 *,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int);
#endif

static bool edgekv_is_c3_v79_profile(const int32_t * p) {
    static const int32_t expected[HTP_EDGEKV_PARAM_COUNT] = {
        2, 16, 8, 4, 2, 1, 2, 32, 16, 256, 2048, 128, 4096, 6144,
    };

    for (int i = 0; i < HTP_EDGEKV_PARAM_COUNT; ++i) {
        if (p[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

int op_edgekv_reconstruct(struct htp_ops_context * octx) {
    const struct htp_tensor * u_storage     = octx->src[0];
    const struct htp_tensor * vh_storage    = octx->src[1];
    const struct htp_tensor * scale_storage = octx->src[2];
    const struct htp_tensor * positions     = octx->src[3];
    const struct htp_tensor * dst           = octx->dst;

    if (!u_storage || !vh_storage || !scale_storage || !positions || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (u_storage->type != HTP_TYPE_I8 || vh_storage->type != HTP_TYPE_I8 || scale_storage->type != HTP_TYPE_F32 ||
        positions->type != HTP_TYPE_I32 || dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!edgekv_is_c3_v79_profile(octx->op_params)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (u_storage->size < 384 || vh_storage->size < 2560 || scale_storage->size < 256 || positions->size < 128 ||
        dst->size < 6144) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (!hex_is_aligned((const void *) (uintptr_t) u_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) vh_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) scale_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) positions->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) dst->data, 128)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

#ifndef HTP_EDGEKV_RECONSTRUCT_AOT
    return HTP_STATUS_NO_SUPPORT;
#else
    int8_t *   u_base        = (int8_t *) (uintptr_t) u_storage->data;
    int8_t *   vh_base       = (int8_t *) (uintptr_t) vh_storage->data;
    float *    scale_base    = (float *) (uintptr_t) scale_storage->data;
    int32_t *  position_base = (int32_t *) (uintptr_t) positions->data;
    uint16_t * dense_base    = (uint16_t *) (uintptr_t) dst->data;

    struct edgekv_memref_i8_3 k_u = {
        .allocated = u_base,
        .aligned   = u_base,
        .offset    = 0,
        .sizes     = { 2,   16, 8 },
        .strides   = { 128, 8,  1 },
    };
    struct edgekv_memref_i8_5 k_vh = {
        .allocated = vh_base,
        .aligned   = vh_base,
        .offset    = 0,
        .sizes     = { 2,    8,   2,   2,  32 },
        .strides   = { 2048, 256, 128, 32, 1  },
    };
    struct edgekv_memref_f32_2 k_scale = {
        .allocated = scale_base,
        .aligned   = scale_base,
        .offset    = 0,
        .sizes     = { 2, 8 },
        .strides   = { 8, 1 },
    };

    int8_t * v_u_base     = u_base + octx->op_params[HTP_EDGEKV_V_U_OFFSET_BYTES];
    int8_t * v_vh_base    = vh_base + octx->op_params[HTP_EDGEKV_V_VH_OFFSET_BYTES];
    float *  v_scale_base = (float *) ((uint8_t *) scale_base + octx->op_params[HTP_EDGEKV_V_SCALE_OFFSET_BYTES]);
    struct edgekv_memref_i8_3 v_u = {
        .allocated = v_u_base,
        .aligned   = v_u_base,
        .offset    = 0,
        .sizes     = { 2,  16, 4 },
        .strides   = { 64, 4,  1 },
    };
    struct edgekv_memref_i8_5 v_vh = {
        .allocated = v_vh_base,
        .aligned   = v_vh_base,
        .offset    = 0,
        .sizes     = { 2,   4,   2,  2,  16 },
        .strides   = { 512, 128, 64, 16, 1  },
    };
    struct edgekv_memref_f32_2 v_scale = {
        .allocated = v_scale_base,
        .aligned   = v_scale_base,
        .offset    = 0,
        .sizes     = { 2, 4 },
        .strides   = { 4, 1 },
    };
    struct edgekv_memref_i32_2 block_positions = {
        .allocated = position_base,
        .aligned   = position_base,
        .offset    = 0,
        .sizes     = { 2,  16 },
        .strides   = { 16, 1  },
    };

    uint16_t * dense_v_base = (uint16_t *) ((uint8_t *) dense_base + octx->op_params[HTP_EDGEKV_DENSE_V_OFFSET_BYTES]);
    struct edgekv_memref_f16_4 dense_k = {
        .allocated = dense_base,
        .aligned   = dense_base,
        .offset    = 0,
        .sizes     = { 2,    2,   16, 32 },
        .strides   = { 1024, 512, 32, 1  },
    };
    struct edgekv_memref_f16_4 dense_v = {
        .allocated = dense_v_base,
        .aligned   = dense_v_base,
        .offset    = 0,
        .sizes     = { 2,   2,   16, 16 },
        .strides   = { 512, 256, 16, 1  },
    };

    const int num_programs_x = 4;
    const int num_programs_y = 1;
    const int num_programs_z = 1;
    for (int pid_x = 0; pid_x < num_programs_x; ++pid_x) {
        edgekv_reconstruct_tile_int8_kernel(0, &k_u, 0, &k_vh, 0, &k_scale, 0, &v_u, 0, &v_vh, 0, &v_scale, 0,
                                            &block_positions, 0, &dense_k, 0, &dense_v, num_programs_x, num_programs_y,
                                            num_programs_z, pid_x, 0, 0);
    }

    return HTP_STATUS_OK;
#endif
}
