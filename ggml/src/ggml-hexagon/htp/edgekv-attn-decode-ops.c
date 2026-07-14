// SPDX-License-Identifier: BSD-3-Clause

#include "hex-utils.h"
#include "htp-ctx.h"

#include <stdint.h>

struct edgekv_memref_f16_2 {
    uint16_t * allocated;
    uint16_t * aligned;
    int64_t    offset;
    int64_t    sizes[2];
    int64_t    strides[2];
};

struct edgekv_memref_i8_3_direct {
    int8_t * allocated;
    int8_t * aligned;
    int64_t  offset;
    int64_t  sizes[3];
    int64_t  strides[3];
};

struct edgekv_memref_i8_5_direct {
    int8_t * allocated;
    int8_t * aligned;
    int64_t  offset;
    int64_t  sizes[5];
    int64_t  strides[5];
};

struct edgekv_memref_f32_2_direct {
    float * allocated;
    float * aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

struct edgekv_memref_i32_2_direct {
    int32_t * allocated;
    int32_t * aligned;
    int64_t   offset;
    int64_t   sizes[2];
    int64_t   strides[2];
};

struct edgekv_memref_f16_3 {
    uint16_t * allocated;
    uint16_t * aligned;
    int64_t    offset;
    int64_t    sizes[3];
    int64_t    strides[3];
};

struct edgekv_memref_i32_1 {
    int32_t * allocated;
    int32_t * aligned;
    int64_t   offset;
    int64_t   sizes[1];
    int64_t   strides[1];
};

#ifdef HTP_EDGEKV_DIRECT_AOT
extern void edgekv_attn_decode_direct_int8_kernel(int64_t,
                                                  struct edgekv_memref_f16_2 *,
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
                                                  struct edgekv_memref_f16_3 *,
                                                  int64_t,
                                                  struct edgekv_memref_f16_3 *,
                                                  int64_t,
                                                  struct edgekv_memref_i32_1 *,
                                                  int64_t,
                                                  struct edgekv_memref_i32_1 *,
                                                  int64_t,
                                                  struct edgekv_memref_f32_2_direct *,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int);
#endif

static bool edgekv_is_direct_small_v79_profile(const int32_t * p) {
    static const int32_t expected[HTP_EDGEKV_ATTN_PARAM_COUNT] = {
        2, 16, 8, 4, 2, 1, 4, 2, 32, 16, 8, 1024, 1536, 1664, 0x3e3504f3,
    };

    for (int i = 0; i < HTP_EDGEKV_ATTN_PARAM_COUNT; ++i) {
        if (p[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

int op_edgekv_attn_decode(struct htp_ops_context * octx) {
    const struct htp_tensor * q_storage         = octx->src[0];
    const struct htp_tensor * u_storage         = octx->src[1];
    const struct htp_tensor * vh_storage        = octx->src[2];
    const struct htp_tensor * scale_storage     = octx->src[3];
    const struct htp_tensor * positions_storage = octx->src[4];
    const struct htp_tensor * recent_storage    = octx->src[5];
    const struct htp_tensor * dst               = octx->dst;

    if (!q_storage || !u_storage || !vh_storage || !scale_storage || !positions_storage || !recent_storage || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (q_storage->type != HTP_TYPE_F16 || u_storage->type != HTP_TYPE_I8 || vh_storage->type != HTP_TYPE_I8 ||
        scale_storage->type != HTP_TYPE_F32 || positions_storage->type != HTP_TYPE_I32 ||
        recent_storage->type != HTP_TYPE_I8 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!edgekv_is_direct_small_v79_profile(octx->op_params)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q_storage->size < 256 || u_storage->size < 384 || vh_storage->size < 2560 || scale_storage->size < 256 ||
        positions_storage->size < 128 || recent_storage->size < 1664 || dst->size < 256) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (!hex_is_aligned((const void *) (uintptr_t) q_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) u_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) vh_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) scale_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) positions_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) recent_storage->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) dst->data, 128)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

#ifndef HTP_EDGEKV_DIRECT_AOT
    return HTP_STATUS_NO_SUPPORT;
#else
    uint16_t * q_base        = (uint16_t *) (uintptr_t) q_storage->data;
    int8_t *   u_base        = (int8_t *) (uintptr_t) u_storage->data;
    int8_t *   vh_base       = (int8_t *) (uintptr_t) vh_storage->data;
    float *    scale_base    = (float *) (uintptr_t) scale_storage->data;
    int32_t *  position_base = (int32_t *) (uintptr_t) positions_storage->data;
    uint8_t *  recent_base   = (uint8_t *) (uintptr_t) recent_storage->data;
    float *    output_base   = (float *) (uintptr_t) dst->data;

    struct edgekv_memref_f16_2 q = {
        .allocated = q_base,
        .aligned   = q_base,
        .offset    = 0,
        .sizes     = { 4,  32 },
        .strides   = { 32, 1  },
    };
    struct edgekv_memref_i8_3_direct k_u = {
        .allocated = u_base,
        .aligned   = u_base,
        .offset    = 0,
        .sizes     = { 2,   16, 8 },
        .strides   = { 128, 8,  1 },
    };
    struct edgekv_memref_i8_5_direct k_vh = {
        .allocated = vh_base,
        .aligned   = vh_base,
        .offset    = 0,
        .sizes     = { 2,    8,   2,   2,  32 },
        .strides   = { 2048, 256, 128, 32, 1  },
    };
    struct edgekv_memref_f32_2_direct k_scale = {
        .allocated = scale_base,
        .aligned   = scale_base,
        .offset    = 0,
        .sizes     = { 2, 8 },
        .strides   = { 8, 1 },
    };

    int8_t * v_u_base     = u_base + 256;
    int8_t * v_vh_base    = vh_base + 2048;
    float *  v_scale_base = (float *) ((uint8_t *) scale_base + 128);
    struct edgekv_memref_i8_3_direct v_u = {
        .allocated = v_u_base,
        .aligned   = v_u_base,
        .offset    = 0,
        .sizes     = { 2,  16, 4 },
        .strides   = { 64, 4,  1 },
    };
    struct edgekv_memref_i8_5_direct v_vh = {
        .allocated = v_vh_base,
        .aligned   = v_vh_base,
        .offset    = 0,
        .sizes     = { 2,   4,   2,  2,  16 },
        .strides   = { 512, 128, 64, 16, 1  },
    };
    struct edgekv_memref_f32_2_direct v_scale = {
        .allocated = v_scale_base,
        .aligned   = v_scale_base,
        .offset    = 0,
        .sizes     = { 2, 4 },
        .strides   = { 4, 1 },
    };
    struct edgekv_memref_i32_2_direct block_positions = {
        .allocated = position_base,
        .aligned   = position_base,
        .offset    = 0,
        .sizes     = { 2,  16 },
        .strides   = { 16, 1  },
    };

    uint16_t * recent_k_base        = (uint16_t *) recent_base;
    uint16_t * recent_v_base        = (uint16_t *) (recent_base + 1024);
    int32_t *  recent_position_base = (int32_t *) (recent_base + 1536);
    int32_t *  query_position_base  = recent_position_base + 8;
    struct edgekv_memref_f16_3 recent_k = {
        .allocated = recent_k_base,
        .aligned   = recent_k_base,
        .offset    = 0,
        .sizes     = { 8,  2,  32 },
        .strides   = { 64, 32, 1  },
    };
    struct edgekv_memref_f16_3 recent_v = {
        .allocated = recent_v_base,
        .aligned   = recent_v_base,
        .offset    = 0,
        .sizes     = { 8,  2,  16 },
        .strides   = { 32, 16, 1  },
    };
    struct edgekv_memref_i32_1 recent_positions = {
        .allocated = recent_position_base,
        .aligned   = recent_position_base,
        .offset    = 0,
        .sizes     = { 8 },
        .strides   = { 1 },
    };
    struct edgekv_memref_i32_1 query_position = {
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
        .sizes     = { 4,  16 },
        .strides   = { 16, 1  },
    };

    edgekv_attn_decode_direct_int8_kernel(0, &q, 0, &k_u, 0, &k_vh, 0, &k_scale, 0, &v_u, 0, &v_vh, 0, &v_scale, 0,
                                          &block_positions, 0, &recent_k, 0, &recent_v, 0, &recent_positions, 0,
                                          &query_position, 0, &output, 1, 1, 1, 0, 0, 0);

    return HTP_STATUS_OK;
#endif
}
