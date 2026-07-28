// SPDX-License-Identifier: BSD-3-Clause

#include "edgekv-blockgtq-attn.h"
#include "../../edgekv-blockgtq-runtime.h"
#include "hex-utils.h"
#include "htp-ctx.h"

#include <stdint.h>

enum {
    EDGEKV_BLOCKGTQ_V2_PACKAGE_ID = 0x7159380b,
    EDGEKV_BLOCKGTQ_V2_CONTRACT_ID = 0x92f8aa7c,
};

static int valid_params_v2(const int32_t * p) {
    return p[HTP_EDGEKV_BLOCKGTQ_ABI_VERSION] == 2 &&
           (uint32_t) p[HTP_EDGEKV_BLOCKGTQ_PACKAGE_ID] ==
               EDGEKV_BLOCKGTQ_V2_PACKAGE_ID &&
           (uint32_t) p[HTP_EDGEKV_BLOCKGTQ_CONTRACT_ID] ==
               EDGEKV_BLOCKGTQ_V2_CONTRACT_ID &&
           p[HTP_EDGEKV_BLOCKGTQ_LAYER_INDEX] >= 0 &&
           p[HTP_EDGEKV_BLOCKGTQ_LAYER_INDEX] <
               EDGEKV_BLOCKGTQ_LAYERS &&
           p[HTP_EDGEKV_BLOCKGTQ_SEQUENCE_LENGTH] > 0 &&
           p[HTP_EDGEKV_BLOCKGTQ_SEQUENCE_LENGTH] <=
               EDGEKV_BLOCKGTQ_CAPACITY &&
           p[HTP_EDGEKV_BLOCKGTQ_CAPACITY] ==
               EDGEKV_BLOCKGTQ_CAPACITY &&
           p[HTP_EDGEKV_BLOCKGTQ_N_HEAD_Q] ==
               EDGEKV_BLOCKGTQ_QUERY_HEADS &&
           p[HTP_EDGEKV_BLOCKGTQ_N_HEAD_KV] ==
               EDGEKV_BLOCKGTQ_KV_HEADS &&
           p[HTP_EDGEKV_BLOCKGTQ_HEAD_DIM] ==
               EDGEKV_BLOCKGTQ_HEAD_DIM &&
           p[HTP_EDGEKV_BLOCKGTQ_GQA] == EDGEKV_BLOCKGTQ_GQA &&
           p[HTP_EDGEKV_BLOCKGTQ_N_LAYERS] ==
               EDGEKV_BLOCKGTQ_LAYERS &&
           p[HTP_EDGEKV_BLOCKGTQ_TRANSFORM_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_PRODUCER_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_CONSUMER_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_CONSUMER_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_SHARED_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_SHARED_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_DYNAMIC_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_DYNAMIC_BYTES;
}

int op_edgekv_blockgtq_attn_decode_v2(struct htp_ops_context * octx) {
    if (octx == NULL || octx->src[0] == NULL || octx->src[1] == NULL ||
        octx->src[2] == NULL || octx->src[3] == NULL ||
        octx->src[4] == NULL || octx->src[5] == NULL ||
        octx->dst == NULL || !valid_params_v2(octx->op_params)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const struct htp_tensor * query = octx->src[0];
    const struct htp_tensor * producer = octx->src[1];
    const struct htp_tensor * consumer = octx->src[2];
    const struct htp_tensor * shared = octx->src[3];
    const struct htp_tensor * history = octx->src[4];
    const struct htp_tensor * current = octx->src[5];
    const struct htp_tensor * output = octx->dst;
    if (query->type != HTP_TYPE_F32 || producer->type != HTP_TYPE_I8 ||
        consumer->type != HTP_TYPE_I8 || shared->type != HTP_TYPE_I8 ||
        history->type != HTP_TYPE_I8 || current->type != HTP_TYPE_I8 ||
        output->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (query->size != EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float) ||
        producer->size != EDGEKV_BLOCKGTQ_PRODUCER_BYTES ||
        consumer->size != EDGEKV_BLOCKGTQ_CONSUMER_BYTES ||
        shared->size != EDGEKV_BLOCKGTQ_SHARED_BYTES ||
        history->size != EDGEKV_BLOCKGTQ_DYNAMIC_BYTES ||
        current->size != EDGEKV_BLOCKGTQ_APPEND_BYTES ||
        output->size != EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (!hex_is_aligned((const void *) (uintptr_t) query->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) producer->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) consumer->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) shared->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) history->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) current->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) output->data, 128)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int layer =
        octx->op_params[HTP_EDGEKV_BLOCKGTQ_LAYER_INDEX];
    const int sequence =
        octx->op_params[HTP_EDGEKV_BLOCKGTQ_SEQUENCE_LENGTH];
    int status = edgekv_blockgtq_commit_token_v2(
        (uint8_t *) (uintptr_t) history->data, history->size,
        EDGEKV_BLOCKGTQ_CAPACITY, layer, sequence - 1,
        (const uint8_t *) (uintptr_t) current->data, current->size);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct edgekv_blockgtq_inputs inputs = {
        .query = (const float *) (uintptr_t) query->data,
        .transform = (const uint8_t *) (uintptr_t) producer->data,
        .transform_bytes = producer->size,
        .transform_v_rotations_offset =
            EDGEKV_BLOCKGTQ_PRODUCER_V_ROTATIONS_OFFSET,
        .consumer = (const uint8_t *) (uintptr_t) consumer->data,
        .consumer_bytes = consumer->size,
        .shared = (const uint8_t *) (uintptr_t) shared->data,
        .shared_bytes = shared->size,
        .history = (const uint8_t *) (uintptr_t) history->data,
        .history_bytes = history->size,
        .layer_index = layer,
        .sequence_length = sequence,
        .capacity = EDGEKV_BLOCKGTQ_CAPACITY,
    };
    status = edgekv_blockgtq_attn_decode(
        &inputs, (float *) (uintptr_t) output->data, NULL);
    return status == EDGEKV_BLOCKGTQ_OK ? HTP_STATUS_OK
                                       : HTP_STATUS_INVAL_PARAMS;
}
