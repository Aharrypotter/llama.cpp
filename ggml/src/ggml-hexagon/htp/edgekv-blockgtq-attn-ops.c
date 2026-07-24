// SPDX-License-Identifier: BSD-3-Clause

#include "edgekv-blockgtq-attn.h"
#include "hex-utils.h"
#include "htp-ctx.h"

#include "HAP_perf.h"

#include <stdint.h>
#include <string.h>

enum {
    EDGEKV_BLOCKGTQ_PACKAGE_ID = 0x7159380b,
    EDGEKV_BLOCKGTQ_CONTRACT_ID = 0x92f8aa7c,
};

static uint64_t last_kernel_pcycles;

#ifdef HTP_EDGEKV_BLOCKGTQ_TEST
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
static float diagnostic_logits[
    EDGEKV_BLOCKGTQ_QUERY_HEADS * EDGEKV_BLOCKGTQ_CAPACITY];
static float diagnostic_weights[
    EDGEKV_BLOCKGTQ_QUERY_HEADS * EDGEKV_BLOCKGTQ_CAPACITY];
#else
static float diagnostic_logits[EDGEKV_BLOCKGTQ_QUERY_HEADS * 5];
static float diagnostic_weights[EDGEKV_BLOCKGTQ_QUERY_HEADS * 5];
#endif
static float diagnostic_denominators[EDGEKV_BLOCKGTQ_QUERY_HEADS];
static float diagnostic_rotated_v[EDGEKV_BLOCKGTQ_OUTPUT_FLOATS];
static struct edgekv_blockgtq_diagnostics test_diagnostics = {
    .logits = diagnostic_logits,
    .weights = diagnostic_weights,
    .denominators = diagnostic_denominators,
    .rotated_v = diagnostic_rotated_v,
};

const struct edgekv_blockgtq_diagnostics *
edgekv_blockgtq_attn_decode_test_diagnostics(void) {
    return &test_diagnostics;
}
#endif

#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
static int test_diagnostics_enabled = 1;
static struct edgekv_blockgtq_operation_attribution last_attribution;

static uint64_t read_pcycles(void * context) {
    (void) context;
    return HAP_perf_get_pcycles();
}

void edgekv_blockgtq_attn_decode_test_set_diagnostics(int enabled) {
    test_diagnostics_enabled = enabled != 0;
}

const struct edgekv_blockgtq_operation_attribution *
edgekv_blockgtq_attn_decode_last_attribution(void) {
    return &last_attribution;
}
#endif

uint64_t edgekv_blockgtq_attn_decode_last_pcycles(void) {
    return last_kernel_pcycles;
}

static int valid_params(const int32_t * p) {
    return p[HTP_EDGEKV_BLOCKGTQ_ABI_VERSION] ==
               EDGEKV_BLOCKGTQ_ABI_VERSION &&
           (uint32_t) p[HTP_EDGEKV_BLOCKGTQ_PACKAGE_ID] ==
               EDGEKV_BLOCKGTQ_PACKAGE_ID &&
           (uint32_t) p[HTP_EDGEKV_BLOCKGTQ_CONTRACT_ID] ==
               EDGEKV_BLOCKGTQ_CONTRACT_ID &&
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
               EDGEKV_BLOCKGTQ_TRANSFORM_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_CONSUMER_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_CONSUMER_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_SHARED_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_SHARED_BYTES &&
           p[HTP_EDGEKV_BLOCKGTQ_DYNAMIC_BYTES_PARAM] ==
               EDGEKV_BLOCKGTQ_DYNAMIC_BYTES;
}

int op_edgekv_blockgtq_attn_decode(struct htp_ops_context * octx) {
    last_kernel_pcycles = 0;
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    memset(&last_attribution, 0, sizeof(last_attribution));
    const uint64_t adapter_start = HAP_perf_get_pcycles();
#endif
    if (octx == NULL || octx->src[0] == NULL || octx->src[1] == NULL ||
        octx->src[2] == NULL || octx->src[3] == NULL ||
        octx->src[4] == NULL || octx->src[5] != NULL ||
        octx->dst == NULL) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const struct htp_tensor * query = octx->src[0];
    const struct htp_tensor * transform = octx->src[1];
    const struct htp_tensor * consumer = octx->src[2];
    const struct htp_tensor * shared = octx->src[3];
    const struct htp_tensor * history = octx->src[4];
    const struct htp_tensor * output = octx->dst;
    if (query->type != HTP_TYPE_F32 || transform->type != HTP_TYPE_I8 ||
        consumer->type != HTP_TYPE_I8 || shared->type != HTP_TYPE_I8 ||
        history->type != HTP_TYPE_I8 || output->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!valid_params(octx->op_params)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (query->size != EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float) ||
        transform->size != EDGEKV_BLOCKGTQ_TRANSFORM_BYTES ||
        consumer->size != EDGEKV_BLOCKGTQ_CONSUMER_BYTES ||
        shared->size != EDGEKV_BLOCKGTQ_SHARED_BYTES ||
        history->size != EDGEKV_BLOCKGTQ_DYNAMIC_BYTES ||
        output->size != EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (!hex_is_aligned((const void *) (uintptr_t) query->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) transform->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) consumer->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) shared->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) history->data, 128) ||
        !hex_is_aligned((const void *) (uintptr_t) output->data, 128)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    last_attribution.adapter_input_validation_pcycles =
        HAP_perf_get_pcycles() - adapter_start;
#endif
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const struct edgekv_blockgtq_inputs inputs = {
        .query = (const float *) (uintptr_t) query->data,
        .transform = (const uint8_t *) (uintptr_t) transform->data,
        .transform_bytes = transform->size,
        .consumer = (const uint8_t *) (uintptr_t) consumer->data,
        .consumer_bytes = consumer->size,
        .shared = (const uint8_t *) (uintptr_t) shared->data,
        .shared_bytes = shared->size,
        .history = (const uint8_t *) (uintptr_t) history->data,
        .history_bytes = history->size,
        .layer_index =
            octx->op_params[HTP_EDGEKV_BLOCKGTQ_LAYER_INDEX],
        .sequence_length =
            octx->op_params[HTP_EDGEKV_BLOCKGTQ_SEQUENCE_LENGTH],
        .capacity = octx->op_params[HTP_EDGEKV_BLOCKGTQ_CAPACITY],
    };
#ifdef HTP_EDGEKV_BLOCKGTQ_TEST
    memset(
        diagnostic_logits, 0,
        (size_t) EDGEKV_BLOCKGTQ_QUERY_HEADS *
            (size_t) inputs.sequence_length * sizeof(float));
    memset(
        diagnostic_weights, 0,
        (size_t) EDGEKV_BLOCKGTQ_QUERY_HEADS *
            (size_t) inputs.sequence_length * sizeof(float));
    memset(diagnostic_denominators, 0, sizeof(diagnostic_denominators));
    memset(diagnostic_rotated_v, 0, sizeof(diagnostic_rotated_v));
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    struct edgekv_blockgtq_diagnostics * diagnostics =
        test_diagnostics_enabled ? &test_diagnostics : NULL;
#else
    struct edgekv_blockgtq_diagnostics * diagnostics =
        inputs.sequence_length <= 5 ? &test_diagnostics : NULL;
#endif
#else
    struct edgekv_blockgtq_diagnostics * diagnostics = NULL;
#endif
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    for (int sample = 0;
         sample < EDGEKV_BLOCKGTQ_COUNTER_OVERHEAD_SAMPLES; ++sample) {
        const uint64_t overhead_start = HAP_perf_get_pcycles();
        last_attribution.counter_overhead_pcycles[sample] =
            HAP_perf_get_pcycles() - overhead_start;
    }
    last_attribution.diagnostics_enabled = diagnostics != NULL;
    last_attribution.core.clock = read_pcycles;
    last_attribution.core.clock_context = NULL;
#endif
    const uint64_t start = HAP_perf_get_pcycles();
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    const int status = edgekv_blockgtq_attn_decode_profiled(
        &inputs, (float *) (uintptr_t) output->data, diagnostics,
        &last_attribution.core);
#else
    const int status = edgekv_blockgtq_attn_decode(
        &inputs, (float *) (uintptr_t) output->data, diagnostics);
#endif
    last_kernel_pcycles = HAP_perf_get_pcycles() - start;
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION
    last_attribution.operation_body_pcycles = last_kernel_pcycles;
#endif
    return status == EDGEKV_BLOCKGTQ_OK ? HTP_STATUS_OK
                                       : HTP_STATUS_INVAL_PARAMS;
}
