// SPDX-License-Identifier: BSD-3-Clause

#include "edgekv-blockgtq-attn.h"
#include "htp-ctx.h"

#include "HAP_compute_res.h"
#include "HAP_perf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern const uint8_t edgekv_blockgtq_fixture_start[];
extern const uint8_t edgekv_blockgtq_fixture_end[];

enum {
    N_THREADS = 4,
#if defined(HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX) || \
    defined(HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX)
    CASE_COUNT = 4,
    MAX_SEQUENCE = EDGEKV_BLOCKGTQ_CAPACITY,
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
    FORMAL_SAMPLES = 5,
#endif
#else
    CASE_COUNT = 36,
    MAX_SEQUENCE = 5,
#endif
    FIXTURE_HEADER_BYTES = 240,
#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
    FIXTURE_RECORD_BYTES = 144,
    H7_IDENTITY_BYTES_PER_HEAD = 40,
#else
    FIXTURE_RECORD_BYTES = 136,
#endif
};

_Static_assert(FIXTURE_HEADER_BYTES == 240, "fixture header schema drift");
#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
_Static_assert(FIXTURE_RECORD_BYTES == 144, "H7 fixture record schema drift");
#else
_Static_assert(FIXTURE_RECORD_BYTES == 136, "fixture record schema drift");
#endif

static uint8_t history[EDGEKV_BLOCKGTQ_DYNAMIC_BYTES]
    __attribute__((aligned(128)));
#ifndef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
static uint8_t mutated_consumer[EDGEKV_BLOCKGTQ_CONSUMER_BYTES]
    __attribute__((aligned(128)));
#endif
static float output[EDGEKV_BLOCKGTQ_OUTPUT_FLOATS]
    __attribute__((aligned(128)));
static _Float16 dense_k[MAX_SEQUENCE * EDGEKV_BLOCKGTQ_KV_HEADS *
                        EDGEKV_BLOCKGTQ_HEAD_DIM]
    __attribute__((aligned(128)));
static _Float16 dense_v[MAX_SEQUENCE * EDGEKV_BLOCKGTQ_KV_HEADS *
                        EDGEKV_BLOCKGTQ_HEAD_DIM]
    __attribute__((aligned(128)));
static _Float16 dense_mask[MAX_SEQUENCE] __attribute__((aligned(128)));
static float dense_output[EDGEKV_BLOCKGTQ_OUTPUT_FLOATS]
    __attribute__((aligned(128)));

static uint32_t read_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static uint64_t read_u64(const uint8_t * p) {
    return (uint64_t) read_u32(p) | ((uint64_t) read_u32(p + 4) << 32);
}

static float read_f32(const uint8_t * p) {
    const uint32_t bits = read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static struct htp_tensor make_tensor(void * data, uint32_t size,
                                     uint16_t type) {
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
    HAP_compute_res_attr_set_vtcm_param_v2(
        &attr, vtcm_size, vtcm_size, vtcm_size);
    ctx->vtcm_rctx = HAP_compute_res_acquire(&attr, 1000000u);
    if (!ctx->vtcm_rctx) {
        return 0;
    }
    void * vtcm_ptr = NULL;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(
            &attr, &vtcm_ptr, &vtcm_size) != 0) {
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

#ifndef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
static uint64_t median3(uint64_t a, uint64_t b, uint64_t c) {
    if (a > b) {
        const uint64_t temporary = a;
        a = b;
        b = temporary;
    }
    if (b > c) {
        const uint64_t temporary = b;
        b = c;
        c = temporary;
    }
    return a > b ? a : b;
}
#endif

#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
static void print_attribution_sample(int sequence, const char * mode,
                                     int sample) {
    const struct edgekv_blockgtq_operation_attribution * attribution =
        edgekv_blockgtq_attn_decode_last_attribution();
    printf(
        "ATTR: sequence=%d mode=%s sample=%d adapter=%llu total=%llu "
        "overhead=[",
        sequence, mode, sample,
        (unsigned long long)
            attribution->adapter_input_validation_pcycles,
        (unsigned long long) attribution->operation_body_pcycles);
    for (int index = 0;
         index < EDGEKV_BLOCKGTQ_COUNTER_OVERHEAD_SAMPLES; ++index) {
        printf(
            "%s%llu", index == 0 ? "" : ",",
            (unsigned long long)
                attribution->counter_overhead_pcycles[index]);
    }
    printf("] stages=[");
    for (int stage = 0; stage < EDGEKV_BLOCKGTQ_STAGE_COUNT; ++stage) {
        printf(
            "%s%s:%llu:%lu", stage == 0 ? "" : ",",
            edgekv_blockgtq_stage_name((enum edgekv_blockgtq_stage) stage),
            (unsigned long long) attribution->core.pcycles[stage],
            (unsigned long) attribution->core.intervals[stage]);
    }
    printf("]\n");
}
#endif

static float max_abs(const float * actual, const uint8_t * expected,
                     size_t count) {
    float maximum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = actual[i];
        const float error = fabsf(value - read_f32(expected + i * 4));
        if (!isfinite(value)) {
            return INFINITY;
        }
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
struct h7_mixed_metrics {
    float max_abs;
    float max_relative;
    float max_ratio;
    int finite;
};

struct h7_head_metrics {
    struct h7_mixed_metrics weights;
    struct h7_mixed_metrics denominator;
    struct h7_mixed_metrics rotated_v;
    struct h7_mixed_metrics output;
    uint32_t denominator_ulp;
    int gate_a;
    int gate_b;
    int gate_c;
};

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float float32_ulp(float reference) {
    const uint32_t bits = float_bits(fabsf(reference));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        return INFINITY;
    }
    if (bits == 0) {
        return bits_float(1);
    }
    return bits_float(bits + 1u) - bits_float(bits);
}

static uint32_t positive_ulp_distance(float actual, float expected) {
    if (!(actual >= 0.0f) || !(expected >= 0.0f) ||
        !isfinite(actual) || !isfinite(expected)) {
        return UINT32_MAX;
    }
    const uint32_t actual_bits = float_bits(actual);
    const uint32_t expected_bits = float_bits(expected);
    return actual_bits > expected_bits ? actual_bits - expected_bits
                                       : expected_bits - actual_bits;
}

static struct h7_mixed_metrics h7_mixed_metrics(
    const float * actual, const uint8_t * expected, size_t count,
    float ulp_limit, float relative_limit, float absolute_floor) {
    struct h7_mixed_metrics metrics = {0.0f, 0.0f, 0.0f, 1};
    for (size_t index = 0; index < count; ++index) {
        const float candidate = actual[index];
        const float reference = read_f32(expected + index * sizeof(float));
        if (!isfinite(candidate) || !isfinite(reference)) {
            metrics.finite = 0;
            metrics.max_abs = INFINITY;
            metrics.max_relative = INFINITY;
            metrics.max_ratio = INFINITY;
            return metrics;
        }
        const float error = fabsf(candidate - reference);
        const float relative =
            error / fmaxf(fabsf(reference), absolute_floor);
        const float bound = fmaxf(
            ulp_limit * float32_ulp(reference),
            fmaxf(relative_limit * fabsf(reference), absolute_floor));
        const float ratio = error / bound;
        if (error > metrics.max_abs) {
            metrics.max_abs = error;
        }
        if (relative > metrics.max_relative) {
            metrics.max_relative = relative;
        }
        if (ratio > metrics.max_ratio) {
            metrics.max_ratio = ratio;
        }
    }
    return metrics;
}

static int h7_observability_equal(
    const struct edgekv_blockgtq_target_observability * left,
    const struct edgekv_blockgtq_target_observability * right) {
    return memcmp(
               left->packed_code_bits_fnv1a64,
               right->packed_code_bits_fnv1a64,
               sizeof(left->packed_code_bits_fnv1a64)) == 0 &&
           memcmp(
               left->lut_bits_fnv1a64, right->lut_bits_fnv1a64,
               sizeof(left->lut_bits_fnv1a64)) == 0 &&
           memcmp(
               left->norm_bits_fnv1a64, right->norm_bits_fnv1a64,
               sizeof(left->norm_bits_fnv1a64)) == 0 &&
           memcmp(
               left->maximum_logit_bits, right->maximum_logit_bits,
               sizeof(left->maximum_logit_bits)) == 0 &&
           memcmp(
               left->logit_bits_fnv1a64, right->logit_bits_fnv1a64,
               sizeof(left->logit_bits_fnv1a64)) == 0 &&
           memcmp(
               left->denominator_bits, right->denominator_bits,
               sizeof(left->denominator_bits)) == 0 &&
           memcmp(
               left->weight_bits_fnv1a64, right->weight_bits_fnv1a64,
               sizeof(left->weight_bits_fnv1a64)) == 0 &&
           memcmp(
               left->rotated_v_bits_fnv1a64,
               right->rotated_v_bits_fnv1a64,
               sizeof(left->rotated_v_bits_fnv1a64)) == 0 &&
           memcmp(
               left->output_bits_fnv1a64, right->output_bits_fnv1a64,
               sizeof(left->output_bits_fnv1a64)) == 0;
}

static void print_h7_head(
    int case_index, int sequence, const char * mode, int query_head,
    const struct edgekv_blockgtq_target_observability * target,
    const struct h7_head_metrics * metrics, int mode_parity) {
    printf(
        "EDGEKV_H7_HEAD schema=1 case=%d sequence=%d mode=%s "
        "query_head=%d storage_head=%d code=%016llx lut=%016llx "
        "norm=%016llx max=%08lx logits=%016llx weights=%016llx "
        "denominator=%08lx rotated_v=%016llx output=%016llx "
        "weight_abs=%.9g weight_rel=%.9g weight_ratio=%.9g "
        "denom_abs=%.9g denom_rel=%.9g denom_ratio=%.9g denom_ulp=%lu "
        "rotated_abs=%.9g rotated_rel=%.9g rotated_ratio=%.9g "
        "output_abs=%.9g output_rel=%.9g output_ratio=%.9g "
        "gate_a=%d gate_b=%d gate_c=%d mode_parity=%d\n",
        case_index, sequence, mode, query_head,
        34 + query_head / EDGEKV_BLOCKGTQ_GQA,
        (unsigned long long)
            target->packed_code_bits_fnv1a64[query_head],
        (unsigned long long) target->lut_bits_fnv1a64[query_head],
        (unsigned long long) target->norm_bits_fnv1a64[query_head],
        (unsigned long) target->maximum_logit_bits[query_head],
        (unsigned long long) target->logit_bits_fnv1a64[query_head],
        (unsigned long long) target->weight_bits_fnv1a64[query_head],
        (unsigned long) target->denominator_bits[query_head],
        (unsigned long long) target->rotated_v_bits_fnv1a64[query_head],
        (unsigned long long) target->output_bits_fnv1a64[query_head],
        (double) metrics->weights.max_abs,
        (double) metrics->weights.max_relative,
        (double) metrics->weights.max_ratio,
        (double) metrics->denominator.max_abs,
        (double) metrics->denominator.max_relative,
        (double) metrics->denominator.max_ratio,
        (unsigned long) metrics->denominator_ulp,
        (double) metrics->rotated_v.max_abs,
        (double) metrics->rotated_v.max_relative,
        (double) metrics->rotated_v.max_ratio,
        (double) metrics->output.max_abs,
        (double) metrics->output.max_relative,
        (double) metrics->output.max_ratio,
        metrics->gate_a, metrics->gate_b, metrics->gate_c, mode_parity);
}
#endif

#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
#ifndef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
#if !defined(EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS) && \
    !defined(EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS)
static uint64_t fnv1a64_bytes(const uint8_t * bytes, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint32_t expected_maximum_bits(const uint8_t * expected_logits,
                                      int query_head, int sequence) {
    const uint8_t * row =
        expected_logits +
        (size_t) query_head * (size_t) sequence * sizeof(float);
    float maximum = read_f32(row);
    uint32_t maximum_bits = read_u32(row);
    for (int token = 1; token < sequence; ++token) {
        const uint8_t * value_bytes = row + (size_t) token * sizeof(float);
        const float value = read_f32(value_bytes);
        if (value > maximum) {
            maximum = value;
            maximum_bits = read_u32(value_bytes);
        }
    }
    return maximum_bits;
}

static void print_target_observability_records(
    const char * mode, int layer, int sequence,
    const uint8_t * expected_logits, const uint8_t * expected_weights,
    const uint8_t * expected_denominators) {
    const struct edgekv_blockgtq_target_observability * target =
        edgekv_blockgtq_attn_decode_test_observability();
    for (int query_head = 0;
         query_head < EDGEKV_BLOCKGTQ_QUERY_HEADS; ++query_head) {
        const uint8_t * expected_weight_row =
            expected_weights +
            (size_t) query_head * (size_t) sequence * sizeof(float);
        const uint64_t expected_weight_hash = fnv1a64_bytes(
            expected_weight_row, (size_t) sequence * sizeof(float));
        printf(
            "EDGEKV_OBS magic=BGTQH3O1 schema=1 "
            "profile=qwen2_5_3b_reference layer=%d sequence=%d "
            "mode=%s head=%d expected_max_logit=%08lx "
            "target_max_logit=%08lx "
            "expected_logits_fnv1a64=%016llx "
            "target_logits_fnv1a64=%016llx "
            "expected_denominator=%08lx "
            "target_denominator=%08lx "
            "expected_weights_fnv1a64=%016llx "
            "target_weights_fnv1a64=%016llx\n",
            layer, sequence, mode, query_head,
            (unsigned long) expected_maximum_bits(
                expected_logits, query_head, sequence),
            (unsigned long) target->maximum_logit_bits[query_head],
            (unsigned long long) fnv1a64_bytes(
                expected_logits +
                    (size_t) query_head * (size_t) sequence *
                        sizeof(float),
                (size_t) sequence * sizeof(float)),
            (unsigned long long)
                target->logit_bits_fnv1a64[query_head],
            (unsigned long) read_u32(
                expected_denominators +
                (size_t) query_head * sizeof(float)),
            (unsigned long) target->denominator_bits[query_head],
            (unsigned long long) expected_weight_hash,
            (unsigned long long)
                target->weight_bits_fnv1a64[query_head]);
    }
}
#endif
#endif

#ifdef EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS
static void print_target_query_rotation_forensics_records(void) {
    const struct edgekv_blockgtq_target_observability * target =
        edgekv_blockgtq_attn_decode_test_observability();
    for (int inner = 0; inner < 22; ++inner) {
        printf(
            "EDGEKV_H5_TERM magic=BGTQH5T1 schema=1 "
            "profile=qwen2_5_3b_reference layer=17 sequence=128 "
            "mode=diagnostics_on query_head=11 storage_head=35 "
            "segment=0 out=2 inner=%d source_dim=%lu query=%08lx "
            "rotation=%08lx product=%08lx separate_acc=%08lx "
            "fma_acc=%08lx\n",
            inner, (unsigned long) target->h5_source_dims[inner],
            (unsigned long) target->h5_query_bits[inner],
            (unsigned long) target->h5_rotation_bits[inner],
            (unsigned long) target->h5_product_bits[inner],
            (unsigned long)
                target->h5_separate_accumulator_bits[inner],
            (unsigned long) target->h5_fma_accumulator_bits[inner]);
    }
    printf(
        "EDGEKV_H5_SUMMARY magic=BGTQH5S1 schema=1 "
        "profile=qwen2_5_3b_reference layer=17 sequence=128 "
        "mode=diagnostics_on query_head=11 storage_head=35 "
        "segment=0 out=2 terms=22 original_query_fnv1a64=%016llx "
        "original_out=%08lx separate_final=%08lx fma_final=%08lx "
        "target_max_logit=%08lx target_logits_fnv1a64=%016llx "
        "target_denominator=%08lx\n",
        (unsigned long long) target->h5_original_query_fnv1a64,
        (unsigned long) target->h5_original_out_bits,
        (unsigned long) target->h5_separate_final_bits,
        (unsigned long) target->h5_fma_final_bits,
        (unsigned long) target->maximum_logit_bits[11],
        (unsigned long long) target->logit_bits_fnv1a64[11],
        (unsigned long) target->denominator_bits[11]);
}
#endif

#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
static void print_target_logit_forensics_records(void) {
    const struct edgekv_blockgtq_target_observability * target =
        edgekv_blockgtq_attn_decode_test_observability();
    for (int dim = 0; dim < EDGEKV_BLOCKGTQ_HEAD_DIM; ++dim) {
        printf(
            "EDGEKV_H4_QUERY magic=BGTQH4Q1 schema=1 "
            "profile=qwen2_5_3b_reference layer=17 sequence=128 "
            "mode=diagnostics_on query_head=11 storage_head=35 "
            "dim=%d target_rotated_query=%08lx\n",
            dim, (unsigned long) target->rotated_query_bits[dim]);
    }
    for (int token = 0; token < 128; ++token) {
        printf(
            "EDGEKV_H4_TOKEN magic=BGTQH4T1 schema=1 "
            "profile=qwen2_5_3b_reference layer=17 sequence=128 "
            "mode=diagnostics_on query_head=11 storage_head=35 token=%d "
            "code0=%016llx code1=%016llx code2=%016llx code3=%016llx "
            "lut0=%016llx lut1=%016llx lut2=%016llx lut3=%016llx "
            "norm0=%08lx norm1=%08lx norm2=%08lx norm3=%08lx "
            "dot0=%08lx dot1=%08lx dot2=%08lx dot3=%08lx "
            "pre_scale_dot=%08lx logit=%08lx\n",
            token,
            (unsigned long long) target->code_fnv1a64[token][0],
            (unsigned long long) target->code_fnv1a64[token][1],
            (unsigned long long) target->code_fnv1a64[token][2],
            (unsigned long long) target->code_fnv1a64[token][3],
            (unsigned long long) target->segment_lut_bits_fnv1a64[token][0],
            (unsigned long long) target->segment_lut_bits_fnv1a64[token][1],
            (unsigned long long) target->segment_lut_bits_fnv1a64[token][2],
            (unsigned long long) target->segment_lut_bits_fnv1a64[token][3],
            (unsigned long) target->norm_bits[token][0],
            (unsigned long) target->norm_bits[token][1],
            (unsigned long) target->norm_bits[token][2],
            (unsigned long) target->norm_bits[token][3],
            (unsigned long) target->cumulative_dot_bits[token][0],
            (unsigned long) target->cumulative_dot_bits[token][1],
            (unsigned long) target->cumulative_dot_bits[token][2],
            (unsigned long) target->cumulative_dot_bits[token][3],
            (unsigned long) target->pre_scale_dot_bits[token],
            (unsigned long) target->logit_bits[token]);
    }
    printf(
        "EDGEKV_H4_SUMMARY magic=BGTQH4S1 schema=1 "
        "profile=qwen2_5_3b_reference layer=17 sequence=128 "
        "mode=diagnostics_on query_head=11 storage_head=35 segments=%lu "
        "target_max_logit=%08lx target_logits_fnv1a64=%016llx "
        "target_denominator=%08lx\n",
        (unsigned long) target->segment_count,
        (unsigned long) target->maximum_logit_bits[11],
        (unsigned long long) target->logit_bits_fnv1a64[11],
        (unsigned long) target->denominator_bits[11]);
}
#endif
#endif

static void scatter_append_payloads(const uint8_t * append, int layer,
                                    int sequence) {
    for (int token = 0; token < sequence; ++token) {
        const uint8_t * payload = append + (size_t) token * 316u;
        for (int kv_head = 0; kv_head < EDGEKV_BLOCKGTQ_KV_HEADS;
             ++kv_head) {
            const int head = layer * EDGEKV_BLOCKGTQ_KV_HEADS + kv_head;
            const size_t k_codes_at =
                (size_t) token * 5472u + (size_t) head * 76u;
            const size_t k_norms_at =
                (size_t) EDGEKV_BLOCKGTQ_CAPACITY * 5472u +
                (size_t) token * 1152u + (size_t) head * 16u;
            const size_t v_codes_at =
                (size_t) EDGEKV_BLOCKGTQ_CAPACITY * (5472u + 1152u) +
                (size_t) token * 4608u + (size_t) head * 64u;
            const size_t v_norms_at =
                (size_t) EDGEKV_BLOCKGTQ_CAPACITY *
                    (5472u + 1152u + 4608u) +
                (size_t) token * 144u + (size_t) head * 2u;
            memcpy(history + k_codes_at, payload + (size_t) kv_head * 76u,
                   76);
            memcpy(history + k_norms_at,
                   payload + 152u + (size_t) kv_head * 16u, 16);
            memcpy(history + v_codes_at,
                   payload + 184u + (size_t) kv_head * 64u, 64);
            memcpy(history + v_norms_at,
                   payload + 312u + (size_t) kv_head * 2u, 2);
        }
    }
}

static void fill_params(int32_t * params, int layer, int sequence) {
    const int32_t values[HTP_EDGEKV_BLOCKGTQ_PARAM_COUNT] = {
        EDGEKV_BLOCKGTQ_ABI_VERSION,
        0x7159380b,
        (int32_t) 0x92f8aa7c,
        layer,
        sequence,
        EDGEKV_BLOCKGTQ_CAPACITY,
        EDGEKV_BLOCKGTQ_QUERY_HEADS,
        EDGEKV_BLOCKGTQ_KV_HEADS,
        EDGEKV_BLOCKGTQ_HEAD_DIM,
        EDGEKV_BLOCKGTQ_GQA,
        EDGEKV_BLOCKGTQ_LAYERS,
        EDGEKV_BLOCKGTQ_TRANSFORM_BYTES,
        EDGEKV_BLOCKGTQ_CONSUMER_BYTES,
        EDGEKV_BLOCKGTQ_SHARED_BYTES,
        EDGEKV_BLOCKGTQ_DYNAMIC_BYTES,
    };
    memcpy(params, values, sizeof(values));
}

static int run_last_slot_address_canary(void) {
    const int capacity = EDGEKV_BLOCKGTQ_CAPACITY;
    const int last_token = capacity - 1;
    const size_t k_codes_end =
        (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
        EDGEKV_BLOCKGTQ_K_CODE_STRIDE;
    const size_t k_norms_end =
        k_codes_end + (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
                          EDGEKV_BLOCKGTQ_K_NORM_STRIDE;
    const size_t v_codes_end =
        k_norms_end + (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
                          EDGEKV_BLOCKGTQ_V_CODE_STRIDE;
    const size_t v_norms_end =
        v_codes_end + (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
                          EDGEKV_BLOCKGTQ_V_NORM_STRIDE;
    for (int head = 0; head < EDGEKV_BLOCKGTQ_HEADS; ++head) {
        struct edgekv_blockgtq_history_addresses addresses;
        if (edgekv_blockgtq_history_addresses(
                capacity, last_token, head, &addresses) !=
                EDGEKV_BLOCKGTQ_OK ||
            addresses.k_codes + EDGEKV_BLOCKGTQ_K_CODE_STRIDE >
                k_codes_end ||
            addresses.k_norms + EDGEKV_BLOCKGTQ_K_NORM_STRIDE >
                k_norms_end ||
            addresses.v_codes + EDGEKV_BLOCKGTQ_V_CODE_STRIDE >
                v_codes_end ||
            addresses.v_norms + EDGEKV_BLOCKGTQ_V_NORM_STRIDE >
                v_norms_end) {
            return 0;
        }
        if (head == EDGEKV_BLOCKGTQ_HEADS - 1 &&
            (addresses.k_codes + EDGEKV_BLOCKGTQ_K_CODE_STRIDE !=
                 k_codes_end ||
             addresses.k_norms + EDGEKV_BLOCKGTQ_K_NORM_STRIDE !=
                 k_norms_end ||
             addresses.v_codes + EDGEKV_BLOCKGTQ_V_CODE_STRIDE !=
                 v_codes_end ||
             addresses.v_norms + EDGEKV_BLOCKGTQ_V_NORM_STRIDE !=
                 v_norms_end)) {
            return 0;
        }
    }
    struct edgekv_blockgtq_history_addresses rejected;
    return v_norms_end == EDGEKV_BLOCKGTQ_DYNAMIC_BYTES &&
           edgekv_blockgtq_history_addresses(
               capacity, capacity, 0, &rejected) ==
               EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
}

struct dense_operation {
    struct htp_tensor q;
    struct htp_tensor k;
    struct htp_tensor v;
    struct htp_tensor mask;
    struct htp_tensor output;
    struct htp_ops_context octx;
};

static void prepare_dense_operation(struct dense_operation * operation,
                                    struct htp_context * ctx,
                                    const float * query, int sequence) {
    operation->q = make_tensor_4d(
        (void *) query, EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float),
        HTP_TYPE_F32, sizeof(float), EDGEKV_BLOCKGTQ_HEAD_DIM, 1,
        EDGEKV_BLOCKGTQ_QUERY_HEADS, 1);
    operation->k = make_tensor_4d(
        dense_k,
        (uint32_t) sequence * EDGEKV_BLOCKGTQ_KV_HEADS *
            EDGEKV_BLOCKGTQ_HEAD_DIM * sizeof(_Float16),
        HTP_TYPE_F16, sizeof(_Float16), EDGEKV_BLOCKGTQ_HEAD_DIM, sequence,
        EDGEKV_BLOCKGTQ_KV_HEADS, 1);
    operation->v = make_tensor_4d(
        dense_v,
        (uint32_t) sequence * EDGEKV_BLOCKGTQ_KV_HEADS *
            EDGEKV_BLOCKGTQ_HEAD_DIM * sizeof(_Float16),
        HTP_TYPE_F16, sizeof(_Float16), EDGEKV_BLOCKGTQ_HEAD_DIM, sequence,
        EDGEKV_BLOCKGTQ_KV_HEADS, 1);
    operation->mask = make_tensor_4d(
        dense_mask, (uint32_t) sequence * sizeof(_Float16), HTP_TYPE_F16,
        sizeof(_Float16), sequence, 1, 1, 1);
    operation->output = make_tensor_4d(
        dense_output, sizeof(dense_output), HTP_TYPE_F32, sizeof(float),
        EDGEKV_BLOCKGTQ_HEAD_DIM, EDGEKV_BLOCKGTQ_QUERY_HEADS, 1, 1);
    memset(&operation->octx, 0, sizeof(operation->octx));
    operation->octx.ctx = ctx;
    operation->octx.op = HTP_OP_FLASH_ATTN_EXT;
    operation->octx.src[0] = &operation->q;
    operation->octx.src[1] = &operation->k;
    operation->octx.src[2] = &operation->v;
    operation->octx.src[3] = &operation->mask;
    operation->octx.dst = &operation->output;
    operation->octx.n_threads = N_THREADS;
    const float scale = 0.08838834764831845f;
    const float zero = 0.0f;
    memcpy(operation->octx.op_params, &scale, sizeof(scale));
    memcpy(operation->octx.op_params + 1, &zero, sizeof(zero));
    memcpy(operation->octx.op_params + 2, &zero, sizeof(zero));
    memset(dense_output, 0, sizeof(dense_output));
}

static int run_dense_prepared(struct dense_operation * operation) {
    return op_flash_attn_ext(&operation->octx);
}

#ifndef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
static int run_dense(struct htp_context * ctx, const float * query,
                     int sequence) {
    struct dense_operation operation;
    prepare_dense_operation(&operation, ctx, query, sequence);
    return run_dense_prepared(&operation);
}
#endif

int main(void) {
    const uint8_t * fixture = edgekv_blockgtq_fixture_start;
    const size_t fixture_bytes =
        (size_t) (edgekv_blockgtq_fixture_end -
                  edgekv_blockgtq_fixture_start);
    if (fixture_bytes < FIXTURE_HEADER_BYTES ||
#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
        memcmp(fixture, "BGTQH71", 7) != 0 ||
#else
        memcmp(fixture, "BGTQH41", 7) != 0 ||
#endif
        read_u32(fixture + 8) != 1 ||
        read_u32(fixture + 12) != CASE_COUNT ||
        read_u32(fixture + 16) != EDGEKV_BLOCKGTQ_CAPACITY ||
        read_u32(fixture + 20) != FIXTURE_HEADER_BYTES ||
        read_u32(fixture + 24) != FIXTURE_RECORD_BYTES ||
        read_u32(fixture + 28) != EDGEKV_BLOCKGTQ_TRANSFORM_BYTES ||
        read_u32(fixture + 32) != EDGEKV_BLOCKGTQ_CONSUMER_BYTES ||
        read_u64(fixture + 60) != fixture_bytes) {
        printf("FAIL: fixture header or length drift\n");
        return 1;
    }
    static const uint8_t expected_package_sha[4] = {0x71, 0x59, 0x38, 0x0b};
    static const uint8_t expected_contract_sha[4] = {0x92, 0xf8, 0xaa, 0x7c};
    if (memcmp(fixture + 68, expected_package_sha, 4) != 0 ||
        memcmp(fixture + 100, expected_contract_sha, 4) != 0) {
        printf("FAIL: fixture package or contract identity drift\n");
        return 2;
    }
    const uint64_t transform_offset = read_u64(fixture + 36);
    const uint64_t consumer_offset = read_u64(fixture + 44);
    const uint64_t shared_offset = read_u64(fixture + 52);
    if (transform_offset + EDGEKV_BLOCKGTQ_TRANSFORM_BYTES > fixture_bytes ||
        consumer_offset + EDGEKV_BLOCKGTQ_CONSUMER_BYTES > fixture_bytes ||
        shared_offset + EDGEKV_BLOCKGTQ_SHARED_BYTES > fixture_bytes) {
        printf("FAIL: fixture static pool bounds drift\n");
        return 3;
    }
    const uint8_t * transform = fixture + transform_offset;
    const uint8_t * consumer = fixture + consumer_offset;
    const uint8_t * shared = fixture + shared_offset;
    if (edgekv_blockgtq_validate_static(
            transform, EDGEKV_BLOCKGTQ_TRANSFORM_BYTES, consumer,
            EDGEKV_BLOCKGTQ_CONSUMER_BYTES, shared,
            EDGEKV_BLOCKGTQ_SHARED_BYTES) != EDGEKV_BLOCKGTQ_OK) {
        printf("FAIL: frozen static pool validation failed\n");
        return 4;
    }
    if (!run_last_slot_address_canary()) {
        printf("FAIL: token-2047 address canary\n");
        return 15;
    }
    printf(
        "PASS: token-2047 address canary fields=K-code/K-norm/V-code/V-norm "
        "token-2048=rejected timed=false\n");

    struct htp_context dense_ctx;
    if (!init_dense_context(&dense_ctx)) {
        release_dense_context(&dense_ctx);
        printf("FAIL: dense HTP context initialization failed\n");
        return 5;
    }
    struct htp_tensor query_tensor;
    struct htp_tensor transform_tensor = make_tensor(
        (void *) transform, EDGEKV_BLOCKGTQ_TRANSFORM_BYTES, HTP_TYPE_I8);
    struct htp_tensor consumer_tensor = make_tensor(
        (void *) consumer, EDGEKV_BLOCKGTQ_CONSUMER_BYTES, HTP_TYPE_I8);
    struct htp_tensor shared_tensor = make_tensor(
        (void *) shared, EDGEKV_BLOCKGTQ_SHARED_BYTES, HTP_TYPE_I8);
    struct htp_tensor history_tensor = make_tensor(
        history, EDGEKV_BLOCKGTQ_DYNAMIC_BYTES, HTP_TYPE_I8);
    struct htp_tensor output_tensor = make_tensor(
        output, sizeof(output), HTP_TYPE_F32);
    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx = &dense_ctx;
    octx.op = HTP_OP_EDGEKV_BLOCKGTQ_ATTN_DECODE;
    octx.src[1] = &transform_tensor;
    octx.src[2] = &consumer_tensor;
    octx.src[3] = &shared_tensor;
    octx.src[4] = &history_tensor;
    octx.dst = &output_tensor;
    octx.n_threads = N_THREADS;

    int result = 0;
    float global_logits = 0.0f;
    float global_weights = 0.0f;
    float global_denominators = 0.0f;
    float global_rotated_v = 0.0f;
    float global_output = 0.0f;
#if defined(EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY) && \
    !defined(HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX)
    for (int index = 1; index < 2; ++index) {
#else
    for (int index = 0; index < CASE_COUNT; ++index) {
#endif
        const uint8_t * record =
            fixture + FIXTURE_HEADER_BYTES +
            (size_t) index * FIXTURE_RECORD_BYTES;
        const int layer = (int) read_u32(record);
        const int sequence = (int) read_u32(record + 4);
#if defined(HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX) || \
    defined(HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX)
        static const int attribution_sequences[CASE_COUNT] = {
            5, 128, 512, 2048};
        if (layer != 17 || sequence != attribution_sequences[index] ||
#else
        if (layer != index || (sequence != 1 && sequence != 3 &&
                               sequence != 5) ||
#endif
            read_u32(record + 16) != (uint32_t) layer * 2u ||
            read_u32(record + 20) != (uint32_t) layer * 2u + 1u) {
            printf("FAIL: case %d identity drift\n", index);
            result = 6;
            goto cleanup;
        }
        const float * query =
            (const float *) (fixture + read_u64(record + 32));
        const uint8_t * append = fixture + read_u64(record + 40);
        const uint8_t * expected_logits =
            fixture + read_u64(record + 48);
        const uint8_t * expected_weights =
            fixture + read_u64(record + 56);
        const uint8_t * expected_denominators =
            fixture + read_u64(record + 64);
        const uint8_t * expected_rotated_v =
            fixture + read_u64(record + 72);
        const uint8_t * expected_output =
            fixture + read_u64(record + 80);
        const uint8_t * expected_dense_k =
            fixture + read_u64(record + 88);
        const uint8_t * expected_dense_v =
            fixture + read_u64(record + 96);
#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
        const uint8_t * expected_identity =
            fixture + read_u64(record + 104);
#endif
        scatter_append_payloads(append, layer, sequence);
        query_tensor = make_tensor(
            (void *) query,
            EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float), HTP_TYPE_F32);
        octx.src[0] = &query_tensor;
        fill_params(octx.op_params, layer, sequence);
        memset(output, 0, sizeof(output));
#if defined(HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX) || \
    defined(HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX)
        edgekv_blockgtq_attn_decode_test_set_diagnostics(1);
#endif
        const int status = op_edgekv_blockgtq_attn_decode(&octx);
        if (status != HTP_STATUS_OK) {
            printf("FAIL: case=%d layer=%d status=%d\n", index, layer,
                   status);
            result = 7;
            goto cleanup;
        }
#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
        if (read_u64(record + 104) +
                (uint64_t) EDGEKV_BLOCKGTQ_QUERY_HEADS *
                    H7_IDENTITY_BYTES_PER_HEAD >
            fixture_bytes) {
            printf("FAIL: H7 identity bounds case=%d\n", index);
            result = 23;
            goto cleanup;
        }
        const struct edgekv_blockgtq_diagnostics * h7_diagnostics =
            edgekv_blockgtq_attn_decode_test_diagnostics();
        const struct edgekv_blockgtq_target_observability *
            on_observability =
                edgekv_blockgtq_attn_decode_test_observability();
        struct edgekv_blockgtq_target_observability on_copy =
            *on_observability;
        struct h7_head_metrics on_metrics[EDGEKV_BLOCKGTQ_QUERY_HEADS];
        int all_on_gates = 1;
        for (int query_head = 0;
             query_head < EDGEKV_BLOCKGTQ_QUERY_HEADS; ++query_head) {
            const uint8_t * identity =
                expected_identity +
                (size_t) query_head * H7_IDENTITY_BYTES_PER_HEAD;
            struct h7_head_metrics * metrics = &on_metrics[query_head];
            memset(metrics, 0, sizeof(*metrics));
            metrics->weights = h7_mixed_metrics(
                h7_diagnostics->weights +
                    (size_t) query_head * (size_t) sequence,
                expected_weights +
                    (size_t) query_head * (size_t) sequence *
                        sizeof(float),
                (size_t) sequence, 8.0f, 0x1p-20f, 0x1p-21f);
            metrics->denominator = h7_mixed_metrics(
                h7_diagnostics->denominators + query_head,
                expected_denominators +
                    (size_t) query_head * sizeof(float),
                1, 8.0f, 0x1p-20f, 0x1p-21f);
            metrics->rotated_v = h7_mixed_metrics(
                h7_diagnostics->rotated_v +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM,
                expected_rotated_v +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM *
                        sizeof(float),
                EDGEKV_BLOCKGTQ_HEAD_DIM, 16.0f, 2.0e-5f, 2.0e-5f);
            metrics->output = h7_mixed_metrics(
                output +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM,
                expected_output +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM *
                        sizeof(float),
                EDGEKV_BLOCKGTQ_HEAD_DIM, 16.0f, 2.0e-5f, 2.0e-5f);
            metrics->denominator_ulp = positive_ulp_distance(
                h7_diagnostics->denominators[query_head],
                read_f32(
                    expected_denominators +
                    (size_t) query_head * sizeof(float)));
            metrics->gate_a =
                on_copy.packed_code_bits_fnv1a64[query_head] ==
                    read_u64(identity) &&
                on_copy.lut_bits_fnv1a64[query_head] ==
                    read_u64(identity + 8) &&
                on_copy.norm_bits_fnv1a64[query_head] ==
                    read_u64(identity + 16) &&
                on_copy.maximum_logit_bits[query_head] ==
                    read_u32(identity + 24) &&
                on_copy.logit_bits_fnv1a64[query_head] ==
                    read_u64(identity + 32);
            metrics->gate_b =
                metrics->weights.finite &&
                metrics->weights.max_ratio <= 1.0f &&
                metrics->denominator.finite &&
                metrics->denominator.max_ratio <= 1.0f &&
                metrics->denominator_ulp <= 16u;
            metrics->gate_c =
                metrics->rotated_v.finite &&
                metrics->rotated_v.max_ratio <= 1.0f &&
                metrics->output.finite &&
                metrics->output.max_ratio <= 1.0f;
            all_on_gates &=
                metrics->gate_a && metrics->gate_b && metrics->gate_c;
            print_h7_head(
                index, sequence, "diagnostics_on", query_head, &on_copy,
                metrics, 1);
        }

        edgekv_blockgtq_attn_decode_test_set_diagnostics(0);
        memset(output, 0, sizeof(output));
        if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
            printf("FAIL: H7 diagnostics-off status case=%d\n", index);
            result = 24;
            goto cleanup;
        }
        const struct edgekv_blockgtq_target_observability *
            off_observability =
                edgekv_blockgtq_attn_decode_test_observability();
        const int mode_parity =
            h7_observability_equal(&on_copy, off_observability);
        int all_off_gates = mode_parity;
        for (int query_head = 0;
             query_head < EDGEKV_BLOCKGTQ_QUERY_HEADS; ++query_head) {
            struct h7_head_metrics metrics = on_metrics[query_head];
            metrics.output = h7_mixed_metrics(
                output +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM,
                expected_output +
                    (size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM *
                        sizeof(float),
                EDGEKV_BLOCKGTQ_HEAD_DIM, 16.0f, 2.0e-5f, 2.0e-5f);
            metrics.gate_a &= mode_parity;
            metrics.gate_b &= mode_parity;
            metrics.gate_c =
                mode_parity && metrics.output.finite &&
                metrics.output.max_ratio <= 1.0f;
            all_off_gates &=
                metrics.gate_a && metrics.gate_b && metrics.gate_c;
            print_h7_head(
                index, sequence, "diagnostics_off", query_head,
                off_observability, &metrics, mode_parity);
        }
        if (!all_on_gates || !all_off_gates) {
            printf(
                "FAIL: H7 layered gate case=%d sequence=%d "
                "on=%d off=%d mode_parity=%d\n",
                index, sequence, all_on_gates, all_off_gates,
                mode_parity);
            result = 25;
            goto cleanup;
        }
        printf(
            "EDGEKV_H7_CASE schema=1 case=%d layer=17 sequence=%d "
            "heads=16 modes=2 gate_a=1 gate_b=1 gate_c=1 "
            "mode_parity=1\n",
            index, sequence);
        continue;
#elif defined(EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS)
        print_target_query_rotation_forensics_records();
        continue;
#elif defined(EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS)
        print_target_logit_forensics_records();
        continue;
#elif defined(EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY)
        print_target_observability_records(
            "diagnostics_on", layer, sequence, expected_logits,
            expected_weights, expected_denominators);
        edgekv_blockgtq_attn_decode_test_set_diagnostics(0);
        memset(output, 0, sizeof(output));
        if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
            printf("FAIL: H3 diagnostics-off status\n");
            result = 22;
            goto cleanup;
        }
        print_target_observability_records(
            "diagnostics_off", layer, sequence, expected_logits,
            expected_weights, expected_denominators);
        continue;
#endif
        const struct edgekv_blockgtq_diagnostics * diagnostics =
            edgekv_blockgtq_attn_decode_test_diagnostics();
        const float logits_error =
            max_abs(diagnostics->logits, expected_logits,
                    (size_t) EDGEKV_BLOCKGTQ_QUERY_HEADS * sequence);
        const float weights_error =
            max_abs(diagnostics->weights, expected_weights,
                    (size_t) EDGEKV_BLOCKGTQ_QUERY_HEADS * sequence);
        const float denominators_error =
            max_abs(diagnostics->denominators, expected_denominators,
                    EDGEKV_BLOCKGTQ_QUERY_HEADS);
        const float rotated_v_error =
            max_abs(diagnostics->rotated_v, expected_rotated_v,
                    EDGEKV_BLOCKGTQ_OUTPUT_FLOATS);
        const float output_error = max_abs(
            output, expected_output, EDGEKV_BLOCKGTQ_OUTPUT_FLOATS);
        if (logits_error > global_logits) {
            global_logits = logits_error;
        }
        if (weights_error > global_weights) {
            global_weights = weights_error;
        }
        if (denominators_error > global_denominators) {
            global_denominators = denominators_error;
        }
        if (rotated_v_error > global_rotated_v) {
            global_rotated_v = rotated_v_error;
        }
        if (output_error > global_output) {
            global_output = output_error;
        }
        if (logits_error > 2.0e-5f || weights_error > 2.0e-5f ||
            denominators_error > 2.0e-5f ||
            rotated_v_error > 2.0e-5f || output_error > 2.0e-5f) {
            printf("FAIL: case=%d errors=%g,%g,%g,%g,%g\n", index,
                   (double) logits_error, (double) weights_error,
                   (double) denominators_error, (double) rotated_v_error,
                   (double) output_error);
            result = 8;
            goto cleanup;
        }
        printf(
            "PASS: case=%d layer=%d sequence=%d segments=%lu/%lu "
            "errors=%g,%g,%g,%g,%g kernel_pcycles=%llu\n",
            index, layer, sequence,
            (unsigned long) read_u32(record + 8),
            (unsigned long) read_u32(record + 12), (double) logits_error,
            (double) weights_error, (double) denominators_error,
            (double) rotated_v_error, (double) output_error,
            (unsigned long long)
                edgekv_blockgtq_attn_decode_last_pcycles());

        if (layer == 17) {
            memcpy(dense_k, expected_dense_k,
                   (size_t) sequence * EDGEKV_BLOCKGTQ_KV_HEADS *
                       EDGEKV_BLOCKGTQ_HEAD_DIM * sizeof(_Float16));
            memcpy(dense_v, expected_dense_v,
                   (size_t) sequence * EDGEKV_BLOCKGTQ_KV_HEADS *
                       EDGEKV_BLOCKGTQ_HEAD_DIM * sizeof(_Float16));
        }
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
        edgekv_blockgtq_attn_decode_test_set_diagnostics(0);
        memset(output, 0, sizeof(output));
        if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
            printf("FAIL: diagnostics-off parity status case=%d\n", index);
            result = 16;
            goto cleanup;
        }
        const float production_output_error = max_abs(
            output, expected_output, EDGEKV_BLOCKGTQ_OUTPUT_FLOATS);
        if (production_output_error > 2.0e-5f) {
            printf(
                "FAIL: diagnostics-off parity case=%d error=%g\n",
                index, (double) production_output_error);
            result = 17;
            goto cleanup;
        }

        static const int diagnostics_modes[2] = {1, 0};
        static const char * const mode_names[2] = {
            "diagnostics_on", "diagnostics_off"};
        for (int mode = 0; mode < 2; ++mode) {
            edgekv_blockgtq_attn_decode_test_set_diagnostics(
                diagnostics_modes[mode]);
            if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
                printf(
                    "FAIL: attribution warmup sequence=%d mode=%s\n",
                    sequence, mode_names[mode]);
                result = 18;
                goto cleanup;
            }
            for (int sample = 0; sample < FORMAL_SAMPLES; ++sample) {
                if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
                    printf(
                        "FAIL: attribution sample sequence=%d mode=%s\n",
                        sequence, mode_names[mode]);
                    result = 19;
                    goto cleanup;
                }
                print_attribution_sample(
                    sequence, mode_names[mode], sample);
            }
        }
        struct dense_operation dense_operation;
        prepare_dense_operation(
            &dense_operation, &dense_ctx, query, sequence);
        if (run_dense_prepared(&dense_operation) != HTP_STATUS_OK) {
            printf("FAIL: dense warmup sequence=%d\n", sequence);
            result = 20;
            goto cleanup;
        }
        for (int sample = 0; sample < FORMAL_SAMPLES; ++sample) {
            const uint64_t start = HAP_perf_get_pcycles();
            const int dense_status =
                run_dense_prepared(&dense_operation);
            const uint64_t pcycles = HAP_perf_get_pcycles() - start;
            if (dense_status != HTP_STATUS_OK) {
                printf("FAIL: dense sample sequence=%d\n", sequence);
                result = 21;
                goto cleanup;
            }
            printf(
                "DENSE: sequence=%d sample=%d total=%llu\n",
                sequence, sample, (unsigned long long) pcycles);
        }
#endif
    }

    // Fail-closed adapter checks happen after the frozen correctness matrix and
    // do not alter any scientific fixture.
#ifndef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
    {
        int32_t saved[HTP_EDGEKV_BLOCKGTQ_PARAM_COUNT];
        memcpy(saved, octx.op_params, sizeof(saved));
        octx.op_params[HTP_EDGEKV_BLOCKGTQ_ABI_VERSION] = 2;
        if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_NO_SUPPORT) {
            printf("FAIL: invalid ABI version accepted\n");
            result = 9;
            goto cleanup;
        }
        memcpy(octx.op_params, saved, sizeof(saved));
        octx.src[5] = &query_tensor;
        if (op_edgekv_blockgtq_attn_decode(&octx) !=
            HTP_STATUS_INVAL_PARAMS) {
            printf("FAIL: sixth/overflow input accepted\n");
            result = 10;
            goto cleanup;
        }
        octx.src[5] = NULL;
        struct htp_tensor short_output = output_tensor;
        short_output.size -= sizeof(float);
        octx.dst = &short_output;
        if (op_edgekv_blockgtq_attn_decode(&octx) !=
            HTP_STATUS_INVAL_PARAMS) {
            printf("FAIL: dense-history-sized output contract not rejected\n");
            result = 11;
            goto cleanup;
        }
        octx.dst = &output_tensor;
        memcpy(mutated_consumer, consumer, sizeof(mutated_consumer));
        mutated_consumer[133760 + 4] ^= 1u;
        struct htp_tensor mutated_tensor = make_tensor(
            mutated_consumer, sizeof(mutated_consumer), HTP_TYPE_I8);
        octx.src[2] = &mutated_tensor;
        if (op_edgekv_blockgtq_attn_decode(&octx) !=
            HTP_STATUS_INVAL_PARAMS) {
            printf("FAIL: semantic descriptor drift accepted\n");
            result = 12;
            goto cleanup;
        }
        octx.src[2] = &consumer_tensor;
    }

#ifndef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
    {
        const uint8_t * record =
            fixture + FIXTURE_HEADER_BYTES +
            17u * FIXTURE_RECORD_BYTES;
        const int sequence = (int) read_u32(record + 4);
        const float * query =
            (const float *) (fixture + read_u64(record + 32));
        query_tensor = make_tensor(
            (void *) query,
            EDGEKV_BLOCKGTQ_OUTPUT_FLOATS * sizeof(float), HTP_TYPE_F32);
        octx.src[0] = &query_tensor;
        fill_params(octx.op_params, 17, sequence);
        uint64_t blockgtq_samples[3];
        uint64_t dense_samples[3];
        for (int sample = 0; sample < 3; ++sample) {
            if (op_edgekv_blockgtq_attn_decode(&octx) != HTP_STATUS_OK) {
                printf("FAIL: Block-GTQ Pcycles sample failed\n");
                result = 13;
                goto cleanup;
            }
            blockgtq_samples[sample] =
                edgekv_blockgtq_attn_decode_last_pcycles();
            const uint64_t start = HAP_perf_get_pcycles();
            const int dense_status = run_dense(&dense_ctx, query, sequence);
            dense_samples[sample] = HAP_perf_get_pcycles() - start;
            if (dense_status != HTP_STATUS_OK) {
                printf("FAIL: dense HTP Pcycles sample status=%d\n",
                       dense_status);
                result = 14;
                goto cleanup;
            }
        }
        const uint64_t blockgtq_median =
            median3(blockgtq_samples[0], blockgtq_samples[1],
                    blockgtq_samples[2]);
        const uint64_t dense_median = median3(
            dense_samples[0], dense_samples[1], dense_samples[2]);
        printf(
            "PERF: blockgtq_op raw=[%llu,%llu,%llu] median=%llu "
            "dense_htp raw=[%llu,%llu,%llu] median=%llu ratio=%.9f "
            "scope=operation_body_only target=v79\n",
            (unsigned long long) blockgtq_samples[0],
            (unsigned long long) blockgtq_samples[1],
            (unsigned long long) blockgtq_samples[2],
            (unsigned long long) blockgtq_median,
            (unsigned long long) dense_samples[0],
            (unsigned long long) dense_samples[1],
            (unsigned long long) dense_samples[2],
            (unsigned long long) dense_median,
            (double) blockgtq_median / (double) dense_median);
    }
#endif
#endif

#ifdef HTP_EDGEKV_BLOCKGTQ_CROSS_LIBM_MATRIX
    printf(
        "PASS: B2-T4P-H7 cross-libm matrix cases=4 layer=17 "
        "sequences=5,128,512,2048 modes=diagnostics_on,diagnostics_off "
        "heads=16 records=128 gate_a=bit_exact gate_b=mixed_cross_libm "
        "gate_c=representation_output\n");
#elif defined(EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS)
    printf(
        "PASS: B2-T4P-H5 query-rotation forensics "
        "profile=qwen2_5_3b_reference layer=17 sequence=128 "
        "mode=diagnostics_on query_head=11 storage_head=35 "
        "segment=0 out=2 term_records=22 summary_records=1\n");
#elif defined(EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS)
    printf(
        "PASS: B2-T4P-H4 target logit-path forensics "
        "profile=qwen2_5_3b_reference layer=17 sequence=128 "
        "mode=diagnostics_on query_head=11 storage_head=35 "
        "query_records=128 token_records=128 summary_records=1\n");
#elif defined(EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY)
    printf(
        "PASS: B2-T4P-H3 target observability profile=qwen2_5_3b_reference "
        "layer=17 sequence=128 modes=diagnostics_on,diagnostics_off "
        "heads=16 records=32\n");
#else
#ifdef HTP_EDGEKV_BLOCKGTQ_ATTRIBUTION_MATRIX
    printf(
        "PASS: B2-T4P v79 attribution matrix cases=4 layer=17 "
        "sequences=5,128,512,2048 modes=diagnostics_on,diagnostics_off "
        "formal_samples=5 "
#else
    printf(
        "PASS: B2-T4 v79 matrix cases=36 layouts=72 groups=319 "
#endif
        "max_errors=%g,%g,%g,%g,%g persistent_dense_history_bytes=0\n",
        (double) global_logits, (double) global_weights,
        (double) global_denominators, (double) global_rotated_v,
        (double) global_output);
#endif

cleanup:
    release_dense_context(&dense_ctx);
    return result;
}
