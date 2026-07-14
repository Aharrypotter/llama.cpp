// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-block.h"
#include "HAP_perf.h"

#include <stdint.h>
#include <stdio.h>

enum {
    PERFORMANCE_SAMPLES = 5,
    MOBILE_N_BLOCKS     = 16,
    MOBILE_N_HEAD_KV    = 8,
};

static float  query[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM] __attribute__((aligned(128)));
static int8_t k_u[EDGEKV_HVX_GQA_BLOCK_SIZE * EDGEKV_HVX_GQA_RANK] __attribute__((aligned(128)));
static int8_t k_vh[EDGEKV_HVX_GQA_RANK * EDGEKV_HVX_GQA_VH_RANK_STRIDE] __attribute__((aligned(128)));
static float  k_scale[EDGEKV_HVX_GQA_RANK] __attribute__((aligned(128)));
static float  weights[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE] __attribute__((aligned(128)));
static int8_t v_u[EDGEKV_HVX_GQA_BLOCK_SIZE * EDGEKV_HVX_GQA_RANK] __attribute__((aligned(128)));
static int8_t v_vh[EDGEKV_HVX_GQA_RANK * EDGEKV_HVX_GQA_VH_RANK_STRIDE] __attribute__((aligned(128)));
static float  v_scale[EDGEKV_HVX_GQA_RANK] __attribute__((aligned(128)));

static float reference_logits[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE] __attribute__((aligned(128)));
static float reference_output[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM] __attribute__((aligned(128)));
static float hvx_logits[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE] __attribute__((aligned(128)));
static float hvx_output[EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM] __attribute__((aligned(128)));

static volatile float perf_sink;

static float absf(float value) {
    return value < 0.0f ? -value : value;
}

static void initialize_inputs(void) {
    for (int i = 0; i < EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM; ++i) {
        query[i] = (float) (((i * 13 + 5) % 29) - 14) * 0.03125f;
    }
    for (int i = 0; i < EDGEKV_HVX_GQA_BLOCK_SIZE * EDGEKV_HVX_GQA_RANK; ++i) {
        k_u[i] = (int8_t) (((i * 7 + 3) % 15) - 7);
        v_u[i] = (int8_t) (((i * 5 + 1) % 13) - 6);
    }
    for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
        for (int dim = 0; dim < EDGEKV_HVX_GQA_HEAD_DIM; ++dim) {
            const int logical                                = rank * EDGEKV_HVX_GQA_HEAD_DIM + dim;
            k_vh[rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE + dim] = (int8_t) (((logical * 11 + 2) % 13) - 6);
            v_vh[rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE + dim] = (int8_t) (((logical * 3 + 4) % 11) - 5);
        }
    }
    for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
        k_scale[rank] = 0.0025f + 0.00003125f * (float) (rank % 7);
        v_scale[rank] = 0.0030f + 0.00002500f * (float) (rank % 5);
    }
    for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
        float sum = 0.0f;
        for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
            const float value                                 = (float) (((token * (head + 3) + 7) % 17) + 1);
            weights[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] = value;
            sum += value;
        }
        for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
            weights[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] /= sum;
        }
    }
}

static float max_error(const float * actual, const float * expected, int count) {
    float result = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float error = absf(actual[i] - expected[i]);
        if (error > result) {
            result = error;
        }
    }
    return result;
}

static uint64_t median5(const uint64_t samples[PERFORMANCE_SAMPLES]) {
    uint64_t sorted[PERFORMANCE_SAMPLES];
    for (int i = 0; i < PERFORMANCE_SAMPLES; ++i) {
        sorted[i] = samples[i];
    }
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

static void print_samples(const char * name, const uint64_t samples[PERFORMANCE_SAMPLES], uint64_t median) {
    printf("PERF: %s raw=[%llu,%llu,%llu,%llu,%llu] median=%llu\n", name, (unsigned long long) samples[0],
           (unsigned long long) samples[1], (unsigned long long) samples[2], (unsigned long long) samples[3],
           (unsigned long long) samples[4], (unsigned long long) median);
}

int main(void) {
    const float attention_scale = 0.08838834764831845f;
    initialize_inputs();

    edgekv_scalar_gqa_block(query, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale, reference_logits,
                            reference_output);
    edgekv_hvx_gqa_block(query, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale, hvx_logits,
                         hvx_output);

    const float logits_error =
        max_error(hvx_logits, reference_logits, EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE);
    const float output_error = max_error(hvx_output, reference_output, EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM);
    if (logits_error > 0.02f || output_error > 0.02f) {
        printf("FAIL: numerical mismatch logits_error=%g output_error=%g\n", (double) logits_error,
               (double) output_error);
        return 1;
    }
    printf("PASS: C4C0 fixed-shape HVX correctness logits_error=%g output_error=%g\n", (double) logits_error,
           (double) output_error);

    uint64_t hvx_k_samples[PERFORMANCE_SAMPLES];
    uint64_t hvx_v_samples[PERFORMANCE_SAMPLES];
    uint64_t hvx_total_samples[PERFORMANCE_SAMPLES];
    uint64_t scalar_total_samples[PERFORMANCE_SAMPLES];

    edgekv_hvx_gqa_block_k(query, k_u, k_vh, k_scale, attention_scale, hvx_logits);
    edgekv_hvx_gqa_block_v(weights, v_u, v_vh, v_scale, hvx_output);
    edgekv_scalar_gqa_block(query, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale, reference_logits,
                            reference_output);

    for (int sample = 0; sample < PERFORMANCE_SAMPLES; ++sample) {
        uint64_t start = HAP_perf_get_pcycles();
        edgekv_hvx_gqa_block_k(query, k_u, k_vh, k_scale, attention_scale, hvx_logits);
        hvx_k_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += hvx_logits[sample];

        start = HAP_perf_get_pcycles();
        edgekv_hvx_gqa_block_v(weights, v_u, v_vh, v_scale, hvx_output);
        hvx_v_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += hvx_output[sample];

        start = HAP_perf_get_pcycles();
        edgekv_hvx_gqa_block(query, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale, hvx_logits,
                             hvx_output);
        hvx_total_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += hvx_logits[sample + 8] + hvx_output[sample + 8];

        start = HAP_perf_get_pcycles();
        edgekv_scalar_gqa_block(query, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale,
                                reference_logits, reference_output);
        scalar_total_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += reference_logits[sample + 16] + reference_output[sample + 16];
    }

    const uint64_t hvx_k_median                   = median5(hvx_k_samples);
    const uint64_t hvx_v_median                   = median5(hvx_v_samples);
    const uint64_t hvx_total_median               = median5(hvx_total_samples);
    const uint64_t scalar_median                  = median5(scalar_total_samples);
    const uint64_t dense_c4b0_cycles              = 746920;
    const uint64_t fanout                         = MOBILE_N_BLOCKS * MOBILE_N_HEAD_KV;
    const uint64_t serial_extrapolated            = hvx_total_median * fanout;
    const uint64_t ideal_four_worker_extrapolated = serial_extrapolated / 4;
    const double   serial_lower_bound_ratio       = (double) serial_extrapolated / (double) dense_c4b0_cycles;
    const double   ideal_four_worker_ratio = (double) ideal_four_worker_extrapolated / (double) dense_c4b0_cycles;
    const double   scalar_speedup          = (double) scalar_median / (double) hvx_total_median;
    const char *   decision                = ideal_four_worker_ratio <= 0.5 ? "CANDIDATE" :
                                             ideal_four_worker_ratio <= 1.0 ? "OPTIMIZE" :
                                                                              "STOP";

    print_samples("hvx_k", hvx_k_samples, hvx_k_median);
    print_samples("hvx_v", hvx_v_samples, hvx_v_median);
    print_samples("hvx_total", hvx_total_samples, hvx_total_median);
    print_samples("scalar_total", scalar_total_samples, scalar_median);
    printf(
        "PERF: scalar_speedup=%.6f serial_16x8=%llu ideal_4worker=%llu "
        "dense_c4b0=%llu serial_ratio=%.6f ideal_4worker_ratio=%.6f\n",
        scalar_speedup, (unsigned long long) serial_extrapolated, (unsigned long long) ideal_four_worker_extrapolated,
        (unsigned long long) dense_c4b0_cycles, serial_lower_bound_ratio, ideal_four_worker_ratio);
    printf(
        "PASS: C4C0 decision=%s scope=one-block-one-hkv-two-hq "
        "excludes=softmax,active-kv,merge,dispatch sink=%g\n",
        decision, (double) perf_sink);
    return 0;
}
