// SPDX-License-Identifier: MIT

#include "edgekv-hvx-gqa-block.h"
#include "HAP_perf.h"
#include "worker-pool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    N_BLOCKS            = 16,
    N_HEAD_KV           = 8,
    N_HEAD_Q            = 16,
    N_THREADS           = 4,
    PERFORMANCE_SAMPLES = 5,
    U_BLOCK_BYTES       = EDGEKV_HVX_GQA_BLOCK_SIZE * EDGEKV_HVX_GQA_RANK,
    U_STORAGE_BYTES     = N_BLOCKS * U_BLOCK_BYTES,
    VH_BLOCK_BYTES      = EDGEKV_HVX_GQA_RANK * EDGEKV_HVX_GQA_VH_RANK_STRIDE,
    VH_STORAGE_BYTES    = N_BLOCKS * VH_BLOCK_BYTES,
    SCALE_BLOCK_ELEMS   = EDGEKV_HVX_GQA_RANK,
    WEIGHT_BLOCK_ELEMS  = EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_BLOCK_SIZE,
    OUTPUT_BLOCK_ELEMS  = EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM,
};

struct mobile_fixture {
    float *  query;
    int8_t * k_u;
    int8_t * k_vh;
    float *  k_scale;
    float *  weights;
    int8_t * v_u;
    int8_t * v_vh;
    float *  v_scale;
    float *  logits;
    float *  output;
    float *  reference_logits;
    float *  reference_output;
};

static volatile float perf_sink;

static void release_fixture(struct mobile_fixture * fixture) {
    free(fixture->reference_output);
    free(fixture->reference_logits);
    free(fixture->output);
    free(fixture->logits);
    free(fixture->v_scale);
    free(fixture->v_vh);
    free(fixture->v_u);
    free(fixture->weights);
    free(fixture->k_scale);
    free(fixture->k_vh);
    free(fixture->k_u);
    free(fixture->query);
    memset(fixture, 0, sizeof(*fixture));
}

static int allocate_fixture(struct mobile_fixture * fixture) {
    memset(fixture, 0, sizeof(*fixture));
    const size_t query_bytes   = N_HEAD_Q * EDGEKV_HVX_GQA_HEAD_DIM * sizeof(float);
    const size_t scale_bytes   = N_BLOCKS * SCALE_BLOCK_ELEMS * sizeof(float);
    const size_t weights_bytes = N_HEAD_KV * N_BLOCKS * WEIGHT_BLOCK_ELEMS * sizeof(float);
    const size_t logits_bytes  = weights_bytes;
    const size_t output_bytes  = N_HEAD_KV * N_BLOCKS * OUTPUT_BLOCK_ELEMS * sizeof(float);

    fixture->query            = (float *) memalign(128, query_bytes);
    fixture->k_u              = (int8_t *) memalign(128, U_STORAGE_BYTES);
    fixture->k_vh             = (int8_t *) memalign(128, VH_STORAGE_BYTES);
    fixture->k_scale          = (float *) memalign(128, scale_bytes);
    fixture->weights          = (float *) memalign(128, weights_bytes);
    fixture->v_u              = (int8_t *) memalign(128, U_STORAGE_BYTES);
    fixture->v_vh             = (int8_t *) memalign(128, VH_STORAGE_BYTES);
    fixture->v_scale          = (float *) memalign(128, scale_bytes);
    fixture->logits           = (float *) memalign(128, logits_bytes);
    fixture->output           = (float *) memalign(128, output_bytes);
    fixture->reference_logits = (float *) memalign(128, logits_bytes);
    fixture->reference_output = (float *) memalign(128, output_bytes);

    if (!fixture->query || !fixture->k_u || !fixture->k_vh || !fixture->k_scale || !fixture->weights || !fixture->v_u ||
        !fixture->v_vh || !fixture->v_scale || !fixture->logits || !fixture->output || !fixture->reference_logits ||
        !fixture->reference_output) {
        release_fixture(fixture);
        return 0;
    }
    return 1;
}

static size_t block_head_offset(int hkv, int block, int block_elems) {
    return ((size_t) hkv * N_BLOCKS + block) * block_elems;
}

static void initialize_fixture(struct mobile_fixture * fixture) {
    memset(fixture->k_vh, 0, VH_STORAGE_BYTES);
    memset(fixture->v_vh, 0, VH_STORAGE_BYTES);

    for (int i = 0; i < N_HEAD_Q * EDGEKV_HVX_GQA_HEAD_DIM; ++i) {
        fixture->query[i] = (float) (((i * 13 + 5) % 29) - 14) * 0.03125f;
    }
    for (int block = 0; block < N_BLOCKS; ++block) {
        for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
            for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
                const int logical = (block * EDGEKV_HVX_GQA_BLOCK_SIZE + token) * EDGEKV_HVX_GQA_RANK + rank;
                fixture->k_u[block * U_BLOCK_BYTES + token * EDGEKV_HVX_GQA_RANK + rank] =
                    (int8_t) (((logical * 7 + 3) % 15) - 7);
                fixture->v_u[block * U_BLOCK_BYTES + token * EDGEKV_HVX_GQA_RANK + rank] =
                    (int8_t) (((logical * 5 + 1) % 13) - 6);
            }
        }
        for (int rank = 0; rank < EDGEKV_HVX_GQA_RANK; ++rank) {
            fixture->k_scale[block * SCALE_BLOCK_ELEMS + rank] = 0.0025f + 0.00003125f * (float) ((block + rank) % 7);
            fixture->v_scale[block * SCALE_BLOCK_ELEMS + rank] = 0.0030f + 0.00002500f * (float) ((block + rank) % 5);
            for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
                for (int dim = 0; dim < EDGEKV_HVX_GQA_HEAD_DIM; ++dim) {
                    const int logical =
                        (((block * EDGEKV_HVX_GQA_RANK + rank) * N_HEAD_KV + hkv) * EDGEKV_HVX_GQA_HEAD_DIM + dim);
                    const size_t offset = (size_t) block * VH_BLOCK_BYTES +
                                          (size_t) rank * EDGEKV_HVX_GQA_VH_RANK_STRIDE +
                                          (size_t) hkv * EDGEKV_HVX_GQA_HEAD_DIM + dim;
                    fixture->k_vh[offset] = (int8_t) (((logical * 11 + 2) % 13) - 6);
                    fixture->v_vh[offset] = (int8_t) (((logical * 3 + 4) % 11) - 5);
                }
            }
        }
    }

    for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
        for (int block = 0; block < N_BLOCKS; ++block) {
            float * block_weights = fixture->weights + block_head_offset(hkv, block, WEIGHT_BLOCK_ELEMS);
            for (int head = 0; head < EDGEKV_HVX_GQA_HEADS; ++head) {
                float sum = 0.0f;
                for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
                    const float value = (float) (((token * (head + 3) + block + hkv + 7) % 17) + 1);
                    block_weights[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] = value;
                    sum += value;
                }
                for (int token = 0; token < EDGEKV_HVX_GQA_BLOCK_SIZE; ++token) {
                    block_weights[head * EDGEKV_HVX_GQA_BLOCK_SIZE + token] /= sum;
                }
            }
        }
    }
}

static void run_block(const struct mobile_fixture * fixture, int hkv, int block, int scalar) {
    const float    attention_scale = 0.08838834764831845f;
    const float *  q               = fixture->query + hkv * EDGEKV_HVX_GQA_HEADS * EDGEKV_HVX_GQA_HEAD_DIM;
    const int8_t * k_u             = fixture->k_u + block * U_BLOCK_BYTES;
    const int8_t * k_vh            = fixture->k_vh + (size_t) block * VH_BLOCK_BYTES + hkv * EDGEKV_HVX_GQA_HEAD_DIM;
    const float *  k_scale         = fixture->k_scale + block * SCALE_BLOCK_ELEMS;
    const float *  weights         = fixture->weights + block_head_offset(hkv, block, WEIGHT_BLOCK_ELEMS);
    const int8_t * v_u             = fixture->v_u + block * U_BLOCK_BYTES;
    const int8_t * v_vh            = fixture->v_vh + (size_t) block * VH_BLOCK_BYTES + hkv * EDGEKV_HVX_GQA_HEAD_DIM;
    const float *  v_scale         = fixture->v_scale + block * SCALE_BLOCK_ELEMS;
    const size_t   logits_offset   = block_head_offset(hkv, block, WEIGHT_BLOCK_ELEMS);
    const size_t   output_offset   = block_head_offset(hkv, block, OUTPUT_BLOCK_ELEMS);

    if (scalar) {
        edgekv_scalar_gqa_block(q, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale,
                                fixture->reference_logits + logits_offset, fixture->reference_output + output_offset);
    } else {
        edgekv_hvx_gqa_block(q, k_u, k_vh, k_scale, attention_scale, weights, v_u, v_vh, v_scale,
                             fixture->logits + logits_offset, fixture->output + output_offset);
    }
}

static void run_mobile_worker(unsigned int n, unsigned int i, void * data) {
    const struct mobile_fixture * fixture = (const struct mobile_fixture *) data;
    for (int hkv = (int) i; hkv < N_HEAD_KV; hkv += (int) n) {
        for (int block = 0; block < N_BLOCKS; ++block) {
            run_block(fixture, hkv, block, 0);
        }
    }
}

static float max_error(const float * actual, const float * expected, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
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
    struct mobile_fixture fixture;
    if (!allocate_fixture(&fixture)) {
        printf("FAIL: could not allocate C4C1 mobile fixture\n");
        return 1;
    }
    initialize_fixture(&fixture);

    for (int hkv = 0; hkv < N_HEAD_KV; ++hkv) {
        for (int block = 0; block < N_BLOCKS; ++block) {
            run_block(&fixture, hkv, block, 1);
        }
    }

    worker_pool_context_t worker_pool = NULL;
    if (worker_pool_init(&worker_pool, N_THREADS) != AEE_SUCCESS) {
        release_fixture(&fixture);
        printf("FAIL: could not initialize C4C1 worker pool\n");
        return 2;
    }
    if (worker_pool_run_func(worker_pool, run_mobile_worker, &fixture, N_THREADS) != AEE_SUCCESS) {
        worker_pool_release(&worker_pool);
        release_fixture(&fixture);
        printf("FAIL: C4C1 correctness dispatch failed\n");
        return 3;
    }

    const size_t logits_count = N_HEAD_KV * N_BLOCKS * WEIGHT_BLOCK_ELEMS;
    const size_t output_count = N_HEAD_KV * N_BLOCKS * OUTPUT_BLOCK_ELEMS;
    const float  logits_error = max_error(fixture.logits, fixture.reference_logits, logits_count);
    const float  output_error = max_error(fixture.output, fixture.reference_output, output_count);
    if (logits_error > 0.02f || output_error > 0.02f) {
        worker_pool_release(&worker_pool);
        release_fixture(&fixture);
        printf("FAIL: C4C1 numerical mismatch logits_error=%g output_error=%g\n", (double) logits_error,
               (double) output_error);
        return 4;
    }
    printf("PASS: C4C1 mobile factor correctness logits_error=%g output_error=%g\n", (double) logits_error,
           (double) output_error);

    uint64_t serial_samples[PERFORMANCE_SAMPLES];
    uint64_t worker_samples[PERFORMANCE_SAMPLES];
    run_mobile_worker(1, 0, &fixture);
    worker_pool_run_func(worker_pool, run_mobile_worker, &fixture, N_THREADS);
    for (int sample = 0; sample < PERFORMANCE_SAMPLES; ++sample) {
        uint64_t start = HAP_perf_get_pcycles();
        run_mobile_worker(1, 0, &fixture);
        serial_samples[sample] = HAP_perf_get_pcycles() - start;
        perf_sink += fixture.output[sample];

        start                  = HAP_perf_get_pcycles();
        const AEEResult status = worker_pool_run_func(worker_pool, run_mobile_worker, &fixture, N_THREADS);
        worker_samples[sample] = HAP_perf_get_pcycles() - start;
        if (status != AEE_SUCCESS) {
            worker_pool_release(&worker_pool);
            release_fixture(&fixture);
            printf("FAIL: C4C1 performance dispatch %d failed\n", sample);
            return 5;
        }
        perf_sink += fixture.logits[sample];
    }

    const uint64_t serial_median       = median5(serial_samples);
    const uint64_t worker_median       = median5(worker_samples);
    const uint64_t dense_c4b0_cycles   = 746920;
    const double   parallel_efficiency = (double) serial_median / ((double) worker_median * N_THREADS);
    const double   dense_ratio         = (double) worker_median / (double) dense_c4b0_cycles;
    const char *   decision            = dense_ratio <= 0.5 ? "CANDIDATE" : dense_ratio <= 1.0 ? "OPTIMIZE" : "STOP";

    print_samples("mobile_serial_factor", serial_samples, serial_median);
    print_samples("mobile_4worker_factor", worker_samples, worker_median);
    printf("PERF: parallel_efficiency=%.6f dense_c4b0=%llu factor_only_ratio=%.6f\n", parallel_efficiency,
           (unsigned long long) dense_c4b0_cycles, dense_ratio);
    printf(
        "PASS: C4C1 decision=%s shape=B64/R32/G4/Hq16/Hkv8/D128 blocks=16 "
        "excludes=softmax,active-kv,merge sink=%g\n",
        decision, (double) perf_sink);

    worker_pool_release(&worker_pool);
    release_fixture(&fixture);
    return 0;
}
