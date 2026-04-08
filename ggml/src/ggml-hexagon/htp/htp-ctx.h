#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "hex-dma.h"
#include "worker-pool.h"

#include <assert.h>
#include <stddef.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>

#define HTP_MAX_NTHREADS 10
#define HTP_WEIGHT_CACHE_NWAYS 4
#define HTP_WEIGHT_TILE_CACHE_NWAYS 2

struct htp_weight_cache_entry {
    uintptr_t src_addr;
    size_t    size;
    uint32_t  epoch;
    uint32_t  type;
    uint32_t  ne0;
    uint32_t  ne1;
    uint32_t  nb0;
    uint32_t  nb1;
    uint8_t * vtcm_ptr;
    uint8_t   valid;
};

struct htp_weight_cache {
    uint8_t * base;
    size_t    size;
    size_t    slot_size;
    uint32_t  epoch;
    uint32_t  next_victim;
    struct htp_weight_cache_entry entries[HTP_WEIGHT_CACHE_NWAYS];
};

struct htp_weight_tile_cache_entry {
    uintptr_t src_addr;
    size_t    payload_size;
    uint32_t  epoch;
    uint32_t  type;
    uint32_t  k;
    uint32_t  n;
    uint32_t  row_stride;
    uint32_t  chunk_col;
    uint32_t  chunk_cols;
    uint8_t * vtcm_ptr;
    uint8_t   valid;
};

struct htp_weight_tile_cache {
    uint8_t * base;
    size_t    size;
    size_t    slot_size;
    uint32_t  epoch;
    uint32_t  next_victim;
    struct htp_weight_tile_cache_entry entries[HTP_WEIGHT_TILE_CACHE_NWAYS];
};

#define HTP_WEIGHT_CACHE_STATS_TYPES 5

struct htp_weight_cache_stats {
    uint64_t raw_hit;
    uint64_t raw_miss;
    uint64_t tile_hit[HTP_WEIGHT_CACHE_STATS_TYPES];
    uint64_t tile_miss[HTP_WEIGHT_CACHE_STATS_TYPES];
    uint64_t tile_fill[HTP_WEIGHT_CACHE_STATS_TYPES];
    uint64_t tile_evict[HTP_WEIGHT_CACHE_STATS_TYPES];
};

struct htp_matmul_runtime_config {
    uint8_t * weight_cache_base;
    size_t    weight_cache_size;
    uint8_t * act_cache_base;
    size_t    act_cache_size;
    int       prefer_preload;
};

// Main context for htp DSP backend
struct htp_context {
    dspqueue_t            queue;
    dma_queue *           dma[HTP_MAX_NTHREADS];
    worker_pool_context_t worker_pool;
    uint32_t              n_threads;

    int thread_id;
    int thread_prio;

    uint8_t * vtcm_base;
    size_t    vtcm_size;
    uint32_t  vtcm_rctx;

    atomic_bool vtcm_valid;
    atomic_bool vtcm_inuse;
    atomic_bool vtcm_needs_release;

    uint32_t opmask;
    uint32_t weight_cache_epoch;
    struct htp_matmul_runtime_config matmul_cfg;
    struct htp_weight_cache weight_cache;
    struct htp_weight_tile_cache weight_tile_cache;
    struct htp_weight_cache_stats weight_cache_stats;

    // Cached src1 spad position from the last quantize pass.
    // When SKIP_QUANTIZE is set the Q8 activation data is already in VTCM
    // at this address; the matmul must read from here instead of recomputing
    // the offset (which depends on the current op's src0 size).
    uint8_t * prev_src1_spad;

    // HMX acceleration fields (v73+, enabled by compile-time HTP_HAS_HMX)
#ifdef HTP_HAS_HMX
    int        hmx_enabled;       // Runtime flag: HMX initialisation succeeded
    size_t     vtcm_scratch_size; // Usable dynamic scratch (vtcm_size minus tail reservation)
#endif
};

#endif /* HTP_CTX_H */
