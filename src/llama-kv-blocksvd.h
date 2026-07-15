#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Backend-agnostic core for Block SVD KV cache compression.
// This header lives in src/ so llama core code can call the CPU reference
// compression/reconstruction path without depending on common/.

// Quantized SVD factors for one K or V matrix.
// Used by both the xKV cross-layer chunk and the per-layer chunk below.
struct llama_kv_blocksvd_xkv_factors {
    int32_t rank = 0;
    int32_t quant_bits = 8;

    // Quantized U: (n_tokens, rank)
    std::vector<int8_t> u_q;
    float u_scale = 1.0f;

    // Quantized S: (rank,)
    std::vector<int8_t> s_q;
    float s_scale = 1.0f;

    // Quantized Vh: (rank, n_dim)
    std::vector<int8_t> vh_q;
    float vh_scale = 1.0f;
};

// Per-layer Block SVD chunk. Stores K and V as separate low-rank factors so
// they can carry independent ranks and are not forced to share a single joint
// SVD basis.
struct llama_kv_blocksvd_chunk {
    int32_t seq_start = 0;
    int32_t seq_end   = 0;

    // Total sequence length of the layer at compression time. Required to
    // compute the correct stride when writing back the transposed V cache.
    int32_t n_kv = 0;

    // Flat feature dimensions: n_head_kv * head_dim_k and n_head_kv * head_dim_v.
    int32_t n_flat_k = 0;
    int32_t n_flat_v = 0;

    int32_t quant_bits = 8;

    llama_kv_blocksvd_xkv_factors k_factors;
    llama_kv_blocksvd_xkv_factors v_factors;
};

// Persistent storage for one compressed xKV cross-layer block.
struct llama_kv_blocksvd_xkv_chunk {
    int32_t  layer_start = 0;      // first KV layer in the group
    int32_t  group_size  = 0;      // number of layers in the group
    uint32_t stream      = 0;      // KV cache stream
    llama_seq_id seq_id  = 0;      // logical sequence id

    // one entry per token in the compressed block
    std::vector<uint32_t> slots;
    std::vector<llama_pos> pos;

    int32_t n_head_kv  = 0;
    int32_t head_dim_k = 0;
    int32_t head_dim_v = 0;

    llama_kv_blocksvd_xkv_factors k_factors;
    llama_kv_blocksvd_xkv_factors v_factors;

    bool materialized = false;     // true when already decompressed into dense cache
};

// Per-stream staging buffer metadata for memory-reduction mode.
// This type is intentionally standalone so llama-kv-blocksvd.h can be used
// without including llama-kv-cache.h.
struct llama_kv_blocksvd_staging_t {
    uint32_t capacity = 0;
    // per-stream: logical cell index -> staging slot, -1 = not staged
    std::vector<std::vector<int32_t>> cell_to_slot;
    // per-stream: staging slot -> logical cell index, -1 = free
    std::vector<std::vector<int32_t>> slot_to_cell;
};

struct llama_kv_blocksvd_execution_pool_registry;

struct llama_kv_blocksvd_context {
    llama_kv_blocksvd_params params;
    // layers -> chunks
    std::vector<std::vector<llama_kv_blocksvd_chunk>> layers;

    // For backend mode: persistent compressed xKV cross-layer chunks.
    std::vector<llama_kv_blocksvd_xkv_chunk> xkv_chunks;
    // Incremented whenever the packed-factor set changes. Graph inputs use it
    // as a fast change signal before refreshing a fixed-capacity factor pool.
    uint64_t                                 xkv_generation = 0;
    // Host factor pools and statically allocated per-backend replicas. This is
    // mutable because graph construction refreshes execution state without
    // changing the compressed xKV archive itself.
    mutable std::shared_ptr<llama_kv_blocksvd_execution_pool_registry> execution_pools;

    // For reconstruct mode: accumulate tokens until a full block is ready
    struct pending_layer {
        std::vector<float> k;
        std::vector<float> v;
        std::vector<uint32_t> slots; // global cell slots
        int32_t n_kv_total = 0;       // total n_kv at first pending token (for chunk seq_start)
    };
    std::vector<pending_layer> pending;

    // For cross-layer reconstruct/backend mode: accumulate tokens per layer group.
    struct pending_xkv_group {
        int32_t layer_start = -1;     // first layer in the group
        int32_t group_size  = 0;      // number of layers in the group
        uint32_t stream     = 0;      // KV cache stream
        llama_seq_id seq_id = 0;      // logical sequence id
        int32_t n_kv_total  = 0;      // total n_kv at first pending token
        std::vector<std::vector<float>> k; // per-layer K values
        std::vector<std::vector<float>> v; // per-layer V values
        std::vector<uint32_t> slots;       // global cell slots
        std::vector<llama_pos> pos;        // token positions
    };
    std::vector<pending_xkv_group> pending_xkv;

    // Lazy decode cache for chunked attention. Indexed by xkv_chunks index; each
    // entry remains valid while the append-only archive is alive, so a chunk is
    // not decompressed again on every graph reuse. Cleared with the archive.
    struct decoded_chunk {
        // Per-layer flat K rows: [group_size][n_tokens * n_head_kv * head_dim_k]
        std::vector<std::vector<float>> k;
        // Per-layer flat V rows: [group_size][n_tokens * n_head_kv * head_dim_v]
        std::vector<std::vector<float>> v;
        int32_t n_tokens = 0;
    };
    mutable std::vector<std::unique_ptr<decoded_chunk>> decode_cache;

    // Lazy rank-domain factor cache for direct low-rank attention. Populated by
    // the single-threaded refs hoist. All layers in
    // an xKV chunk share the same U/S/Vh factors (that's the xKV cross-layer
    // property); only the layer offset into Vh's "combined_dim" varies. Cached
    // buffers are the fp32-dequant + US premultiply results.
    struct rank_chunk {
        int32_t n_tokens = 0;        // tokens in this compressed block
        int32_t r_k      = 0;        // K-side rank
        int32_t r_v      = 0;        // V-side rank
        int32_t combined_k_dim = 0;  // group_size * n_head_kv * head_dim_k
        int32_t combined_v_dim = 0;  // group_size * n_head_kv * head_dim_v
        // US_k: (n_tokens, r_k) row-major; premultiplied U_k * diag(S_k)
        std::vector<float> us_k;
        // US_v: (n_tokens, r_v) row-major
        std::vector<float> us_v;
        // Vh_k dequantized: (r_k, combined_k_dim) row-major
        std::vector<float> vh_k;
        // Vh_v dequantized: (r_v, combined_v_dim) row-major
        std::vector<float> vh_v;

        // Optional packed-arithmetic reference data. Built lazily only for the
        // C4C5 diagnostic mode so the default portable path does not duplicate
        // the compressed INT8 archive.
        bool                folded_ready = false;
        std::vector<int8_t> u_k_q;
        std::vector<int8_t> u_v_q;
        std::vector<int8_t> vh_k_q;
        std::vector<int8_t> vh_v_q;
        std::vector<float>  rank_scale_k;
        std::vector<float>  rank_scale_v;
    };
    mutable std::vector<std::unique_ptr<rank_chunk>> rank_cache;
};

llama_kv_blocksvd_context * llama_kv_blocksvd_init(const llama_kv_blocksvd_params & params);
void                        llama_kv_blocksvd_free(llama_kv_blocksvd_context * ctx);

// k/v layout follows WHLR copy_current_kv_chunk_f32():
// k: (head_dim_k, n_head_kv, n_kv)
// v: (n_kv, n_head_kv, head_dim_v) when v_transposed == true
//    (head_dim_v, n_head_kv, n_kv) when v_transposed == false
bool llama_kv_blocksvd_compress_chunk(
        llama_kv_blocksvd_context * ctx,
        int32_t layer,
        const float * k,
        const float * v,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv,
        int32_t n_kv_start,
        bool v_transposed);

bool llama_kv_blocksvd_decompress_chunk(
        const llama_kv_blocksvd_chunk & chunk,
        float * k,
        float * v,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        bool v_transposed);

// xKV-style cross-layer SVD: group consecutive layers, stack their K/V along the
// feature (head) dimension, run a single SVD for K and one for V, then reconstruct
// and split back per-layer.
bool llama_kv_blocksvd_compress_xkv_group(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        int32_t n_tokens,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err = nullptr);

// Accumulate dense K/V for a layer group. When pending length reaches block_size,
// run cross-layer SVD reconstruction and return the per-layer reconstructed rows.
bool llama_kv_blocksvd_append_and_reconstruct_xkv_group(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        const uint32_t * slots,
        int32_t n_tokens,
        uint32_t stream,
        llama_seq_id seq_id,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv_start,
        std::vector<uint32_t> & out_slots,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err = nullptr);

// Accumulate dense K/V for a layer. When pending length reaches block_size,
// compress, decompress, and return the reconstructed rows + slots so the caller
// can write them back into the dense KV cache.
bool llama_kv_blocksvd_append_and_reconstruct(
        llama_kv_blocksvd_context * ctx,
        int32_t layer,
        const float * k,
        const float * v,
        const uint32_t * slots,
        int32_t n_tokens,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv_start,
        bool v_transposed,
        std::vector<uint32_t> & out_slots,
        std::vector<float> & out_k,
        std::vector<float> & out_v,
        std::string * err = nullptr);

// xKV-style cross-layer persistent backend: accumulate tokens, compress when a full
// block is ready, store the quantized factors in ctx->xkv_chunks, and return the
// compressed slots/positions.  Tail tokens remain in pending.
bool llama_kv_blocksvd_append_and_compress_xkv_group_store(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        const uint32_t * slots,
        const llama_pos * pos,
        int32_t n_tokens,
        uint32_t stream,
        llama_seq_id seq_id,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        std::vector<uint32_t> & out_compressed_slots,
        std::vector<llama_pos>  & out_compressed_pos,
        std::string * err = nullptr);

// Decompress one persistent xKV chunk back into per-layer flat K/V vectors.
// out_k/out_v are resized to [group_size][n_tokens * d_k_or_v].
bool llama_kv_blocksvd_decompress_xkv_chunk(
        const llama_kv_blocksvd_xkv_chunk & chunk,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err = nullptr);

// Reconstruct one layer's K/V rows for logical cells [0, n_kv) across all streams.
// k_staged/v_staged are f32 buffers in the same layout as the dense KV cache view
// (capacity cells per stream, n_stream streams, n_embd_gqa floats per cell).
// out_k/out_v are written in logical cell order: for each stream s and cell c,
// the slice starts at ((s * n_kv) + c) * d_k (or d_v).
bool llama_kv_blocksvd_reconstruct_layer(
        const llama_kv_blocksvd_context & bctx,
        const llama_kv_blocksvd_staging_t & staging,
        int32_t il,
        uint32_t n_kv,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        const std::vector<float> & k_staged,
        const std::vector<float> & v_staged,
        std::vector<float> & out_k,
        std::vector<float> & out_v,
        std::string * err = nullptr);

// Clear all pending K/V/slots and persistent chunks (e.g. on KV cache clear).
void llama_kv_blocksvd_clear_pending(llama_kv_blocksvd_context * ctx);
void llama_kv_blocksvd_clear(llama_kv_blocksvd_context * ctx);

// Invalidate the per-forward decode cache. Call before graph build in each ubatch.
void llama_kv_blocksvd_reset_decode_cache(llama_kv_blocksvd_context * ctx);

// --- Chunked Attention (Tier 3: iterative materialize + online softmax) ---

struct ggml_tensor;

// Per-forward refs built single-threaded in set_input (llama_chunked_attn_build_refs)
// and consumed read-only by all worker threads in the parallel compute op.
//
// `pos` is owned by value (a per-forward snapshot copied from the chunk during
// the hoist) rather than a pointer into bctx->xkv_chunks. xkv_chunks is an
// append-only but *growable* std::vector: a push_back during compression
// reallocates its storage and would leave any &xkv_chunks[ci].pos captured in a
// ref dangling, which caused the heap-buffer-overflow. Owning the positions here
// keeps them valid for the whole forward regardless of xkv_chunks storage, and
// the ref stays valid even if chunks_dense/chunks_rank itself reallocates
// (vector move transfers the buffer), so correctness never relies on reserve().
struct llama_chunked_attn_chunk_ref {
    const float * k_data;
    const float * v_data;
    int32_t n_tokens;
    std::vector<llama_pos> pos;
};

struct llama_kv_lowrank_direct_chunk_ref {
    const void * rc; // rank_chunk *, opaque here to avoid header exposing internals
    std::vector<llama_pos> pos;
    int32_t layer_offset_k;
    int32_t layer_offset_v;
};

// Parameters passed via userdata to the chunked attention custom op.
struct llama_chunked_attn_params {
    const llama_kv_blocksvd_context * bctx;
    const llama_kv_blocksvd_staging_t * staging;
    int32_t il;
    uint32_t n_kv;
    int32_t n_head_kv;
    int32_t n_head_q;
    int32_t head_dim_k;
    int32_t head_dim_v;
    float scale;
    uint32_t n_stream;
    uint32_t cache_size;
    std::vector<llama_pos> q_pos;
    std::vector<llama_pos> slot_pos;

    // Optional GGML/HTP reconstruction input. When enabled, src[3]/src[4]
    // contain dense FP16 compressed-history K/V in [H,N,B,D] storage order;
    // the active staging window remains in src[1]/src[2].
    bool edgekv_reconstructed = false;
    int32_t edgekv_n_blocks   = 0;
    int32_t edgekv_block_size = 0;
    std::vector<llama_pos> edgekv_positions;

    // Selects which compute path this node runs: true = kv_lowrank_direct
    // (rank-domain, uses chunks_rank), false = chunked_attn (dense, uses
    // chunks_dense). Set by build_attn; read by the set_input hoist to build
    // the matching refs list.
    bool use_lowrank_direct = false;

    // C4C5 CPU-reference diagnostics. Defaults preserve the historical
    // portable path: split dequantization and active-window-first traversal.
    // These flags do not alter the packed GGML/HTP operation.
    bool direct_folded_scale     = false;
    bool direct_compressed_first = false;

    // Per-forward refs built by the single-threaded set_input hoist and consumed
    // read-only by all worker threads during the parallel compute op. Only one
    // list is populated per node, per use_lowrank_direct above.
    //
    // These are built at set_input (main thread, before the parallel region and
    // strictly after the previous forward's compute has drained) rather than
    // inside the compute op. Building inside the op required a hand-rolled
    // spin-latch to gate workers behind thread 0's rebuild, which raced: a
    // worker could still be reading the previous invocation's refs while thread
    // 0 cleared and rebuilt this list (ASan: heap-use-after-free). A custom op
    // has no access to ggml_barrier, so the op must stay strictly read-only and
    // all mutation must happen at the single-threaded graph boundary.
    mutable std::vector<llama_chunked_attn_chunk_ref>        chunks_dense;
    mutable std::vector<llama_kv_lowrank_direct_chunk_ref>   chunks_rank;

    // Reservation-side workspace footprint of this op invocation.
    // Includes: the params struct itself, both position vectors (capacity),
    // the transient VKQ_acc accumulator (head_dim_v F32), and the transient
    // chunks list built per forward (reserved to xkv_chunks.size()).
    // The output tensor is allocated by ggml and is *not* included here.
    size_t bytes() const {
        size_t b = sizeof(*this);
        b += q_pos.capacity()    * sizeof(llama_pos);
        b += slot_pos.capacity() * sizeof(llama_pos);
        b += edgekv_positions.capacity() * sizeof(llama_pos);
        b += (size_t) head_dim_v * sizeof(float);
        if (bctx && !edgekv_reconstructed) {
            for (const auto & chunk : bctx->xkv_chunks) {
                if (chunk.stream != 0 || il < chunk.layer_start || il >= chunk.layer_start + chunk.group_size) {
                    continue;
                }
                b += use_lowrank_direct ? sizeof(llama_kv_lowrank_direct_chunk_ref) :
                                          sizeof(llama_chunked_attn_chunk_ref);
                b += chunk.pos.size() * sizeof(llama_pos);
            }
        }
        return b;
    }
};

// Custom op compute function: chunked attention with online softmax.
// dst: [head_dim_v * n_head_q, n_tokens]
// dst->src[0] (q):        [head_dim_k, n_head_q,  n_tokens] (F32, pre-stream-split)
// dst->src[1] (k_active): [head_dim_k, n_head_kv, cache_size, n_stream] (F32)
// dst->src[2] (v_active): [head_dim_v, n_head_kv, cache_size, n_stream] (F32, non-transposed)
// dst->src[3] (k_archive, optional): [head_dim_k, block_size, n_blocks, n_head_kv] (F16)
// dst->src[4] (v_archive, optional): [head_dim_v, block_size, n_blocks, n_head_kv] (F16)
// n_stream must be 1 in this milestone.
void llama_chunked_attn_compute(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata);

// Direct low-rank attention consumer: same params + dst/src layout as
// llama_chunked_attn_compute, but consumes compressed factors without
// materializing dense per-token K/V for each block.
//
// The implementation evaluates K logits and V values directly in the rank
// domain and partitions independent query-head/token work across ggml workers.
void llama_kv_lowrank_direct_attn_compute(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata);

// Single-threaded hoist: build the per-forward refs list (chunks_dense or
// chunks_rank, per p->use_lowrank_direct) that the parallel compute op above
// consumes read-only. Called from set_input on the main thread, before the
// parallel region and after the previous forward's compute has drained, so the
// compute op never mutates shared state. Decompresses / dequantizes chunk
// factors into bctx->decode_cache / rank_cache as needed (both are
// vector<unique_ptr<...>>, so their pointees are stable across the op).
void llama_chunked_attn_build_refs(llama_chunked_attn_params * p);
