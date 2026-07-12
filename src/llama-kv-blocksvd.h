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

struct llama_kv_blocksvd_context {
    llama_kv_blocksvd_params params;
    // layers -> chunks
    std::vector<std::vector<llama_kv_blocksvd_chunk>> layers;

    // For backend mode: persistent compressed xKV cross-layer chunks.
    std::vector<llama_kv_blocksvd_xkv_chunk> xkv_chunks;

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

    // Per-forward decode cache for chunked attention.
    // Indexed by xkv_chunks index; each slot holds the group-decompressed K/V for
    // that chunk so the compute op does not decode the same chunk N_layer times per forward.
    // Reset at the start of each ubatch via llama_kv_blocksvd_reset_decode_cache.
    struct decoded_chunk {
        // Per-layer flat K rows: [group_size][n_tokens * n_head_kv * head_dim_k]
        std::vector<std::vector<float>> k;
        // Per-layer flat V rows: [group_size][n_tokens * n_head_kv * head_dim_v]
        std::vector<std::vector<float>> v;
        int32_t n_tokens = 0;
    };
    mutable std::vector<std::unique_ptr<decoded_chunk>> decode_cache;
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

    // Reservation-side workspace footprint of this op invocation.
    // Includes: the params struct itself, both position vectors (capacity),
    // the transient VKQ_acc accumulator (head_dim_v F32), and the transient
    // chunks list built per forward (reserved to xkv_chunks.size()).
    // The output tensor is allocated by ggml and is *not* included here.
    size_t bytes() const {
        size_t b = sizeof(*this);
        b += q_pos.capacity()    * sizeof(llama_pos);
        b += slot_pos.capacity() * sizeof(llama_pos);
        b += (size_t) head_dim_v * sizeof(float);
        // chunk_ref inside compute: 2 pointers + int32 + pointer-to-vector.
        // Approximate with 4 pointers per chunk to stay portable.
        const size_t chunk_ref_sz = 4 * sizeof(void *);
        const size_t n_chunks = bctx ? bctx->xkv_chunks.size() : 0;
        b += n_chunks * chunk_ref_sz;
        return b;
    }
};

// Custom op compute function: chunked attention with online softmax.
// dst: [head_dim_v * n_head_q, n_tokens]
// dst->src[0] (q):        [head_dim_k, n_head_q,  n_tokens] (F32, pre-stream-split)
// dst->src[1] (k_active): [head_dim_k, n_head_kv, cache_size, n_stream] (F32)
// dst->src[2] (v_active): [head_dim_v, n_head_kv, cache_size, n_stream] (F32, non-transposed)
// n_stream must be 1 in this milestone.
void llama_chunked_attn_compute(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata);
