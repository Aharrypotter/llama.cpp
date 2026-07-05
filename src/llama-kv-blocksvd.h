#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Backend-agnostic core for Block SVD KV cache compression.
// This header lives in src/ so llama core code can call the CPU reference
// compression/reconstruction path without depending on common/.

struct llama_kv_blocksvd_chunk {
    int32_t seq_start = 0;
    int32_t seq_end   = 0;

    // Total sequence length of the layer.  Required to compute the correct
    // stride when writing back the transposed V cache.
    int32_t n_kv = 0;

    // Flat feature dimension: n_head_kv * (head_dim_k + head_dim_v)
    int32_t n_flat = 0;

    // Quantization metadata.  The quantized buffers below store raw bytes;
    // element size is quant_bits / 8.
    int32_t quant_bits = 8;

    // Quantized U: (block_size, rank)
    std::vector<int8_t> u_q;
    float u_scale = 1.0f;

    // Quantized S: (rank,)
    std::vector<int8_t> s_q;
    float s_scale = 1.0f;

    // Quantized Vh: (rank, n_flat)
    std::vector<int8_t> vh_q;
    float vh_scale = 1.0f;
};

struct llama_kv_blocksvd_context {
    llama_kv_blocksvd_params params;
    // layers -> chunks
    std::vector<std::vector<llama_kv_blocksvd_chunk>> layers;

    // For reconstruct mode: accumulate tokens until a full block is ready
    struct pending_layer {
        std::vector<float> k;
        std::vector<float> v;
        std::vector<uint32_t> slots; // global cell slots
        int32_t n_kv_total = 0;       // total n_kv at first pending token (for chunk seq_start)
    };
    std::vector<pending_layer> pending;
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

// Clear all pending K/V/slots (e.g. on KV cache clear).
void llama_kv_blocksvd_clear_pending(llama_kv_blocksvd_context * ctx);
