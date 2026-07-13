#pragma once

#include "llama-kv-blocksvd.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Consumer-native view for one INT8 U/S/Vh factorization. U/Vh retain the
// archive bytes; S and all three archive scales are folded into rank_scale.
struct llama_kv_blocksvd_int8_execution_factors {
    int32_t rank     = 0;
    int32_t n_tokens = 0;
    int32_t n_dim    = 0;

    std::vector<int8_t> u_q;
    std::vector<int8_t> vh_q;
    std::vector<float>  rank_scale;
};

// HTP-facing storage contract for reconstructing one layer from multiple xKV
// blocks. K and V components share four aligned input buffers; the output is
// one aligned FP16 buffer containing dense K followed by dense V. Byte offsets
// fit in HTP op_params and point at 128-byte-aligned component starts.
struct llama_kv_blocksvd_int8_reconstruct_dispatch {
    static constexpr int32_t buffer_alignment   = 128;
    static constexpr size_t  htp_op_param_count = 14;

    // n_blocks is the static pool capacity serialized into HTP op_params.
    // valid_blocks is host-only state; unused tail blocks carry position -1.
    int32_t n_blocks     = 0;
    int32_t valid_blocks = 0;
    int32_t block_size   = 0;
    int32_t rank_k      = 0;
    int32_t rank_v      = 0;
    int32_t group_size  = 0;
    int32_t layer_start = 0;
    int32_t layer_index = 0;
    int32_t n_head_kv   = 0;
    int32_t head_dim_k  = 0;
    int32_t head_dim_v  = 0;

    uint32_t     stream = 0;
    llama_seq_id seq_id = 0;

    int32_t v_u_offset_bytes          = 0;
    int32_t v_vh_offset_bytes         = 0;
    int32_t v_rank_scale_offset_bytes = 0;
    int32_t dense_v_offset_bytes      = 0;
    int32_t dense_total_bytes         = 0;

    // U_q: K [N,B,Rk], aligned padding, V [N,B,Rv].
    std::vector<int8_t>  u_q;
    // Vh_q: K [N,Rk,G,H,Dk], aligned padding, V [N,Rv,G,H,Dv].
    std::vector<int8_t>  vh_q;
    // rank_scale: K [N,Rk], aligned padding, V [N,Rv].
    std::vector<float>   rank_scale;
    // Position-sorted [N,B]. Dense output layout is [H,N,B,D].
    std::vector<int32_t> block_positions;

    // Stable serialization for htp_op_desc.params:
    // N, B, Rk, Rv, G, layer_index, H, Dk, Dv, then five byte offsets/sizes.
    std::array<int32_t, htp_op_param_count> htp_op_params() const {
        return {
            n_blocks,
            block_size,
            rank_k,
            rank_v,
            group_size,
            layer_index,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            v_u_offset_bytes,
            v_vh_offset_bytes,
            v_rank_scale_offset_bytes,
            dense_v_offset_bytes,
            dense_total_bytes,
        };
    }
};

struct llama_kv_blocksvd_int8_pool_update {
    size_t first_dirty_block = 0;
    size_t last_dirty_block  = 0;
    bool   full_repack       = false;
};

struct llama_kv_blocksvd_int8_backend_pool;

// A graph-stable view of one statically allocated backend factor pool. The
// owner keeps both tensor metadata and the backend buffer alive across graph
// topology rebuilds.
struct llama_kv_blocksvd_int8_backend_view {
    std::shared_ptr<llama_kv_blocksvd_int8_backend_pool> owner;

    ggml_tensor * u_q             = nullptr;
    ggml_tensor * vh_q            = nullptr;
    ggml_tensor * rank_scale      = nullptr;
    ggml_tensor * block_positions = nullptr;

    const llama_kv_blocksvd_int8_reconstruct_dispatch * storage = nullptr;
    ggml_backend_t                                       backend = nullptr;
};

bool llama_kv_blocksvd_pack_int8_execution_factors(
        const llama_kv_blocksvd_xkv_factors & factors,
        int32_t n_tokens,
        int32_t n_dim,
        llama_kv_blocksvd_int8_execution_factors & out,
        std::string * err = nullptr);

bool llama_kv_blocksvd_pack_int8_reconstruct_dispatch(const llama_kv_blocksvd_context &             ctx,
                                                      int32_t                                       layer,
                                                      uint32_t                                      stream,
                                                      llama_seq_id                                  seq_id,
                                                      llama_kv_blocksvd_int8_reconstruct_dispatch & out,
                                                      std::string *                                 err = nullptr);

bool llama_kv_blocksvd_pack_int8_reconstruct_pool(const llama_kv_blocksvd_context &             ctx,
                                                  int32_t                                       layer,
                                                  uint32_t                                      stream,
                                                  llama_seq_id                                  seq_id,
                                                  int32_t                                       block_capacity,
                                                  llama_kv_blocksvd_int8_reconstruct_dispatch & out,
                                                  std::string *                                 err = nullptr);

// Refresh an existing fixed-capacity host pool. The append-only fast path
// packs only newly appended xKV chunks and reports their block range.
bool llama_kv_blocksvd_refresh_int8_reconstruct_pool(
        const llama_kv_blocksvd_context &             ctx,
        int32_t                                       layer,
        uint32_t                                      stream,
        llama_seq_id                                  seq_id,
        llama_kv_blocksvd_int8_reconstruct_dispatch & pool,
        llama_kv_blocksvd_int8_pool_update &          update,
        std::string *                                 err = nullptr);

// Acquire a statically allocated backend replica. The selected backend is the
// highest-priority scheduler backend that supports the reconstruct op for this
// layer, with backend_cpu as the required fallback.
bool llama_kv_blocksvd_acquire_int8_backend_pool(
        const llama_kv_blocksvd_context &       ctx,
        int32_t                                 layer,
        uint32_t                                stream,
        llama_seq_id                            seq_id,
        int32_t                                 block_capacity,
        ggml_backend_sched_t                    sched,
        ggml_backend_t                          backend_cpu,
        llama_kv_blocksvd_int8_backend_view &   out,
        std::string *                           err = nullptr);

// Synchronize host changes into an acquired backend replica. Append-only
// growth uploads only blocks not yet present in that replica.
bool llama_kv_blocksvd_sync_int8_backend_pool(
        const llama_kv_blocksvd_context &     ctx,
        llama_kv_blocksvd_int8_backend_view & view,
        std::string *                         err = nullptr);
