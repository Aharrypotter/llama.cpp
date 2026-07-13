#pragma once

#include "llama-kv-blocksvd.h"

#include <array>
#include <cstddef>
#include <cstdint>
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

    int32_t n_blocks    = 0;
    int32_t block_size  = 0;
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
