#pragma once

#include "llama-kv-blocksvd.h"

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

bool llama_kv_blocksvd_pack_int8_execution_factors(
        const llama_kv_blocksvd_xkv_factors & factors,
        int32_t n_tokens,
        int32_t n_dim,
        llama_kv_blocksvd_int8_execution_factors & out,
        std::string * err = nullptr);
