#include "llama-kv-blocksvd-execution.h"

#include <cmath>
#include <limits>
#include <utility>

bool llama_kv_blocksvd_pack_int8_execution_factors(
        const llama_kv_blocksvd_xkv_factors & factors,
        int32_t n_tokens,
        int32_t n_dim,
        llama_kv_blocksvd_int8_execution_factors & out,
        std::string * err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return false;
    };

    if (factors.quant_bits != 8) {
        return fail("INT8 execution view requires quant_bits == 8");
    }
    if (factors.rank <= 0 || n_tokens <= 0 || n_dim <= 0) {
        return fail("rank, n_tokens, and n_dim must be positive");
    }

    const size_t rank = static_cast<size_t>(factors.rank);
    if (static_cast<size_t>(n_tokens) > std::numeric_limits<size_t>::max() / rank ||
        rank > std::numeric_limits<size_t>::max() / static_cast<size_t>(n_dim)) {
        return fail("execution-view dimensions overflow size_t");
    }

    const size_t expected_u  = static_cast<size_t>(n_tokens) * rank;
    const size_t expected_vh = rank * static_cast<size_t>(n_dim);
    if (factors.u_q.size() != expected_u) {
        return fail("U factor size does not match n_tokens * rank");
    }
    if (factors.s_q.size() != rank) {
        return fail("S factor size does not match rank");
    }
    if (factors.vh_q.size() != expected_vh) {
        return fail("Vh factor size does not match rank * n_dim");
    }
    if (!std::isfinite(factors.u_scale) || factors.u_scale <= 0.0f ||
        !std::isfinite(factors.s_scale) || factors.s_scale <= 0.0f ||
        !std::isfinite(factors.vh_scale) || factors.vh_scale <= 0.0f) {
        return fail("factor scales must be finite and positive");
    }

    llama_kv_blocksvd_int8_execution_factors next;
    next.rank     = factors.rank;
    next.n_tokens = n_tokens;
    next.n_dim    = n_dim;
    next.u_q      = factors.u_q;
    next.vh_q     = factors.vh_q;
    next.rank_scale.resize(rank);

    for (size_t r = 0; r < rank; ++r) {
        float scale = static_cast<float>(factors.s_q[r]);
        scale *= factors.u_scale;
        scale *= factors.s_scale;
        scale *= factors.vh_scale;
        if (!std::isfinite(scale)) {
            return fail("folded rank scale is not finite");
        }
        next.rank_scale[r] = scale;
    }

    out = std::move(next);
    return true;
}
