#include "llama-kv-blocksvd-execution.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
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

bool llama_kv_blocksvd_pack_int8_reconstruct_dispatch(const llama_kv_blocksvd_context &             ctx,
                                                      int32_t                                       layer,
                                                      uint32_t                                      stream,
                                                      llama_seq_id                                  seq_id,
                                                      llama_kv_blocksvd_int8_reconstruct_dispatch & out,
                                                      std::string *                                 err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return false;
    };

    std::vector<const llama_kv_blocksvd_xkv_chunk *> chunks;
    for (const auto & chunk : ctx.xkv_chunks) {
        const int64_t layer_end = static_cast<int64_t>(chunk.layer_start) + chunk.group_size;
        if (chunk.stream == stream && chunk.seq_id == seq_id && layer >= chunk.layer_start &&
            static_cast<int64_t>(layer) < layer_end) {
            if (chunk.pos.empty()) {
                return fail("matching xKV chunk has no positions");
            }
            chunks.push_back(&chunk);
        }
    }
    if (chunks.empty()) {
        return fail("no xKV chunks match layer, stream, and sequence");
    }

    std::sort(chunks.begin(), chunks.end(), [](const auto * lhs, const auto * rhs) {
        return std::lexicographical_compare(lhs->pos.begin(), lhs->pos.end(), rhs->pos.begin(), rhs->pos.end());
    });

    if (chunks.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return fail("dispatch block count exceeds int32_t");
    }

    const auto & first = *chunks.front();
    if (first.group_size <= 0 || first.n_head_kv <= 0 || first.head_dim_k <= 0 || first.head_dim_v <= 0 ||
        first.k_factors.rank <= 0 || first.v_factors.rank <= 0) {
        return fail("dispatch chunk dimensions must be positive");
    }
    if (first.slots.empty() || first.slots.size() != first.pos.size() ||
        first.slots.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return fail("dispatch chunk slots and positions must have the same positive size");
    }

    const int32_t block_size = static_cast<int32_t>(first.slots.size());
    if (ctx.params.block_size > 0 && block_size != ctx.params.block_size) {
        return fail("dispatch chunk size does not match configured block size");
    }

    const int64_t combined_k_dim64 = static_cast<int64_t>(first.group_size) * first.n_head_kv * first.head_dim_k;
    const int64_t combined_v_dim64 = static_cast<int64_t>(first.group_size) * first.n_head_kv * first.head_dim_v;
    if (combined_k_dim64 > std::numeric_limits<int32_t>::max() ||
        combined_v_dim64 > std::numeric_limits<int32_t>::max()) {
        return fail("combined factor dimension exceeds int32_t");
    }
    const int32_t combined_k_dim = static_cast<int32_t>(combined_k_dim64);
    const int32_t combined_v_dim = static_cast<int32_t>(combined_v_dim64);

    llama_kv_blocksvd_int8_reconstruct_dispatch next;
    next.n_blocks    = static_cast<int32_t>(chunks.size());
    next.block_size  = block_size;
    next.rank_k      = first.k_factors.rank;
    next.rank_v      = first.v_factors.rank;
    next.group_size  = first.group_size;
    next.layer_start = first.layer_start;
    next.layer_index = layer - first.layer_start;
    next.n_head_kv   = first.n_head_kv;
    next.head_dim_k  = first.head_dim_k;
    next.head_dim_v  = first.head_dim_v;
    next.stream      = stream;
    next.seq_id      = seq_id;

    std::vector<int8_t> k_u_q;
    std::vector<int8_t> v_u_q;
    std::vector<int8_t> k_vh_q;
    std::vector<int8_t> v_vh_q;
    std::vector<float>  k_rank_scale;
    std::vector<float>  v_rank_scale;

    llama_pos previous_position      = 0;
    bool      have_previous_position = false;
    for (const auto * chunk_ptr : chunks) {
        const auto & chunk = *chunk_ptr;
        if (chunk.layer_start != next.layer_start || chunk.group_size != next.group_size ||
            chunk.n_head_kv != next.n_head_kv || chunk.head_dim_k != next.head_dim_k ||
            chunk.head_dim_v != next.head_dim_v || chunk.k_factors.rank != next.rank_k ||
            chunk.v_factors.rank != next.rank_v || chunk.slots.size() != static_cast<size_t>(next.block_size) ||
            chunk.pos.size() != static_cast<size_t>(next.block_size)) {
            return fail("matching xKV chunks do not share one dispatch shape");
        }

        for (llama_pos position : chunk.pos) {
            if (position < 0 || position > std::numeric_limits<int32_t>::max()) {
                return fail("dispatch position is outside non-negative int32_t range");
            }
            if (have_previous_position && position <= previous_position) {
                return fail("dispatch positions must be globally strictly increasing");
            }
            next.block_positions.push_back(static_cast<int32_t>(position));
            previous_position      = position;
            have_previous_position = true;
        }

        llama_kv_blocksvd_int8_execution_factors k_view;
        llama_kv_blocksvd_int8_execution_factors v_view;
        std::string                              factor_error;
        if (!llama_kv_blocksvd_pack_int8_execution_factors(chunk.k_factors, next.block_size, combined_k_dim, k_view,
                                                           &factor_error)) {
            return fail("could not pack K execution factors: " + factor_error);
        }
        if (!llama_kv_blocksvd_pack_int8_execution_factors(chunk.v_factors, next.block_size, combined_v_dim, v_view,
                                                           &factor_error)) {
            return fail("could not pack V execution factors: " + factor_error);
        }

        k_u_q.insert(k_u_q.end(), k_view.u_q.begin(), k_view.u_q.end());
        v_u_q.insert(v_u_q.end(), v_view.u_q.begin(), v_view.u_q.end());
        k_vh_q.insert(k_vh_q.end(), k_view.vh_q.begin(), k_view.vh_q.end());
        v_vh_q.insert(v_vh_q.end(), v_view.vh_q.begin(), v_view.vh_q.end());
        k_rank_scale.insert(k_rank_scale.end(), k_view.rank_scale.begin(), k_view.rank_scale.end());
        v_rank_scale.insert(v_rank_scale.end(), v_view.rank_scale.begin(), v_view.rank_scale.end());
    }

    auto align_bytes = [&fail](size_t bytes, size_t & aligned) {
        constexpr size_t alignment = llama_kv_blocksvd_int8_reconstruct_dispatch::buffer_alignment;
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            return fail("dispatch buffer alignment overflows size_t");
        }
        aligned = ((bytes + alignment - 1) / alignment) * alignment;
        if (aligned > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return fail("dispatch buffer exceeds signed HTP op-param range");
        }
        return true;
    };

    auto pack_aligned_pair = [&fail, &align_bytes](const auto & k_values, const auto & v_values, auto & packed,
                                                   int32_t & v_offset_bytes) {
        using value_type = typename std::decay_t<decltype(k_values)>::value_type;
        if (k_values.size() > std::numeric_limits<size_t>::max() / sizeof(value_type) ||
            v_values.size() > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
            return fail("dispatch component byte size overflows size_t");
        }
        const size_t k_bytes  = k_values.size() * sizeof(value_type);
        const size_t v_bytes  = v_values.size() * sizeof(value_type);
        size_t       v_offset = 0;
        if (!align_bytes(k_bytes, v_offset)) {
            return false;
        }
        if (v_bytes > std::numeric_limits<size_t>::max() - v_offset) {
            return fail("dispatch packed buffer size overflows size_t");
        }
        size_t total_bytes = 0;
        if (!align_bytes(v_offset + v_bytes, total_bytes)) {
            return false;
        }
        if (v_offset % sizeof(value_type) != 0 || total_bytes % sizeof(value_type) != 0) {
            return fail("dispatch alignment is incompatible with component type");
        }

        packed = k_values;
        packed.resize(v_offset / sizeof(value_type), value_type{});
        packed.insert(packed.end(), v_values.begin(), v_values.end());
        packed.resize(total_bytes / sizeof(value_type), value_type{});
        v_offset_bytes = static_cast<int32_t>(v_offset);
        return true;
    };

    if (!pack_aligned_pair(k_u_q, v_u_q, next.u_q, next.v_u_offset_bytes) ||
        !pack_aligned_pair(k_vh_q, v_vh_q, next.vh_q, next.v_vh_offset_bytes) ||
        !pack_aligned_pair(k_rank_scale, v_rank_scale, next.rank_scale, next.v_rank_scale_offset_bytes)) {
        return false;
    }

    auto checked_mul = [&fail](size_t lhs, size_t rhs, size_t & product) {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
            return fail("dispatch dense output size overflows size_t");
        }
        product = lhs * rhs;
        return true;
    };

    size_t dense_k_elements = static_cast<size_t>(next.n_head_kv);
    size_t dense_v_elements = static_cast<size_t>(next.n_head_kv);
    for (size_t dimension : { static_cast<size_t>(next.n_blocks), static_cast<size_t>(next.block_size),
                              static_cast<size_t>(next.head_dim_k) }) {
        if (!checked_mul(dense_k_elements, dimension, dense_k_elements)) {
            return false;
        }
    }
    for (size_t dimension : { static_cast<size_t>(next.n_blocks), static_cast<size_t>(next.block_size),
                              static_cast<size_t>(next.head_dim_v) }) {
        if (!checked_mul(dense_v_elements, dimension, dense_v_elements)) {
            return false;
        }
    }
    size_t dense_k_bytes = 0;
    size_t dense_v_bytes = 0;
    if (!checked_mul(dense_k_elements, 2, dense_k_bytes) || !checked_mul(dense_v_elements, 2, dense_v_bytes)) {
        return false;
    }
    size_t dense_v_offset = 0;
    if (!align_bytes(dense_k_bytes, dense_v_offset)) {
        return false;
    }
    if (dense_v_bytes > std::numeric_limits<size_t>::max() - dense_v_offset) {
        return fail("dispatch dense output buffer size overflows size_t");
    }
    size_t dense_total_bytes = 0;
    if (!align_bytes(dense_v_offset + dense_v_bytes, dense_total_bytes)) {
        return false;
    }
    next.dense_v_offset_bytes = static_cast<int32_t>(dense_v_offset);
    next.dense_total_bytes    = static_cast<int32_t>(dense_total_bytes);

    out = std::move(next);
    return true;
}
