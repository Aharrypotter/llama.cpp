#include "llama-kv-blocksvd-execution.h"

#include "llama-impl.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
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

static bool llama_kv_blocksvd_pack_int8_reconstruct_impl(const llama_kv_blocksvd_context &             ctx,
                                                         int32_t                                       layer,
                                                         uint32_t                                      stream,
                                                         llama_seq_id                                  seq_id,
                                                         int32_t                                       block_capacity,
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

    const int32_t valid_blocks = static_cast<int32_t>(chunks.size());
    if (block_capacity == 0) {
        block_capacity = valid_blocks;
    }
    if (block_capacity < valid_blocks) {
        return fail("dispatch block capacity is smaller than the matching block count");
    }
    if (block_capacity <= 0) {
        return fail("dispatch block capacity must be positive");
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
    next.n_blocks     = block_capacity;
    next.valid_blocks = valid_blocks;
    next.block_size   = block_size;
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

    auto checked_mul = [&fail](size_t lhs, size_t rhs, size_t & product) {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
            return fail("dispatch size overflows size_t");
        }
        product = lhs * rhs;
        return true;
    };

    auto checked_product = [&checked_mul](std::initializer_list<size_t> dimensions, size_t & product) {
        product = 1;
        for (size_t dimension : dimensions) {
            if (!checked_mul(product, dimension, product)) {
                return false;
            }
        }
        return true;
    };

    size_t k_u_capacity      = 0;
    size_t v_u_capacity      = 0;
    size_t k_vh_capacity     = 0;
    size_t v_vh_capacity     = 0;
    size_t k_scale_capacity  = 0;
    size_t v_scale_capacity  = 0;
    size_t position_capacity = 0;
    const size_t capacity    = static_cast<size_t>(next.n_blocks);
    if (!checked_product({ capacity, static_cast<size_t>(next.block_size), static_cast<size_t>(next.rank_k) },
                         k_u_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.block_size), static_cast<size_t>(next.rank_v) },
                         v_u_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.rank_k), static_cast<size_t>(next.group_size),
                           static_cast<size_t>(next.n_head_kv), static_cast<size_t>(next.head_dim_k) },
                         k_vh_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.rank_v), static_cast<size_t>(next.group_size),
                           static_cast<size_t>(next.n_head_kv), static_cast<size_t>(next.head_dim_v) },
                         v_vh_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.rank_k) }, k_scale_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.rank_v) }, v_scale_capacity) ||
        !checked_product({ capacity, static_cast<size_t>(next.block_size) }, position_capacity)) {
        return false;
    }
    if (k_u_q.size() > k_u_capacity || v_u_q.size() > v_u_capacity || k_vh_q.size() > k_vh_capacity ||
        v_vh_q.size() > v_vh_capacity || k_rank_scale.size() > k_scale_capacity ||
        v_rank_scale.size() > v_scale_capacity || next.block_positions.size() > position_capacity) {
        return fail("dispatch active factors exceed the requested block capacity");
    }

    k_u_q.resize(k_u_capacity, 0);
    v_u_q.resize(v_u_capacity, 0);
    k_vh_q.resize(k_vh_capacity, 0);
    v_vh_q.resize(v_vh_capacity, 0);
    k_rank_scale.resize(k_scale_capacity, 0.0f);
    v_rank_scale.resize(v_scale_capacity, 0.0f);
    next.block_positions.resize(position_capacity, -1);

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

bool llama_kv_blocksvd_pack_int8_reconstruct_dispatch(const llama_kv_blocksvd_context &             ctx,
                                                      int32_t                                       layer,
                                                      uint32_t                                      stream,
                                                      llama_seq_id                                  seq_id,
                                                      llama_kv_blocksvd_int8_reconstruct_dispatch & out,
                                                      std::string *                                 err) {
    return llama_kv_blocksvd_pack_int8_reconstruct_impl(ctx, layer, stream, seq_id, 0, out, err);
}

bool llama_kv_blocksvd_pack_int8_reconstruct_pool(const llama_kv_blocksvd_context &             ctx,
                                                  int32_t                                       layer,
                                                  uint32_t                                      stream,
                                                  llama_seq_id                                  seq_id,
                                                  int32_t                                       block_capacity,
                                                  llama_kv_blocksvd_int8_reconstruct_dispatch & out,
                                                  std::string *                                 err) {
    return llama_kv_blocksvd_pack_int8_reconstruct_impl(
        ctx, layer, stream, seq_id, block_capacity, out, err);
}

namespace {

struct blocksvd_execution_pool_key {
    int32_t      layer_start;
    uint32_t     stream;
    llama_seq_id seq_id;

    bool operator<(const blocksvd_execution_pool_key & other) const {
        if (layer_start != other.layer_start) {
            return layer_start < other.layer_start;
        }
        if (stream != other.stream) {
            return stream < other.stream;
        }
        return seq_id < other.seq_id;
    }
};

struct llama_kv_blocksvd_int8_host_pool {
    llama_kv_blocksvd_int8_reconstruct_dispatch dispatch;
    uint64_t                                     observed_generation = 0;
    uint64_t                                     epoch               = 1;
};

struct blocksvd_backend_pool_key {
    const llama_kv_blocksvd_int8_host_pool * host;
    ggml_backend_t                            backend;
    ggml_backend_buffer_type_t                buft;

    bool operator<(const blocksvd_backend_pool_key & other) const {
        const auto less_pointer = [](const void * lhs, const void * rhs) {
            return std::less<const void *>{}(lhs, rhs);
        };
        if (host != other.host) {
            return less_pointer(host, other.host);
        }
        if (backend != other.backend) {
            return less_pointer(backend, other.backend);
        }
        return less_pointer(buft, other.buft);
    }
};

static bool blocksvd_same_static_layout(
        const llama_kv_blocksvd_int8_reconstruct_dispatch & lhs,
        const llama_kv_blocksvd_int8_reconstruct_dispatch & rhs) {
    return lhs.n_blocks == rhs.n_blocks && lhs.block_size == rhs.block_size && lhs.rank_k == rhs.rank_k &&
           lhs.rank_v == rhs.rank_v && lhs.group_size == rhs.group_size && lhs.layer_start == rhs.layer_start &&
           lhs.layer_index == rhs.layer_index && lhs.n_head_kv == rhs.n_head_kv &&
           lhs.head_dim_k == rhs.head_dim_k && lhs.head_dim_v == rhs.head_dim_v && lhs.stream == rhs.stream &&
           lhs.seq_id == rhs.seq_id && lhs.v_u_offset_bytes == rhs.v_u_offset_bytes &&
           lhs.v_vh_offset_bytes == rhs.v_vh_offset_bytes &&
           lhs.v_rank_scale_offset_bytes == rhs.v_rank_scale_offset_bytes &&
           lhs.dense_v_offset_bytes == rhs.dense_v_offset_bytes &&
           lhs.dense_total_bytes == rhs.dense_total_bytes && lhs.u_q.size() == rhs.u_q.size() &&
           lhs.vh_q.size() == rhs.vh_q.size() && lhs.rank_scale.size() == rhs.rank_scale.size() &&
           lhs.block_positions.size() == rhs.block_positions.size();
}

static bool blocksvd_same_chunk_shape(
        const llama_kv_blocksvd_int8_reconstruct_dispatch & pool,
        const llama_kv_blocksvd_int8_reconstruct_dispatch & chunk) {
    return chunk.n_blocks == 1 && chunk.valid_blocks == 1 && pool.block_size == chunk.block_size &&
           pool.rank_k == chunk.rank_k && pool.rank_v == chunk.rank_v && pool.group_size == chunk.group_size &&
           pool.layer_start == chunk.layer_start && pool.layer_index == chunk.layer_index &&
           pool.n_head_kv == chunk.n_head_kv && pool.head_dim_k == chunk.head_dim_k &&
           pool.head_dim_v == chunk.head_dim_v && pool.stream == chunk.stream && pool.seq_id == chunk.seq_id;
}

static std::vector<const llama_kv_blocksvd_xkv_chunk *> blocksvd_matching_chunks(
        const llama_kv_blocksvd_context & ctx,
        int32_t                           layer,
        uint32_t                          stream,
        llama_seq_id                     seq_id) {
    std::vector<const llama_kv_blocksvd_xkv_chunk *> chunks;
    for (const auto & chunk : ctx.xkv_chunks) {
        const int64_t layer_end = static_cast<int64_t>(chunk.layer_start) + chunk.group_size;
        if (chunk.stream == stream && chunk.seq_id == seq_id && layer >= chunk.layer_start &&
            static_cast<int64_t>(layer) < layer_end) {
            chunks.push_back(&chunk);
        }
    }
    std::sort(chunks.begin(), chunks.end(), [](const auto * lhs, const auto * rhs) {
        return std::lexicographical_compare(lhs->pos.begin(), lhs->pos.end(), rhs->pos.begin(), rhs->pos.end());
    });
    return chunks;
}

template<typename T>
static void blocksvd_copy_component_block(
        std::vector<T> &       dst,
        size_t                 dst_base,
        const std::vector<T> & src,
        size_t                 src_base,
        size_t                 stride,
        size_t                 dst_block) {
    const size_t dst_offset = dst_base + dst_block * stride;
    GGML_ASSERT(dst_offset + stride <= dst.size());
    GGML_ASSERT(src_base + stride <= src.size());
    std::copy_n(src.begin() + src_base, stride, dst.begin() + dst_offset);
}

template<typename T>
static size_t blocksvd_upload_component_blocks(
        ggml_tensor *         tensor,
        const std::vector<T> & values,
        size_t                 base,
        size_t                 stride,
        size_t                 first_block,
        size_t                 block_count) {
    const size_t element_offset = base + first_block * stride;
    const size_t element_count  = block_count * stride;
    GGML_ASSERT(element_offset + element_count <= values.size());
    ggml_backend_tensor_set(
        tensor, values.data() + element_offset, element_offset * sizeof(T), element_count * sizeof(T));
    return element_count * sizeof(T);
}

static size_t blocksvd_dispatch_bytes(const llama_kv_blocksvd_int8_reconstruct_dispatch & dispatch) {
    return dispatch.u_q.size() * sizeof(int8_t) + dispatch.vh_q.size() * sizeof(int8_t) +
           dispatch.rank_scale.size() * sizeof(float) + dispatch.block_positions.size() * sizeof(int32_t);
}

static ggml_edgekv_reconstruct_params blocksvd_ggml_params(
        const llama_kv_blocksvd_int8_reconstruct_dispatch & dispatch,
        int32_t                                               layer_index) {
    return {
        dispatch.n_blocks,
        dispatch.block_size,
        dispatch.rank_k,
        dispatch.rank_v,
        dispatch.group_size,
        layer_index,
        dispatch.n_head_kv,
        dispatch.head_dim_k,
        dispatch.head_dim_v,
        dispatch.v_u_offset_bytes,
        dispatch.v_vh_offset_bytes,
        dispatch.v_rank_scale_offset_bytes,
        dispatch.dense_v_offset_bytes,
        dispatch.dense_total_bytes,
    };
}

static bool blocksvd_refresh_host_pool(
        const llama_kv_blocksvd_context &            ctx,
        llama_kv_blocksvd_int8_host_pool &           host,
        llama_kv_blocksvd_int8_pool_update &         update,
        std::string *                                err) {
    if (host.observed_generation == ctx.xkv_generation) {
        const size_t valid = static_cast<size_t>(host.dispatch.valid_blocks);
        update = { valid, valid, false };
        return true;
    }

    if (!llama_kv_blocksvd_refresh_int8_reconstruct_pool(
            ctx, host.dispatch.layer_start + host.dispatch.layer_index, host.dispatch.stream,
            host.dispatch.seq_id, host.dispatch, update, err)) {
        return false;
    }
    if (update.full_repack) {
        ++host.epoch;
    }
    host.observed_generation = ctx.xkv_generation;
    return true;
}

static bool blocksvd_find_pool_key(
        const llama_kv_blocksvd_context & ctx,
        int32_t                           layer,
        uint32_t                          stream,
        llama_seq_id                     seq_id,
        blocksvd_execution_pool_key &     key,
        std::string *                     err) {
    bool found = false;
    for (const auto & chunk : ctx.xkv_chunks) {
        const int64_t layer_end = static_cast<int64_t>(chunk.layer_start) + chunk.group_size;
        if (chunk.stream != stream || chunk.seq_id != seq_id || layer < chunk.layer_start ||
            static_cast<int64_t>(layer) >= layer_end) {
            continue;
        }
        const blocksvd_execution_pool_key candidate = { chunk.layer_start, stream, seq_id };
        if (found && candidate.layer_start != key.layer_start) {
            if (err) {
                *err = "overlapping xKV layer groups match one execution pool request";
            }
            return false;
        }
        key   = candidate;
        found = true;
    }
    if (!found && err) {
        *err = "no xKV chunks match layer, stream, and sequence";
    }
    return found;
}

static void blocksvd_set_backend_view(
        const std::shared_ptr<llama_kv_blocksvd_int8_backend_pool> & pool,
        llama_kv_blocksvd_int8_backend_view &                       view);

} // namespace

struct llama_kv_blocksvd_int8_backend_pool {
    std::shared_ptr<llama_kv_blocksvd_int8_host_pool> host;
    ggml_backend_t                                    backend = nullptr;
    ggml_backend_buffer_type_t                        buft    = nullptr;

    // buffer must be destroyed before tensor_ctx.
    ggml_context_ptr        tensor_ctx;
    ggml_backend_buffer_ptr buffer;

    ggml_tensor * u_q             = nullptr;
    ggml_tensor * vh_q            = nullptr;
    ggml_tensor * rank_scale      = nullptr;
    ggml_tensor * block_positions = nullptr;

    uint64_t host_epoch     = 0;
    int32_t  uploaded_valid = 0;
};

struct llama_kv_blocksvd_execution_pool_registry {
    std::map<blocksvd_execution_pool_key, std::shared_ptr<llama_kv_blocksvd_int8_host_pool>> hosts;
    std::map<blocksvd_backend_pool_key, std::shared_ptr<llama_kv_blocksvd_int8_backend_pool>> replicas;
};

namespace {

static void blocksvd_set_backend_view(
        const std::shared_ptr<llama_kv_blocksvd_int8_backend_pool> & pool,
        llama_kv_blocksvd_int8_backend_view &                       view) {
    view.owner           = pool;
    view.u_q             = pool->u_q;
    view.vh_q            = pool->vh_q;
    view.rank_scale      = pool->rank_scale;
    view.block_positions = pool->block_positions;
    view.storage         = &pool->host->dispatch;
    view.backend         = pool->backend;
}

static bool blocksvd_upload_full_backend_pool(llama_kv_blocksvd_int8_backend_pool & pool) {
    const auto & dispatch = pool.host->dispatch;
    if (ggml_nbytes(pool.u_q) != dispatch.u_q.size() * sizeof(int8_t) ||
        ggml_nbytes(pool.vh_q) != dispatch.vh_q.size() * sizeof(int8_t) ||
        ggml_nbytes(pool.rank_scale) != dispatch.rank_scale.size() * sizeof(float) ||
        ggml_nbytes(pool.block_positions) != dispatch.block_positions.size() * sizeof(int32_t)) {
        return false;
    }
    ggml_backend_tensor_set(pool.u_q, dispatch.u_q.data(), 0, ggml_nbytes(pool.u_q));
    ggml_backend_tensor_set(pool.vh_q, dispatch.vh_q.data(), 0, ggml_nbytes(pool.vh_q));
    ggml_backend_tensor_set(pool.rank_scale, dispatch.rank_scale.data(), 0, ggml_nbytes(pool.rank_scale));
    ggml_backend_tensor_set(
        pool.block_positions, dispatch.block_positions.data(), 0, ggml_nbytes(pool.block_positions));
    pool.host_epoch     = pool.host->epoch;
    pool.uploaded_valid = dispatch.valid_blocks;
    return true;
}

static std::shared_ptr<llama_kv_blocksvd_int8_backend_pool> blocksvd_create_backend_pool(
        const std::shared_ptr<llama_kv_blocksvd_int8_host_pool> & host,
        ggml_backend_t                                            backend,
        ggml_backend_buffer_type_t                                buft,
        std::string *                                             err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return std::shared_ptr<llama_kv_blocksvd_int8_backend_pool>{};
    };

    ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr tensor_ctx(ggml_init(params));
    if (!tensor_ctx) {
        return fail("could not allocate EdgeKV backend-pool tensor context");
    }

    const auto & dispatch = host->dispatch;
    auto pool             = std::make_shared<llama_kv_blocksvd_int8_backend_pool>();
    pool->host            = host;
    pool->backend         = backend;
    pool->buft            = buft;
    pool->tensor_ctx      = std::move(tensor_ctx);
    pool->u_q             = ggml_new_tensor_1d(pool->tensor_ctx.get(), GGML_TYPE_I8, dispatch.u_q.size());
    pool->vh_q            = ggml_new_tensor_1d(pool->tensor_ctx.get(), GGML_TYPE_I8, dispatch.vh_q.size());
    pool->rank_scale =
        ggml_new_tensor_1d(pool->tensor_ctx.get(), GGML_TYPE_F32, dispatch.rank_scale.size());
    pool->block_positions =
        ggml_new_tensor_1d(pool->tensor_ctx.get(), GGML_TYPE_I32, dispatch.block_positions.size());

    ggml_format_name(pool->u_q, "edgekv_u_q-%d", dispatch.layer_start);
    ggml_format_name(pool->vh_q, "edgekv_vh_q-%d", dispatch.layer_start);
    ggml_format_name(pool->rank_scale, "edgekv_scale-%d", dispatch.layer_start);
    ggml_format_name(pool->block_positions, "edgekv_pos-%d", dispatch.layer_start);

    pool->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(pool->tensor_ctx.get(), buft));
    if (!pool->buffer) {
        return fail(std::string("could not allocate EdgeKV backend pool from ") + ggml_backend_buft_name(buft));
    }
    ggml_backend_buffer_set_usage(pool->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(pool->buffer.get(), 0);
    if (!blocksvd_upload_full_backend_pool(*pool)) {
        return fail("EdgeKV backend-pool tensor sizes do not match host storage");
    }

    LLAMA_LOG_DEBUG(
        "%s: allocated pool=%p backend=%s buft=%s layer_group=[%d,%d) capacity=%d valid=%d bytes=%zu\n",
        __func__, static_cast<void *>(pool.get()), ggml_backend_name(backend), ggml_backend_buft_name(buft),
        dispatch.layer_start, dispatch.layer_start + dispatch.group_size, dispatch.n_blocks,
        dispatch.valid_blocks, blocksvd_dispatch_bytes(dispatch));
    return pool;
}

} // namespace

bool llama_kv_blocksvd_refresh_int8_reconstruct_pool(
        const llama_kv_blocksvd_context &             ctx,
        int32_t                                       layer,
        uint32_t                                      stream,
        llama_seq_id                                  seq_id,
        llama_kv_blocksvd_int8_reconstruct_dispatch & pool,
        llama_kv_blocksvd_int8_pool_update &          update,
        std::string *                                 err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return false;
    };

    const size_t previous_valid = pool.valid_blocks >= 0 ? static_cast<size_t>(pool.valid_blocks) : 0;
    update = { previous_valid, previous_valid, false };
    if (pool.n_blocks <= 0 || pool.valid_blocks < 0 || pool.valid_blocks > pool.n_blocks) {
        return fail("existing reconstruct pool has invalid capacity metadata");
    }
    const size_t position_capacity = static_cast<size_t>(pool.n_blocks) * pool.block_size;
    if (pool.block_positions.size() != position_capacity) {
        return fail("existing reconstruct pool position storage does not match its capacity");
    }
    if (layer < pool.layer_start || layer >= pool.layer_start + pool.group_size || pool.stream != stream ||
        pool.seq_id != seq_id) {
        return fail("refresh request does not match the existing reconstruct pool identity");
    }

    const auto chunks = blocksvd_matching_chunks(ctx, layer, stream, seq_id);
    if (chunks.empty()) {
        return fail("no xKV chunks match layer, stream, and sequence");
    }
    if (chunks.size() > static_cast<size_t>(pool.n_blocks)) {
        return fail("matching xKV chunks exceed the existing reconstruct pool capacity");
    }

    llama_pos previous_position      = 0;
    bool      have_previous_position = false;
    bool      prefix_matches         = chunks.size() >= previous_valid;
    for (size_t block = 0; block < chunks.size(); ++block) {
        const auto & chunk = *chunks[block];
        if (chunk.pos.size() != static_cast<size_t>(pool.block_size)) {
            return fail("matching xKV chunk position count changed reconstruct pool shape");
        }
        for (size_t token = 0; token < chunk.pos.size(); ++token) {
            const llama_pos position = chunk.pos[token];
            if (position < 0 || position > std::numeric_limits<int32_t>::max()) {
                return fail("reconstruct pool position is outside non-negative int32_t range");
            }
            if (have_previous_position && position <= previous_position) {
                return fail("reconstruct pool positions must be globally strictly increasing");
            }
            if (block < previous_valid &&
                pool.block_positions[block * static_cast<size_t>(pool.block_size) + token] != position) {
                prefix_matches = false;
            }
            previous_position      = position;
            have_previous_position = true;
        }
    }

    if (chunks.size() == previous_valid && prefix_matches) {
        return true;
    }

    if (!prefix_matches) {
        llama_kv_blocksvd_int8_reconstruct_dispatch next;
        std::string                                    pack_error;
        if (!llama_kv_blocksvd_pack_int8_reconstruct_pool(
                ctx, layer, stream, seq_id, pool.n_blocks, next, &pack_error)) {
            return fail("could not repack reconstruct pool: " + pack_error);
        }
        if (!blocksvd_same_static_layout(pool, next)) {
            return fail("matching xKV chunks changed reconstruct pool static layout");
        }
        pool   = std::move(next);
        update = { 0, static_cast<size_t>(pool.n_blocks), true };
        return true;
    }

    std::vector<llama_kv_blocksvd_int8_reconstruct_dispatch> additions;
    additions.reserve(chunks.size() - previous_valid);
    for (size_t block = previous_valid; block < chunks.size(); ++block) {
        llama_kv_blocksvd_context single;
        single.params = ctx.params;
        single.xkv_chunks.push_back(*chunks[block]);

        llama_kv_blocksvd_int8_reconstruct_dispatch addition;
        std::string                                    pack_error;
        if (!llama_kv_blocksvd_pack_int8_reconstruct_pool(
                single, layer, stream, seq_id, 1, addition, &pack_error)) {
            return fail("could not pack appended xKV chunk: " + pack_error);
        }
        if (!blocksvd_same_chunk_shape(pool, addition)) {
            return fail("appended xKV chunk changed reconstruct pool static shape");
        }
        additions.push_back(std::move(addition));
    }

    const size_t k_u_stride = static_cast<size_t>(pool.block_size) * pool.rank_k;
    const size_t v_u_stride = static_cast<size_t>(pool.block_size) * pool.rank_v;
    const size_t k_vh_stride = static_cast<size_t>(pool.rank_k) * pool.group_size * pool.n_head_kv * pool.head_dim_k;
    const size_t v_vh_stride = static_cast<size_t>(pool.rank_v) * pool.group_size * pool.n_head_kv * pool.head_dim_v;
    const size_t k_scale_stride = static_cast<size_t>(pool.rank_k);
    const size_t v_scale_stride = static_cast<size_t>(pool.rank_v);
    const size_t position_stride = static_cast<size_t>(pool.block_size);

    for (size_t index = 0; index < additions.size(); ++index) {
        const size_t block = previous_valid + index;
        const auto & addition = additions[index];
        blocksvd_copy_component_block(pool.u_q, 0, addition.u_q, 0, k_u_stride, block);
        blocksvd_copy_component_block(
            pool.u_q, static_cast<size_t>(pool.v_u_offset_bytes), addition.u_q,
            static_cast<size_t>(addition.v_u_offset_bytes), v_u_stride, block);
        blocksvd_copy_component_block(pool.vh_q, 0, addition.vh_q, 0, k_vh_stride, block);
        blocksvd_copy_component_block(
            pool.vh_q, static_cast<size_t>(pool.v_vh_offset_bytes), addition.vh_q,
            static_cast<size_t>(addition.v_vh_offset_bytes), v_vh_stride, block);
        blocksvd_copy_component_block(pool.rank_scale, 0, addition.rank_scale, 0, k_scale_stride, block);
        blocksvd_copy_component_block(
            pool.rank_scale, static_cast<size_t>(pool.v_rank_scale_offset_bytes) / sizeof(float),
            addition.rank_scale, static_cast<size_t>(addition.v_rank_scale_offset_bytes) / sizeof(float),
            v_scale_stride, block);
        blocksvd_copy_component_block(
            pool.block_positions, 0, addition.block_positions, 0, position_stride, block);
    }

    pool.valid_blocks = static_cast<int32_t>(chunks.size());
    update = { previous_valid, chunks.size(), false };
    return true;
}

bool llama_kv_blocksvd_acquire_int8_backend_pool(
        const llama_kv_blocksvd_context &     ctx,
        int32_t                               layer,
        uint32_t                              stream,
        llama_seq_id                          seq_id,
        int32_t                               block_capacity,
        ggml_backend_sched_t                  sched,
        ggml_backend_t                        backend_cpu,
        llama_kv_blocksvd_int8_backend_view & out,
        std::string *                         err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return false;
    };

    out = {};
    if (!sched || !backend_cpu) {
        return fail("EdgeKV backend pool requires a scheduler and CPU fallback backend");
    }
    if (block_capacity <= 0) {
        return fail("EdgeKV backend pool capacity must be positive");
    }

    blocksvd_execution_pool_key key{};
    if (!blocksvd_find_pool_key(ctx, layer, stream, seq_id, key, err)) {
        return false;
    }
    if (!ctx.execution_pools) {
        ctx.execution_pools = std::make_shared<llama_kv_blocksvd_execution_pool_registry>();
    }
    auto & registry = *ctx.execution_pools;

    std::shared_ptr<llama_kv_blocksvd_int8_host_pool> host;
    auto                                               host_it = registry.hosts.find(key);
    if (host_it != registry.hosts.end()) {
        host = host_it->second;
    }

    bool replace_host = !host || host->dispatch.n_blocks < block_capacity;
    if (host && !replace_host) {
        llama_kv_blocksvd_int8_pool_update update;
        std::string                         refresh_error;
        if (!blocksvd_refresh_host_pool(ctx, *host, update, &refresh_error)) {
            replace_host = true;
        }
    }

    if (replace_host) {
        llama_kv_blocksvd_int8_reconstruct_dispatch dispatch;
        std::string                                    pack_error;
        if (!llama_kv_blocksvd_pack_int8_reconstruct_pool(
                ctx, layer, stream, seq_id, block_capacity, dispatch, &pack_error)) {
            return fail("could not create EdgeKV host pool: " + pack_error);
        }

        const auto * replaced_host = host.get();
        if (replaced_host) {
            for (auto it = registry.replicas.begin(); it != registry.replicas.end();) {
                if (it->first.host == replaced_host) {
                    it = registry.replicas.erase(it);
                } else {
                    ++it;
                }
            }
        }
        host                       = std::make_shared<llama_kv_blocksvd_int8_host_pool>();
        host->dispatch             = std::move(dispatch);
        host->observed_generation  = ctx.xkv_generation;
        registry.hosts[key]        = host;

        LLAMA_LOG_DEBUG(
            "%s: created host pool=%p generation=%llu layer_group=[%d,%d) capacity=%d valid=%d\n",
            __func__, static_cast<void *>(host.get()), (unsigned long long) host->observed_generation,
            host->dispatch.layer_start, host->dispatch.layer_start + host->dispatch.group_size,
            host->dispatch.n_blocks, host->dispatch.valid_blocks);
    }

    ggml_init_params prototype_params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr prototype_ctx(ggml_init(prototype_params));
    if (!prototype_ctx) {
        return fail("could not allocate EdgeKV backend-probe context");
    }
    const auto & dispatch = host->dispatch;
    ggml_tensor * prototype_u =
        ggml_new_tensor_1d(prototype_ctx.get(), GGML_TYPE_I8, static_cast<int64_t>(dispatch.u_q.size()));
    ggml_tensor * prototype_vh =
        ggml_new_tensor_1d(prototype_ctx.get(), GGML_TYPE_I8, static_cast<int64_t>(dispatch.vh_q.size()));
    ggml_tensor * prototype_scale =
        ggml_new_tensor_1d(prototype_ctx.get(), GGML_TYPE_F32, static_cast<int64_t>(dispatch.rank_scale.size()));
    ggml_tensor * prototype_positions = ggml_new_tensor_1d(
        prototype_ctx.get(), GGML_TYPE_I32, static_cast<int64_t>(dispatch.block_positions.size()));
    const auto prototype_op_params = blocksvd_ggml_params(dispatch, layer - dispatch.layer_start);
    ggml_tensor * prototype_op = ggml_edgekv_reconstruct(
        prototype_ctx.get(), prototype_u, prototype_vh, prototype_scale, prototype_positions,
        &prototype_op_params);

    std::vector<ggml_backend_t> candidates;
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int index = 0; index < n_backends; ++index) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, index);
        if (backend != backend_cpu && ggml_backend_supports_op(backend, prototype_op)) {
            candidates.push_back(backend);
        }
    }
    if (ggml_backend_supports_op(backend_cpu, prototype_op)) {
        candidates.push_back(backend_cpu);
    }
    if (candidates.empty()) {
        return fail("no scheduler backend supports the EdgeKV reconstruct shape");
    }

    std::string allocation_errors;
    for (ggml_backend_t backend : candidates) {
        ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, backend);
        const blocksvd_backend_pool_key replica_key = { host.get(), backend, buft };
        auto replica_it = registry.replicas.find(replica_key);
        if (replica_it != registry.replicas.end()) {
            blocksvd_set_backend_view(replica_it->second, out);
            return true;
        }

        std::string create_error;
        auto replica = blocksvd_create_backend_pool(host, backend, buft, &create_error);
        if (replica) {
            registry.replicas.emplace(replica_key, replica);
            blocksvd_set_backend_view(replica, out);
            return true;
        }
        if (!allocation_errors.empty()) {
            allocation_errors += "; ";
        }
        allocation_errors += std::string(ggml_backend_name(backend)) + ": " + create_error;
    }

    return fail("could not allocate an EdgeKV backend replica: " + allocation_errors);
}

bool llama_kv_blocksvd_sync_int8_backend_pool(
        const llama_kv_blocksvd_context &     ctx,
        llama_kv_blocksvd_int8_backend_view & view,
        std::string *                         err) {
    auto fail = [err](std::string message) {
        if (err) {
            *err = std::move(message);
        }
        return false;
    };

    if (!view.owner || !view.owner->host) {
        return fail("EdgeKV backend view has no owner");
    }
    auto & pool = *view.owner;
    auto & host = *pool.host;

    llama_kv_blocksvd_int8_pool_update update;
    if (!blocksvd_refresh_host_pool(ctx, host, update, err)) {
        return false;
    }

    const auto & dispatch = host.dispatch;
    const int32_t previous_valid = pool.uploaded_valid;
    size_t        first_dirty    = static_cast<size_t>(dispatch.valid_blocks);
    size_t        last_dirty     = first_dirty;
    size_t        dirty_bytes    = 0;
    bool          full_upload    = pool.host_epoch != host.epoch || pool.uploaded_valid > dispatch.valid_blocks;
    if (full_upload) {
        if (!blocksvd_upload_full_backend_pool(pool)) {
            return fail("EdgeKV backend-pool tensor sizes changed during full synchronization");
        }
        first_dirty = 0;
        last_dirty  = static_cast<size_t>(dispatch.n_blocks);
        dirty_bytes = blocksvd_dispatch_bytes(dispatch);
    } else if (pool.uploaded_valid < dispatch.valid_blocks) {
        first_dirty = static_cast<size_t>(pool.uploaded_valid);
        last_dirty  = static_cast<size_t>(dispatch.valid_blocks);
        const size_t block_count = last_dirty - first_dirty;
        const size_t k_u_stride = static_cast<size_t>(dispatch.block_size) * dispatch.rank_k;
        const size_t v_u_stride = static_cast<size_t>(dispatch.block_size) * dispatch.rank_v;
        const size_t k_vh_stride = static_cast<size_t>(dispatch.rank_k) * dispatch.group_size *
                                   dispatch.n_head_kv * dispatch.head_dim_k;
        const size_t v_vh_stride = static_cast<size_t>(dispatch.rank_v) * dispatch.group_size *
                                   dispatch.n_head_kv * dispatch.head_dim_v;
        const size_t k_scale_stride = static_cast<size_t>(dispatch.rank_k);
        const size_t v_scale_stride = static_cast<size_t>(dispatch.rank_v);
        const size_t position_stride = static_cast<size_t>(dispatch.block_size);

        dirty_bytes += blocksvd_upload_component_blocks(
            pool.u_q, dispatch.u_q, 0, k_u_stride, first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.u_q, dispatch.u_q, static_cast<size_t>(dispatch.v_u_offset_bytes), v_u_stride,
            first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.vh_q, dispatch.vh_q, 0, k_vh_stride, first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.vh_q, dispatch.vh_q, static_cast<size_t>(dispatch.v_vh_offset_bytes), v_vh_stride,
            first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.rank_scale, dispatch.rank_scale, 0, k_scale_stride, first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.rank_scale, dispatch.rank_scale,
            static_cast<size_t>(dispatch.v_rank_scale_offset_bytes) / sizeof(float), v_scale_stride,
            first_dirty, block_count);
        dirty_bytes += blocksvd_upload_component_blocks(
            pool.block_positions, dispatch.block_positions, 0, position_stride, first_dirty, block_count);
        pool.uploaded_valid = dispatch.valid_blocks;
    }
    pool.host_epoch = host.epoch;
    blocksvd_set_backend_view(view.owner, view);

    if (dirty_bytes > 0) {
        LLAMA_LOG_DEBUG(
            "%s: synchronized pool=%p generation=%llu backend=%s layer_group=[%d,%d) capacity=%d "
            "valid=%d->%d dirty=[%zu,%zu) bytes=%zu full=%d\n",
            __func__, static_cast<void *>(view.owner.get()), (unsigned long long) host.observed_generation,
            ggml_backend_name(pool.backend), dispatch.layer_start, dispatch.layer_start + dispatch.group_size,
            dispatch.n_blocks, previous_valid, dispatch.valid_blocks, first_dirty, last_dirty, dirty_bytes,
            full_upload ? 1 : 0);
    }
    return true;
}
