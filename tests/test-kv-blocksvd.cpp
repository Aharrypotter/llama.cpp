#include "llama-kv-blocksvd.h"
#include "llama-kv-blocksvd-execution.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

template<typename T>
static bool write_raw(const std::filesystem::path & path, const std::vector<T> & values) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
    return output.good();
}

static llama_kv_blocksvd_context * make_int8_dispatch_context(std::string & err) {
    llama_kv_blocksvd_params params{};
    params.block_size       = 16;
    params.rank             = 8;
    params.rank_v           = 4;
    params.quant_bits       = 8;
    params.cross_layer      = true;
    params.layer_group_size = 2;

    auto * ctx = llama_kv_blocksvd_init(params);
    if (!ctx) {
        err = "could not initialize dispatch context";
        return nullptr;
    }

    constexpr int32_t group_size = 2;
    constexpr int32_t n_head_kv  = 2;
    constexpr int32_t head_dim_k = 32;
    constexpr int32_t head_dim_v = 16;
    constexpr int32_t n_tokens   = 16;
    constexpr int32_t d_k        = n_head_kv * head_dim_k;
    constexpr int32_t d_v        = n_head_kv * head_dim_v;

    std::mt19937                          generator(20260716);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    // Insert the later block first so the dispatch packer must establish the
    // position-sorted consumer order rather than inheriting storage order.
    for (int32_t first_position : { 16, 0 }) {
        std::vector<std::vector<float>> k(group_size), v(group_size);
        std::vector<const float *>      kp(group_size), vp(group_size);
        for (int32_t layer = 0; layer < group_size; ++layer) {
            k[layer].resize(static_cast<size_t>(n_tokens) * d_k);
            v[layer].resize(static_cast<size_t>(n_tokens) * d_v);
            for (float & value : k[layer]) {
                value = distribution(generator);
            }
            for (float & value : v[layer]) {
                value = distribution(generator);
            }
            kp[layer] = k[layer].data();
            vp[layer] = v[layer].data();
        }

        std::vector<uint32_t>  slots(n_tokens);
        std::vector<llama_pos> positions(n_tokens);
        for (int32_t token = 0; token < n_tokens; ++token) {
            slots[token]     = static_cast<uint32_t>(first_position + token);
            positions[token] = first_position + token;
        }

        std::vector<uint32_t>  compressed_slots;
        std::vector<llama_pos> compressed_positions;
        if (!llama_kv_blocksvd_append_and_compress_xkv_group_store(ctx, 0, kp, vp, slots.data(), positions.data(),
                                                                   n_tokens, 0, 0, n_head_kv, head_dim_k, head_dim_v,
                                                                   compressed_slots, compressed_positions, &err)) {
            llama_kv_blocksvd_free(ctx);
            return nullptr;
        }
    }

    if (ctx->xkv_chunks.size() != 2) {
        err = "dispatch fixture did not produce two xKV chunks";
        llama_kv_blocksvd_free(ctx);
        return nullptr;
    }
    if (ctx->xkv_generation != 2) {
        err = "dispatch fixture generation did not track both xKV chunks";
        llama_kv_blocksvd_free(ctx);
        return nullptr;
    }
    return ctx;
}

static size_t align_dispatch_bytes(size_t bytes) {
    constexpr size_t alignment = llama_kv_blocksvd_int8_reconstruct_dispatch::buffer_alignment;
    return ((bytes + alignment - 1) / alignment) * alignment;
}

static void test_int8_reconstruct_dispatch() {
    std::string err;
    auto *      ctx = make_int8_dispatch_context(err);
    assert(ctx != nullptr);

    llama_kv_blocksvd_int8_reconstruct_dispatch dispatch;
    bool ok = llama_kv_blocksvd_pack_int8_reconstruct_dispatch(*ctx, 1, 0, 0, dispatch, &err);
    assert(ok);
    assert(dispatch.n_blocks == 2);
    assert(dispatch.valid_blocks == dispatch.n_blocks);
    assert(dispatch.block_size == 16);
    assert(dispatch.rank_k == 8);
    assert(dispatch.rank_v == 4);
    assert(dispatch.group_size == 2);
    assert(dispatch.layer_start == 0);
    assert(dispatch.layer_index == 1);
    assert(dispatch.n_head_kv == 2);
    assert(dispatch.head_dim_k == 32);
    assert(dispatch.head_dim_v == 16);
    assert(dispatch.block_positions.size() == 32);
    for (int32_t position = 0; position < 32; ++position) {
        assert(dispatch.block_positions[position] == position);
    }

    llama_kv_blocksvd_int8_reconstruct_dispatch layer_zero_dispatch;
    ok = llama_kv_blocksvd_pack_int8_reconstruct_dispatch(*ctx, 0, 0, 0, layer_zero_dispatch, &err);
    assert(ok);
    assert(layer_zero_dispatch.layer_index == 0);
    assert(layer_zero_dispatch.u_q == dispatch.u_q);
    assert(layer_zero_dispatch.vh_q == dispatch.vh_q);
    assert(layer_zero_dispatch.rank_scale == dispatch.rank_scale);
    assert(layer_zero_dispatch.block_positions == dispatch.block_positions);

    const size_t k_u_count  = static_cast<size_t>(dispatch.n_blocks) * dispatch.block_size * dispatch.rank_k;
    const size_t v_u_count  = static_cast<size_t>(dispatch.n_blocks) * dispatch.block_size * dispatch.rank_v;
    const size_t k_vh_count = static_cast<size_t>(dispatch.n_blocks) * dispatch.rank_k * dispatch.group_size *
                              dispatch.n_head_kv * dispatch.head_dim_k;
    const size_t v_vh_count = static_cast<size_t>(dispatch.n_blocks) * dispatch.rank_v * dispatch.group_size *
                              dispatch.n_head_kv * dispatch.head_dim_v;
    const size_t k_scale_count = static_cast<size_t>(dispatch.n_blocks) * dispatch.rank_k;
    const size_t v_scale_count = static_cast<size_t>(dispatch.n_blocks) * dispatch.rank_v;

    assert(dispatch.v_u_offset_bytes == static_cast<int32_t>(align_dispatch_bytes(k_u_count)));
    assert(dispatch.v_vh_offset_bytes == static_cast<int32_t>(align_dispatch_bytes(k_vh_count)));
    assert(dispatch.v_rank_scale_offset_bytes ==
           static_cast<int32_t>(align_dispatch_bytes(k_scale_count * sizeof(float))));
    assert(dispatch.u_q.size() * sizeof(int8_t) == align_dispatch_bytes(dispatch.v_u_offset_bytes + v_u_count));
    assert(dispatch.vh_q.size() * sizeof(int8_t) == align_dispatch_bytes(dispatch.v_vh_offset_bytes + v_vh_count));
    assert(dispatch.rank_scale.size() * sizeof(float) ==
           align_dispatch_bytes(dispatch.v_rank_scale_offset_bytes + v_scale_count * sizeof(float)));

    for (size_t i = k_u_count; i < static_cast<size_t>(dispatch.v_u_offset_bytes); ++i) {
        assert(dispatch.u_q[i] == 0);
    }
    for (size_t i = k_vh_count; i < static_cast<size_t>(dispatch.v_vh_offset_bytes); ++i) {
        assert(dispatch.vh_q[i] == 0);
    }
    for (size_t i = k_scale_count; i < static_cast<size_t>(dispatch.v_rank_scale_offset_bytes) / sizeof(float); ++i) {
        assert(dispatch.rank_scale[i] == 0.0f);
    }

    const size_t dense_k_bytes =
        static_cast<size_t>(dispatch.n_head_kv) * dispatch.n_blocks * dispatch.block_size * dispatch.head_dim_k * 2;
    const size_t dense_v_bytes =
        static_cast<size_t>(dispatch.n_head_kv) * dispatch.n_blocks * dispatch.block_size * dispatch.head_dim_v * 2;
    assert(dispatch.dense_v_offset_bytes == static_cast<int32_t>(align_dispatch_bytes(dense_k_bytes)));
    assert(dispatch.dense_total_bytes ==
           static_cast<int32_t>(align_dispatch_bytes(dispatch.dense_v_offset_bytes + dense_v_bytes)));
    const auto htp_op_params = dispatch.htp_op_params();
    static_assert(llama_kv_blocksvd_int8_reconstruct_dispatch::htp_op_param_count <= 16);
    assert(htp_op_params[0] == dispatch.n_blocks);
    assert(htp_op_params[5] == dispatch.layer_index);
    assert(htp_op_params[9] == dispatch.v_u_offset_bytes);
    assert(htp_op_params[13] == dispatch.dense_total_bytes);

    const auto * k_u_q   = dispatch.u_q.data();
    const auto * v_u_q   = dispatch.u_q.data() + dispatch.v_u_offset_bytes;
    const auto * k_vh_q  = dispatch.vh_q.data();
    const auto * v_vh_q  = dispatch.vh_q.data() + dispatch.v_vh_offset_bytes;
    const auto * k_scale = dispatch.rank_scale.data();
    const auto * v_scale = dispatch.rank_scale.data() + dispatch.v_rank_scale_offset_bytes / sizeof(float);

    ggml_init_params ggml_params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    assert(ggml_ctx != nullptr);

    ggml_tensor * t_u = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, dispatch.u_q.size());
    ggml_tensor * t_vh = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, dispatch.vh_q.size());
    ggml_tensor * t_scale = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, dispatch.rank_scale.size());
    ggml_tensor * t_positions = ggml_new_tensor_1d(
        ggml_ctx, GGML_TYPE_I32, dispatch.block_positions.size());
    std::memcpy(t_u->data, dispatch.u_q.data(), ggml_nbytes(t_u));
    std::memcpy(t_vh->data, dispatch.vh_q.data(), ggml_nbytes(t_vh));
    std::memcpy(t_scale->data, dispatch.rank_scale.data(), ggml_nbytes(t_scale));
    std::memcpy(t_positions->data, dispatch.block_positions.data(), ggml_nbytes(t_positions));

    const ggml_edgekv_reconstruct_params reconstruct_params = {
        dispatch.n_blocks,
        dispatch.block_size,
        dispatch.rank_k,
        dispatch.rank_v,
        dispatch.group_size,
        dispatch.layer_index,
        dispatch.n_head_kv,
        dispatch.head_dim_k,
        dispatch.head_dim_v,
        dispatch.v_u_offset_bytes,
        dispatch.v_vh_offset_bytes,
        dispatch.v_rank_scale_offset_bytes,
        dispatch.dense_v_offset_bytes,
        dispatch.dense_total_bytes,
    };
    ggml_tensor * t_dense = ggml_edgekv_reconstruct(
        ggml_ctx, t_u, t_vh, t_scale, t_positions, &reconstruct_params);
    ggml_cgraph * graph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(graph, t_dense);
    ggml_cplan plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work_buffer(plan.work_size);
    plan.work_data = work_buffer.empty() ? nullptr : work_buffer.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto * dense_k_fp16 = (const ggml_fp16_t *) t_dense->data;
    const auto * dense_v_fp16 = (const ggml_fp16_t *) (
        (const char *) t_dense->data + dispatch.dense_v_offset_bytes);

    std::vector<const llama_kv_blocksvd_xkv_chunk *> sorted_chunks;
    for (const auto & chunk : ctx->xkv_chunks) {
        sorted_chunks.push_back(&chunk);
    }
    std::sort(sorted_chunks.begin(), sorted_chunks.end(),
              [](const auto * lhs, const auto * rhs) { return lhs->pos.front() < rhs->pos.front(); });

    for (int32_t block = 0; block < dispatch.n_blocks; ++block) {
        std::vector<std::vector<float>> dense_k;
        std::vector<std::vector<float>> dense_v;
        ok = llama_kv_blocksvd_decompress_xkv_chunk(*sorted_chunks[block], dense_k, dense_v, &err);
        assert(ok);
        for (int32_t token = 0; token < dispatch.block_size; ++token) {
            for (int32_t head = 0; head < dispatch.n_head_kv; ++head) {
                for (int32_t d = 0; d < dispatch.head_dim_k; ++d) {
                    float actual = 0.0f;
                    for (int32_t rank = 0; rank < dispatch.rank_k; ++rank) {
                        const size_t u_index =
                            (static_cast<size_t>(block) * dispatch.block_size + token) * dispatch.rank_k + rank;
                        const size_t vh_index =
                            ((((static_cast<size_t>(block) * dispatch.rank_k + rank) * dispatch.group_size +
                               dispatch.layer_index) *
                                  dispatch.n_head_kv +
                              head) *
                             dispatch.head_dim_k) +
                            d;
                        actual += static_cast<float>(k_u_q[u_index]) *
                                  k_scale[static_cast<size_t>(block) * dispatch.rank_k + rank] *
                                  static_cast<float>(k_vh_q[vh_index]);
                    }
                    const size_t reference_index =
                        (static_cast<size_t>(token) * dispatch.n_head_kv + head) * dispatch.head_dim_k + d;
                    assert(std::fabs(actual - dense_k[dispatch.layer_index][reference_index]) < 1e-5f);
                    const size_t output_index =
                        (((static_cast<size_t>(head) * dispatch.n_blocks + block) * dispatch.block_size + token) *
                         dispatch.head_dim_k) + d;
                    assert(ggml_fp16_to_fp32(dense_k_fp16[output_index]) == ggml_fp16_to_fp32(ggml_fp32_to_fp16(actual)));
                }
                for (int32_t d = 0; d < dispatch.head_dim_v; ++d) {
                    float actual = 0.0f;
                    for (int32_t rank = 0; rank < dispatch.rank_v; ++rank) {
                        const size_t u_index =
                            (static_cast<size_t>(block) * dispatch.block_size + token) * dispatch.rank_v + rank;
                        const size_t vh_index =
                            ((((static_cast<size_t>(block) * dispatch.rank_v + rank) * dispatch.group_size +
                               dispatch.layer_index) *
                                  dispatch.n_head_kv +
                              head) *
                             dispatch.head_dim_v) +
                            d;
                        actual += static_cast<float>(v_u_q[u_index]) *
                                  v_scale[static_cast<size_t>(block) * dispatch.rank_v + rank] *
                                  static_cast<float>(v_vh_q[vh_index]);
                    }
                    const size_t reference_index =
                        (static_cast<size_t>(token) * dispatch.n_head_kv + head) * dispatch.head_dim_v + d;
                    assert(std::fabs(actual - dense_v[dispatch.layer_index][reference_index]) < 1e-5f);
                    const size_t output_index =
                        (((static_cast<size_t>(head) * dispatch.n_blocks + block) * dispatch.block_size + token) *
                         dispatch.head_dim_v) + d;
                    assert(ggml_fp16_to_fp32(dense_v_fp16[output_index]) == ggml_fp16_to_fp32(ggml_fp32_to_fp16(actual)));
                }
            }
        }
    }

    {
        constexpr int32_t n_head_q = 4;
        ggml_tensor * q = ggml_new_tensor_3d(
            ggml_ctx, GGML_TYPE_F32, dispatch.head_dim_k, n_head_q, 1);
        ggml_tensor * active_k =
            ggml_new_tensor_4d(ggml_ctx, GGML_TYPE_F16, dispatch.head_dim_k, dispatch.n_head_kv, 1, 1);
        ggml_tensor * active_v =
            ggml_new_tensor_4d(ggml_ctx, GGML_TYPE_F16, dispatch.head_dim_v, dispatch.n_head_kv, 1, 1);
        ggml_tensor * archive_k = ggml_view_4d(
            ggml_ctx, t_dense, dispatch.head_dim_k, dispatch.block_size,
            dispatch.n_blocks, dispatch.n_head_kv,
            (size_t) dispatch.head_dim_k*sizeof(ggml_fp16_t),
            (size_t) dispatch.head_dim_k*dispatch.block_size*sizeof(ggml_fp16_t),
            (size_t) dispatch.head_dim_k*dispatch.block_size*dispatch.n_blocks*sizeof(ggml_fp16_t), 0);
        ggml_tensor * archive_v = ggml_view_4d(
            ggml_ctx, t_dense, dispatch.head_dim_v, dispatch.block_size,
            dispatch.n_blocks, dispatch.n_head_kv,
            (size_t) dispatch.head_dim_v*sizeof(ggml_fp16_t),
            (size_t) dispatch.head_dim_v*dispatch.block_size*sizeof(ggml_fp16_t),
            (size_t) dispatch.head_dim_v*dispatch.block_size*dispatch.n_blocks*sizeof(ggml_fp16_t),
            dispatch.dense_v_offset_bytes);

        std::mt19937 consumer_generator(20260717);
        std::uniform_real_distribution<float> consumer_distribution(-0.25f, 0.25f);
        for (int64_t i = 0; i < ggml_nelements(q); ++i) {
            ((float *) q->data)[i] = consumer_distribution(consumer_generator);
        }
        for (int64_t i = 0; i < ggml_nelements(active_k); ++i) {
            ((ggml_fp16_t *) active_k->data)[i] = ggml_fp32_to_fp16(consumer_distribution(consumer_generator));
        }
        for (int64_t i = 0; i < ggml_nelements(active_v); ++i) {
            ((ggml_fp16_t *) active_v->data)[i] = ggml_fp32_to_fp16(consumer_distribution(consumer_generator));
        }

        llama_kv_blocksvd_staging_t staging;
        staging.capacity = 1;
        staging.cell_to_slot = { std::vector<int32_t>(41, -1) };
        staging.slot_to_cell = { std::vector<int32_t>{ 40 } };
        staging.cell_to_slot[0][40] = 0;

        const auto configure_consumer = [&](llama_chunked_attn_params & p) {
            p.bctx         = ctx;
            p.staging      = &staging;
            p.il           = 1;
            p.n_kv         = 41;
            p.n_head_kv    = dispatch.n_head_kv;
            p.n_head_q     = n_head_q;
            p.head_dim_k   = dispatch.head_dim_k;
            p.head_dim_v   = dispatch.head_dim_v;
            p.scale        = 1.0f/std::sqrt((float) dispatch.head_dim_k);
            p.n_stream     = 1;
            p.cache_size   = 1;
            p.q_pos        = { 40 };
            p.slot_pos     = { 40 };
        };

        llama_chunked_attn_params baseline_params;
        configure_consumer(baseline_params);
        llama_chunked_attn_build_refs(&baseline_params);

        // Reproduce the original lifetime hazard directly: growing xkv_chunks
        // after the hoist must not invalidate the per-forward position snapshot.
        const size_t original_chunk_count    = ctx->xkv_chunks.size();
        const size_t original_chunk_capacity = ctx->xkv_chunks.capacity();
        do {
            ctx->xkv_chunks.emplace_back();
        } while (ctx->xkv_chunks.capacity() == original_chunk_capacity);
        ggml_tensor * baseline_out = ggml_new_tensor_2d(
            ggml_ctx, GGML_TYPE_F32, (int64_t) dispatch.head_dim_v*n_head_q, 1);
        baseline_out->src[0] = q;
        baseline_out->src[1] = active_k;
        baseline_out->src[2] = active_v;
        llama_chunked_attn_compute(baseline_out, 0, 1, &baseline_params);
        ctx->xkv_chunks.resize(original_chunk_count);

        llama_chunked_attn_params edgekv_params;
        configure_consumer(edgekv_params);
        edgekv_params.edgekv_reconstructed = true;
        edgekv_params.edgekv_n_blocks       = dispatch.n_blocks;
        edgekv_params.edgekv_block_size     = dispatch.block_size;
        edgekv_params.edgekv_positions.assign(
            dispatch.block_positions.begin(), dispatch.block_positions.end());
        llama_chunked_attn_build_refs(&edgekv_params);
        ggml_tensor * edgekv_out = ggml_new_tensor_2d(
            ggml_ctx, GGML_TYPE_F32, (int64_t) dispatch.head_dim_v*n_head_q, 1);
        edgekv_out->src[0] = q;
        edgekv_out->src[1] = active_k;
        edgekv_out->src[2] = active_v;
        edgekv_out->src[3] = archive_k;
        edgekv_out->src[4] = archive_v;
        llama_chunked_attn_compute(edgekv_out, 0, 1, &edgekv_params);

        for (int64_t i = 0; i < ggml_nelements(edgekv_out); ++i) {
            const float baseline = ((const float *) baseline_out->data)[i];
            const float edgekv   = ((const float *) edgekv_out->data)[i];
            assert(std::isfinite(baseline));
            assert(std::isfinite(edgekv));
            assert(std::fabs(baseline - edgekv) < 2e-4f);
        }

        llama_chunked_attn_params direct_params;
        configure_consumer(direct_params);
        direct_params.use_lowrank_direct = true;
        llama_chunked_attn_build_refs(&direct_params);
        assert(direct_params.chunks_rank.size() == original_chunk_count);

        const size_t direct_chunk_capacity = ctx->xkv_chunks.capacity();
        do {
            ctx->xkv_chunks.emplace_back();
        } while (ctx->xkv_chunks.capacity() == direct_chunk_capacity);

        ggml_tensor * direct_serial_out =
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, (int64_t) dispatch.head_dim_v * n_head_q, 1);
        direct_serial_out->src[0] = q;
        direct_serial_out->src[1] = active_k;
        direct_serial_out->src[2] = active_v;
        llama_kv_lowrank_direct_attn_compute(direct_serial_out, 0, 1, &direct_params);

        ggml_tensor * direct_parallel_out =
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, (int64_t) dispatch.head_dim_v * n_head_q, 1);
        direct_parallel_out->src[0] = q;
        direct_parallel_out->src[1] = active_k;
        direct_parallel_out->src[2] = active_v;

        constexpr int32_t        n_workers = 4;
        std::vector<std::thread> workers;
        workers.reserve(n_workers);
        for (int32_t ith = 0; ith < n_workers; ++ith) {
            workers.emplace_back([=, &direct_params]() {
                llama_kv_lowrank_direct_attn_compute(direct_parallel_out, ith, n_workers, &direct_params);
            });
        }
        for (auto & worker : workers) {
            worker.join();
        }

        for (int64_t i = 0; i < ggml_nelements(direct_parallel_out); ++i) {
            const float serial   = ((const float *) direct_serial_out->data)[i];
            const float parallel = ((const float *) direct_parallel_out->data)[i];
            assert(std::isfinite(serial));
            assert(std::isfinite(parallel));
            assert(std::fabs(serial - parallel) < 1e-6f);
        }

        ctx->xkv_chunks.resize(original_chunk_count);
        llama_kv_blocksvd_int8_reconstruct_dispatch pooled_dispatch;
        ok = llama_kv_blocksvd_pack_int8_reconstruct_pool(*ctx, 1, 0, 0, 16, pooled_dispatch, &err);
        assert(ok);
        ggml_tensor * pooled_u  = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, pooled_dispatch.u_q.size());
        ggml_tensor * pooled_vh = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, pooled_dispatch.vh_q.size());
        std::memcpy(pooled_u->data, pooled_dispatch.u_q.data(), pooled_dispatch.u_q.size());
        std::memcpy(pooled_vh->data, pooled_dispatch.vh_q.data(), pooled_dispatch.vh_q.size());

        llama_kv_blocksvd_int8_direct_metadata_layout direct_layout;
        ok = llama_kv_blocksvd_make_int8_direct_metadata_layout(pooled_dispatch, 1, direct_layout, &err);
        assert(ok);
        std::vector<int8_t> direct_metadata;
        ok = llama_kv_blocksvd_pack_int8_direct_metadata(pooled_dispatch, direct_layout, std::vector<llama_pos>{ 40 },
                                                         40, direct_metadata, &err);
        assert(ok);
        ggml_tensor * metadata_tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, direct_metadata.size());
        std::memcpy(metadata_tensor->data, direct_metadata.data(), direct_metadata.size());

        ggml_edgekv_attn_decode_params packed_params;
        ok = llama_kv_blocksvd_make_int8_direct_params(pooled_dispatch, 1, n_head_q, 1, direct_params.scale,
                                                       packed_params, &err);
        assert(ok);
        ggml_tensor * packed_direct_out   = ggml_edgekv_attn_decode(ggml_ctx, q, pooled_u, pooled_vh, metadata_tensor,
                                                                    active_k, active_v, &packed_params);
        ggml_cgraph * packed_direct_graph = ggml_new_graph(ggml_ctx);
        ggml_build_forward_expand(packed_direct_graph, packed_direct_out);
        ggml_cplan           packed_direct_plan = ggml_graph_plan(packed_direct_graph, 4, nullptr);
        std::vector<uint8_t> packed_direct_work(packed_direct_plan.work_size);
        packed_direct_plan.work_data = packed_direct_work.empty() ? nullptr : packed_direct_work.data();
        assert(ggml_graph_compute(packed_direct_graph, &packed_direct_plan) == GGML_STATUS_SUCCESS);

        float packed_max_error = 0.0f;
        for (int64_t i = 0; i < ggml_nelements(packed_direct_out); ++i) {
            const float portable = ((const float *) direct_serial_out->data)[i];
            const float packed   = ((const float *) packed_direct_out->data)[i];
            packed_max_error     = std::max(packed_max_error, std::fabs(portable - packed));
        }
        assert(packed_max_error < 1e-5f);
    }

    ggml_free(ggml_ctx);

    const auto  before_failure = dispatch;
    std::string failure_error;
    ok = llama_kv_blocksvd_pack_int8_reconstruct_dispatch(*ctx, 99, 0, 0, dispatch, &failure_error);
    assert(!ok);
    assert(!failure_error.empty());
    assert(dispatch.n_blocks == before_failure.n_blocks);
    assert(dispatch.u_q == before_failure.u_q);

    assert(!ctx->decode_cache.empty());
    assert(!ctx->rank_cache.empty());
    const uint64_t generation_before_clear = ctx->xkv_generation;
    llama_kv_blocksvd_clear(ctx);
    assert(ctx->xkv_chunks.empty());
    assert(ctx->decode_cache.empty());
    assert(ctx->rank_cache.empty());
    assert(ctx->xkv_generation == generation_before_clear + 1);

    std::printf("test-kv-blocksvd: INT8 reconstruct dispatch OK\n");
    llama_kv_blocksvd_free(ctx);
}

static void test_int8_reconstruct_pool() {
    std::string err;
    auto *      ctx = make_int8_dispatch_context(err);
    assert(ctx != nullptr);

    llama_kv_blocksvd_int8_reconstruct_dispatch exact;
    bool ok = llama_kv_blocksvd_pack_int8_reconstruct_dispatch(*ctx, 1, 0, 0, exact, &err);
    assert(ok);

    llama_kv_blocksvd_int8_reconstruct_dispatch pool;
    ok = llama_kv_blocksvd_pack_int8_reconstruct_pool(*ctx, 1, 0, 0, 4, pool, &err);
    assert(ok);
    assert(pool.n_blocks == 4);
    assert(pool.valid_blocks == 2);
    assert(pool.block_positions.size() == static_cast<size_t>(pool.n_blocks) * pool.block_size);

    const auto assert_component_pool = [](const auto & exact_values,
                                          size_t       exact_base,
                                          const auto & pool_values,
                                          size_t       pool_base,
                                          size_t       stride,
                                          size_t       valid_blocks,
                                          size_t       capacity) {
        using value_type = typename std::decay_t<decltype(pool_values)>::value_type;
        for (size_t index = 0; index < valid_blocks * stride; ++index) {
            assert(pool_values[pool_base + index] == exact_values[exact_base + index]);
        }
        for (size_t index = valid_blocks * stride; index < capacity * stride; ++index) {
            assert(pool_values[pool_base + index] == value_type{});
        }
    };

    const size_t valid_blocks = static_cast<size_t>(pool.valid_blocks);
    const size_t capacity     = static_cast<size_t>(pool.n_blocks);
    const size_t k_u_stride   = static_cast<size_t>(pool.block_size) * pool.rank_k;
    const size_t v_u_stride   = static_cast<size_t>(pool.block_size) * pool.rank_v;
    const size_t k_vh_stride  = static_cast<size_t>(pool.rank_k) * pool.group_size * pool.n_head_kv * pool.head_dim_k;
    const size_t v_vh_stride  = static_cast<size_t>(pool.rank_v) * pool.group_size * pool.n_head_kv * pool.head_dim_v;

    assert_component_pool(exact.u_q, 0, pool.u_q, 0, k_u_stride, valid_blocks, capacity);
    assert_component_pool(exact.u_q, static_cast<size_t>(exact.v_u_offset_bytes), pool.u_q,
                          static_cast<size_t>(pool.v_u_offset_bytes), v_u_stride, valid_blocks, capacity);
    assert_component_pool(exact.vh_q, 0, pool.vh_q, 0, k_vh_stride, valid_blocks, capacity);
    assert_component_pool(exact.vh_q, static_cast<size_t>(exact.v_vh_offset_bytes), pool.vh_q,
                          static_cast<size_t>(pool.v_vh_offset_bytes), v_vh_stride, valid_blocks, capacity);
    assert_component_pool(exact.rank_scale, 0, pool.rank_scale, 0, static_cast<size_t>(pool.rank_k), valid_blocks,
                          capacity);
    assert_component_pool(exact.rank_scale,
                          static_cast<size_t>(exact.v_rank_scale_offset_bytes) / sizeof(float), pool.rank_scale,
                          static_cast<size_t>(pool.v_rank_scale_offset_bytes) / sizeof(float),
                          static_cast<size_t>(pool.rank_v), valid_blocks, capacity);

    assert(std::equal(exact.block_positions.begin(), exact.block_positions.end(), pool.block_positions.begin()));
    for (size_t index = exact.block_positions.size(); index < pool.block_positions.size(); ++index) {
        assert(pool.block_positions[index] == -1);
    }

    const auto assert_component_prefix_unchanged = [](const auto & before,
                                                       size_t       before_base,
                                                       const auto & after,
                                                       size_t       after_base,
                                                       size_t       stride,
                                                       size_t       blocks) {
        assert(std::equal(
            before.begin() + before_base, before.begin() + before_base + blocks * stride,
            after.begin() + after_base));
    };
    const auto append_fixture_block = [ctx](int32_t first_position) {
        auto chunk = ctx->xkv_chunks.front();
        for (int32_t token = 0; token < static_cast<int32_t>(chunk.pos.size()); ++token) {
            chunk.slots[token] = static_cast<uint32_t>(first_position + token);
            chunk.pos[token]   = first_position + token;
        }
        chunk.materialized = false;
        ctx->xkv_chunks.push_back(std::move(chunk));
        ++ctx->xkv_generation;
    };

    auto before_append = pool;
    append_fixture_block(32);
    llama_kv_blocksvd_int8_pool_update update;
    ok = llama_kv_blocksvd_refresh_int8_reconstruct_pool(*ctx, 1, 0, 0, pool, update, &err);
    assert(ok);
    assert(!update.full_repack);
    assert(update.first_dirty_block == 2);
    assert(update.last_dirty_block == 3);
    assert(pool.valid_blocks == 3);
    assert_component_prefix_unchanged(before_append.u_q, 0, pool.u_q, 0, k_u_stride, 2);
    assert_component_prefix_unchanged(before_append.u_q, static_cast<size_t>(before_append.v_u_offset_bytes),
                                      pool.u_q, static_cast<size_t>(pool.v_u_offset_bytes), v_u_stride, 2);
    assert_component_prefix_unchanged(before_append.vh_q, 0, pool.vh_q, 0, k_vh_stride, 2);
    assert_component_prefix_unchanged(before_append.vh_q, static_cast<size_t>(before_append.v_vh_offset_bytes),
                                      pool.vh_q, static_cast<size_t>(pool.v_vh_offset_bytes), v_vh_stride, 2);
    assert_component_prefix_unchanged(before_append.rank_scale, 0, pool.rank_scale, 0,
                                      static_cast<size_t>(pool.rank_k), 2);
    assert_component_prefix_unchanged(
        before_append.rank_scale,
        static_cast<size_t>(before_append.v_rank_scale_offset_bytes) / sizeof(float), pool.rank_scale,
        static_cast<size_t>(pool.v_rank_scale_offset_bytes) / sizeof(float), static_cast<size_t>(pool.rank_v), 2);
    assert_component_prefix_unchanged(before_append.block_positions, 0, pool.block_positions, 0,
                                      static_cast<size_t>(pool.block_size), 2);
    for (int32_t position = 32; position < 48; ++position) {
        assert(pool.block_positions[position] == position);
    }

    ok = llama_kv_blocksvd_refresh_int8_reconstruct_pool(*ctx, 1, 0, 0, pool, update, &err);
    assert(ok);
    assert(!update.full_repack);
    assert(update.first_dirty_block == 3);
    assert(update.last_dirty_block == 3);

    before_append = pool;
    append_fixture_block(48);
    ok = llama_kv_blocksvd_refresh_int8_reconstruct_pool(*ctx, 1, 0, 0, pool, update, &err);
    assert(ok);
    assert(!update.full_repack);
    assert(update.first_dirty_block == 3);
    assert(update.last_dirty_block == 4);
    assert(pool.valid_blocks == 4);
    assert_component_prefix_unchanged(before_append.u_q, 0, pool.u_q, 0, k_u_stride, 3);
    assert_component_prefix_unchanged(before_append.vh_q, 0, pool.vh_q, 0, k_vh_stride, 3);
    assert_component_prefix_unchanged(before_append.rank_scale, 0, pool.rank_scale, 0,
                                      static_cast<size_t>(pool.rank_k), 3);
    assert_component_prefix_unchanged(before_append.block_positions, 0, pool.block_positions, 0,
                                      static_cast<size_t>(pool.block_size), 3);
    for (int32_t position = 48; position < 64; ++position) {
        assert(pool.block_positions[position] == position);
    }

    const auto before_failure = pool;
    ok = llama_kv_blocksvd_pack_int8_reconstruct_pool(*ctx, 1, 0, 0, 1, pool, &err);
    assert(!ok);
    assert(!err.empty());
    assert(pool.n_blocks == before_failure.n_blocks);
    assert(pool.valid_blocks == before_failure.valid_blocks);
    assert(pool.u_q == before_failure.u_q);
    pool = before_failure;

    ggml_init_params ggml_params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    assert(ggml_ctx != nullptr);

    ggml_tensor * t_u         = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, pool.u_q.size());
    ggml_tensor * t_vh        = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, pool.vh_q.size());
    ggml_tensor * t_scale     = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, pool.rank_scale.size());
    ggml_tensor * t_positions = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, pool.block_positions.size());
    std::memcpy(t_u->data, pool.u_q.data(), ggml_nbytes(t_u));
    std::memcpy(t_vh->data, pool.vh_q.data(), ggml_nbytes(t_vh));
    std::memcpy(t_scale->data, pool.rank_scale.data(), ggml_nbytes(t_scale));
    std::memcpy(t_positions->data, pool.block_positions.data(), ggml_nbytes(t_positions));

    const ggml_edgekv_reconstruct_params reconstruct_params = {
        pool.n_blocks,
        pool.block_size,
        pool.rank_k,
        pool.rank_v,
        pool.group_size,
        pool.layer_index,
        pool.n_head_kv,
        pool.head_dim_k,
        pool.head_dim_v,
        pool.v_u_offset_bytes,
        pool.v_vh_offset_bytes,
        pool.v_rank_scale_offset_bytes,
        pool.dense_v_offset_bytes,
        pool.dense_total_bytes,
    };
    ggml_tensor * t_dense =
        ggml_edgekv_reconstruct(ggml_ctx, t_u, t_vh, t_scale, t_positions, &reconstruct_params);
    ggml_cgraph * graph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(graph, t_dense);
    ggml_cplan plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work_buffer(plan.work_size);
    plan.work_data = work_buffer.empty() ? nullptr : work_buffer.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto * dense_k = static_cast<const ggml_fp16_t *>(t_dense->data);
    const auto * dense_v = reinterpret_cast<const ggml_fp16_t *>(
        static_cast<const char *>(t_dense->data) + pool.dense_v_offset_bytes);
    for (int32_t head = 0; head < pool.n_head_kv; ++head) {
        for (int32_t block = pool.valid_blocks; block < pool.n_blocks; ++block) {
            for (int32_t token = 0; token < pool.block_size; ++token) {
                for (int32_t dim = 0; dim < pool.head_dim_k; ++dim) {
                    const size_t index =
                        (((static_cast<size_t>(head) * pool.n_blocks + block) * pool.block_size + token) *
                         pool.head_dim_k) + dim;
                    assert(dense_k[index] == ggml_fp32_to_fp16(0.0f));
                }
                for (int32_t dim = 0; dim < pool.head_dim_v; ++dim) {
                    const size_t index =
                        (((static_cast<size_t>(head) * pool.n_blocks + block) * pool.block_size + token) *
                         pool.head_dim_v) + dim;
                    assert(dense_v[index] == ggml_fp32_to_fp16(0.0f));
                }
            }
        }
    }

    ggml_free(ggml_ctx);
    llama_kv_blocksvd_free(ctx);
    std::printf("test-kv-blocksvd: INT8 reconstruct fixed-capacity pool OK\n");
}

static void test_int8_direct_backend_pool() {
    std::string err;
    auto *      ctx = make_int8_dispatch_context(err);
    assert(ctx != nullptr);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);
    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    assert(sched != nullptr);

    constexpr int32_t                   n_head_q    = 4;
    constexpr int32_t                   recent_size = 3;
    const float                         scale       = 1.0f / std::sqrt(32.0f);
    llama_kv_blocksvd_int8_backend_view view;
    bool ok = llama_kv_blocksvd_acquire_int8_direct_backend_pool(*ctx, 1, 0, 0, 4, n_head_q, recent_size, scale, sched,
                                                                 backend, view, &err);
    assert(ok);
    assert(view.backend == backend);
    assert(view.storage != nullptr);
    assert(view.metadata != nullptr);
    assert(view.direct_metadata.recent_size == recent_size);
    assert(view.direct_metadata.v_scale_offset_bytes == 128);
    assert(view.direct_metadata.block_positions_offset_bytes == 256);
    assert(view.direct_metadata.recent_positions_offset_bytes == 512);
    assert(view.direct_metadata.total_bytes == 640);

    ggml_edgekv_attn_decode_params direct_params;
    ok = llama_kv_blocksvd_make_int8_direct_params(*view.storage, 1, n_head_q, recent_size, scale, direct_params, &err);
    assert(ok);
    assert(direct_params.n_blocks == 4);
    assert(direct_params.layer_index == 1);
    assert(direct_params.metadata_total_bytes == view.direct_metadata.total_bytes);

    const std::vector<llama_pos> recent_positions = { 31, -1, 33 };
    ok = llama_kv_blocksvd_update_int8_direct_metadata(view, recent_positions, 33, &err);
    assert(ok);

    std::vector<int8_t> expected;
    ok = llama_kv_blocksvd_pack_int8_direct_metadata(*view.storage, view.direct_metadata, recent_positions, 33,
                                                     expected, &err);
    assert(ok);
    std::vector<int8_t> actual(expected.size());
    ggml_backend_tensor_get(view.metadata, actual.data(), 0, actual.size());
    assert(actual == expected);

    const size_t dynamic_bytes =
        expected.size() - static_cast<size_t>(view.direct_metadata.recent_positions_offset_bytes);
    assert(dynamic_bytes == 128);
    assert((recent_positions.size() + 1) * sizeof(int32_t) == 16);

    view = {};
    llama_kv_blocksvd_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    std::printf("test-kv-blocksvd: INT8 direct backend pool and metadata OK\n");
}

static void test_mobile_direct_parity() {
    constexpr int32_t n_blocks_pool = 16;
    constexpr int32_t block_size    = 64;
    constexpr int32_t rank          = 32;
    constexpr int32_t group_size    = 4;
    constexpr int32_t n_head_q      = 16;
    constexpr int32_t n_head_kv     = 8;
    constexpr int32_t head_dim      = 128;
    constexpr int32_t recent_size   = 128;
    constexpr int32_t layer         = 2;
    constexpr int32_t combined_dim  = group_size * n_head_kv * head_dim;

    llama_kv_blocksvd_params init_params{};
    init_params.block_size       = block_size;
    init_params.rank             = rank;
    init_params.rank_v           = rank;
    init_params.quant_bits       = 8;
    init_params.cross_layer      = true;
    init_params.layer_group_size = group_size;
    auto * bctx                  = llama_kv_blocksvd_init(init_params);
    assert(bctx != nullptr);

    llama_kv_blocksvd_xkv_chunk chunk;
    chunk.layer_start = 0;
    chunk.group_size  = group_size;
    chunk.stream      = 0;
    chunk.seq_id      = 0;
    chunk.n_head_kv   = n_head_kv;
    chunk.head_dim_k  = head_dim;
    chunk.head_dim_v  = head_dim;
    chunk.slots.resize(block_size);
    chunk.pos.resize(block_size);
    for (int32_t token = 0; token < block_size; ++token) {
        chunk.slots[token] = static_cast<uint32_t>(token);
        chunk.pos[token]   = token;
    }

    const auto initialize_factors = [](llama_kv_blocksvd_xkv_factors & factors, int seed) {
        factors.rank       = rank;
        factors.quant_bits = 8;
        factors.u_scale    = 0.0078125f;
        factors.s_scale    = 0.03125f;
        factors.vh_scale   = 0.0078125f;
        factors.u_q.resize(static_cast<size_t>(block_size) * rank);
        factors.s_q.resize(rank);
        factors.vh_q.resize(static_cast<size_t>(rank) * combined_dim);
        for (size_t index = 0; index < factors.u_q.size(); ++index) {
            factors.u_q[index] = static_cast<int8_t>((index * (seed + 3) + seed) % 31 - 15);
        }
        for (size_t index = 0; index < factors.s_q.size(); ++index) {
            factors.s_q[index] = static_cast<int8_t>((index * (seed + 5) + 7) % 23 - 11);
        }
        for (size_t index = 0; index < factors.vh_q.size(); ++index) {
            factors.vh_q[index] = static_cast<int8_t>((index * (seed + 7) + 3) % 29 - 14);
        }
    };
    initialize_factors(chunk.k_factors, 1);
    initialize_factors(chunk.v_factors, 2);
    bctx->xkv_chunks.push_back(std::move(chunk));
    bctx->xkv_generation = 1;

    llama_kv_blocksvd_int8_reconstruct_dispatch dispatch;
    std::string                                 err;
    bool ok = llama_kv_blocksvd_pack_int8_reconstruct_pool(*bctx, layer, 0, 0, n_blocks_pool, dispatch, &err);
    assert(ok);

    ggml_init_params ggml_params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    assert(ggml_ctx != nullptr);
    ggml_tensor * q        = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, head_dim, n_head_q, 1);
    ggml_tensor * active_k = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F16, head_dim, n_head_kv, recent_size);
    ggml_tensor * active_v = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F16, head_dim, n_head_kv, recent_size);
    for (int64_t index = 0; index < ggml_nelements(q); ++index) {
        ((float *) q->data)[index] = static_cast<float>((index * 5 + 3) % 37 - 18) * 0.00390625f;
    }
    for (int64_t index = 0; index < ggml_nelements(active_k); ++index) {
        ((ggml_fp16_t *) active_k->data)[index] =
            ggml_fp32_to_fp16(static_cast<float>((index * 3 + 1) % 31 - 15) * 0.0078125f);
        ((ggml_fp16_t *) active_v->data)[index] =
            ggml_fp32_to_fp16(static_cast<float>((index * 7 + 2) % 29 - 14) * 0.0078125f);
    }

    llama_kv_blocksvd_staging_t staging;
    staging.capacity                    = recent_size;
    staging.cell_to_slot                = { std::vector<int32_t>(block_size + 1, -1) };
    staging.slot_to_cell                = { std::vector<int32_t>(recent_size, -1) };
    staging.cell_to_slot[0][block_size] = 0;
    staging.slot_to_cell[0][0]          = block_size;

    llama_chunked_attn_params portable_params;
    portable_params.bctx       = bctx;
    portable_params.staging    = &staging;
    portable_params.il         = layer;
    portable_params.n_kv       = block_size + 1;
    portable_params.n_head_kv  = n_head_kv;
    portable_params.n_head_q   = n_head_q;
    portable_params.head_dim_k = head_dim;
    portable_params.head_dim_v = head_dim;
    portable_params.scale      = 1.0f / std::sqrt(static_cast<float>(head_dim));
    portable_params.n_stream   = 1;
    portable_params.cache_size = recent_size;
    portable_params.q_pos      = { block_size };
    portable_params.slot_pos.assign(recent_size, -1);
    portable_params.slot_pos[0]        = block_size;
    portable_params.use_lowrank_direct = true;
    llama_chunked_attn_build_refs(&portable_params);

    ggml_tensor * portable_out =
        ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, static_cast<int64_t>(head_dim) * n_head_q, 1);
    portable_out->src[0] = q;
    portable_out->src[1] = active_k;
    portable_out->src[2] = active_v;
    llama_kv_lowrank_direct_attn_compute(portable_out, 0, 1, &portable_params);

    ggml_tensor * packed_u  = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, dispatch.u_q.size());
    ggml_tensor * packed_vh = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, dispatch.vh_q.size());
    std::memcpy(packed_u->data, dispatch.u_q.data(), dispatch.u_q.size());
    std::memcpy(packed_vh->data, dispatch.vh_q.data(), dispatch.vh_q.size());
    llama_kv_blocksvd_int8_direct_metadata_layout layout;
    ok = llama_kv_blocksvd_make_int8_direct_metadata_layout(dispatch, recent_size, layout, &err);
    assert(ok);
    assert(layout.v_scale_offset_bytes == 2048);
    assert(layout.block_positions_offset_bytes == 4096);
    assert(layout.recent_positions_offset_bytes == 8192);
    assert(layout.total_bytes == 8832);
    std::vector<llama_pos> recent_positions(recent_size, -1);
    recent_positions[0] = block_size;
    std::vector<int8_t> metadata;
    ok = llama_kv_blocksvd_pack_int8_direct_metadata(dispatch, layout, recent_positions, block_size, metadata, &err);
    assert(ok);
    ggml_tensor * packed_metadata = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I8, metadata.size());
    std::memcpy(packed_metadata->data, metadata.data(), metadata.size());
    ggml_edgekv_attn_decode_params direct_params;
    ok = llama_kv_blocksvd_make_int8_direct_params(dispatch, layer, n_head_q, recent_size, portable_params.scale,
                                                   direct_params, &err);
    assert(ok);
    ggml_tensor * direct_out =
        ggml_edgekv_attn_decode(ggml_ctx, q, packed_u, packed_vh, packed_metadata, active_k, active_v, &direct_params);
    ggml_cgraph * graph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(graph, direct_out);
    ggml_cplan           plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    float max_error = 0.0f;
    for (int64_t index = 0; index < ggml_nelements(direct_out); ++index) {
        max_error = std::max(max_error, std::fabs(((const float *) portable_out->data)[index] -
                                                  ((const float *) direct_out->data)[index]));
    }
    assert(max_error < 1e-5f);
    std::printf("test-kv-blocksvd: mobile direct parity max_error=%g OK\n", (double) max_error);

    ggml_free(ggml_ctx);
    llama_kv_blocksvd_free(bctx);
}

static bool export_int8_dispatch_golden(const std::filesystem::path & output_dir) {
    std::string err;
    auto *      ctx = make_int8_dispatch_context(err);
    if (!ctx) {
        std::fprintf(stderr, "INT8 dispatch fixture setup failed: %s\n", err.c_str());
        return false;
    }

    llama_kv_blocksvd_int8_reconstruct_dispatch dispatch;
    bool ok = llama_kv_blocksvd_pack_int8_reconstruct_dispatch(*ctx, 1, 0, 0, dispatch, &err);
    if (!ok) {
        std::fprintf(stderr, "INT8 dispatch pack failed: %s\n", err.c_str());
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(output_dir, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "Could not create dispatch golden directory: %s\n", filesystem_error.message().c_str());
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    ok = write_raw(output_dir / "u_q.i8", dispatch.u_q) && write_raw(output_dir / "vh_q.i8", dispatch.vh_q) &&
         write_raw(output_dir / "rank_scale.f32", dispatch.rank_scale) &&
         write_raw(output_dir / "block_positions.i32", dispatch.block_positions);
    if (!ok) {
        std::fprintf(stderr, "Could not write one or more dispatch golden data files\n");
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    std::ofstream manifest(output_dir / "manifest.json");
    manifest << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"storage_abi\": \"htp_4src_1dst_aligned_v1\",\n"
             << "  \"producer\": \"llama.cpp BlockSVD xKV dispatch packer\",\n"
             << "  \"source_seed\": 20260716,\n"
             << "  \"n_blocks\": " << dispatch.n_blocks << ",\n"
             << "  \"block_size\": " << dispatch.block_size << ",\n"
             << "  \"r_k\": " << dispatch.rank_k << ",\n"
             << "  \"r_v\": " << dispatch.rank_v << ",\n"
             << "  \"group_size\": " << dispatch.group_size << ",\n"
             << "  \"layer_start\": " << dispatch.layer_start << ",\n"
             << "  \"layer_index\": " << dispatch.layer_index << ",\n"
             << "  \"h_kv\": " << dispatch.n_head_kv << ",\n"
             << "  \"d_k\": " << dispatch.head_dim_k << ",\n"
             << "  \"d_v\": " << dispatch.head_dim_v << ",\n"
             << "  \"buffer_alignment\": " << dispatch.buffer_alignment << ",\n"
             << "  \"v_u_offset_bytes\": " << dispatch.v_u_offset_bytes << ",\n"
             << "  \"v_vh_offset_bytes\": " << dispatch.v_vh_offset_bytes << ",\n"
             << "  \"v_rank_scale_offset_bytes\": " << dispatch.v_rank_scale_offset_bytes << ",\n"
             << "  \"dense_v_offset_bytes\": " << dispatch.dense_v_offset_bytes << ",\n"
             << "  \"dense_total_bytes\": " << dispatch.dense_total_bytes << ",\n"
             << "  \"u_buffer_bytes\": " << dispatch.u_q.size() * sizeof(int8_t) << ",\n"
             << "  \"vh_buffer_bytes\": " << dispatch.vh_q.size() * sizeof(int8_t) << ",\n"
             << "  \"rank_scale_buffer_bytes\": " << dispatch.rank_scale.size() * sizeof(float) << ",\n"
             << "  \"htp_op_params\": [" << dispatch.n_blocks << ", " << dispatch.block_size << ", " << dispatch.rank_k
             << ", " << dispatch.rank_v << ", " << dispatch.group_size << ", " << dispatch.layer_index << ", "
             << dispatch.n_head_kv << ", " << dispatch.head_dim_k << ", " << dispatch.head_dim_v << ", "
             << dispatch.v_u_offset_bytes << ", " << dispatch.v_vh_offset_bytes << ", "
             << dispatch.v_rank_scale_offset_bytes << ", " << dispatch.dense_v_offset_bytes << ", "
             << dispatch.dense_total_bytes << "]\n"
             << "}\n";
    ok = manifest.good();

    llama_kv_blocksvd_free(ctx);
    return ok;
}

static bool export_int8_golden(const std::filesystem::path & output_dir) {
    llama_kv_blocksvd_params params{};
    params.block_size      = 16;
    params.rank            = 8;
    params.rank_v          = 4;
    params.quant_bits      = 8;
    params.cross_layer     = true;
    params.layer_group_size = 2;

    auto * ctx = llama_kv_blocksvd_init(params);
    if (!ctx) {
        return false;
    }
    assert(ctx->xkv_generation == 0);

    const int32_t group_size = 2;
    const int32_t n_head_kv  = 2;
    const int32_t head_dim_k = 32;
    const int32_t head_dim_v = 16;
    const int32_t d_k        = n_head_kv * head_dim_k;
    const int32_t d_v        = n_head_kv * head_dim_v;
    const int32_t n_tokens   = params.block_size;

    std::mt19937 generator(20260714);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<std::vector<float>> k(group_size), v(group_size);
    std::vector<const float *> kp(group_size), vp(group_size);
    for (int32_t layer = 0; layer < group_size; ++layer) {
        k[layer].resize(static_cast<size_t>(n_tokens) * d_k);
        v[layer].resize(static_cast<size_t>(n_tokens) * d_v);
        for (float & value : k[layer]) {
            value = distribution(generator);
        }
        for (float & value : v[layer]) {
            value = distribution(generator);
        }
        kp[layer] = k[layer].data();
        vp[layer] = v[layer].data();
    }

    std::vector<uint32_t> slots(n_tokens);
    std::vector<llama_pos> positions(n_tokens);
    for (int32_t token = 0; token < n_tokens; ++token) {
        slots[token] = static_cast<uint32_t>(token);
        positions[token] = token;
    }

    std::vector<uint32_t> compressed_slots;
    std::vector<llama_pos> compressed_positions;
    std::string err;
    bool ok = llama_kv_blocksvd_append_and_compress_xkv_group_store(
            ctx, 0, kp, vp, slots.data(), positions.data(), n_tokens,
            0, 0, n_head_kv, head_dim_k, head_dim_v,
            compressed_slots, compressed_positions, &err);
    if (!ok || ctx->xkv_chunks.size() != 1) {
        std::fprintf(stderr, "INT8 golden compression failed: %s\n", err.c_str());
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    const auto & chunk = ctx->xkv_chunks.front();
    const int32_t combined_k_dim = group_size * d_k;
    const int32_t combined_v_dim = group_size * d_v;
    llama_kv_blocksvd_int8_execution_factors k_view;
    llama_kv_blocksvd_int8_execution_factors v_view;
    ok = llama_kv_blocksvd_pack_int8_execution_factors(
            chunk.k_factors, n_tokens, combined_k_dim, k_view, &err);
    ok = ok && llama_kv_blocksvd_pack_int8_execution_factors(
            chunk.v_factors, n_tokens, combined_v_dim, v_view, &err);
    if (!ok) {
        std::fprintf(stderr, "INT8 golden pack failed: %s\n", err.c_str());
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    std::vector<int32_t> block_positions;
    block_positions.reserve(chunk.pos.size());
    for (llama_pos position : chunk.pos) {
        if (position < std::numeric_limits<int32_t>::min() ||
            position > std::numeric_limits<int32_t>::max()) {
            llama_kv_blocksvd_free(ctx);
            return false;
        }
        block_positions.push_back(static_cast<int32_t>(position));
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(output_dir, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "Could not create golden directory: %s\n", filesystem_error.message().c_str());
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    ok = write_raw(output_dir / "k_u_q.i8", k_view.u_q) &&
         write_raw(output_dir / "k_s_q.i8", chunk.k_factors.s_q) &&
         write_raw(output_dir / "k_vh_q.i8", k_view.vh_q) &&
         write_raw(output_dir / "k_rank_scale.f32", k_view.rank_scale) &&
         write_raw(output_dir / "v_u_q.i8", v_view.u_q) &&
         write_raw(output_dir / "v_s_q.i8", chunk.v_factors.s_q) &&
         write_raw(output_dir / "v_vh_q.i8", v_view.vh_q) &&
         write_raw(output_dir / "v_rank_scale.f32", v_view.rank_scale) &&
         write_raw(output_dir / "block_positions.i32", block_positions) &&
         write_raw(output_dir / "slots.u32", chunk.slots);
    if (!ok) {
        std::fprintf(stderr, "Could not write one or more golden data files\n");
        llama_kv_blocksvd_free(ctx);
        return false;
    }

    std::ofstream manifest(output_dir / "manifest.json");
    manifest << std::setprecision(9)
             << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"producer\": \"llama.cpp BlockSVD xKV production compressor\",\n"
             << "  \"source_seed\": 20260714,\n"
             << "  \"runtime_seed\": 20260715,\n"
             << "  \"n_blocks\": 1,\n"
             << "  \"block_size\": " << n_tokens << ",\n"
             << "  \"r_k\": " << k_view.rank << ",\n"
             << "  \"r_v\": " << v_view.rank << ",\n"
             << "  \"group_size\": " << group_size << ",\n"
             << "  \"layer_index\": 1,\n"
             << "  \"h_q\": 4,\n"
             << "  \"h_kv\": " << n_head_kv << ",\n"
             << "  \"d_k\": " << head_dim_k << ",\n"
             << "  \"d_v\": " << head_dim_v << ",\n"
             << "  \"recent_size\": 8,\n"
             << "  \"recent_future\": 2,\n"
             << "  \"num_programs\": 1,\n"
             << "  \"archive_scales\": {\n"
             << "    \"k_u\": " << chunk.k_factors.u_scale << ",\n"
             << "    \"k_s\": " << chunk.k_factors.s_scale << ",\n"
             << "    \"k_vh\": " << chunk.k_factors.vh_scale << ",\n"
             << "    \"v_u\": " << chunk.v_factors.u_scale << ",\n"
             << "    \"v_s\": " << chunk.v_factors.s_scale << ",\n"
             << "    \"v_vh\": " << chunk.v_factors.vh_scale << "\n"
             << "  }\n"
             << "}\n";
    ok = manifest.good();

    llama_kv_blocksvd_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const char * export_dir          = nullptr;
    const char * dispatch_export_dir = nullptr;
    if (argc != 1) {
        if (argc == 3 && std::string(argv[1]) == "--export-int8-golden") {
            export_dir = argv[2];
        } else if (argc == 3 && std::string(argv[1]) == "--export-int8-dispatch-golden") {
            dispatch_export_dir = argv[2];
        } else {
            std::fprintf(stderr, "usage: %s [--export-int8-golden DIR | --export-int8-dispatch-golden DIR]\n", argv[0]);
            return 2;
        }
    }

    srand(42);

    llama_kv_blocksvd_params params{};
    params.block_size = 8;
    params.rank = 4;
    params.quant_bits = 8;
    params.cross_layer = true;
    params.layer_group_size = 2;

    auto * ctx = llama_kv_blocksvd_init(params);
    assert(ctx != nullptr);
    assert(ctx->xkv_generation == 0);

    const int group_size = 2;
    const int n_head_kv  = 4;
    const int head_dim_k = 16;
    const int head_dim_v = 16;
    const int d_k = n_head_kv * head_dim_k;
    const int d_v = n_head_kv * head_dim_v;
    const int n_tokens = 8;

    std::vector<std::vector<float>> k(group_size), v(group_size);
    std::vector<const float *> kp(group_size), vp(group_size);
    for (int l = 0; l < group_size; ++l) {
        k[l].resize(n_tokens * d_k);
        v[l].resize(n_tokens * d_v);
        for (auto & x : k[l]) x = (float) rand() / RAND_MAX - 0.5f;
        for (auto & x : v[l]) x = (float) rand() / RAND_MAX - 0.5f;
        kp[l] = k[l].data();
        vp[l] = v[l].data();
    }

    std::vector<uint32_t> slots(n_tokens);
    std::vector<llama_pos> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) { slots[i] = i; pos[i] = i; }

    std::vector<uint32_t> cslots;
    std::vector<llama_pos> cpos;
    std::string err;
    bool ok = llama_kv_blocksvd_append_and_compress_xkv_group_store(
            ctx, 0, kp, vp, slots.data(), pos.data(), n_tokens,
            0, 0, n_head_kv, head_dim_k, head_dim_v,
            cslots, cpos, &err);
    assert(ok);
    assert(ctx->xkv_chunks.size() == 1);
    assert(ctx->xkv_generation == 1);

    const auto & chunk = ctx->xkv_chunks.back();
    {
        llama_kv_blocksvd_int8_execution_factors k_view;
        llama_kv_blocksvd_int8_execution_factors v_view;
        const int32_t combined_k_dim = group_size * d_k;
        const int32_t combined_v_dim = group_size * d_v;
        ok = llama_kv_blocksvd_pack_int8_execution_factors(
                chunk.k_factors, n_tokens, combined_k_dim, k_view, &err);
        assert(ok);
        ok = llama_kv_blocksvd_pack_int8_execution_factors(
                chunk.v_factors, n_tokens, combined_v_dim, v_view, &err);
        assert(ok);
        assert(k_view.u_q == chunk.k_factors.u_q);
        assert(k_view.vh_q == chunk.k_factors.vh_q);
        assert(v_view.u_q == chunk.v_factors.u_q);
        assert(v_view.vh_q == chunk.v_factors.vh_q);

        for (int32_t r = 0; r < k_view.rank; ++r) {
            float expected = static_cast<float>(chunk.k_factors.s_q[r]);
            expected *= chunk.k_factors.u_scale;
            expected *= chunk.k_factors.s_scale;
            expected *= chunk.k_factors.vh_scale;
            assert(k_view.rank_scale[r] == expected);
        }
        for (int32_t r = 0; r < v_view.rank; ++r) {
            float expected = static_cast<float>(chunk.v_factors.s_q[r]);
            expected *= chunk.v_factors.u_scale;
            expected *= chunk.v_factors.s_scale;
            expected *= chunk.v_factors.vh_scale;
            assert(v_view.rank_scale[r] == expected);
        }

        auto invalid = chunk.k_factors;
        invalid.quant_bits = 16;
        std::string invalid_error;
        ok = llama_kv_blocksvd_pack_int8_execution_factors(
                invalid, n_tokens, combined_k_dim, k_view, &invalid_error);
        assert(!ok);
        assert(!invalid_error.empty());
        printf("test-kv-blocksvd: INT8 execution view OK\n");
    }

    std::vector<std::vector<float>> rk, rv;
    ok = llama_kv_blocksvd_decompress_xkv_chunk(chunk, rk, rv, &err);
    assert(ok);

    double mse = 0;
    for (int l = 0; l < group_size; ++l) {
        for (size_t i = 0; i < k[l].size(); ++i) mse += std::pow(k[l][i] - rk[l][i], 2);
        for (size_t i = 0; i < v[l].size(); ++i) mse += std::pow(v[l][i] - rv[l][i], 2);
    }
    mse /= (double)(group_size * n_tokens * (d_k + d_v));
    printf("test-kv-blocksvd: mse = %g\n", mse);
    assert(mse < 0.05);

    // Second test: reconstruct one layer from persisted xKV chunks plus a staging buffer.
    {
        const uint32_t n_stream = 2;
        const uint32_t n_kv     = 10;
        const uint32_t capacity = (uint32_t) n_tokens;

        // Create a second compressed chunk on stream 1.
        std::vector<std::vector<float>> k2(group_size), v2(group_size);
        std::vector<const float *> kp2(group_size), vp2(group_size);
        for (int l = 0; l < group_size; ++l) {
            k2[l].resize(n_tokens * d_k);
            v2[l].resize(n_tokens * d_v);
            for (auto & x : k2[l]) x = (float) rand() / RAND_MAX - 0.5f;
            for (auto & x : v2[l]) x = (float) rand() / RAND_MAX - 0.5f;
            kp2[l] = k2[l].data();
            vp2[l] = v2[l].data();
        }

        std::vector<uint32_t> slots2(n_tokens);
        std::vector<llama_pos> pos2(n_tokens);
        for (int i = 0; i < n_tokens; ++i) { slots2[i] = i; pos2[i] = i; }

        std::vector<uint32_t> cslots2;
        std::vector<llama_pos> cpos2;
        ok = llama_kv_blocksvd_append_and_compress_xkv_group_store(
                ctx, 0, kp2, vp2, slots2.data(), pos2.data(), n_tokens,
                1, 1, n_head_kv, head_dim_k, head_dim_v,
                cslots2, cpos2, &err);
        assert(ok);
        assert(cslots2.size() == (size_t) n_tokens);

        // Staging metadata: cells 8 and 9 are staged on both streams.
        llama_kv_blocksvd_staging_t staging;
        staging.capacity = capacity;
        staging.cell_to_slot.resize(n_stream, std::vector<int32_t>(n_kv, -1));
        staging.slot_to_cell.resize(n_stream, std::vector<int32_t>(capacity, -1));
        staging.cell_to_slot[0][8] = 0;
        staging.cell_to_slot[0][9] = 1;
        staging.slot_to_cell[0][0] = 8;
        staging.slot_to_cell[0][1] = 9;
        staging.cell_to_slot[1][8] = 0;
        staging.cell_to_slot[1][9] = 1;
        staging.slot_to_cell[1][0] = 8;
        staging.slot_to_cell[1][1] = 9;

        // Synthetic staged values for cells 8 and 9 on both streams.
        std::vector<float> k_staged((size_t) n_stream * capacity * d_k, 0.0f);
        std::vector<float> v_staged((size_t) n_stream * capacity * d_v, 0.0f);
        for (int d = 0; d < d_k; ++d) {
            k_staged[((size_t) 0 * capacity + 0) * d_k + d] = 10.0f + (float) d;
            k_staged[((size_t) 0 * capacity + 1) * d_k + d] = 20.0f + (float) d;
            k_staged[((size_t) 1 * capacity + 0) * d_k + d] = 100.0f + (float) d;
            k_staged[((size_t) 1 * capacity + 1) * d_k + d] = 200.0f + (float) d;
        }
        for (int d = 0; d < d_v; ++d) {
            v_staged[((size_t) 0 * capacity + 0) * d_v + d] = 30.0f + (float) d;
            v_staged[((size_t) 0 * capacity + 1) * d_v + d] = 40.0f + (float) d;
            v_staged[((size_t) 1 * capacity + 0) * d_v + d] = 300.0f + (float) d;
            v_staged[((size_t) 1 * capacity + 1) * d_v + d] = 400.0f + (float) d;
        }

        std::vector<float> out_k, out_v;
        ok = llama_kv_blocksvd_reconstruct_layer(
                *ctx, staging, 0, n_kv, n_head_kv, head_dim_k, head_dim_v,
                k_staged, v_staged, out_k, out_v, &err);
        assert(ok);

        auto cell_offset = [&](uint32_t s, uint32_t c, int32_t d) {
            return ((size_t) s * n_kv + c) * (size_t) d;
        };

        // Cells 0..7 for both streams must match the compressed originals.
        double mse2 = 0.0;
        for (uint32_t s : {0u, 1u}) {
            const auto & ref_k = (s == 0 ? k[0] : k2[0]);
            const auto & ref_v = (s == 0 ? v[0] : v2[0]);
            for (uint32_t c = 0; c < 8; ++c) {
                size_t off_out_k = cell_offset(s, c, d_k);
                size_t off_ref_k = (size_t) c * d_k;
                for (int d = 0; d < d_k; ++d) {
                    mse2 += std::pow(ref_k[off_ref_k + d] - out_k[off_out_k + d], 2);
                }
                size_t off_out_v = cell_offset(s, c, d_v);
                size_t off_ref_v = (size_t) c * d_v;
                for (int d = 0; d < d_v; ++d) {
                    mse2 += std::pow(ref_v[off_ref_v + d] - out_v[off_out_v + d], 2);
                }
            }
        }
        mse2 /= (double)(2 * 8 * (d_k + d_v));
        printf("test-kv-blocksvd: reconstruct mse = %g\n", mse2);
        assert(mse2 < 0.05);

        // Cells 8 and 9 on both streams must match the synthetic staged values exactly.
        for (uint32_t s : {0u, 1u}) {
            for (uint32_t c : {8u, 9u}) {
                uint32_t slot = (c == 8 ? 0u : 1u);
                size_t off_out_k = cell_offset(s, c, d_k);
                size_t off_staged_k = ((size_t) s * capacity + slot) * d_k;
                for (int d = 0; d < d_k; ++d) {
                    assert(out_k[off_out_k + d] == k_staged[off_staged_k + d]);
                }
                size_t off_out_v = cell_offset(s, c, d_v);
                size_t off_staged_v = ((size_t) s * capacity + slot) * d_v;
                for (int d = 0; d < d_v; ++d) {
                    assert(out_v[off_out_v + d] == v_staged[off_staged_v + d]);
                }
            }
        }

        printf("test-kv-blocksvd: reconstruct helper OK\n");
    }

    // Third test: an unbacked logical cell must make the helper fail.
    {
        const uint32_t n_stream = 2;
        const uint32_t n_kv     = 12;
        const uint32_t capacity = (uint32_t) n_tokens;

        llama_kv_blocksvd_staging_t staging;
        staging.capacity = capacity;
        staging.cell_to_slot.resize(n_stream, std::vector<int32_t>(n_kv, -1));
        staging.slot_to_cell.resize(n_stream, std::vector<int32_t>(capacity, -1));

        std::vector<float> k_staged((size_t) n_stream * capacity * d_k, 0.0f);
        std::vector<float> v_staged((size_t) n_stream * capacity * d_v, 0.0f);

        std::vector<float> out_k, out_v;
        std::string err2;
        bool ok2 = llama_kv_blocksvd_reconstruct_layer(
                *ctx, staging, 0, n_kv, n_head_kv, head_dim_k, head_dim_v,
                k_staged, v_staged, out_k, out_v, &err2);
        assert(!ok2);
        assert(!err2.empty());
        printf("test-kv-blocksvd: unbacked cell error OK\n");
    }

    if (export_dir && !export_int8_golden(export_dir)) {
        llama_kv_blocksvd_free(ctx);
        return 1;
    }
    test_int8_reconstruct_dispatch();
    test_int8_reconstruct_pool();
    test_int8_direct_backend_pool();
    test_mobile_direct_parity();
    if (dispatch_export_dir && !export_int8_dispatch_golden(dispatch_export_dir)) {
        llama_kv_blocksvd_free(ctx);
        return 1;
    }

    llama_kv_blocksvd_free(ctx);
    return 0;
}
