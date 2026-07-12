#include "llama-kv-blocksvd.h"
#include "llama-kv-blocksvd-execution.h"
#include "llama.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
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
    const char * export_dir = nullptr;
    if (argc != 1) {
        if (argc == 3 && std::string(argv[1]) == "--export-int8-golden") {
            export_dir = argv[2];
        } else {
            std::fprintf(stderr, "usage: %s [--export-int8-golden DIR]\n", argv[0]);
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

    llama_kv_blocksvd_free(ctx);
    return 0;
}
