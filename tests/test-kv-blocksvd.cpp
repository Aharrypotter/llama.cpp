#include "llama-kv-blocksvd.h"
#include "llama.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
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

    llama_kv_blocksvd_free(ctx);
    return 0;
}
