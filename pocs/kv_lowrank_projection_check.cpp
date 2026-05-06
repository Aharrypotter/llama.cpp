#include "kv_lowrank.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void append_f32(std::vector<uint8_t> & dst, float value) {
    const size_t offset = dst.size();
    dst.resize(offset + sizeof(float));
    std::memcpy(dst.data() + offset, &value, sizeof(float));
}

static std::vector<uint8_t> make_identity_like_basis(int rank, int d_kv) {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(rank) * static_cast<size_t>(d_kv) * sizeof(float));

    for (int r = 0; r < rank; ++r) {
        for (int c = 0; c < d_kv; ++c) {
            append_f32(out, r == c ? 1.0f : 0.0f);
        }
    }

    return out;
}

static bool near(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-6f;
}

static bool expect_true(bool value, const char * msg) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return false;
    }

    return true;
}

static bool expect_near_vec(const std::vector<float> & actual, const std::vector<float> & expected, const char * name) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "FAIL: %s size actual=%zu expected=%zu\n", name, actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!near(actual[i], expected[i])) {
            std::fprintf(stderr, "FAIL: %s[%zu] actual=%f expected=%f\n", name, i, actual[i], expected[i]);
            return false;
        }
    }

    return true;
}

int main() {
    common_kv_lowrank_basis_manifest manifest;
    manifest.dtype = "f32";
    manifest.layout = "row-major";
    manifest.rank = 2;
    manifest.n_layer = 1;
    manifest.head_dim = 4;
    manifest.n_head_kv = 1;

    common_kv_lowrank_basis_layer_data basis;
    basis.layer = 0;
    basis.k = make_identity_like_basis(manifest.rank, manifest.head_dim * manifest.n_head_kv);
    basis.v = make_identity_like_basis(manifest.rank, manifest.head_dim * manifest.n_head_kv);

    const std::vector<float> k_dense = {
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    const std::vector<float> v_dense = {
        30.0f, 31.0f, 32.0f, 33.0f,
        40.0f, 41.0f, 42.0f, 43.0f,
    };

    std::string err;
    std::vector<float> a_k;
    std::vector<float> a_v;
    if (!expect_true(common_kv_lowrank_project_chunk(
                manifest, basis, k_dense.data(), v_dense.data(), 2, a_k, a_v, &err), err.c_str())) {
        return 1;
    }

    if (!expect_near_vec(a_k, { 10.0f, 11.0f, 20.0f, 21.0f }, "a_k")) {
        return 1;
    }
    if (!expect_near_vec(a_v, { 30.0f, 31.0f, 40.0f, 41.0f }, "a_v")) {
        return 1;
    }

    common_kv_lowrank_layer_state state;
    state.layer = 0;
    state.rank = manifest.rank;
    state.d_kv = manifest.head_dim * manifest.n_head_kv;

    if (!expect_true(common_kv_lowrank_layer_append_projected_chunk(
                state, a_k.data(), a_v.data(), 2, &err), err.c_str())) {
        return 1;
    }
    if (!expect_true(state.n_hist_tokens == 2, "n_hist_tokens should be 2")) {
        return 1;
    }
    if (!expect_true(state.n_chunks == 1, "n_chunks should be 1")) {
        return 1;
    }
    if (!expect_true(state.a_k.size() == 4, "a_k state size should be 4")) {
        return 1;
    }
    if (!expect_true(state.a_v.size() == 4, "a_v state size should be 4")) {
        return 1;
    }
    if (!expect_true(common_kv_lowrank_layer_memory_bytes(state) == 32, "state memory should be 32 bytes")) {
        return 1;
    }

    std::vector<float> k_recon;
    std::vector<float> v_recon;
    if (!expect_true(common_kv_lowrank_reconstruct_chunk(
                manifest, basis, state.a_k.data(), state.a_v.data(), state.n_hist_tokens, k_recon, v_recon, &err), err.c_str())) {
        return 1;
    }

    if (!expect_near_vec(k_recon, {
                10.0f, 11.0f, 0.0f, 0.0f,
                20.0f, 21.0f, 0.0f, 0.0f,
            }, "k_recon")) {
        return 1;
    }
    if (!expect_near_vec(v_recon, {
                30.0f, 31.0f, 0.0f, 0.0f,
                40.0f, 41.0f, 0.0f, 0.0f,
            }, "v_recon")) {
        return 1;
    }

    common_kv_lowrank_layer_clear(state);
    if (!expect_true(state.n_hist_tokens == 0, "n_hist_tokens should clear to 0")) {
        return 1;
    }
    if (!expect_true(state.n_chunks == 0, "n_chunks should clear to 0")) {
        return 1;
    }
    if (!expect_true(common_kv_lowrank_layer_memory_bytes(state) == 0, "state memory should clear to 0")) {
        return 1;
    }

    common_kv_lowrank_context ctx;
    ctx.params.enabled = true;
    ctx.params.rank = manifest.rank;
    ctx.params.window = 2;
    ctx.params.chunk = 2;
    ctx.basis.manifest = manifest;
    ctx.basis.layers.push_back(basis);

    common_kv_lowrank_layer_state policy_state;
    policy_state.layer = 0;
    policy_state.rank = manifest.rank;
    policy_state.d_kv = manifest.head_dim * manifest.n_head_kv;
    ctx.layers.push_back(policy_state);

    llama_kv_lowrank_error_stats stats;
    if (!expect_true(common_kv_lowrank_context_append_policy_project_reconstruct_error(
                ctx, 0, k_dense.data(), v_dense.data(), 2, stats, &err), err.c_str())) {
        return 1;
    }
    if (!expect_true(stats.n_projected_tokens == 0, "policy should keep first window dense")) {
        return 1;
    }
    if (!expect_true(stats.n_pending_tokens == 2, "policy pending tokens should equal window")) {
        return 1;
    }
    if (!expect_true(ctx.layers[0].n_hist_tokens == 0, "policy history should be empty inside window")) {
        return 1;
    }

    const std::vector<float> k_dense_next = {
        50.0f, 51.0f, 52.0f, 53.0f,
        60.0f, 61.0f, 62.0f, 63.0f,
    };
    const std::vector<float> v_dense_next = {
        70.0f, 71.0f, 72.0f, 73.0f,
        80.0f, 81.0f, 82.0f, 83.0f,
    };
    if (!expect_true(common_kv_lowrank_context_append_policy_project_reconstruct_error(
                ctx, 0, k_dense_next.data(), v_dense_next.data(), 2, stats, &err), err.c_str())) {
        return 1;
    }
    if (!expect_true(stats.n_projected_tokens == 2, "policy should project one full historical chunk")) {
        return 1;
    }
    if (!expect_true(stats.n_chunks_projected == 1, "policy should project one chunk")) {
        return 1;
    }
    if (!expect_true(stats.n_pending_tokens == 2, "policy should leave recent window pending")) {
        return 1;
    }
    if (!expect_true(ctx.layers[0].n_hist_tokens == 2, "policy history should contain projected chunk")) {
        return 1;
    }
    if (!expect_true(ctx.layers[0].n_chunks == 1, "policy history should count one chunk")) {
        return 1;
    }
    if (!expect_near_vec(ctx.layers[0].a_k, { 10.0f, 11.0f, 20.0f, 21.0f }, "policy_a_k")) {
        return 1;
    }
    if (!expect_near_vec(ctx.layers[0].pending_k, k_dense_next, "policy_pending_k")) {
        return 1;
    }

    std::printf("PASS: WHLR-KV projection, append/state/memory, reconstruction, and policy checks passed\n");
    return 0;
}
