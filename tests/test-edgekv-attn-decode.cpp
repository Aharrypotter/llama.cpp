#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int   N_BLOCKS          = 2;
constexpr int   BLOCK_SIZE        = 16;
constexpr int   RANK_K            = 8;
constexpr int   RANK_V            = 4;
constexpr int   GROUP_SIZE        = 2;
constexpr int   LAYER_INDEX       = 1;
constexpr int   N_HEAD_Q          = 4;
constexpr int   N_HEAD_KV         = 2;
constexpr int   HEAD_DIM_K        = 32;
constexpr int   HEAD_DIM_V        = 16;
constexpr int   RECENT_SIZE       = 8;
constexpr int   V_U_OFFSET        = 256;
constexpr int   V_VH_OFFSET       = 2048;
constexpr int   V_SCALE_OFFSET    = 128;
constexpr int   RECENT_V_OFFSET   = 1024;
constexpr int   RECENT_POS_OFFSET = 1536;
constexpr int   RECENT_TOTAL      = 1664;
constexpr float ATTN_SCALE        = 0.17677669529663687f;

alignas(128) std::array<ggml_fp16_t, N_HEAD_Q * HEAD_DIM_K> q;
alignas(128) std::array<int8_t, 384> u_storage;
alignas(128) std::array<int8_t, 2560> vh_storage;
alignas(128) std::array<uint8_t, 256> scale_storage;
alignas(128) std::array<int32_t, N_BLOCKS * BLOCK_SIZE> block_positions;
alignas(128) std::array<int8_t, RECENT_TOTAL> recent_storage;

void initialize_inputs() {
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = ggml_fp32_to_fp16((float) ((int) ((i * 5 + 3) % 19) - 9) * 0.0078125f);
    }
    for (int i = 0; i < V_U_OFFSET; ++i) {
        u_storage[i] = (int8_t) ((i * 3 + 1) % 11 - 5);
    }
    for (size_t i = 0; i < u_storage.size() - V_U_OFFSET; ++i) {
        u_storage[V_U_OFFSET + i] = (int8_t) ((int) ((i * 5 + 2) % 9) - 4);
    }
    for (int i = 0; i < V_VH_OFFSET; ++i) {
        vh_storage[i] = (int8_t) ((i * 7 + 3) % 13 - 6);
    }
    for (size_t i = 0; i < vh_storage.size() - V_VH_OFFSET; ++i) {
        vh_storage[V_VH_OFFSET + i] = (int8_t) ((int) ((i * 2 + 4) % 11) - 5);
    }

    scale_storage.fill(0);
    auto * k_scale = reinterpret_cast<float *>(scale_storage.data());
    auto * v_scale = reinterpret_cast<float *>(scale_storage.data() + V_SCALE_OFFSET);
    for (int i = 0; i < N_BLOCKS * RANK_K; ++i) {
        k_scale[i] = 0.0015f + 0.0001f * (float) (i % 5);
    }
    for (int i = 0; i < N_BLOCKS * RANK_V; ++i) {
        v_scale[i] = 0.0020f + 0.0002f * (float) (i % 3);
    }

    for (size_t i = 0; i < block_positions.size(); ++i) {
        block_positions[i] = (int32_t) i;
    }
    for (size_t i = block_positions.size() - 3; i < block_positions.size(); ++i) {
        block_positions[i] = -1;
    }

    recent_storage.fill(0);
    auto * recent_k   = reinterpret_cast<ggml_fp16_t *>(recent_storage.data());
    auto * recent_v   = reinterpret_cast<ggml_fp16_t *>(recent_storage.data() + RECENT_V_OFFSET);
    auto * recent_pos = reinterpret_cast<int32_t *>(recent_storage.data() + RECENT_POS_OFFSET);
    for (int i = 0; i < RECENT_SIZE * N_HEAD_KV * HEAD_DIM_K; ++i) {
        recent_k[i] = ggml_fp32_to_fp16((float) ((i * 3 + 2) % 17 - 8) * 0.015625f);
    }
    for (int i = 0; i < RECENT_SIZE * N_HEAD_KV * HEAD_DIM_V; ++i) {
        recent_v[i] = ggml_fp32_to_fp16((float) ((i * 7 + 1) % 15 - 7) * 0.015625f);
    }
    for (int i = 0; i < RECENT_SIZE; ++i) {
        recent_pos[i] = 29 + i;
    }
    recent_pos[RECENT_SIZE] = 33;
}

std::array<float, N_HEAD_Q * HEAD_DIM_V> reference() {
    const auto * k_scale    = reinterpret_cast<const float *>(scale_storage.data());
    const auto * v_scale    = reinterpret_cast<const float *>(scale_storage.data() + V_SCALE_OFFSET);
    const auto * recent_k   = reinterpret_cast<const ggml_fp16_t *>(recent_storage.data());
    const auto * recent_v   = reinterpret_cast<const ggml_fp16_t *>(recent_storage.data() + RECENT_V_OFFSET);
    const auto * recent_pos = reinterpret_cast<const int32_t *>(recent_storage.data() + RECENT_POS_OFFSET);
    const int    query_pos  = recent_pos[RECENT_SIZE];

    std::array<float, N_HEAD_Q * HEAD_DIM_V> result{};
    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        const int                                  hkv = hq / (N_HEAD_Q / N_HEAD_KV);
        std::vector<float>                         logits;
        std::vector<std::array<float, HEAD_DIM_V>> values;

        for (int block = 0; block < N_BLOCKS; ++block) {
            for (int token = 0; token < BLOCK_SIZE; ++token) {
                const int token_index = block * BLOCK_SIZE + token;
                if (block_positions[token_index] < 0 || block_positions[token_index] > query_pos) {
                    continue;
                }

                float logit = 0.0f;
                for (int d = 0; d < HEAD_DIM_K; ++d) {
                    float dense_k = 0.0f;
                    for (int rank = 0; rank < RANK_K; ++rank) {
                        const int vh_index =
                            ((((block * RANK_K + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + hkv) * HEAD_DIM_K + d);
                        dense_k += (float) u_storage[token_index * RANK_K + rank] * k_scale[block * RANK_K + rank] *
                                   (float) vh_storage[vh_index];
                    }
                    logit += dense_k * ggml_fp16_to_fp32(q[hq * HEAD_DIM_K + d]);
                }
                logits.push_back(logit * ATTN_SCALE);

                std::array<float, HEAD_DIM_V> dense_v{};
                for (int d = 0; d < HEAD_DIM_V; ++d) {
                    for (int rank = 0; rank < RANK_V; ++rank) {
                        const int vh_index =
                            ((((block * RANK_V + rank) * GROUP_SIZE + LAYER_INDEX) * N_HEAD_KV + hkv) * HEAD_DIM_V + d);
                        dense_v[d] += (float) u_storage[V_U_OFFSET + token_index * RANK_V + rank] *
                                      v_scale[block * RANK_V + rank] * (float) vh_storage[V_VH_OFFSET + vh_index];
                    }
                }
                values.push_back(dense_v);
            }
        }

        for (int token = 0; token < RECENT_SIZE; ++token) {
            if (recent_pos[token] < 0 || recent_pos[token] > query_pos) {
                continue;
            }
            float     logit  = 0.0f;
            const int k_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
            for (int d = 0; d < HEAD_DIM_K; ++d) {
                logit += ggml_fp16_to_fp32(recent_k[k_base + d]) * ggml_fp16_to_fp32(q[hq * HEAD_DIM_K + d]);
            }
            logits.push_back(logit * ATTN_SCALE);

            std::array<float, HEAD_DIM_V> dense_v{};
            const int                     v_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_V;
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                dense_v[d] = ggml_fp16_to_fp32(recent_v[v_base + d]);
            }
            values.push_back(dense_v);
        }

        const float max_logit = *std::max_element(logits.begin(), logits.end());
        float       sum_exp   = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            const float weight = std::exp(logits[i] - max_logit);
            sum_exp += weight;
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                result[hq * HEAD_DIM_V + d] += weight * values[i][d];
            }
        }
        for (int d = 0; d < HEAD_DIM_V; ++d) {
            result[hq * HEAD_DIM_V + d] /= sum_exp;
        }
    }
    return result;
}

}  // namespace

int main() {
    initialize_inputs();

    ggml_init_params init_params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * tq  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HEAD_DIM_K, N_HEAD_Q);
    ggml_tensor * tu  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, u_storage.size());
    ggml_tensor * tvh = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, vh_storage.size());
    ggml_tensor * ts  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, scale_storage.size() / sizeof(float));
    ggml_tensor * tp  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, block_positions.size());
    ggml_tensor * tr  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, recent_storage.size());
    std::memcpy(tq->data, q.data(), ggml_nbytes(tq));
    std::memcpy(tu->data, u_storage.data(), ggml_nbytes(tu));
    std::memcpy(tvh->data, vh_storage.data(), ggml_nbytes(tvh));
    std::memcpy(ts->data, scale_storage.data(), ggml_nbytes(ts));
    std::memcpy(tp->data, block_positions.data(), ggml_nbytes(tp));
    std::memcpy(tr->data, recent_storage.data(), ggml_nbytes(tr));

    const ggml_edgekv_attn_decode_params params = {
        N_BLOCKS,   BLOCK_SIZE, RANK_K,      RANK_V,          GROUP_SIZE,        LAYER_INDEX,  N_HEAD_Q,   N_HEAD_KV,
        HEAD_DIM_K, HEAD_DIM_V, RECENT_SIZE, RECENT_V_OFFSET, RECENT_POS_OFFSET, RECENT_TOTAL, ATTN_SCALE,
    };
    static_assert(sizeof(params) == GGML_EDGEKV_ATTN_DECODE_PARAM_COUNT * sizeof(int32_t));
    ggml_tensor * output = ggml_edgekv_attn_decode(ctx, tq, tu, tvh, ts, tp, tr, &params);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_cplan           plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work_buffer(plan.work_size);
    plan.work_data = work_buffer.empty() ? nullptr : work_buffer.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto   expected  = reference();
    const auto * actual    = static_cast<const float *>(output->data);
    float        max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    assert(max_error < 1e-5f);
    std::printf("test-edgekv-attn-decode: max_error=%g OK\n", (double) max_error);

    ggml_free(ctx);
    return 0;
}
