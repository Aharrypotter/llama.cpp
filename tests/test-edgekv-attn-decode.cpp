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

constexpr int   N_BLOCKS                   = 2;
constexpr int   BLOCK_SIZE                 = 16;
constexpr int   RANK_K                     = 8;
constexpr int   RANK_V                     = 4;
constexpr int   GROUP_SIZE                 = 2;
constexpr int   N_HEAD_Q                   = 4;
constexpr int   N_HEAD_KV                  = 2;
constexpr int   HEAD_DIM_K                 = 32;
constexpr int   HEAD_DIM_V                 = 16;
constexpr int   RECENT_SIZE                = 8;
constexpr int   V_U_OFFSET                 = 256;
constexpr int   V_VH_OFFSET                = 2048;
constexpr int   METADATA_V_SCALE_OFFSET    = 128;
constexpr int   METADATA_BLOCK_POS_OFFSET  = 256;
constexpr int   METADATA_RECENT_POS_OFFSET = 384;
constexpr int   METADATA_TOTAL             = 512;
constexpr float ATTN_SCALE                 = 0.17677669529663687f;

alignas(128) std::array<float, N_HEAD_Q * HEAD_DIM_K> q;
alignas(128) std::array<int8_t, 384> u_storage;
alignas(128) std::array<int8_t, 2560> vh_storage;
alignas(128) std::array<uint8_t, METADATA_TOTAL> metadata;
alignas(128) std::array<ggml_fp16_t, RECENT_SIZE * N_HEAD_KV * HEAD_DIM_K> active_k;
alignas(128) std::array<ggml_fp16_t, RECENT_SIZE * N_HEAD_KV * HEAD_DIM_V> active_v;

void initialize_inputs() {
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = (float) ((int) ((i * 5 + 3) % 19) - 9) * 0.0078125f;
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

    metadata.fill(0);
    auto * k_scale = reinterpret_cast<float *>(metadata.data());
    auto * v_scale = reinterpret_cast<float *>(metadata.data() + METADATA_V_SCALE_OFFSET);
    for (int i = 0; i < N_BLOCKS * RANK_K; ++i) {
        k_scale[i] = 0.0015f + 0.0001f * (float) (i % 5);
    }
    for (int i = 0; i < N_BLOCKS * RANK_V; ++i) {
        v_scale[i] = 0.0020f + 0.0002f * (float) (i % 3);
    }

    auto * block_positions = reinterpret_cast<int32_t *>(metadata.data() + METADATA_BLOCK_POS_OFFSET);
    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; ++i) {
        block_positions[i] = i;
    }
    for (int i = N_BLOCKS * BLOCK_SIZE - 3; i < N_BLOCKS * BLOCK_SIZE; ++i) {
        block_positions[i] = -1;
    }

    auto * recent_positions = reinterpret_cast<int32_t *>(metadata.data() + METADATA_RECENT_POS_OFFSET);
    for (int i = 0; i < RECENT_SIZE; ++i) {
        recent_positions[i] = 29 + i;
    }
    recent_positions[RECENT_SIZE] = 33;

    for (size_t i = 0; i < active_k.size(); ++i) {
        active_k[i] = ggml_fp32_to_fp16((float) ((int) ((i * 3 + 2) % 17) - 8) * 0.015625f);
    }
    for (size_t i = 0; i < active_v.size(); ++i) {
        active_v[i] = ggml_fp32_to_fp16((float) ((int) ((i * 7 + 1) % 15) - 7) * 0.015625f);
    }
}

std::array<float, N_HEAD_Q * HEAD_DIM_V> reference(int layer_index) {
    const auto * k_scale = reinterpret_cast<const float *>(metadata.data());
    const auto * v_scale = reinterpret_cast<const float *>(metadata.data() + METADATA_V_SCALE_OFFSET);
    const auto * block_positions =
        reinterpret_cast<const int32_t *>(metadata.data() + METADATA_BLOCK_POS_OFFSET);
    const auto * recent_positions =
        reinterpret_cast<const int32_t *>(metadata.data() + METADATA_RECENT_POS_OFFSET);
    const int query_position = recent_positions[RECENT_SIZE];

    std::array<float, N_HEAD_Q * HEAD_DIM_V> result{};
    for (int hq = 0; hq < N_HEAD_Q; ++hq) {
        const int                                  hkv = hq / (N_HEAD_Q / N_HEAD_KV);
        std::vector<float>                         logits;
        std::vector<std::array<float, HEAD_DIM_V>> values;

        for (int block = 0; block < N_BLOCKS; ++block) {
            for (int token = 0; token < BLOCK_SIZE; ++token) {
                const int token_index = block * BLOCK_SIZE + token;
                if (block_positions[token_index] < 0 || block_positions[token_index] > query_position) {
                    continue;
                }

                float logit = 0.0f;
                for (int d = 0; d < HEAD_DIM_K; ++d) {
                    float dense_k = 0.0f;
                    for (int rank = 0; rank < RANK_K; ++rank) {
                        const int vh_index =
                            ((((block * RANK_K + rank) * GROUP_SIZE + layer_index) * N_HEAD_KV + hkv) * HEAD_DIM_K + d);
                        dense_k += (float) u_storage[token_index * RANK_K + rank] * k_scale[block * RANK_K + rank] *
                                   (float) vh_storage[vh_index];
                    }
                    logit += dense_k * q[hq * HEAD_DIM_K + d];
                }
                logits.push_back(logit * ATTN_SCALE);

                std::array<float, HEAD_DIM_V> dense_v{};
                for (int d = 0; d < HEAD_DIM_V; ++d) {
                    for (int rank = 0; rank < RANK_V; ++rank) {
                        const int vh_index =
                            ((((block * RANK_V + rank) * GROUP_SIZE + layer_index) * N_HEAD_KV + hkv) * HEAD_DIM_V + d);
                        dense_v[d] += (float) u_storage[V_U_OFFSET + token_index * RANK_V + rank] *
                                      v_scale[block * RANK_V + rank] * (float) vh_storage[V_VH_OFFSET + vh_index];
                    }
                }
                values.push_back(dense_v);
            }
        }

        for (int token = 0; token < RECENT_SIZE; ++token) {
            if (recent_positions[token] < 0 || recent_positions[token] > query_position) {
                continue;
            }
            float     logit  = 0.0f;
            const int k_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_K;
            for (int d = 0; d < HEAD_DIM_K; ++d) {
                logit += ggml_fp16_to_fp32(active_k[k_base + d]) * q[hq * HEAD_DIM_K + d];
            }
            logits.push_back(logit * ATTN_SCALE);

            std::array<float, HEAD_DIM_V> dense_v{};
            const int                     v_base = (token * N_HEAD_KV + hkv) * HEAD_DIM_V;
            for (int d = 0; d < HEAD_DIM_V; ++d) {
                dense_v[d] = ggml_fp16_to_fp32(active_v[v_base + d]);
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

void copy_active_tensor(ggml_tensor * tensor, const ggml_fp16_t * source, size_t count) {
    if (tensor->type == GGML_TYPE_F16) {
        std::memcpy(tensor->data, source, count * sizeof(*source));
        return;
    }
    auto * output = static_cast<float *>(tensor->data);
    for (size_t i = 0; i < count; ++i) {
        output[i] = ggml_fp16_to_fp32(source[i]);
    }
}

void run_case(ggml_type active_type, int layer_index) {
    ggml_init_params init_params = {
        /* .mem_size   = */ 2 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * tq  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM_K, N_HEAD_Q);
    ggml_tensor * tu  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, u_storage.size());
    ggml_tensor * tvh = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, vh_storage.size());
    ggml_tensor * tm  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, metadata.size());
    ggml_tensor * tk  = ggml_new_tensor_3d(ctx, active_type, HEAD_DIM_K, N_HEAD_KV, RECENT_SIZE);
    ggml_tensor * tv  = ggml_new_tensor_3d(ctx, active_type, HEAD_DIM_V, N_HEAD_KV, RECENT_SIZE);
    std::memcpy(tq->data, q.data(), ggml_nbytes(tq));
    std::memcpy(tu->data, u_storage.data(), ggml_nbytes(tu));
    std::memcpy(tvh->data, vh_storage.data(), ggml_nbytes(tvh));
    std::memcpy(tm->data, metadata.data(), ggml_nbytes(tm));
    copy_active_tensor(tk, active_k.data(), active_k.size());
    copy_active_tensor(tv, active_v.data(), active_v.size());

    const ggml_edgekv_attn_decode_params params = {
        N_BLOCKS,
        BLOCK_SIZE,
        RANK_K,
        RANK_V,
        GROUP_SIZE,
        layer_index,
        N_HEAD_Q,
        N_HEAD_KV,
        HEAD_DIM_K,
        HEAD_DIM_V,
        RECENT_SIZE,
        METADATA_V_SCALE_OFFSET,
        METADATA_BLOCK_POS_OFFSET,
        METADATA_RECENT_POS_OFFSET,
        METADATA_TOTAL,
        ATTN_SCALE,
    };
    static_assert(sizeof(params) == GGML_EDGEKV_ATTN_DECODE_PARAM_COUNT * sizeof(int32_t));
    ggml_tensor * output = ggml_edgekv_attn_decode(ctx, tq, tu, tvh, tm, tk, tv, &params);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_cplan           plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work_buffer(plan.work_size);
    plan.work_data = work_buffer.empty() ? nullptr : work_buffer.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto   expected  = reference(layer_index);
    const auto * actual    = static_cast<const float *>(output->data);
    float        max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    assert(max_error < 1e-5f);
    std::printf("test-edgekv-attn-decode: active=%s layer=%d max_error=%g OK\n", ggml_type_name(active_type),
                layer_index, (double) max_error);

    ggml_free(ctx);
}

}  // namespace

int main() {
    initialize_inputs();
    for (int layer_index = 0; layer_index < GROUP_SIZE; ++layer_index) {
        run_case(GGML_TYPE_F16, layer_index);
        run_case(GGML_TYPE_F32, layer_index);
    }
    return 0;
}
