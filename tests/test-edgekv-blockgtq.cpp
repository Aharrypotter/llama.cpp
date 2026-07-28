#include "ggml.h"
#include "ggml-cpu.h"
#include "llama-kv-blockgtq.h"

#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr size_t HEADER_BYTES         = 240;
constexpr size_t RECORD_BYTES         = 144;
constexpr size_t PRODUCER_BYTES       = 958144;
constexpr size_t LEGACY_TRANSFORM     = 948800;
constexpr size_t LEGACY_V_OFFSET      = 817728;
constexpr size_t PRODUCER_V_OFFSET    = 827008;
constexpr size_t CONSUMER_BYTES       = 143168;
constexpr size_t SHARED_BYTES         = 23616;
constexpr size_t DYNAMIC_BYTES        = 23298048;
constexpr size_t APPEND_BYTES         = 316;
constexpr int    CAPACITY             = 2048;
constexpr int    LAYER                = 17;
constexpr int    SEQUENCE             = 5;
constexpr int    N_HEAD_Q             = 16;
constexpr int    N_HEAD_KV            = 2;
constexpr int    HEAD_DIM              = 128;
constexpr size_t K_CODE_STRIDE         = 76;
constexpr size_t K_NORM_STRIDE         = 16;
constexpr size_t V_CODE_STRIDE         = 64;
constexpr size_t V_NORM_STRIDE         = 2;
constexpr size_t K_CODES_PER_TOKEN     = 72 * K_CODE_STRIDE;
constexpr size_t K_NORMS_PER_TOKEN     = 72 * K_NORM_STRIDE;
constexpr size_t V_CODES_PER_TOKEN     = 72 * V_CODE_STRIDE;
constexpr size_t V_NORMS_PER_TOKEN     = 72 * V_NORM_STRIDE;
constexpr size_t HISTORY_K_NORMS       = CAPACITY * K_CODES_PER_TOKEN;
constexpr size_t HISTORY_V_CODES       = HISTORY_K_NORMS + CAPACITY * K_NORMS_PER_TOKEN;
constexpr size_t HISTORY_V_NORMS       = HISTORY_V_CODES + CAPACITY * V_CODES_PER_TOKEN;

uint32_t read_u32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const uint8_t * p) {
    return static_cast<uint64_t>(read_u32(p)) |
           (static_cast<uint64_t>(read_u32(p + 4)) << 32);
}

std::vector<uint8_t> read_file(const char * path) {
    std::ifstream stream(path, std::ios::binary);
    assert(stream.good());
    return {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>(),
    };
}

void commit_token(
        std::vector<uint8_t> & history,
        int token,
        const uint8_t payload[APPEND_BYTES]) {
    const size_t head = static_cast<size_t>(LAYER) * N_HEAD_KV;
    for (int kv = 0; kv < N_HEAD_KV; ++kv) {
        const size_t storage_head = head + static_cast<size_t>(kv);
        std::memcpy(
            history.data() +
                static_cast<size_t>(token) * K_CODES_PER_TOKEN +
                storage_head * K_CODE_STRIDE,
            payload + static_cast<size_t>(kv) * K_CODE_STRIDE,
            K_CODE_STRIDE);
        std::memcpy(
            history.data() + HISTORY_K_NORMS +
                static_cast<size_t>(token) * K_NORMS_PER_TOKEN +
                storage_head * K_NORM_STRIDE,
            payload + 152 + static_cast<size_t>(kv) * K_NORM_STRIDE,
            K_NORM_STRIDE);
        std::memcpy(
            history.data() + HISTORY_V_CODES +
                static_cast<size_t>(token) * V_CODES_PER_TOKEN +
                storage_head * V_CODE_STRIDE,
            payload + 184 + static_cast<size_t>(kv) * V_CODE_STRIDE,
            V_CODE_STRIDE);
        std::memcpy(
            history.data() + HISTORY_V_NORMS +
                static_cast<size_t>(token) * V_NORMS_PER_TOKEN +
                storage_head * V_NORM_STRIDE,
            payload + 312 + static_cast<size_t>(kv) * V_NORM_STRIDE,
            V_NORM_STRIDE);
    }
}

void assert_token_committed(
        const std::vector<uint8_t> & history,
        int token,
        const uint8_t payload[APPEND_BYTES]) {
    std::vector<uint8_t> expected(DYNAMIC_BYTES);
    commit_token(expected, token, payload);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != 0) {
            assert(history[i] == expected[i]);
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    assert(argc == 4);
    const std::vector<uint8_t> h7_fixture = read_file(argv[1]);
    const llama_kv_blockgtq_package package =
        llama_kv_blockgtq_package::load(argv[2]);
    const std::vector<uint8_t> graph_fixture = read_file(argv[3]);
    assert(h7_fixture.size() > HEADER_BYTES + 4 * RECORD_BYTES);
    assert(std::memcmp(h7_fixture.data(), "BGTQH71", 7) == 0);
    assert(read_u32(h7_fixture.data() + 8) == 1);
    assert(read_u32(h7_fixture.data() + 12) == 4);
    assert(read_u32(h7_fixture.data() + 16) == CAPACITY);
    assert(read_u32(h7_fixture.data() + 20) == HEADER_BYTES);
    assert(read_u32(h7_fixture.data() + 24) == RECORD_BYTES);
    assert(read_u32(h7_fixture.data() + 28) == LEGACY_TRANSFORM);
    assert(read_u32(h7_fixture.data() + 32) == CONSUMER_BYTES);

    const uint64_t transform_offset = read_u64(h7_fixture.data() + 36);
    const uint64_t consumer_offset  = read_u64(h7_fixture.data() + 44);
    const uint64_t shared_offset    = read_u64(h7_fixture.data() + 52);
    assert(read_u64(h7_fixture.data() + 60) == h7_fixture.size());
    assert(transform_offset + LEGACY_TRANSFORM <= h7_fixture.size());
    assert(consumer_offset + CONSUMER_BYTES <= h7_fixture.size());
    assert(shared_offset + SHARED_BYTES <= h7_fixture.size());

    assert(graph_fixture.size() >= 256);
    assert(std::memcmp(graph_fixture.data(), "BGTQLG1", 7) == 0);
    assert(read_u32(graph_fixture.data() + 8) == 1);
    assert(read_u32(graph_fixture.data() + 12) == LAYER);
    assert(read_u32(graph_fixture.data() + 16) == SEQUENCE);
    assert(read_u32(graph_fixture.data() + 20) == CAPACITY);
    assert(read_u32(graph_fixture.data() + 24) == N_HEAD_Q);
    assert(read_u32(graph_fixture.data() + 28) == N_HEAD_KV);
    assert(read_u32(graph_fixture.data() + 32) == HEAD_DIM);
    assert(read_u32(graph_fixture.data() + 36) == APPEND_BYTES);
    const uint64_t k_offset       = read_u64(graph_fixture.data() + 40);
    const uint64_t v_offset       = read_u64(graph_fixture.data() + 48);
    const uint64_t query_offset   = read_u64(graph_fixture.data() + 56);
    const uint64_t append_offset  = read_u64(graph_fixture.data() + 64);
    const uint64_t output_offset  = read_u64(graph_fixture.data() + 72);
    assert(read_u64(graph_fixture.data() + 80) == graph_fixture.size());
    constexpr size_t DENSE_TOKEN_BYTES = N_HEAD_KV * HEAD_DIM * sizeof(float);
    assert(k_offset + SEQUENCE * DENSE_TOKEN_BYTES <= graph_fixture.size());
    assert(v_offset + SEQUENCE * DENSE_TOKEN_BYTES <= graph_fixture.size());
    assert(query_offset + N_HEAD_Q * HEAD_DIM * sizeof(float) <= graph_fixture.size());
    assert(append_offset + SEQUENCE * APPEND_BYTES <= graph_fixture.size());
    assert(output_offset + N_HEAD_Q * HEAD_DIM * sizeof(float) <= graph_fixture.size());

    const std::vector<uint8_t> & producer = package.producer();
    const std::vector<uint8_t> & consumer = package.consumer();
    const std::vector<uint8_t> & shared = package.shared();
    assert(package.sha256() ==
           "7159380b3b2bbc580c3e55173e1e7dacaa56c8b2c6a7b8b72f6c28f6ccb5bc5b");
    assert(std::memcmp(
               producer.data(),
               h7_fixture.data() + transform_offset,
               LEGACY_V_OFFSET) == 0);
    assert(std::memcmp(
               producer.data() + PRODUCER_V_OFFSET,
               h7_fixture.data() + transform_offset + LEGACY_V_OFFSET,
               LEGACY_TRANSFORM - LEGACY_V_OFFSET) == 0);
    assert(std::memcmp(
               consumer.data(),
               h7_fixture.data() + consumer_offset,
               CONSUMER_BYTES) == 0);
    assert(std::memcmp(
               shared.data(),
               h7_fixture.data() + shared_offset,
               SHARED_BYTES) == 0);
    std::vector<uint8_t> history(DYNAMIC_BYTES);
    const uint8_t * append = graph_fixture.data() + append_offset;
    for (int token = 0; token < SEQUENCE - 1; ++token) {
        commit_token(history, token, append + static_cast<size_t>(token) * APPEND_BYTES);
    }

    const ggml_init_params init_params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_Q);
    ggml_tensor * producer_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, PRODUCER_BYTES);
    ggml_tensor * consumer_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, CONSUMER_BYTES);
    ggml_tensor * shared_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, SHARED_BYTES);
    ggml_tensor * history_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, DYNAMIC_BYTES);
    std::memcpy(query->data, graph_fixture.data() + query_offset, ggml_nbytes(query));
    std::memcpy(producer_tensor->data, producer.data(), producer.size());
    std::memcpy(consumer_tensor->data, consumer.data(), consumer.size());
    std::memcpy(shared_tensor->data, shared.data(), shared.size());
    std::memcpy(history_tensor->data, history.data(), history.size());

    std::array<ggml_tensor *, SEQUENCE> packed{};
    for (int token = 0; token < SEQUENCE; ++token) {
        ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV);
        ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV);
        std::memcpy(
            k->data,
            graph_fixture.data() + k_offset +
                static_cast<size_t>(token) * DENSE_TOKEN_BYTES,
            DENSE_TOKEN_BYTES);
        std::memcpy(
            v->data,
            graph_fixture.data() + v_offset +
                static_cast<size_t>(token) * DENSE_TOKEN_BYTES,
            DENSE_TOKEN_BYTES);
        const ggml_edgekv_blockgtq_pack_token_params pack_params = {
            2,
            LAYER,
            N_HEAD_KV,
            HEAD_DIM,
        };
        packed[token] = ggml_edgekv_blockgtq_pack_token(
            ctx,
            k,
            v,
            producer_tensor,
            consumer_tensor,
            shared_tensor,
            &pack_params);
    }

    const ggml_edgekv_blockgtq_attn_params params = {
        2,
        static_cast<int32_t>(UINT32_C(0x7159380b)),
        static_cast<int32_t>(UINT32_C(0x92f8aa7c)),
        LAYER,
        SEQUENCE,
        CAPACITY,
        N_HEAD_Q,
        N_HEAD_KV,
        HEAD_DIM,
        8,
        36,
        static_cast<int32_t>(PRODUCER_BYTES),
        static_cast<int32_t>(CONSUMER_BYTES),
        static_cast<int32_t>(SHARED_BYTES),
        static_cast<int32_t>(DYNAMIC_BYTES),
    };
    ggml_tensor * output = ggml_edgekv_blockgtq_attn_decode(
        ctx,
        query,
        producer_tensor,
        consumer_tensor,
        shared_tensor,
        history_tensor,
        packed[SEQUENCE - 1],
        &params);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * token : packed) {
        ggml_build_forward_expand(graph, token);
    }
    ggml_build_forward_expand(graph, output);
    ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const float * actual   = static_cast<const float *>(output->data);
    const float * expected =
        reinterpret_cast<const float *>(graph_fixture.data() + output_offset);
    float max_error = 0.0f;
    for (int i = 0; i < N_HEAD_Q * HEAD_DIM; ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    assert(max_error <= 2.0e-5f);
    for (int token = 0; token < SEQUENCE; ++token) {
        assert(std::memcmp(
                   packed[token]->data,
                   append + static_cast<size_t>(token) * APPEND_BYTES,
                   APPEND_BYTES) == 0);
    }

    std::memcpy(history.data(), history_tensor->data, history.size());
    assert_token_committed(
        history,
        SEQUENCE - 1,
        append + static_cast<size_t>(SEQUENCE - 1) * APPEND_BYTES);

    ggml_tensor * batch_k =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV, SEQUENCE);
    ggml_tensor * batch_v =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV, SEQUENCE);
    ggml_tensor * batch_history =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I8, DYNAMIC_BYTES);
    std::memcpy(
        batch_k->data,
        graph_fixture.data() + k_offset,
        SEQUENCE * DENSE_TOKEN_BYTES);
    std::memcpy(
        batch_v->data,
        graph_fixture.data() + v_offset,
        SEQUENCE * DENSE_TOKEN_BYTES);
    std::memset(batch_history->data, 0, DYNAMIC_BYTES);
    const ggml_edgekv_blockgtq_pack_batch_params batch_params = {
        2,
        LAYER,
        0,
        SEQUENCE,
        CAPACITY,
    };
    ggml_tensor * batch_commit = ggml_edgekv_blockgtq_pack_batch(
        ctx,
        batch_k,
        batch_v,
        producer_tensor,
        consumer_tensor,
        shared_tensor,
        batch_history,
        &batch_params);
    ggml_cgraph * batch_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(batch_graph, batch_commit);
    ggml_cplan batch_plan = ggml_graph_plan(batch_graph, 1, nullptr);
    std::vector<uint8_t> batch_work(batch_plan.work_size);
    batch_plan.work_data =
        batch_work.empty() ? nullptr : batch_work.data();
    assert(ggml_graph_compute(batch_graph, &batch_plan) == GGML_STATUS_SUCCESS);
    assert(static_cast<const uint8_t *>(batch_commit->data)[0] == 1);

    std::vector<uint8_t> expected_batch_history(DYNAMIC_BYTES);
    for (int token = 0; token < SEQUENCE; ++token) {
        commit_token(
            expected_batch_history,
            token,
            append + static_cast<size_t>(token) * APPEND_BYTES);
    }
    assert(std::memcmp(
               batch_history->data,
               expected_batch_history.data(),
               DYNAMIC_BYTES) == 0);

    ggml_backend_t runtime_backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(runtime_backend != nullptr);
    std::string runtime_error;
    auto runtime = llama_kv_blockgtq_runtime::create(
        argv[2], runtime_backend, &runtime_error);
    assert(runtime != nullptr);
    assert(runtime->valid());
    assert(runtime->sequence_length() == 0);
    assert(runtime->prepare(0, SEQUENCE, &runtime_error));
    assert(!runtime->prepare(0, 1, &runtime_error));
    runtime->finish();
    assert(runtime->sequence_length() == SEQUENCE);
    assert(!runtime->prepare(SEQUENCE - 1, 1, &runtime_error));
    assert(runtime->prepare(SEQUENCE, 1, &runtime_error));
    runtime->fail();
    assert(!runtime->valid());
    assert(!runtime->prepare(SEQUENCE, 1, &runtime_error));
    runtime.reset();
    ggml_backend_free(runtime_backend);

    std::printf(
        "test-edgekv-blockgtq: layer=%d sequence=%d max_error=%g batch=OK state=OK\n",
        LAYER,
        SEQUENCE,
        static_cast<double>(max_error));

    ggml_free(ctx);
    return 0;
}
