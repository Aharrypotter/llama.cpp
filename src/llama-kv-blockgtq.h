#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

enum {
    LLAMA_KV_BLOCKGTQ_LAYERS         = 36,
    LLAMA_KV_BLOCKGTQ_QUERY_HEADS    = 16,
    LLAMA_KV_BLOCKGTQ_KV_HEADS       = 2,
    LLAMA_KV_BLOCKGTQ_HEAD_DIM       = 128,
    LLAMA_KV_BLOCKGTQ_CAPACITY       = 2048,
    LLAMA_KV_BLOCKGTQ_APPEND_BYTES   = 316,
    LLAMA_KV_BLOCKGTQ_PRODUCER_BYTES = 958144,
    LLAMA_KV_BLOCKGTQ_CONSUMER_BYTES = 143168,
    LLAMA_KV_BLOCKGTQ_SHARED_BYTES   = 23616,
    LLAMA_KV_BLOCKGTQ_HISTORY_BYTES  = 23298048,
};

class llama_kv_blockgtq_package {
public:
    static llama_kv_blockgtq_package load(const std::string & path);

    const std::vector<uint8_t> & producer() const;
    const std::vector<uint8_t> & consumer() const;
    const std::vector<uint8_t> & shared() const;
    const std::string & sha256() const;

private:
    std::vector<uint8_t> producer_;
    std::vector<uint8_t> consumer_;
    std::vector<uint8_t> shared_;
    std::string sha256_;
};

class llama_kv_blockgtq_runtime {
public:
    static std::unique_ptr<llama_kv_blockgtq_runtime> create(
        const std::string & package_path,
        ggml_backend_t backend,
        std::string * error);

    ~llama_kv_blockgtq_runtime();

    ggml_tensor * producer() const;
    ggml_tensor * consumer() const;
    ggml_tensor * shared() const;
    ggml_tensor * history() const;
    ggml_backend_t backend() const;

    bool prepare(int32_t token_start, int32_t token_count, std::string * error);
    void finish();
    void fail();
    bool valid() const;
    int32_t sequence_length() const;

private:
    struct impl;
    explicit llama_kv_blockgtq_runtime(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};
