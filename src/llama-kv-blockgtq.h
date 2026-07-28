#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama-memory.h"

class llama_batch_allocr;
class llama_io_read_i;
class llama_io_write_i;

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
    bool reset(bool clear_data, std::string * error);
    bool truncate(int32_t sequence_length, std::string * error);
    bool valid() const;
    int32_t sequence_length() const;

private:
    struct impl;
    explicit llama_kv_blockgtq_runtime(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};

class llama_memory_blockgtq final : public llama_memory_i {
public:
    llama_memory_blockgtq(
        llama_kv_blockgtq_runtime * runtime,
        uint32_t n_seq_max);

    llama_memory_context_ptr init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(
        llama_context * lctx,
        bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) override;
    void seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos shift) override;
    void seq_div(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown()
        const override;

    void state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id = -1,
        llama_state_seq_flags flags = 0) const override;
    void state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id = -1,
        llama_state_seq_flags flags = 0) override;

private:
    void invalidate_unsupported(const char * operation);

    llama_kv_blockgtq_runtime * runtime_;
};
