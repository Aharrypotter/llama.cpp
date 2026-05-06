#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Backend-agnostic core for the Windowed Historical Low-Rank KV prototype.
//
// This header intentionally lives in src/ so llama core code can call the CPU
// reference projection/reconstruction path without depending on common/.
struct llama_kv_lowrank_params {
    bool        enabled     = false;
    int32_t     rank        = 32;
    int32_t     window      = 256;
    int32_t     chunk       = 64;
    int32_t     sample_max_tokens = 4096;
    bool        reconstruct = true;
    std::string basis_path;
    std::string samples_path;
};

struct llama_kv_lowrank_basis_info {
    std::string path;
    size_t      size = 0;
    bool        exists = false;
};

struct llama_kv_lowrank_basis_layer {
    int32_t layer = -1;

    llama_kv_lowrank_basis_info k;
    llama_kv_lowrank_basis_info v;
};

struct llama_kv_lowrank_basis_layer_data {
    int32_t layer = -1;

    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
};

struct llama_kv_lowrank_basis_manifest {
    std::string path;
    std::string format = "whlr-kv-basis";
    std::string dtype  = "f16";
    std::string layout = "row-major";

    int32_t version   = 1;
    int32_t rank      = 0;
    int32_t n_layer   = 0;
    int32_t head_dim  = 0;
    int32_t n_head_kv = 0;

    std::vector<llama_kv_lowrank_basis_layer> layers;
};

struct llama_kv_lowrank_basis_data {
    llama_kv_lowrank_basis_manifest manifest;
    std::vector<llama_kv_lowrank_basis_layer_data> layers;
};

struct llama_kv_lowrank_layer_state {
    int32_t layer         = -1;
    int32_t n_hist_tokens = 0;
    int32_t n_chunks      = 0;
    int32_t rank          = 0;
    int32_t d_kv          = 0;
    int32_t n_pending_tokens = 0;
    int32_t n_sample_tokens  = 0;

    std::vector<float> a_k;
    std::vector<float> a_v;
    std::vector<float> pending_k;
    std::vector<float> pending_v;
    std::vector<float> sample_k;
    std::vector<float> sample_v;
};

struct llama_kv_lowrank_context {
    llama_kv_lowrank_params params;
    llama_kv_lowrank_basis_data basis;
    std::vector<llama_kv_lowrank_layer_state> layers;

    bool enabled() const {
        return params.enabled;
    }
};

struct llama_kv_lowrank_error_stats {
    size_t n_values = 0;

    int32_t n_observed_tokens  = 0;
    int32_t n_projected_tokens = 0;
    int32_t n_pending_tokens   = 0;
    int32_t n_chunks_projected = 0;

    float k_max_abs  = 0.0f;
    float k_mean_abs = 0.0f;
    float v_max_abs  = 0.0f;
    float v_mean_abs = 0.0f;
};

bool llama_kv_lowrank_validate(const llama_kv_lowrank_params & params, std::string * err = nullptr);

llama_kv_lowrank_basis_info llama_kv_lowrank_basis_probe(const std::string & path);

bool llama_kv_lowrank_basis_manifest_load(
        const std::string & path,
        llama_kv_lowrank_basis_manifest & out,
        std::string * err = nullptr);

bool llama_kv_lowrank_basis_load_all(
        const std::string & path,
        llama_kv_lowrank_basis_data & out,
        std::string * err = nullptr);

bool llama_kv_lowrank_context_init(
        const llama_kv_lowrank_params & params,
        llama_kv_lowrank_context & out,
        std::string * err = nullptr);

bool llama_kv_lowrank_layer_append_projected_chunk(
        llama_kv_lowrank_layer_state & layer,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::string * err = nullptr);

bool llama_kv_lowrank_project_chunk(
        const llama_kv_lowrank_basis_manifest & manifest,
        const llama_kv_lowrank_basis_layer_data & basis,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::vector<float> & out_a_k,
        std::vector<float> & out_a_v,
        std::string * err = nullptr);

bool llama_kv_lowrank_context_project_and_append(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err = nullptr);

bool llama_kv_lowrank_context_project_append_reconstruct_error(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err = nullptr);

bool llama_kv_lowrank_context_append_policy_project_reconstruct_error(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err = nullptr);

bool llama_kv_lowrank_reconstruct_chunk(
        const llama_kv_lowrank_basis_manifest & manifest,
        const llama_kv_lowrank_basis_layer_data & basis,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::vector<float> & out_k_dense,
        std::vector<float> & out_v_dense,
        std::string * err = nullptr);

void llama_kv_lowrank_layer_clear(llama_kv_lowrank_layer_state & layer);

size_t llama_kv_lowrank_layer_memory_bytes(const llama_kv_lowrank_layer_state & layer);

size_t llama_kv_lowrank_context_history_memory_bytes(const llama_kv_lowrank_context & ctx);

bool llama_kv_lowrank_context_collect_samples(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err = nullptr);

bool llama_kv_lowrank_context_write_samples_npz(
        const llama_kv_lowrank_context & ctx,
        const std::string & path,
        std::string * err = nullptr);
