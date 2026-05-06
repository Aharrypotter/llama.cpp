#pragma once

#include "common.h"
#include "llama-kv-lowrank.h"

// Common-layer compatibility wrappers for WHLR-KV.
//
// The algorithmic core lives in src/llama-kv-lowrank.*. This header keeps the
// existing common_kv_lowrank_* names available for CLI/common code while letting
// llama core call llama_kv_lowrank_* directly.
using common_kv_lowrank_params           = llama_kv_lowrank_params;
using common_kv_lowrank_basis_info       = llama_kv_lowrank_basis_info;
using common_kv_lowrank_basis_layer      = llama_kv_lowrank_basis_layer;
using common_kv_lowrank_basis_layer_data = llama_kv_lowrank_basis_layer_data;
using common_kv_lowrank_basis_manifest   = llama_kv_lowrank_basis_manifest;
using common_kv_lowrank_basis_data       = llama_kv_lowrank_basis_data;
using common_kv_lowrank_layer_state      = llama_kv_lowrank_layer_state;
using common_kv_lowrank_context          = llama_kv_lowrank_context;

common_kv_lowrank_params common_kv_lowrank_params_from_common(const common_params & params);

bool common_kv_lowrank_validate(const common_kv_lowrank_params & params, std::string * err = nullptr);

common_kv_lowrank_basis_info common_kv_lowrank_basis_probe(const std::string & path);

bool common_kv_lowrank_basis_manifest_load(
        const std::string & path,
        common_kv_lowrank_basis_manifest & out,
        std::string * err = nullptr);

bool common_kv_lowrank_basis_load_all(
        const std::string & path,
        common_kv_lowrank_basis_data & out,
        std::string * err = nullptr);

bool common_kv_lowrank_context_init(
        const common_kv_lowrank_params & params,
        common_kv_lowrank_context & out,
        std::string * err = nullptr);

bool common_kv_lowrank_layer_append_projected_chunk(
        common_kv_lowrank_layer_state & layer,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::string * err = nullptr);

bool common_kv_lowrank_project_chunk(
        const common_kv_lowrank_basis_manifest & manifest,
        const common_kv_lowrank_basis_layer_data & basis,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::vector<float> & out_a_k,
        std::vector<float> & out_a_v,
        std::string * err = nullptr);

bool common_kv_lowrank_context_project_and_append(
        common_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err = nullptr);

bool common_kv_lowrank_context_append_policy_project_reconstruct_error(
        common_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err = nullptr);

bool common_kv_lowrank_reconstruct_chunk(
        const common_kv_lowrank_basis_manifest & manifest,
        const common_kv_lowrank_basis_layer_data & basis,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::vector<float> & out_k_dense,
        std::vector<float> & out_v_dense,
        std::string * err = nullptr);

void common_kv_lowrank_layer_clear(common_kv_lowrank_layer_state & layer);

size_t common_kv_lowrank_layer_memory_bytes(const common_kv_lowrank_layer_state & layer);

size_t common_kv_lowrank_context_history_memory_bytes(const common_kv_lowrank_context & ctx);

void common_kv_lowrank_log_context(const common_kv_lowrank_context & ctx);

void common_kv_lowrank_log_config(const common_kv_lowrank_params & params);
