#include "kv_lowrank.h"

#include "log.h"

common_kv_lowrank_params common_kv_lowrank_params_from_common(const common_params & params) {
    common_kv_lowrank_params out;
    out.enabled     = params.kv_lowrank;
    out.rank        = params.kv_lowrank_rank;
    out.rank_k      = params.kv_lowrank_rank_k;
    out.rank_v      = params.kv_lowrank_rank_v;
    out.window      = params.kv_lowrank_window;
    out.chunk       = params.kv_lowrank_chunk;
    out.sample_max_tokens = params.kv_lowrank_sample_max_tokens;
    out.reconstruct = params.kv_lowrank_reconstruct;
    out.reconstruct_cache = params.kv_lowrank_reconstruct_cache;
    out.basis_path  = params.kv_lowrank_basis_path;
    out.samples_path = params.kv_lowrank_samples_path;
    return out;
}

bool common_kv_lowrank_validate(const common_kv_lowrank_params & params, std::string * err) {
    return llama_kv_lowrank_validate(params, err);
}

common_kv_lowrank_basis_info common_kv_lowrank_basis_probe(const std::string & path) {
    return llama_kv_lowrank_basis_probe(path);
}

bool common_kv_lowrank_basis_manifest_load(
        const std::string & path,
        common_kv_lowrank_basis_manifest & out,
        std::string * err) {
    return llama_kv_lowrank_basis_manifest_load(path, out, err);
}

bool common_kv_lowrank_basis_load_all(
        const std::string & path,
        common_kv_lowrank_basis_data & out,
        std::string * err) {
    return llama_kv_lowrank_basis_load_all(path, out, err);
}

bool common_kv_lowrank_context_init(
        const common_kv_lowrank_params & params,
        common_kv_lowrank_context & out,
        std::string * err) {
    return llama_kv_lowrank_context_init(params, out, err);
}

bool common_kv_lowrank_layer_append_projected_chunk(
        common_kv_lowrank_layer_state & layer,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::string * err) {
    return llama_kv_lowrank_layer_append_projected_chunk(layer, a_k, a_v, n_tokens, err);
}

bool common_kv_lowrank_project_chunk(
        const common_kv_lowrank_basis_manifest & manifest,
        const common_kv_lowrank_basis_layer_data & basis,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::vector<float> & out_a_k,
        std::vector<float> & out_a_v,
        std::string * err) {
    return llama_kv_lowrank_project_chunk(manifest, basis, k_dense, v_dense, n_tokens, out_a_k, out_a_v, err);
}

bool common_kv_lowrank_context_project_and_append(
        common_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err) {
    return llama_kv_lowrank_context_project_and_append(ctx, layer, k_dense, v_dense, n_tokens, err);
}

bool common_kv_lowrank_context_append_policy_project_reconstruct_error(
        common_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err) {
    return llama_kv_lowrank_context_append_policy_project_reconstruct_error(
            ctx, layer, k_dense, v_dense, n_tokens, out_stats, nullptr, err);
}

bool common_kv_lowrank_reconstruct_chunk(
        const common_kv_lowrank_basis_manifest & manifest,
        const common_kv_lowrank_basis_layer_data & basis,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::vector<float> & out_k_dense,
        std::vector<float> & out_v_dense,
        std::string * err) {
    return llama_kv_lowrank_reconstruct_chunk(manifest, basis, a_k, a_v, n_tokens, out_k_dense, out_v_dense, err);
}

void common_kv_lowrank_layer_clear(common_kv_lowrank_layer_state & layer) {
    llama_kv_lowrank_layer_clear(layer);
}

size_t common_kv_lowrank_layer_memory_bytes(const common_kv_lowrank_layer_state & layer) {
    return llama_kv_lowrank_layer_memory_bytes(layer);
}

size_t common_kv_lowrank_context_history_memory_bytes(const common_kv_lowrank_context & ctx) {
    return llama_kv_lowrank_context_history_memory_bytes(ctx);
}

void common_kv_lowrank_log_context(const common_kv_lowrank_context & ctx) {
    if (!ctx.enabled()) {
        return;
    }

    size_t basis_bytes = 0;
    for (const common_kv_lowrank_basis_layer_data & layer : ctx.basis.layers) {
        basis_bytes += layer.k.size();
        basis_bytes += layer.v.size();
    }

    LOG_INF("%s: loaded basis layers=%zu states=%zu total_basis_bytes=%zu\n",
            __func__,
            ctx.basis.layers.size(),
            ctx.layers.size(),
            basis_bytes);
    LOG_INF("%s: historical low-rank cache bytes=%zu\n",
            __func__,
            common_kv_lowrank_context_history_memory_bytes(ctx));
}

void common_kv_lowrank_log_config(const common_kv_lowrank_params & params) {
    if (!params.enabled) {
        return;
    }

    const common_kv_lowrank_basis_info info = common_kv_lowrank_basis_probe(params.basis_path);
    const int32_t rank_k = params.rank_k > 0 ? params.rank_k : params.rank;
    const int32_t rank_v = params.rank_v > 0 ? params.rank_v : params.rank;
    LOG_INF("%s: enabled rank=%d rank_k=%d rank_v=%d window=%d chunk=%d mode=%s reconstruct_cache=%s basis=%s",
            __func__,
            params.rank,
            rank_k,
            rank_v,
            params.window,
            params.chunk,
            params.reconstruct ? "reconstruct" : "direct",
            params.reconstruct_cache ? "on" : "off",
            params.basis_path.c_str());
    if (info.exists) {
        LOG_CNT(" (%zu bytes)\n", info.size);
    } else {
        LOG_CNT(" (missing)\n");
    }

    if (!params.samples_path.empty()) {
        LOG_INF("%s: dense KV samples will be written to %s max_tokens/layer=%d\n",
                __func__,
                params.samples_path.c_str(),
                params.sample_max_tokens);
    }

    common_kv_lowrank_basis_manifest manifest;
    std::string manifest_error;
    if (!common_kv_lowrank_basis_manifest_load(params.basis_path, manifest, &manifest_error)) {
        LOG_ERR("%s: %s\n", __func__, manifest_error.c_str());
        return;
    }

    LOG_INF("%s: manifest version=%d rank=%d rank_k=%d rank_v=%d layers=%zu dtype=%s layout=%s head_dim=%d n_head_kv=%d\n",
            __func__,
            manifest.version,
            manifest.rank,
            manifest.rank_k,
            manifest.rank_v,
            manifest.layers.size(),
            manifest.dtype.c_str(),
            manifest.layout.c_str(),
            manifest.head_dim,
            manifest.n_head_kv);
}
