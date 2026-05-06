#include "llama-kv-lowrank.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

using json = nlohmann::ordered_json;

struct llama_kv_lowrank_npz_entry {
    std::string name;
    std::vector<uint8_t> data;
    uint32_t crc = 0;
    uint32_t offset = 0;
};

static std::string llama_kv_lowrank_resolve_path(const std::string & manifest_path, const std::string & value) {
    if (value.empty()) {
        return value;
    }

    const std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path.string();
    }

    const std::filesystem::path base = std::filesystem::path(manifest_path).parent_path();
    if (base.empty()) {
        return path.string();
    }

    return (base / path).lexically_normal().string();
}

static int32_t llama_kv_lowrank_json_i32(const json & obj, const char * key, int32_t fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return fallback;
    }

    return obj.at(key).get<int32_t>();
}

static std::string llama_kv_lowrank_json_string(const json & obj, const char * key, const std::string & fallback = "") {
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return fallback;
    }

    return obj.at(key).get<std::string>();
}

static size_t llama_kv_lowrank_dtype_size(const std::string & dtype) {
    if (dtype == "f16") {
        return 2;
    }
    if (dtype == "f32") {
        return 4;
    }

    return 0;
}

static bool llama_kv_lowrank_read_binary_file(const std::string & path, std::vector<uint8_t> & out, std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return fail("failed to open low-rank KV basis file: " + path);
    }

    const std::streamoff size = file.tellg();
    if (size < 0) {
        return fail("failed to determine low-rank KV basis file size: " + path);
    }

    out.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!out.empty() && !file.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()))) {
        return fail("failed to read low-rank KV basis file: " + path);
    }

    return true;
}

static void llama_kv_lowrank_push_u16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
}

static void llama_kv_lowrank_push_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xffu));
}

static uint32_t llama_kv_lowrank_crc32(const uint8_t * data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return crc ^ 0xffffffffu;
}

static std::vector<uint8_t> llama_kv_lowrank_make_npy_f32(
        const std::vector<float> & values,
        int32_t n_tokens,
        int32_t d_kv) {
    std::vector<uint8_t> out;
    out.insert(out.end(), { 0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00 });

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        + std::to_string(n_tokens) + ", " + std::to_string(d_kv) + "), }";
    const size_t prefix = 10;
    const size_t padded_len = ((prefix + header.size() + 1 + 15) / 16) * 16 - prefix;
    header.resize(padded_len - 1, ' ');
    header.push_back('\n');

    llama_kv_lowrank_push_u16(out, static_cast<uint16_t>(header.size()));
    out.insert(out.end(), header.begin(), header.end());

    const uint8_t * data = reinterpret_cast<const uint8_t *>(values.data());
    out.insert(out.end(), data, data + values.size() * sizeof(float));
    return out;
}

static void llama_kv_lowrank_zip_local_header(std::vector<uint8_t> & out, const llama_kv_lowrank_npz_entry & entry) {
    llama_kv_lowrank_push_u32(out, 0x04034b50u);
    llama_kv_lowrank_push_u16(out, 20);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u32(out, entry.crc);
    llama_kv_lowrank_push_u32(out, static_cast<uint32_t>(entry.data.size()));
    llama_kv_lowrank_push_u32(out, static_cast<uint32_t>(entry.data.size()));
    llama_kv_lowrank_push_u16(out, static_cast<uint16_t>(entry.name.size()));
    llama_kv_lowrank_push_u16(out, 0);
    out.insert(out.end(), entry.name.begin(), entry.name.end());
}

static void llama_kv_lowrank_zip_central_header(std::vector<uint8_t> & out, const llama_kv_lowrank_npz_entry & entry) {
    llama_kv_lowrank_push_u32(out, 0x02014b50u);
    llama_kv_lowrank_push_u16(out, 20);
    llama_kv_lowrank_push_u16(out, 20);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u32(out, entry.crc);
    llama_kv_lowrank_push_u32(out, static_cast<uint32_t>(entry.data.size()));
    llama_kv_lowrank_push_u32(out, static_cast<uint32_t>(entry.data.size()));
    llama_kv_lowrank_push_u16(out, static_cast<uint16_t>(entry.name.size()));
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u16(out, 0);
    llama_kv_lowrank_push_u32(out, 0);
    llama_kv_lowrank_push_u32(out, entry.offset);
    out.insert(out.end(), entry.name.begin(), entry.name.end());
}

static float llama_kv_lowrank_basis_value(const std::vector<uint8_t> & basis, const std::string & dtype, size_t index) {
    if (dtype == "f16") {
        ggml_fp16_t value;
        std::memcpy(&value, basis.data() + index * sizeof(value), sizeof(value));
        return ggml_fp16_to_fp32(value);
    }

    float value;
    std::memcpy(&value, basis.data() + index * sizeof(value), sizeof(value));
    return value;
}

static llama_kv_lowrank_basis_layer_data * llama_kv_lowrank_find_basis_layer(llama_kv_lowrank_context & ctx, int32_t layer) {
    for (llama_kv_lowrank_basis_layer_data & item : ctx.basis.layers) {
        if (item.layer == layer) {
            return &item;
        }
    }

    return nullptr;
}

static llama_kv_lowrank_layer_state * llama_kv_lowrank_find_layer_state(llama_kv_lowrank_context & ctx, int32_t layer) {
    for (llama_kv_lowrank_layer_state & item : ctx.layers) {
        if (item.layer == layer) {
            return &item;
        }
    }

    return nullptr;
}

static llama_kv_lowrank_error_stats llama_kv_lowrank_compute_error_stats(
        const float * k_dense,
        const float * v_dense,
        const std::vector<float> & k_recon,
        const std::vector<float> & v_recon) {
    llama_kv_lowrank_error_stats stats;
    stats.n_values = k_recon.size();

    double k_sum_abs = 0.0;
    double v_sum_abs = 0.0;
    for (size_t i = 0; i < stats.n_values; ++i) {
        const float k_abs = std::fabs(k_dense[i] - k_recon[i]);
        const float v_abs = std::fabs(v_dense[i] - v_recon[i]);

        stats.k_max_abs = std::max(stats.k_max_abs, k_abs);
        stats.v_max_abs = std::max(stats.v_max_abs, v_abs);
        k_sum_abs += k_abs;
        v_sum_abs += v_abs;
    }

    if (stats.n_values > 0) {
        stats.k_mean_abs = static_cast<float>(k_sum_abs / static_cast<double>(stats.n_values));
        stats.v_mean_abs = static_cast<float>(v_sum_abs / static_cast<double>(stats.n_values));
    }

    return stats;
}

static void llama_kv_lowrank_accumulate_error_stats(
        llama_kv_lowrank_error_stats & dst,
        const llama_kv_lowrank_error_stats & src) {
    const double k_sum = static_cast<double>(dst.k_mean_abs) * static_cast<double>(dst.n_values)
                       + static_cast<double>(src.k_mean_abs) * static_cast<double>(src.n_values);
    const double v_sum = static_cast<double>(dst.v_mean_abs) * static_cast<double>(dst.n_values)
                       + static_cast<double>(src.v_mean_abs) * static_cast<double>(src.n_values);

    dst.n_values += src.n_values;
    dst.k_max_abs = std::max(dst.k_max_abs, src.k_max_abs);
    dst.v_max_abs = std::max(dst.v_max_abs, src.v_max_abs);

    if (dst.n_values > 0) {
        dst.k_mean_abs = static_cast<float>(k_sum / static_cast<double>(dst.n_values));
        dst.v_mean_abs = static_cast<float>(v_sum / static_cast<double>(dst.n_values));
    }
}

bool llama_kv_lowrank_validate(const llama_kv_lowrank_params & params, std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!params.enabled) {
        return true;
    }

    if (params.rank <= 0) {
        return fail("low-rank KV rank must be positive");
    }
    if (params.window <= 0) {
        return fail("low-rank KV recent window must be positive");
    }
    if (params.chunk <= 0) {
        return fail("low-rank KV chunk size must be positive");
    }
    if (params.sample_max_tokens <= 0) {
        return fail("low-rank KV sample max tokens must be positive");
    }
    if (params.window < params.chunk) {
        return fail("low-rank KV recent window must be greater than or equal to chunk size");
    }
    if (params.basis_path.empty()) {
        return fail("low-rank KV requires --kv-lowrank-basis-path for the first prototype");
    }

    const llama_kv_lowrank_basis_info info = llama_kv_lowrank_basis_probe(params.basis_path);
    if (!info.exists) {
        return fail("low-rank KV basis path does not exist: " + params.basis_path);
    }

    llama_kv_lowrank_basis_manifest manifest;
    std::string manifest_error;
    if (!llama_kv_lowrank_basis_manifest_load(params.basis_path, manifest, &manifest_error)) {
        return fail(manifest_error);
    }
    if (manifest.rank != params.rank) {
        return fail("low-rank KV basis manifest rank does not match --kv-lowrank-rank");
    }

    return true;
}

llama_kv_lowrank_basis_info llama_kv_lowrank_basis_probe(const std::string & path) {
    llama_kv_lowrank_basis_info info;
    info.path = path;

    if (path.empty()) {
        return info;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return info;
    }

    info.exists = true;
    info.size   = static_cast<size_t>(file.tellg());
    return info;
}

bool llama_kv_lowrank_basis_manifest_load(
        const std::string & path,
        llama_kv_lowrank_basis_manifest & out,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    std::ifstream file(path);
    if (!file) {
        return fail("failed to open low-rank KV basis manifest: " + path);
    }

    json manifest;
    try {
        manifest = json::parse(file);
    } catch (const std::exception & ex) {
        return fail("failed to parse low-rank KV basis manifest: " + std::string(ex.what()));
    }

    llama_kv_lowrank_basis_manifest next;
    next.path      = path;
    next.format    = llama_kv_lowrank_json_string(manifest, "format", next.format);
    next.dtype     = llama_kv_lowrank_json_string(manifest, "dtype", next.dtype);
    next.layout    = llama_kv_lowrank_json_string(manifest, "layout", next.layout);
    next.version   = llama_kv_lowrank_json_i32(manifest, "version", next.version);
    next.rank      = llama_kv_lowrank_json_i32(manifest, "rank", next.rank);
    next.n_layer   = llama_kv_lowrank_json_i32(manifest, "n_layer", next.n_layer);
    next.head_dim  = llama_kv_lowrank_json_i32(manifest, "head_dim", next.head_dim);
    next.n_head_kv = llama_kv_lowrank_json_i32(manifest, "n_head_kv", next.n_head_kv);

    if (next.format != "whlr-kv-basis") {
        return fail("unsupported low-rank KV basis manifest format: " + next.format);
    }
    if (next.version != 1) {
        return fail("unsupported low-rank KV basis manifest version: " + std::to_string(next.version));
    }
    if (next.rank <= 0) {
        return fail("low-rank KV basis manifest rank must be positive");
    }
    if (next.layout != "row-major") {
        return fail("unsupported low-rank KV basis layout: " + next.layout);
    }
    const size_t dtype_size = llama_kv_lowrank_dtype_size(next.dtype);
    if (dtype_size == 0) {
        return fail("unsupported low-rank KV basis dtype: " + next.dtype);
    }
    if (!manifest.contains("layers") || !manifest.at("layers").is_array()) {
        return fail("low-rank KV basis manifest must contain a layers array");
    }

    const bool check_basis_size = next.head_dim > 0 && next.n_head_kv > 0;
    const size_t expected_basis_size = check_basis_size
        ? static_cast<size_t>(next.rank) * static_cast<size_t>(next.n_head_kv) * static_cast<size_t>(next.head_dim) * dtype_size
        : 0;

    for (const json & layer_json : manifest.at("layers")) {
        llama_kv_lowrank_basis_layer layer;
        layer.layer = llama_kv_lowrank_json_i32(layer_json, "layer", -1);
        if (layer.layer < 0) {
            return fail("low-rank KV basis manifest layer index must be non-negative");
        }

        const std::string k_path = llama_kv_lowrank_resolve_path(path, llama_kv_lowrank_json_string(layer_json, "k"));
        const std::string v_path = llama_kv_lowrank_resolve_path(path, llama_kv_lowrank_json_string(layer_json, "v"));
        if (k_path.empty() || v_path.empty()) {
            return fail("low-rank KV basis manifest layer is missing k or v path");
        }

        layer.k = llama_kv_lowrank_basis_probe(k_path);
        layer.v = llama_kv_lowrank_basis_probe(v_path);
        if (!layer.k.exists) {
            return fail("low-rank KV K basis file does not exist: " + k_path);
        }
        if (!layer.v.exists) {
            return fail("low-rank KV V basis file does not exist: " + v_path);
        }
        if (check_basis_size && layer.k.size != expected_basis_size) {
            return fail("low-rank KV K basis file has unexpected size: " + k_path);
        }
        if (check_basis_size && layer.v.size != expected_basis_size) {
            return fail("low-rank KV V basis file has unexpected size: " + v_path);
        }

        next.layers.push_back(std::move(layer));
    }

    if (next.n_layer > 0 && static_cast<int32_t>(next.layers.size()) != next.n_layer) {
        return fail("low-rank KV basis manifest n_layer does not match layers size");
    }

    out = std::move(next);
    return true;
}

bool llama_kv_lowrank_basis_load_all(
        const std::string & path,
        llama_kv_lowrank_basis_data & out,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    llama_kv_lowrank_basis_manifest manifest;
    std::string manifest_error;
    if (!llama_kv_lowrank_basis_manifest_load(path, manifest, &manifest_error)) {
        return fail(manifest_error);
    }

    llama_kv_lowrank_basis_data next;
    next.manifest = std::move(manifest);
    next.layers.reserve(next.manifest.layers.size());

    for (const llama_kv_lowrank_basis_layer & layer_meta : next.manifest.layers) {
        llama_kv_lowrank_basis_layer_data layer;
        layer.layer = layer_meta.layer;

        std::string read_error;
        if (!llama_kv_lowrank_read_binary_file(layer_meta.k.path, layer.k, &read_error)) {
            return fail(read_error);
        }
        if (!llama_kv_lowrank_read_binary_file(layer_meta.v.path, layer.v, &read_error)) {
            return fail(read_error);
        }
        if (layer.k.size() != layer_meta.k.size) {
            return fail("low-rank KV K basis size changed while loading: " + layer_meta.k.path);
        }
        if (layer.v.size() != layer_meta.v.size) {
            return fail("low-rank KV V basis size changed while loading: " + layer_meta.v.path);
        }

        next.layers.push_back(std::move(layer));
    }

    out = std::move(next);
    return true;
}

bool llama_kv_lowrank_context_init(
        const llama_kv_lowrank_params & params,
        llama_kv_lowrank_context & out,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    llama_kv_lowrank_context next;
    next.params = params;

    if (!params.enabled) {
        out = std::move(next);
        return true;
    }

    std::string validate_error;
    if (!llama_kv_lowrank_validate(params, &validate_error)) {
        return fail(validate_error);
    }

    std::string load_error;
    if (!llama_kv_lowrank_basis_load_all(params.basis_path, next.basis, &load_error)) {
        return fail(load_error);
    }

    const int32_t d_kv = next.basis.manifest.head_dim * next.basis.manifest.n_head_kv;
    next.layers.reserve(next.basis.layers.size());
    for (const llama_kv_lowrank_basis_layer_data & layer_data : next.basis.layers) {
        llama_kv_lowrank_layer_state layer;
        layer.layer = layer_data.layer;
        layer.rank  = next.basis.manifest.rank;
        layer.d_kv  = d_kv;
        next.layers.push_back(std::move(layer));
    }

    out = std::move(next);
    return true;
}

bool llama_kv_lowrank_layer_append_projected_chunk(
        llama_kv_lowrank_layer_state & layer,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (layer.rank <= 0) {
        return fail("low-rank KV layer rank must be positive before appending a chunk");
    }
    if (n_tokens <= 0) {
        return fail("low-rank KV projected chunk must contain at least one token");
    }
    if (a_k == nullptr || a_v == nullptr) {
        return fail("low-rank KV projected chunk input pointers must not be null");
    }

    const size_t n_values = static_cast<size_t>(n_tokens) * static_cast<size_t>(layer.rank);
    layer.a_k.insert(layer.a_k.end(), a_k, a_k + n_values);
    layer.a_v.insert(layer.a_v.end(), a_v, a_v + n_values);
    layer.n_hist_tokens += n_tokens;
    layer.n_chunks += 1;

    return true;
}

bool llama_kv_lowrank_project_chunk(
        const llama_kv_lowrank_basis_manifest & manifest,
        const llama_kv_lowrank_basis_layer_data & basis,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::vector<float> & out_a_k,
        std::vector<float> & out_a_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    const int32_t rank = manifest.rank;
    const int32_t d_kv = manifest.head_dim * manifest.n_head_kv;
    const size_t dtype_size = llama_kv_lowrank_dtype_size(manifest.dtype);

    if (rank <= 0 || d_kv <= 0) {
        return fail("low-rank KV projection requires positive rank and d_kv");
    }
    if (dtype_size == 0) {
        return fail("low-rank KV projection has unsupported basis dtype: " + manifest.dtype);
    }
    if (n_tokens <= 0) {
        return fail("low-rank KV projection chunk must contain at least one token");
    }
    if (k_dense == nullptr || v_dense == nullptr) {
        return fail("low-rank KV projection dense input pointers must not be null");
    }

    const size_t expected_basis_size = static_cast<size_t>(rank) * static_cast<size_t>(d_kv) * dtype_size;
    if (basis.k.size() != expected_basis_size) {
        return fail("low-rank KV K basis size does not match projection shape");
    }
    if (basis.v.size() != expected_basis_size) {
        return fail("low-rank KV V basis size does not match projection shape");
    }

    out_a_k.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(rank), 0.0f);
    out_a_v.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(rank), 0.0f);

    for (int32_t t = 0; t < n_tokens; ++t) {
        for (int32_t r = 0; r < rank; ++r) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;

            for (int32_t c = 0; c < d_kv; ++c) {
                const size_t basis_index = static_cast<size_t>(r) * static_cast<size_t>(d_kv) + static_cast<size_t>(c);
                const size_t dense_index = static_cast<size_t>(t) * static_cast<size_t>(d_kv) + static_cast<size_t>(c);

                sum_k += k_dense[dense_index] * llama_kv_lowrank_basis_value(basis.k, manifest.dtype, basis_index);
                sum_v += v_dense[dense_index] * llama_kv_lowrank_basis_value(basis.v, manifest.dtype, basis_index);
            }

            const size_t out_index = static_cast<size_t>(t) * static_cast<size_t>(rank) + static_cast<size_t>(r);
            out_a_k[out_index] = sum_k;
            out_a_v[out_index] = sum_v;
        }
    }

    return true;
}

bool llama_kv_lowrank_context_project_and_append(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx.enabled()) {
        return fail("low-rank KV context is not enabled");
    }

    llama_kv_lowrank_basis_layer_data * basis = llama_kv_lowrank_find_basis_layer(ctx, layer);
    if (basis == nullptr) {
        return fail("low-rank KV basis not found for layer: " + std::to_string(layer));
    }

    llama_kv_lowrank_layer_state * state = llama_kv_lowrank_find_layer_state(ctx, layer);
    if (state == nullptr) {
        return fail("low-rank KV layer state not found for layer: " + std::to_string(layer));
    }

    std::vector<float> a_k;
    std::vector<float> a_v;
    std::string project_error;
    if (!llama_kv_lowrank_project_chunk(ctx.basis.manifest, *basis, k_dense, v_dense, n_tokens, a_k, a_v, &project_error)) {
        return fail(project_error);
    }

    std::string append_error;
    if (!llama_kv_lowrank_layer_append_projected_chunk(*state, a_k.data(), a_v.data(), n_tokens, &append_error)) {
        return fail(append_error);
    }

    return true;
}

bool llama_kv_lowrank_context_project_append_reconstruct_error(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx.enabled()) {
        return fail("low-rank KV context is not enabled");
    }

    llama_kv_lowrank_basis_layer_data * basis = llama_kv_lowrank_find_basis_layer(ctx, layer);
    if (basis == nullptr) {
        return fail("low-rank KV basis not found for layer: " + std::to_string(layer));
    }

    llama_kv_lowrank_layer_state * state = llama_kv_lowrank_find_layer_state(ctx, layer);
    if (state == nullptr) {
        return fail("low-rank KV layer state not found for layer: " + std::to_string(layer));
    }

    std::vector<float> a_k;
    std::vector<float> a_v;
    std::string project_error;
    if (!llama_kv_lowrank_project_chunk(ctx.basis.manifest, *basis, k_dense, v_dense, n_tokens, a_k, a_v, &project_error)) {
        return fail(project_error);
    }

    std::vector<float> k_recon;
    std::vector<float> v_recon;
    std::string reconstruct_error;
    if (!llama_kv_lowrank_reconstruct_chunk(
                ctx.basis.manifest, *basis, a_k.data(), a_v.data(), n_tokens, k_recon, v_recon, &reconstruct_error)) {
        return fail(reconstruct_error);
    }

    out_stats = llama_kv_lowrank_compute_error_stats(k_dense, v_dense, k_recon, v_recon);
    out_stats.n_observed_tokens  = n_tokens;
    out_stats.n_projected_tokens = n_tokens;
    out_stats.n_chunks_projected = 1;

    std::string append_error;
    if (!llama_kv_lowrank_layer_append_projected_chunk(*state, a_k.data(), a_v.data(), n_tokens, &append_error)) {
        return fail(append_error);
    }

    return true;
}

bool llama_kv_lowrank_context_append_policy_project_reconstruct_error(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        llama_kv_lowrank_error_stats & out_stats,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    out_stats = {};
    out_stats.n_observed_tokens = n_tokens;

    if (!ctx.enabled()) {
        return fail("low-rank KV context is not enabled");
    }
    if (n_tokens <= 0) {
        return fail("low-rank KV policy append requires at least one token");
    }
    if (k_dense == nullptr || v_dense == nullptr) {
        return fail("low-rank KV policy append dense input pointers must not be null");
    }
    if (ctx.params.window <= 0 || ctx.params.chunk <= 0) {
        return fail("low-rank KV policy requires positive window and chunk");
    }

    llama_kv_lowrank_basis_layer_data * basis = llama_kv_lowrank_find_basis_layer(ctx, layer);
    if (basis == nullptr) {
        return fail("low-rank KV basis not found for layer: " + std::to_string(layer));
    }

    llama_kv_lowrank_layer_state * state = llama_kv_lowrank_find_layer_state(ctx, layer);
    if (state == nullptr) {
        return fail("low-rank KV layer state not found for layer: " + std::to_string(layer));
    }
    if (state->d_kv <= 0) {
        return fail("low-rank KV policy requires positive d_kv");
    }

    const size_t n_dense_values = static_cast<size_t>(n_tokens) * static_cast<size_t>(state->d_kv);
    state->pending_k.insert(state->pending_k.end(), k_dense, k_dense + n_dense_values);
    state->pending_v.insert(state->pending_v.end(), v_dense, v_dense + n_dense_values);
    state->n_pending_tokens += n_tokens;

    const int32_t n_eligible = state->n_pending_tokens - ctx.params.window;
    if (n_eligible < ctx.params.chunk) {
        out_stats.n_pending_tokens = state->n_pending_tokens;
        return true;
    }

    const int32_t n_project = (n_eligible / ctx.params.chunk) * ctx.params.chunk;
    for (int32_t offset = 0; offset < n_project; offset += ctx.params.chunk) {
        const size_t dense_offset = static_cast<size_t>(offset) * static_cast<size_t>(state->d_kv);
        const float * k_chunk = state->pending_k.data() + dense_offset;
        const float * v_chunk = state->pending_v.data() + dense_offset;

        std::vector<float> a_k;
        std::vector<float> a_v;
        std::string project_error;
        if (!llama_kv_lowrank_project_chunk(
                    ctx.basis.manifest, *basis, k_chunk, v_chunk, ctx.params.chunk, a_k, a_v, &project_error)) {
            return fail(project_error);
        }

        std::vector<float> k_recon;
        std::vector<float> v_recon;
        std::string reconstruct_error;
        if (!llama_kv_lowrank_reconstruct_chunk(
                    ctx.basis.manifest, *basis, a_k.data(), a_v.data(), ctx.params.chunk,
                    k_recon, v_recon, &reconstruct_error)) {
            return fail(reconstruct_error);
        }

        const llama_kv_lowrank_error_stats chunk_stats =
            llama_kv_lowrank_compute_error_stats(k_chunk, v_chunk, k_recon, v_recon);
        llama_kv_lowrank_accumulate_error_stats(out_stats, chunk_stats);

        std::string append_error;
        if (!llama_kv_lowrank_layer_append_projected_chunk(
                    *state, a_k.data(), a_v.data(), ctx.params.chunk, &append_error)) {
            return fail(append_error);
        }

        out_stats.n_projected_tokens += ctx.params.chunk;
        out_stats.n_chunks_projected += 1;
    }

    const size_t n_erase_values = static_cast<size_t>(n_project) * static_cast<size_t>(state->d_kv);
    state->pending_k.erase(state->pending_k.begin(), state->pending_k.begin() + n_erase_values);
    state->pending_v.erase(state->pending_v.begin(), state->pending_v.begin() + n_erase_values);
    state->n_pending_tokens -= n_project;
    out_stats.n_pending_tokens = state->n_pending_tokens;

    return true;
}

bool llama_kv_lowrank_reconstruct_chunk(
        const llama_kv_lowrank_basis_manifest & manifest,
        const llama_kv_lowrank_basis_layer_data & basis,
        const float * a_k,
        const float * a_v,
        int32_t n_tokens,
        std::vector<float> & out_k_dense,
        std::vector<float> & out_v_dense,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    const int32_t rank = manifest.rank;
    const int32_t d_kv = manifest.head_dim * manifest.n_head_kv;
    const size_t dtype_size = llama_kv_lowrank_dtype_size(manifest.dtype);

    if (rank <= 0 || d_kv <= 0) {
        return fail("low-rank KV reconstruction requires positive rank and d_kv");
    }
    if (dtype_size == 0) {
        return fail("low-rank KV reconstruction has unsupported basis dtype: " + manifest.dtype);
    }
    if (n_tokens <= 0) {
        return fail("low-rank KV reconstruction chunk must contain at least one token");
    }
    if (a_k == nullptr || a_v == nullptr) {
        return fail("low-rank KV reconstruction input pointers must not be null");
    }

    const size_t expected_basis_size = static_cast<size_t>(rank) * static_cast<size_t>(d_kv) * dtype_size;
    if (basis.k.size() != expected_basis_size) {
        return fail("low-rank KV K basis size does not match reconstruction shape");
    }
    if (basis.v.size() != expected_basis_size) {
        return fail("low-rank KV V basis size does not match reconstruction shape");
    }

    out_k_dense.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(d_kv), 0.0f);
    out_v_dense.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(d_kv), 0.0f);

    for (int32_t t = 0; t < n_tokens; ++t) {
        for (int32_t c = 0; c < d_kv; ++c) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;

            for (int32_t r = 0; r < rank; ++r) {
                const size_t basis_index = static_cast<size_t>(r) * static_cast<size_t>(d_kv) + static_cast<size_t>(c);
                const size_t a_index = static_cast<size_t>(t) * static_cast<size_t>(rank) + static_cast<size_t>(r);

                sum_k += a_k[a_index] * llama_kv_lowrank_basis_value(basis.k, manifest.dtype, basis_index);
                sum_v += a_v[a_index] * llama_kv_lowrank_basis_value(basis.v, manifest.dtype, basis_index);
            }

            const size_t out_index = static_cast<size_t>(t) * static_cast<size_t>(d_kv) + static_cast<size_t>(c);
            out_k_dense[out_index] = sum_k;
            out_v_dense[out_index] = sum_v;
        }
    }

    return true;
}

void llama_kv_lowrank_layer_clear(llama_kv_lowrank_layer_state & layer) {
    layer.n_hist_tokens = 0;
    layer.n_chunks = 0;
    layer.n_pending_tokens = 0;
    layer.n_sample_tokens = 0;
    layer.a_k.clear();
    layer.a_v.clear();
    layer.pending_k.clear();
    layer.pending_v.clear();
    layer.sample_k.clear();
    layer.sample_v.clear();
}

size_t llama_kv_lowrank_layer_memory_bytes(const llama_kv_lowrank_layer_state & layer) {
    return (layer.a_k.size() + layer.a_v.size()) * sizeof(float);
}

size_t llama_kv_lowrank_context_history_memory_bytes(const llama_kv_lowrank_context & ctx) {
    size_t total = 0;
    for (const llama_kv_lowrank_layer_state & layer : ctx.layers) {
        total += llama_kv_lowrank_layer_memory_bytes(layer);
    }
    return total;
}

bool llama_kv_lowrank_context_collect_samples(
        llama_kv_lowrank_context & ctx,
        int32_t layer,
        const float * k_dense,
        const float * v_dense,
        int32_t n_tokens,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx.enabled()) {
        return fail("low-rank KV context is not enabled");
    }
    if (ctx.params.samples_path.empty()) {
        return true;
    }
    if (n_tokens <= 0) {
        return fail("low-rank KV sample collection requires at least one token");
    }
    if (k_dense == nullptr || v_dense == nullptr) {
        return fail("low-rank KV sample collection dense input pointers must not be null");
    }

    llama_kv_lowrank_layer_state * state = llama_kv_lowrank_find_layer_state(ctx, layer);
    if (state == nullptr) {
        return fail("low-rank KV layer state not found for layer: " + std::to_string(layer));
    }
    if (state->d_kv <= 0) {
        return fail("low-rank KV sample collection requires positive d_kv");
    }
    if (state->n_sample_tokens >= ctx.params.sample_max_tokens) {
        return true;
    }

    const int32_t n_take = std::min(n_tokens, ctx.params.sample_max_tokens - state->n_sample_tokens);
    const size_t n_values = static_cast<size_t>(n_take) * static_cast<size_t>(state->d_kv);
    state->sample_k.insert(state->sample_k.end(), k_dense, k_dense + n_values);
    state->sample_v.insert(state->sample_v.end(), v_dense, v_dense + n_values);
    state->n_sample_tokens += n_take;
    return true;
}

bool llama_kv_lowrank_context_write_samples_npz(
        const llama_kv_lowrank_context & ctx,
        const std::string & path,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (path.empty()) {
        return true;
    }

    std::vector<llama_kv_lowrank_npz_entry> entries;
    for (const llama_kv_lowrank_layer_state & layer : ctx.layers) {
        if (layer.n_sample_tokens <= 0) {
            continue;
        }
        if (layer.d_kv <= 0) {
            return fail("low-rank KV sample export requires positive d_kv");
        }

        const std::string prefix = "layer_" + (layer.layer < 10 ? std::string("00") :
                (layer.layer < 100 ? std::string("0") : std::string())) + std::to_string(layer.layer);

        llama_kv_lowrank_npz_entry k_entry;
        k_entry.name = prefix + ".k.npy";
        k_entry.data = llama_kv_lowrank_make_npy_f32(layer.sample_k, layer.n_sample_tokens, layer.d_kv);
        k_entry.crc = llama_kv_lowrank_crc32(k_entry.data.data(), k_entry.data.size());
        entries.push_back(std::move(k_entry));

        llama_kv_lowrank_npz_entry v_entry;
        v_entry.name = prefix + ".v.npy";
        v_entry.data = llama_kv_lowrank_make_npy_f32(layer.sample_v, layer.n_sample_tokens, layer.d_kv);
        v_entry.crc = llama_kv_lowrank_crc32(v_entry.data.data(), v_entry.data.size());
        entries.push_back(std::move(v_entry));
    }

    if (entries.empty()) {
        return true;
    }

    std::vector<uint8_t> zip;
    for (llama_kv_lowrank_npz_entry & entry : entries) {
        if (zip.size() > std::numeric_limits<uint32_t>::max()) {
            return fail("low-rank KV sample npz is too large");
        }
        entry.offset = static_cast<uint32_t>(zip.size());
        llama_kv_lowrank_zip_local_header(zip, entry);
        zip.insert(zip.end(), entry.data.begin(), entry.data.end());
    }

    const uint32_t central_offset = static_cast<uint32_t>(zip.size());
    for (const llama_kv_lowrank_npz_entry & entry : entries) {
        llama_kv_lowrank_zip_central_header(zip, entry);
    }
    const uint32_t central_size = static_cast<uint32_t>(zip.size() - central_offset);

    llama_kv_lowrank_push_u32(zip, 0x06054b50u);
    llama_kv_lowrank_push_u16(zip, 0);
    llama_kv_lowrank_push_u16(zip, 0);
    llama_kv_lowrank_push_u16(zip, static_cast<uint16_t>(entries.size()));
    llama_kv_lowrank_push_u16(zip, static_cast<uint16_t>(entries.size()));
    llama_kv_lowrank_push_u32(zip, central_size);
    llama_kv_lowrank_push_u32(zip, central_offset);
    llama_kv_lowrank_push_u16(zip, 0);

    const std::filesystem::path out_path(path);
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return fail("failed to open low-rank KV sample npz for writing: " + path);
    }
    file.write(reinterpret_cast<const char *>(zip.data()), static_cast<std::streamsize>(zip.size()));
    if (!file) {
        return fail("failed to write low-rank KV sample npz: " + path);
    }

    return true;
}
