#include "htp-debug.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml-impl.h"  // for GGML_LOG_* macros
#include "ggml.h"

// ============================================================
// Environment flag helpers
// ============================================================

bool env_flag_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && strcmp(value, "0") != 0;
}

bool htp_debug_mul_mat_enabled() {
    static const bool enabled = env_flag_enabled("HTP_DEBUG_MUL_MAT");
    return enabled;
}

bool htp_debug_logits_enabled() {
    static const bool enabled = getenv("HTP_DEBUG_LOGITS") != nullptr;
    return enabled;
}

bool htp_debug_rpcmem_enabled() {
    static const bool enabled = getenv("HTP_DEBUG_RPCMEM") != nullptr;
    return enabled;
}

bool htp_force_cpu_output_layer_enabled() {
    static const bool enabled = env_flag_enabled("HTP_FORCE_CPU_OUTPUT_LAYER");
    return enabled;
}

bool htp_debug_result_norm_enabled() {
    static const bool enabled = getenv("HTP_DEBUG_RESULT_NORM") != nullptr;
    return enabled;
}

namespace {

bool str_matches_filter_terms(const char * filter, const char * value) {
    if (!filter || !value || !*value) {
        return false;
    }

    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '|') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }

        const char * start = p;
        while (*p && *p != ',' && *p != '|') {
            ++p;
        }

        const size_t len = (size_t) (p - start);
        if (len > 0) {
            std::string needle(start, len);
            if (strstr(value, needle.c_str()) != nullptr) {
                return true;
            }
        }
    }

    return false;
}

bool str_matches_any_term(const char * value, const char * const * terms, size_t n_terms) {
    if (!value || !*value) {
        return false;
    }

    for (size_t i = 0; i < n_terms; ++i) {
        const char * term = terms[i];
        if (term && strstr(value, term) != nullptr) {
            return true;
        }
    }

    return false;
}

} // namespace

bool ggml_backend_htp_should_force_cpu_tensor(const ggml_tensor * tensor) {
    if (!tensor || tensor->name[0] == '\0') {
        return false;
    }

    // Backend-owned debug placement policy for Qwen3.5 recurrent / DeltaNet nodes.
    // This helper deliberately stays name-based for now: the immediate goal is to get
    // a clean "true CPU rebind before alloc" experiment, not to commit to a final
    // graph-structure-aware scheduler design.
    static const char * const linear_attn_debug_terms[] = {
        "linear_attn",
        "conv_states",
        "conv_input",
        "conv_output",
        "q_conv",
        "k_conv",
        "v_conv",
        "predelta",
        "z_silu",
        "attn_output_normed",
        "attn_output_gated",
        "final_output",
        "state_predelta",
        "state_update_target",
        "last_conv_states",
        "new_state",
        "output_state",
        "ssm_alpha",
        "ssm_beta",
        "ssm_scan",
        "gated_delta_net",
        "deltanet",
        "dnet_add_",
        "__fgdn_ar__",
        "__fgdn_ch__",
    };

    return str_matches_any_term(
            tensor->name,
            linear_attn_debug_terms,
            sizeof(linear_attn_debug_terms) / sizeof(linear_attn_debug_terms[0]));
}

bool htp_debug_mul_mat_target_enabled(const ggml_tensor * dst) {
    const char * filter = getenv("HTP_DEBUG_MUL_MAT_FILTER");
    if (!filter || !dst || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }

    if (dst->name[0] != '\0' && str_matches_filter_terms(filter, dst->name)) {
        return true;
    }

    const ggml_tensor * weight = dst->src[0];
    if (weight && weight->name[0] != '\0' && str_matches_filter_terms(filter, weight->name)) {
        return true;
    }

    const ggml_tensor * act = dst->src[1];
    if (act && act->name[0] != '\0' && str_matches_filter_terms(filter, act->name)) {
        return true;
    }

    return false;
}

// ============================================================
// Tensor identification helpers
// ============================================================

bool is_output_layer_mul_mat(const ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * act    = dst->src[1];
    if (!weight || !act) {
        return false;
    }

    const bool weight_is_output = weight->name[0] != '\0' && strstr(weight->name, "token_embd") != nullptr;
    const bool dst_is_output    = dst->name[0] != '\0' && strstr(dst->name, "result_output") != nullptr;
    const bool act_is_norm      = act->name[0] != '\0' && strstr(act->name, "result_norm") != nullptr;

    return weight_is_output || (dst_is_output && act_is_norm);
}

bool is_tied_token_embd_output_mul_mat(const ggml_tensor * dst) {
    if (!is_output_layer_mul_mat(dst)) {
        return false;
    }

    const ggml_tensor * weight = dst->src[0];
    return weight && weight->name[0] != '\0' && strstr(weight->name, "token_embd") != nullptr;
}

bool is_named_mul_mat(const ggml_tensor * dst, const char * name) {
    if (!dst || dst->op != GGML_OP_MUL_MAT || dst->name[0] == '\0' || !name) {
        return false;
    }
    return strcmp(dst->name, name) == 0;
}

bool should_debug_named_mul_mat_once(const ggml_tensor * dst, const char * name, bool & printed) {
    if (printed || !is_named_mul_mat(dst, name)) {
        return false;
    }
    printed = true;
    return true;
}

bool should_debug_ffn_up_once(const ggml_tensor * dst) {
    static bool printed = false;
    return should_debug_named_mul_mat_once(dst, "ffn_up-0", printed);
}

bool should_debug_ffn_gate_once(const ggml_tensor * dst) {
    static bool printed = false;
    return should_debug_named_mul_mat_once(dst, "ffn_gate-0", printed);
}

bool should_debug_ffn_out_once(const ggml_tensor * dst) {
    static bool printed = false;
    return should_debug_named_mul_mat_once(dst, "ffn_out-0", printed);
}

bool should_debug_mul_mat_target_once(const ggml_tensor * dst) {
    static std::unordered_set<std::string> printed;

    if (!htp_debug_mul_mat_target_enabled(dst)) {
        return false;
    }

    const char * name = (dst && dst->name[0] != '\0') ? dst->name : nullptr;
    const ggml_tensor * weight = dst ? dst->src[0] : nullptr;
    const char * wname = (weight && weight->name[0] != '\0') ? weight->name : nullptr;

    std::string key = name ? name : (wname ? wname : "(unnamed_mul_mat)");
    return printed.insert(key).second;
}

// ============================================================
// Tensor printing utilities
// ============================================================

void print_tensor_f32_stats(const char * tag, const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_F32 || t->data == nullptr) {
        GGML_LOG_DEBUG("HTP STATS DEBUG: %s invalid tensor/type/data\n", tag ? tag : "unknown");
        return;
    }

    const int64_t n = ggml_nelements(t);
    if (n <= 0) {
        GGML_LOG_DEBUG("HTP STATS DEBUG: %s empty tensor\n", tag ? tag : "unknown");
        return;
    }

    const float * x = reinterpret_cast<const float *>(t->data);

    float min_v =  FLT_MAX;
    float max_v = -FLT_MAX;
    double sum  = 0.0;
    int64_t nan_count = 0;
    int64_t inf_count = 0;

    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (std::isnan(v)) {
            ++nan_count;
            continue;
        }
        if (std::isinf(v)) {
            ++inf_count;
            continue;
        }
        min_v = v < min_v ? v : min_v;
        max_v = v > max_v ? v : max_v;
        sum += v;
    }

    const int64_t finite_count = n - nan_count - inf_count;
    const double mean = finite_count > 0 ? sum / (double) finite_count : 0.0;
    GGML_LOG_DEBUG("HTP STATS DEBUG: %s name=%s ne=[%lld,%lld,%lld,%lld] n=%lld finite=%lld nan=%lld inf=%lld min=%.6f max=%.6f mean=%.6f\n",
            tag ? tag : "unknown", t->name,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (long long) n, (long long) finite_count, (long long) nan_count, (long long) inf_count,
            min_v, max_v, mean);

    const int preview = n < 16 ? (int) n : 16;
    GGML_LOG_DEBUG("HTP STATS DEBUG: %s first_%d =", tag ? tag : "unknown", preview);
    for (int i = 0; i < preview; ++i) {
        GGML_LOG_CONT(" %.6f", x[i]);
    }
    GGML_LOG_CONT("\n");
}

void print_tensor_shape_and_layout(const char * tag, const ggml_tensor * t) {
    fprintf(stderr,
            "HTP MUL_MAT DEBUG: %s name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nrows=%lld\n",
            tag, t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
            (long long) ggml_nrows(t));
}

void print_tensor_memory_detail(const char * tag, const ggml_tensor * t) {
    fprintf(stderr,
            "HTP MEMORY DEBUG: %s name=%s type=%s\n",
            tag, t->name, ggml_type_name(t->type));
    fprintf(stderr,
            "  shape: ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nrows=%lld\n",
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
            (long long) ggml_nrows(t));
    fprintf(stderr,
            "  data: %p, buffer: %p, buft: %s\n",
            t->data,
            t->buffer ? (void *) t->buffer : nullptr,
            t->buffer ? (t->buffer->buft ? t->buffer->buft->iface.get_name(t->buffer->buft) : "null") : "null");

    if (t->buffer && ggml_backend_buft_is_rpcmem(t->buffer->buft)) {
        const auto & mapper = ggml_backend_htp_context::instance()->mapper;
        auto [fd, offset] = mapper.get_tensor_mapping(t);
        void * base = t->buffer->context;
        fprintf(stderr,
                "  RPCMEM: fd=%d offset=%lld base=%p (data-base=%lld)\n",
                fd, (long long) offset, base, (long long)((char*)t->data - (char*)base));
    }
}

void print_rpcmem_mapping_detail(const char * tag, int fd, ssize_t offset, const ggml_tensor * t) {
    void * base = t->buffer ? t->buffer->context : nullptr;
    fprintf(stderr,
            "HTP RPCMEM DEBUG: %s tensor=%s fd=%d offset=%lld base=%p\n",
            tag, t->name, fd, (long long) offset, base);
    fprintf(stderr,
            "  tensor_data=%p (offset_from_base=%lld) size=%lld bytes\n",
            t->data, (long long)((char*)t->data - (char*)base), (long long)ggml_nbytes(t));
}

void print_logits_top_k(const ggml_tensor * output, int k) {
    if (!output || output->type != GGML_TYPE_F32) {
        GGML_LOG_DEBUG("HTP LOGITS DEBUG: invalid output tensor\n");
        return;
    }

    float * logits = (float *) output->data;
    int vocab_size = output->ne[0];
    int n_tokens = ggml_nrows(output);

    GGML_LOG_DEBUG("HTP LOGITS DEBUG: vocab_size=%d n_tokens=%d\n", vocab_size, n_tokens);

    if (n_tokens > 0) {
        GGML_LOG_DEBUG("HTP LOGITS DEBUG: token[0] top-%d logits:\n", k);

        std::vector<std::pair<float, int>> logit_idx;
        for (int i = 0; i < vocab_size && i < 1000; i++) {
            logit_idx.push_back({logits[i], i});
        }

        std::sort(logit_idx.begin(), logit_idx.end(), [](auto & a, auto & b) {
            return a.first > b.first;
        });

        for (int i = 0; i < k && i < (int)logit_idx.size(); i++) {
            GGML_LOG_DEBUG("  [%d] token_id=%d logit=%.6f\n", i, logit_idx[i].second, logit_idx[i].first);
        }

        bool has_nan = false;
        bool has_inf = false;
        for (int i = 0; i < vocab_size && i < 1000; i++) {
            if (std::isnan(logits[i])) has_nan = true;
            if (std::isinf(logits[i])) has_inf = true;
        }
        if (has_nan || has_inf) {
            GGML_LOG_WARN("HTP LOGITS DEBUG: WARNING - has_nan=%d has_inf=%d\n", has_nan, has_inf);
        }
    }
}

void print_split_activation_detail(const ggml_tensor * dst, const ggml_tensor * activation) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    GGML_LOG_DEBUG("HTP SPLIT DEBUG: dst=%s activation=%s\n", dst->name, activation->name);
    GGML_LOG_DEBUG("  dst: data=%p buffer=%s\n",
            dst->data, dst->buffer ? (dst->buffer->buft->iface.get_name(dst->buffer->buft)) : "null");
    GGML_LOG_DEBUG("  activation: data=%p buffer=%s\n",
            activation->data, activation->buffer ? (activation->buffer->buft->iface.get_name(activation->buffer->buft)) : "null");

    if (dst->buffer && ggml_backend_buft_is_rpcmem(dst->buffer->buft)) {
        auto [fd, offset] = mapper.get_tensor_mapping(dst);
        GGML_LOG_DEBUG("  dst RPCMEM: fd=%d offset=%lld\n", fd, (long long)offset);
    }
    if (activation->buffer && ggml_backend_buft_is_rpcmem(activation->buffer->buft)) {
        auto [fd, offset] = mapper.get_tensor_mapping(activation);
        GGML_LOG_DEBUG("  activation RPCMEM: fd=%d offset=%lld\n", fd, (long long)offset);
    }
}
