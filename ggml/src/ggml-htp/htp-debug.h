#pragma once

#include <stdbool.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Environment flag helpers
bool env_flag_enabled(const char * name);

// Debug enable/disable flags (backed by environment variables)
bool htp_debug_mul_mat_enabled(void);
bool htp_debug_logits_enabled(void);
bool htp_debug_rpcmem_enabled(void);
bool htp_force_cpu_output_layer_enabled(void);
bool htp_debug_result_norm_enabled(void);
bool htp_debug_mul_mat_target_enabled(const struct ggml_tensor * dst);

// Tensor identification helpers
bool is_output_layer_mul_mat(const struct ggml_tensor * dst);
bool is_tied_token_embd_output_mul_mat(const struct ggml_tensor * dst);
bool is_named_mul_mat(const struct ggml_tensor * dst, const char * name);

#ifdef __cplusplus
// C++ only: these use references
bool should_debug_named_mul_mat_once(const ggml_tensor * dst, const char * name, bool & printed);
bool should_debug_ffn_up_once(const ggml_tensor * dst);
bool should_debug_ffn_gate_once(const ggml_tensor * dst);
bool should_debug_ffn_out_once(const ggml_tensor * dst);
bool should_debug_mul_mat_target_once(const ggml_tensor * dst);

// Tensor printing utilities
void print_tensor_f32_stats(const char * tag, const ggml_tensor * t);
void print_tensor_shape_and_layout(const char * tag, const ggml_tensor * t);
void print_tensor_memory_detail(const char * tag, const ggml_tensor * t);
void print_rpcmem_mapping_detail(const char * tag, int fd, ssize_t offset, const ggml_tensor * t);
void print_logits_top_k(const ggml_tensor * output, int k);
void print_split_activation_detail(const ggml_tensor * dst, const ggml_tensor * activation);
#endif

#ifdef __cplusplus
}
#endif
