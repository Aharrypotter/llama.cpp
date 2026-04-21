#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operator entry functions
void ggml_compute_forward_acc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_arange(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argmax(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argsort(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_clamp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_concat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cont(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_count_equal(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cpy(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_div(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_group_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col_back_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_l2_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_leaky_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom2(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom3(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mean(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_opt_step_adamw(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_out_prod(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad_reflect_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_permute(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_reshape(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv6(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_scale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_silu_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ssm_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ssm_scan(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sub(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_timestep_embedding(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_transpose(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_upscale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_view(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// Dispatch function
void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

// Externally defined (may be in NPU ops)
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
