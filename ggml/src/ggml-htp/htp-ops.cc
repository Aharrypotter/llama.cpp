#include "htp-ops.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <string.h>
#include <stdlib.h>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "ggml-impl.h"  // for GGML_LOG_* macros
#include "htp-debug.h"

////// Special headers intended for CPU-NPU communication. Keep them in sync with ops backend.
#include "message.h"
#include "op_reg.h"

namespace {

auto get_all_rpcmem_mappings(const ggml_tensor * dst) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    std::vector<std::pair<int, ssize_t>> mappings;
    if (dst->buffer && ggml_backend_buft_is_rpcmem(dst->buffer->buft)) {
        mappings.push_back(mapper.get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src && src->buffer && ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
            mappings.push_back(mapper.get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];  // TODO(hzx): better implementation

bool is_recurrent_weight_name(const char * wname) {
    if (!wname || wname[0] == '\0') {
        return false;
    }

    return std::strstr(wname, ".attn_qkv.weight")  != nullptr ||
           std::strstr(wname, ".attn_gate.weight") != nullptr ||
           std::strstr(wname, ".ssm_alpha.weight") != nullptr ||
           std::strstr(wname, ".ssm_beta.weight")  != nullptr ||
           std::strstr(wname, ".ssm_ba.weight")    != nullptr ||
           std::strstr(wname, ".ssm_out.weight")   != nullptr;
}

bool is_recurrent_common_layout_weight(const ggml_tensor * weight) {
    if (!weight || weight->name[0] == '\0') {
        return false;
    }

    const char * wname = weight->name;
    return std::strstr(wname, ".attn_qkv.weight")  != nullptr ||
           std::strstr(wname, ".attn_gate.weight") != nullptr ||
           std::strstr(wname, ".ssm_out.weight")   != nullptr;
}

}  // namespace

// Send a standalone RPCMEM_MAP unmap-only message to DSP, wait for ack,
// then do host-side fastrpc_munmap. This is used as RpcMemMapper::DspFlushFn
// callback when deferred LRU eviction needs to free device mappings before new fastrpc_mmap.
void dsp_flush_pending_unmaps(RpcMemMapper & mapper) {
    auto * ctx = ggml_backend_htp_context::instance();

    int n_unmap_fds = mapper.get_pending_unmap_reqs().size();
    if (n_unmap_fds == 0) {
        return;
    }

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);
    auto * d_ptr   = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, 0);

    msg_hdr->n_reqs         = 1;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);

    size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + map_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & [fd, _base, _len] : mapper.get_pending_unmap_reqs()) {
            write_buf(p, fd);
        }
    }

    // checksum
    {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;
        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;
        msg_hdr->checksum = -sum;

#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    // issue request & poll
    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));
    std::atomic_store_explicit(v0_ptr, 1, std::memory_order_release);

    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        usleep(1);
    }
    d_ptr->store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    // DSP has released references, now safe to do host-side fastrpc_munmap
    mapper.unmap_all_pending_buffers();
}

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->skip_htp_ops) {
        return false;
    }
    if (!ctx->ops_backend_initialized) {
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            if (dst->src[0] != nullptr &&
                dst->type == GGML_TYPE_F32 &&
                dst->src[0]->type == GGML_TYPE_F32) {
                // NOTE: RPC version is mainly for testing
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                // Keep matmul gate conservative:
                // kernel constraints from mat_mul.c: k % 32 == 0 && n % 32 == 0
                size_t k = weight->ne[0];
                size_t n = weight->ne[1];

                bool shape_ok = k % 32 == 0 && n % 32 == 0 &&
                                ggml_nrows(dst) == dst->ne[1] &&
                                ggml_nrows(activation) == activation->ne[1];

                // FP16 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q4_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q8_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) IQ4_NL weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                    activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                return false;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                float scale         = *reinterpret_cast<const float *>(&dst->op_params[0]);
                float max_bias      = *reinterpret_cast<const float *>(&dst->op_params[1]);
                float logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);

                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                bool mask_type_ok = !mask || mask->type == GGML_TYPE_F16;

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && mask_type_ok && max_bias == 0 && logit_softcap == 0;
            }
        case GGML_OP_SSM_CONV:
            {
                auto * src0 = dst->src[0];
                auto * src1 = dst->src[1];

                if (!src0 || !src1) {
                    return false;
                }

                const bool type_ok = dst->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                                     src1->type == GGML_TYPE_F32;
                const bool shape_ok = dst->ne[0] == src0->ne[1] && dst->ne[0] == src1->ne[1] &&
                                      src0->ne[0] == src1->ne[0] - 1 + dst->ne[1] &&
                                      dst->ne[2] == src0->ne[2] &&
                                      src1->ne[2] == 1 && src1->ne[3] == 1 &&
                                      dst->ne[3] == 1 && src0->ne[3] == 1;
                const bool layout_ok = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);

                return type_ok && shape_ok && layout_ok;
            }
        default:
            return false;
    }
}

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    prepare_tensor_rpcmem_mapping(dst);

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    constexpr bool prefer_rpc = false;

    int op_index  = -1;
    int args_size = 0;  // strictly 32 bits

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                if (prefer_rpc) {
                    using fn_type = int(int, int, int, int, int, int);

                    auto op_fn = reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32"));
                    GGML_ASSERT(op_fn);

                    return op_fn(dst_fd, dst_offset, src_fd, src_offset, dst->ne[0], ggml_nrows(dst));
                }

                RmsNormF32Params params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

                // ggml_mul_mat: dst[M, N] = weight[K, M]^T @ activation[K, N]
                //   where: K = weight->ne[0], M = weight->ne[1], N = activation->ne[1]
                //
                // ggml tensors are column-major (ne[0] is contiguous dimension):
                //   - weight: [K, M] with stride = K (nb1/nb0 = K)
                //   - activation: [K, N] with stride = K (nb1/nb0 = K)
                //   - dst: [M, N] with stride = M (nb1/nb0 = M)
                //
                // Kernel expects row-major layout:
                //   output[m, n] = activation[m, k] @ permuted_weight[n, k]^T
                //   - activation: m rows, k cols, stride = k
                //   - weight: n rows, k cols, stride = k
                //   - output: m rows, n cols, stride = n
                //
                // Column-major [K, N] is equivalent to row-major [N, K] (same stride, just dimension names swapped).
                // So for kernel operation:
                //   - kernel's m = ggml's activation rows
                //   - kernel's k = ggml's K (weight->ne[0], inner dim) ✓ stride matches
                //   - kernel's n = ggml's M (weight->ne[1], output features)

                int m = ggml_nrows(activation);  // N * ne[2] * ne[3]... - handles multi-dimensional activations
                int k = weight->ne[0];           // K (inner dim) - stride matches ggml
                int n = weight->ne[1];           // M (output features) - kernel outputs M cols
                const bool use_common_layout = is_recurrent_common_layout_weight(weight);

                debug_ffn_up   = should_debug_ffn_up_once(dst);
                debug_ffn_gate = should_debug_ffn_gate_once(dst);
                debug_ffn_out  = should_debug_ffn_out_once(dst);
                debug_mul_mat_target = should_debug_mul_mat_target_once(dst);
                const bool debug_ffn = debug_ffn_up || debug_ffn_gate || debug_ffn_out;
                const bool debug_mul_mat = debug_ffn || debug_mul_mat_target;

                if (debug_mul_mat) {
                    const char * debug_prefix = debug_ffn_up ? "HTP FFN_UP DEBUG" :
                                                debug_ffn_gate ? "HTP FFN_GATE DEBUG" :
                                                debug_ffn_out ? "HTP FFN_OUT DEBUG" :
                                                "HTP MUL_MAT TARGET DEBUG";
                    print_tensor_shape_and_layout("weight", weight);
                    print_tensor_shape_and_layout("activation", activation);
                    print_tensor_shape_and_layout("dst", dst);
                    print_tensor_memory_detail("weight", weight);
                    print_tensor_memory_detail("activation", activation);
                    print_tensor_memory_detail("dst", dst);
                    fprintf(stderr,
                            "%s: activation nrows=%lld vs ne[1]=%lld, kernel m=%d k=%d n=%d\n",
                            debug_prefix,
                            (long long) ggml_nrows(activation), (long long) activation->ne[1], m, k, n);
                    fprintf(stderr,
                            "%s: fd+offset dst=(%d,%lld) weight=(%d,%lld) activation=(%d,%lld)\n",
                            debug_prefix,
                            output_fd, (long long) output_offset,
                            weight_fd, (long long) weight_offset,
                            activation_fd, (long long) activation_offset);
                    fprintf(stderr,
                            "%s: weight_name=%s activation_name=%s dst_name=%s recurrent_weight=%d\n",
                            debug_prefix,
                            weight->name[0] != '\0' ? weight->name : "(unnamed)",
                            activation->name[0] != '\0' ? activation->name : "(unnamed)",
                            dst->name[0] != '\0' ? dst->name : "(unnamed)",
                            weight->name[0] != '\0' && is_recurrent_weight_name(weight->name));
                    if (debug_ffn_up || debug_mul_mat_target) {
                        print_tensor_f32_stats("mul_mat.activation BEFORE HTP", activation);
                    }
                    if (debug_ffn_up) {
                        print_tensor_f32_stats("ffn_up.activation BEFORE HTP", activation);
                    } else if (debug_ffn_gate) {
                        print_tensor_f32_stats("ffn_gate.activation BEFORE HTP", activation);
                    } else if (debug_ffn_out) {
                        print_tensor_f32_stats("ffn_out.activation BEFORE HTP", activation);
                    }
                }

                // 启用 RPCMEM 详细映射调试（用于对比内存布局差异）
                if (htp_debug_rpcmem_enabled() || debug_mul_mat_target) {
                    print_rpcmem_mapping_detail("output", output_fd, output_offset, dst);
                    print_rpcmem_mapping_detail("weight", weight_fd, weight_offset, weight);
                    print_rpcmem_mapping_detail("activation", activation_fd, activation_offset, activation);
                }

                // NOTE: output stride issue: kernel expects stride=n=M, but ggml dst has stride=M ✓ (matches!)
                // So after this correction, all strides match correctly!

                ///////////////////////
                // k &= ~31;
                // n &= ~31;
                ///////////////////////

                MatMulParams params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = m,
                    .k          = k,
                    .n          = n,
                };
                *reinterpret_cast<MatMulParams *>(param_buf) = params;

                args_size = sizeof(MatMulParams);

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (prefer_rpc) {
                        using fn_type = int(int, int, int, int, int, int, int, int, int);

                        auto op_fn =
                            reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32"));
                        GGML_ASSERT(op_fn);

                        return op_fn(output_fd, output_offset, activation_fd, activation_offset, weight_fd,
                                     weight_offset, m, k, n);
                    }

                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = use_common_layout ? HTP_OPS_MAT_MUL_COMMON_W8D16A32
                                                 : HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = use_common_layout ? HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL
                                                 : HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }

            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                auto mappings = get_all_rpcmem_mappings(dst);

                // NOTE(hzx): `mask` is allowed to be null
                if (mappings.size() == 4) {
                    mappings.push_back({ -1, 0 });
                }
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

                int head_dim   = q->ne[0];
                int qo_len     = q->ne[1];
                int kv_len     = k->ne[1];
                int n_heads    = q->ne[2];
                int n_kv_heads = k->ne[2];

                FlashAttnParams params{
                    .o          = { o_fd,    (int32_t) o_offset    },
                    .q          = { q_fd,    (int32_t) q_offset    },
                    .k          = { k_fd,    (int32_t) k_offset    },
                    .v          = { v_fd,    (int32_t) v_offset    },
                    .mask       = { mask_fd, (int32_t) mask_offset },
                    .qo_len     = qo_len,
                    .kv_len     = kv_len,
                    .n_heads    = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim   = head_dim,
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);
            }
            break;

        case GGML_OP_SSM_CONV:
            {
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [dst_fd,  dst_offset]  = mappings[0];
                auto [src0_fd, src0_offset] = mappings[1];
                auto [src1_fd, src1_offset] = mappings[2];

                auto * src0 = dst->src[0];
                auto * src1 = dst->src[1];

                SsmConvParams params{
                    .dst     = { dst_fd,  (int32_t) dst_offset  },
                    .src0    = { src0_fd, (int32_t) src0_offset },
                    .src1    = { src1_fd, (int32_t) src1_offset },
                    .d_conv  = (int32_t) src1->ne[0],
                    .d_inner = (int32_t) dst->ne[0],
                    .n_t     = (int32_t) dst->ne[1],
                    .n_s     = (int32_t) dst->ne[2],
                };
                *reinterpret_cast<SsmConvParams *>(param_buf) = params;

                op_index  = HTP_OPS_SSM_CONV_F32;
                args_size = sizeof(SsmConvParams);
            }
            break;

        default:
            break;
    }

    // TODO(hzx): make sure only one thread can arrive here
    int  n_reqs         = 1;
    int  n_unmap_fds    = ctx->mapper.get_pending_unmap_reqs().size();
    bool has_unmap_reqs = n_unmap_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    // FIXME: this is very ugly
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    // std::atomic_store_explicit(d_ptr, 0, std::memory_order_release);

    // The memory order here is not very important
    std::atomic_store(d_ptr, 0);

    msg_hdr->n_reqs         = n_reqs;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + op_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_OP_COMPUTE,
        };
        OpComputeRequest op_req{
            .op = (uint32_t) op_index,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, op_req);
        write_buf(p, param_buf, args_size);
    }

    if (has_unmap_reqs) {
        size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & [fd, _base, _len] : ctx->mapper.get_pending_unmap_reqs()) {
            write_buf(p, fd);
        }
    }

    // compute checksum
    if (1) {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;  // skip state & checksum
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;  // value of `state`

        msg_hdr->checksum = -sum;

#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    // issue request
    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));

    // NOTE(hzx): make sure memory_order_release is used here to ensure all previous writes are valid
    std::atomic_store_explicit(v0_ptr, 1, std::memory_order_release);

    // poll for response
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        // TODO(hzx): use cpu_relax here
        usleep(1);
    }
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        ctx->mapper.unmap_all_pending_buffers();
    }

    std::atomic_thread_fence(std::memory_order_acquire);

    return message_header_get_request_ptr(msg_hdr, 0)->state;
}
}
