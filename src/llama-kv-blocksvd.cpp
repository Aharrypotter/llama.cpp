#include "llama-kv-blocksvd.h"
#include "ggml.h"
#include "llama-impl.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstring>

using llama_kv_blocksvd_matrix_rm = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static int32_t llama_kv_blocksvd_qmax(int32_t quant_bits) {
    return quant_bits == 16 ? 32767 : 127;
}

static int32_t llama_kv_blocksvd_dtype_size(int32_t quant_bits) {
    return quant_bits == 16 ? 2 : 1;
}

// Store a 16-bit quantized value in little-endian byte order at dst[2*i].
// The byte order is fixed regardless of host endianness so that serialized
// chunks are portable across platforms.
static void llama_kv_blocksvd_store_i16(int8_t * dst, int32_t i, int16_t value) {
    uint16_t u = static_cast<uint16_t>(value);
    dst[2*i + 0] = static_cast<int8_t>(u & 0xffu);
    dst[2*i + 1] = static_cast<int8_t>((u >> 8) & 0xffu);
}

static int16_t llama_kv_blocksvd_load_i16(const int8_t * src, int32_t i) {
    uint16_t u = static_cast<uint8_t>(src[2*i + 0]) | (static_cast<uint8_t>(src[2*i + 1]) << 8);
    return static_cast<int16_t>(u);
}

static void llama_kv_blocksvd_quantize_symmetric(
        const float * src,
        int8_t * dst,
        int32_t n,
        int32_t quant_bits,
        float & scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    const int32_t qmax = llama_kv_blocksvd_qmax(quant_bits);
    scale = max_abs > 0.0f ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (quant_bits == 16) {
        for (int i = 0; i < n; ++i) {
            float q = std::round(src[i] / scale);
            q = std::max(-32768.0f, std::min(32767.0f, q));
            llama_kv_blocksvd_store_i16(dst, i, static_cast<int16_t>(q));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float q = std::round(src[i] / scale);
            q = std::max(-128.0f, std::min(127.0f, q));
            dst[i] = static_cast<int8_t>(q);
        }
    }
}

static void llama_kv_blocksvd_dequantize_symmetric(
        const int8_t * src,
        float * dst,
        int32_t n,
        int32_t quant_bits,
        float scale) {
    if (quant_bits == 16) {
        for (int i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(llama_kv_blocksvd_load_i16(src, i)) * scale;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(src[i]) * scale;
        }
    }
}

llama_kv_blocksvd_context * llama_kv_blocksvd_init(const llama_kv_blocksvd_params & params) {
    auto * ctx = new llama_kv_blocksvd_context{params};
    return ctx;
}

void llama_kv_blocksvd_free(llama_kv_blocksvd_context * ctx) {
    delete ctx;
}

bool llama_kv_blocksvd_compress_chunk(
        llama_kv_blocksvd_context * ctx,
        int32_t layer,
        const float * k,
        const float * v,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv,
        int32_t n_kv_start,
        bool v_transposed) {

    if (!ctx) {
        LLAMA_LOG_WARN("%s: null context\n", __func__);
        return false;
    }
    if (!k || !v) {
        LLAMA_LOG_WARN("%s: null K/V pointers\n", __func__);
        return false;
    }
    if (layer < 0) {
        LLAMA_LOG_WARN("%s: invalid layer %d\n", __func__, layer);
        return false;
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        LLAMA_LOG_WARN("%s: invalid head configuration (n_head_kv=%d, head_dim_k=%d, head_dim_v=%d)\n",
                       __func__, n_head_kv, head_dim_k, head_dim_v);
        return false;
    }
    if (n_kv <= 0 || n_kv_start < 0) {
        LLAMA_LOG_WARN("%s: invalid sequence range (n_kv=%d, n_kv_start=%d)\n", __func__, n_kv, n_kv_start);
        return false;
    }

    const int32_t block_size = ctx->params.block_size;
    const int32_t rank       = ctx->params.rank;
    const int32_t quant_bits = ctx->params.quant_bits;

    if (block_size <= 0) {
        LLAMA_LOG_WARN("%s: invalid block_size %d\n", __func__, block_size);
        return false;
    }
    if (rank <= 0) {
        LLAMA_LOG_WARN("%s: invalid rank %d\n", __func__, rank);
        return false;
    }
    if (quant_bits != 8 && quant_bits != 16) {
        LLAMA_LOG_WARN("%s: invalid quant_bits %d (must be 8 or 16)\n", __func__, quant_bits);
        return false;
    }

    if (layer >= (int32_t) ctx->layers.size()) {
        ctx->layers.resize(layer + 1);
    }

    const int32_t n_flat_k = n_head_kv * head_dim_k;
    const int32_t n_flat_v = n_head_kv * head_dim_v;
    const int32_t n_flat   = n_flat_k + n_flat_v;

    // Build dense matrix X of shape (n_kv, n_flat) in row-major order.
    std::vector<float> X(static_cast<size_t>(n_kv) * static_cast<size_t>(n_flat));
    for (int i = 0; i < n_kv; ++i) {
        // K: layout (head_dim_k, n_head_kv, n_kv)
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim_k; ++d) {
                const int k_idx = d + h * head_dim_k + i * n_head_kv * head_dim_k;
                X[static_cast<size_t>(i) * n_flat + h * head_dim_k + d] = k[k_idx];
            }
        }
        // V: layout depends on v_transposed
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim_v; ++d) {
                int v_idx;
                if (v_transposed) {
                    // (n_kv, n_head_kv, head_dim_v)
                    v_idx = i + h * n_kv + d * n_kv * n_head_kv;
                } else {
                    // (head_dim_v, n_head_kv, n_kv)
                    v_idx = d + h * head_dim_v + i * n_head_kv * head_dim_v;
                }
                X[static_cast<size_t>(i) * n_flat + n_flat_k + h * head_dim_v + d] = v[v_idx];
            }
        }
    }

    // Process by blocks along the sequence dimension.  The final block may be
    // smaller than block_size; we still compress it with an adaptive rank.
    const int32_t n_blocks_full = n_kv / block_size;
    const int32_t n_blocks      = n_blocks_full + (n_kv % block_size > 0 ? 1 : 0);

    for (int b = 0; b < n_blocks; ++b) {
        const int32_t start = b * block_size;
        const int32_t end   = std::min(start + block_size, n_kv);
        const int32_t actual_block_size = end - start;

        Eigen::Map<llama_kv_blocksvd_matrix_rm> M(X.data() + static_cast<size_t>(start) * n_flat,
                                                  actual_block_size, n_flat);
        Eigen::JacobiSVD<llama_kv_blocksvd_matrix_rm> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (svd.info() != Eigen::Success) {
            LLAMA_LOG_WARN("%s: SVD failed for layer %d block [%d, %d)\n",
                           __func__, layer, start, end);
            return false;
        }

        const int32_t max_rank = std::min(actual_block_size, n_flat);
        const int32_t r        = std::min(rank, max_rank);

        auto U = svd.matrixU().leftCols(r);
        auto S = svd.singularValues().head(r);
        auto V = svd.matrixV().leftCols(r);

        llama_kv_blocksvd_chunk chunk;
        chunk.seq_start  = n_kv_start + start;
        chunk.seq_end    = n_kv_start + end;
        chunk.n_kv       = n_kv_start + n_kv;
        chunk.n_flat     = n_flat;
        chunk.quant_bits = quant_bits;

        const int32_t dtype_size = llama_kv_blocksvd_dtype_size(quant_bits);
        chunk.u_q.resize(static_cast<size_t>(actual_block_size) * r * dtype_size);
        chunk.s_q.resize(static_cast<size_t>(r) * dtype_size);
        chunk.vh_q.resize(static_cast<size_t>(r) * n_flat * dtype_size);

        std::vector<float> u_f(static_cast<size_t>(actual_block_size) * r);
        std::vector<float> s_f(r);
        std::vector<float> v_f(static_cast<size_t>(r) * n_flat);

        // U: (actual_block_size, r)
        for (int i = 0; i < actual_block_size; ++i) {
            for (int j = 0; j < r; ++j) {
                u_f[static_cast<size_t>(i) * r + j] = U(i, j);
            }
        }
        // S: (r,)
        for (int j = 0; j < r; ++j) {
            s_f[j] = S(j);
        }
        // Vh = V^T: (r, n_flat)
        for (int j = 0; j < r; ++j) {
            for (int i = 0; i < n_flat; ++i) {
                v_f[static_cast<size_t>(j) * n_flat + i] = V(i, j);
            }
        }

        llama_kv_blocksvd_quantize_symmetric(u_f.data(), chunk.u_q.data(), (int32_t) u_f.size(), quant_bits, chunk.u_scale);
        llama_kv_blocksvd_quantize_symmetric(s_f.data(), chunk.s_q.data(), (int32_t) s_f.size(), quant_bits, chunk.s_scale);
        llama_kv_blocksvd_quantize_symmetric(v_f.data(), chunk.vh_q.data(), (int32_t) v_f.size(), quant_bits, chunk.vh_scale);

        ctx->layers[layer].push_back(std::move(chunk));
    }

    return true;
}

bool llama_kv_blocksvd_decompress_chunk(
        const llama_kv_blocksvd_chunk & chunk,
        float * k,
        float * v,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        bool v_transposed) {

    if (!k || !v) {
        LLAMA_LOG_WARN("%s: null K/V pointers\n", __func__);
        return false;
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        LLAMA_LOG_WARN("%s: invalid head configuration (n_head_kv=%d, head_dim_k=%d, head_dim_v=%d)\n",
                       __func__, n_head_kv, head_dim_k, head_dim_v);
        return false;
    }
    if (chunk.seq_start < 0 || chunk.seq_end <= chunk.seq_start) {
        LLAMA_LOG_WARN("%s: invalid chunk sequence range [%d, %d)\n",
                       __func__, chunk.seq_start, chunk.seq_end);
        return false;
    }
    if (chunk.quant_bits != 8 && chunk.quant_bits != 16) {
        LLAMA_LOG_WARN("%s: invalid chunk quant_bits %d (must be 8 or 16)\n",
                       __func__, chunk.quant_bits);
        return false;
    }

    const int32_t block_size = chunk.seq_end - chunk.seq_start;
    if (block_size <= 0) {
        LLAMA_LOG_WARN("%s: invalid block size %d\n", __func__, block_size);
        return false;
    }

    const int32_t n_flat_k   = n_head_kv * head_dim_k;
    const int32_t n_flat_v   = n_head_kv * head_dim_v;
    const int32_t n_flat     = chunk.n_flat;
    const int32_t quant_bits = chunk.quant_bits;
    const int32_t dtype_size = llama_kv_blocksvd_dtype_size(quant_bits);

    if (n_flat != n_flat_k + n_flat_v) {
        LLAMA_LOG_WARN("%s: chunk n_flat %d does not match head dims\n", __func__, n_flat);
        return false;
    }

    const int32_t rank = (int32_t) chunk.s_q.size() / dtype_size;
    if (rank <= 0) {
        LLAMA_LOG_WARN("%s: invalid chunk rank %d\n", __func__, rank);
        return false;
    }

    if ((int32_t) chunk.u_q.size() != block_size * rank * dtype_size) {
        LLAMA_LOG_WARN("%s: chunk U buffer size mismatch (expected %d, got %zu)\n",
                       __func__, block_size * rank * dtype_size, chunk.u_q.size());
        return false;
    }
    if ((int32_t) chunk.vh_q.size() != rank * n_flat * dtype_size) {
        LLAMA_LOG_WARN("%s: chunk Vh buffer size mismatch (expected %d, got %zu)\n",
                       __func__, rank * n_flat * dtype_size, chunk.vh_q.size());
        return false;
    }

    std::vector<float> u_f(static_cast<size_t>(block_size) * rank);
    std::vector<float> s_f(rank);
    std::vector<float> v_f(static_cast<size_t>(rank) * n_flat);

    llama_kv_blocksvd_dequantize_symmetric(chunk.u_q.data(), u_f.data(), (int32_t) u_f.size(), quant_bits, chunk.u_scale);
    llama_kv_blocksvd_dequantize_symmetric(chunk.s_q.data(), s_f.data(), (int32_t) s_f.size(), quant_bits, chunk.s_scale);
    llama_kv_blocksvd_dequantize_symmetric(chunk.vh_q.data(), v_f.data(), (int32_t) v_f.size(), quant_bits, chunk.vh_scale);

    // Reconstruct: X = U * diag(S) * Vh
    std::vector<float> X(static_cast<size_t>(block_size) * n_flat, 0.0f);
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < rank; ++j) {
            const float us = u_f[static_cast<size_t>(i) * rank + j] * s_f[j];
            for (int c = 0; c < n_flat; ++c) {
                X[static_cast<size_t>(i) * n_flat + c] += us * v_f[static_cast<size_t>(j) * n_flat + c];
            }
        }
    }

    // Write back to k/v layout.
    const int32_t n_kv_total = chunk.n_kv;
    for (int i = 0; i < block_size; ++i) {
        const int seq = chunk.seq_start + i;
        if (seq >= n_kv_total) continue;

        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim_k; ++d) {
                const int k_idx = d + h * head_dim_k + seq * n_head_kv * head_dim_k;
                k[k_idx] = X[static_cast<size_t>(i) * n_flat + h * head_dim_k + d];
            }
            for (int d = 0; d < head_dim_v; ++d) {
                int v_idx;
                if (v_transposed) {
                    // (n_kv, n_head_kv, head_dim_v)
                    v_idx = seq + h * n_kv_total + d * n_kv_total * n_head_kv;
                } else {
                    // (head_dim_v, n_head_kv, n_kv)
                    v_idx = d + h * head_dim_v + seq * n_head_kv * head_dim_v;
                }
                v[v_idx] = X[static_cast<size_t>(i) * n_flat + n_flat_k + h * head_dim_v + d];
            }
        }
    }

    return true;
}
