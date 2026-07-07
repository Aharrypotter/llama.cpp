#include "llama-kv-blocksvd.h"
#include "ggml.h"
#include "llama-impl.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

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

// Factor a row-major (n_tokens, n_dim) matrix with SVD, quantize the factors,
// dequantize in place, and reconstruct back into M.  The quantized factors are
// returned in out_factors for persistent storage.
static bool llama_kv_blocksvd_factor_matrix(
        std::vector<float> & M,
        int32_t n_tokens,
        int32_t n_dim,
        int32_t rank,
        int32_t quant_bits,
        llama_kv_blocksvd_xkv_factors & out_factors,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    const int32_t max_rank = std::min(n_tokens, n_dim);
    const int32_t r = std::min(rank, max_rank);
    if (r <= 0) {
        return fail("rank must be > 0");
    }
    if (quant_bits != 8 && quant_bits != 16) {
        return fail("quant_bits must be 8 or 16");
    }

    Eigen::Map<llama_kv_blocksvd_matrix_rm> Mat(M.data(), n_tokens, n_dim);
    Eigen::JacobiSVD<llama_kv_blocksvd_matrix_rm> svd(Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) {
        return fail("SVD failed for cross-layer group");
    }

    auto U = svd.matrixU().leftCols(r);
    auto S = svd.singularValues().head(r);
    auto Vmat = svd.matrixV().leftCols(r);

    std::vector<float> u_f(static_cast<size_t>(n_tokens) * r);
    std::vector<float> s_f(r);
    std::vector<float> v_f(static_cast<size_t>(r) * n_dim);
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < r; ++j) {
            u_f[static_cast<size_t>(i) * r + j] = U(i, j);
        }
    }
    for (int j = 0; j < r; ++j) {
        s_f[j] = S(j);
    }
    for (int j = 0; j < r; ++j) {
        for (int c = 0; c < n_dim; ++c) {
            v_f[static_cast<size_t>(j) * n_dim + c] = Vmat(c, j);
        }
    }

    const int32_t dtype_size = llama_kv_blocksvd_dtype_size(quant_bits);
    out_factors.rank = r;
    out_factors.quant_bits = quant_bits;
    out_factors.u_q.resize(u_f.size() * dtype_size);
    out_factors.s_q.resize(s_f.size() * dtype_size);
    out_factors.vh_q.resize(v_f.size() * dtype_size);
    llama_kv_blocksvd_quantize_symmetric(u_f.data(), out_factors.u_q.data(), (int32_t) u_f.size(), quant_bits, out_factors.u_scale);
    llama_kv_blocksvd_quantize_symmetric(s_f.data(), out_factors.s_q.data(), (int32_t) s_f.size(), quant_bits, out_factors.s_scale);
    llama_kv_blocksvd_quantize_symmetric(v_f.data(), out_factors.vh_q.data(), (int32_t) v_f.size(), quant_bits, out_factors.vh_scale);

    std::vector<float> u_dq(u_f.size());
    std::vector<float> s_dq(s_f.size());
    std::vector<float> v_dq(v_f.size());
    llama_kv_blocksvd_dequantize_symmetric(out_factors.u_q.data(), u_dq.data(), (int32_t) u_dq.size(), quant_bits, out_factors.u_scale);
    llama_kv_blocksvd_dequantize_symmetric(out_factors.s_q.data(), s_dq.data(), (int32_t) s_dq.size(), quant_bits, out_factors.s_scale);
    llama_kv_blocksvd_dequantize_symmetric(out_factors.vh_q.data(), v_dq.data(), (int32_t) v_dq.size(), quant_bits, out_factors.vh_scale);

    std::fill(M.begin(), M.end(), 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < r; ++j) {
            const float us = u_dq[static_cast<size_t>(i) * r + j] * s_dq[j];
            for (int c = 0; c < n_dim; ++c) {
                M[static_cast<size_t>(i) * n_dim + c] += us * v_dq[static_cast<size_t>(j) * n_dim + c];
            }
        }
    }
    return true;
}

llama_kv_blocksvd_context * llama_kv_blocksvd_init(const llama_kv_blocksvd_params & params) {
    auto * ctx = new llama_kv_blocksvd_context{params, {}, {}, {}, {}};
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

        // Avoid duplicate/overlapping chunks for the same sequence range.
        auto & chunks = ctx->layers[layer];
        chunks.erase(
            std::remove_if(chunks.begin(), chunks.end(),
                [&chunk](const llama_kv_blocksvd_chunk & existing) {
                    return existing.seq_start == chunk.seq_start;
                }),
            chunks.end());

        chunks.push_back(std::move(chunk));
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

// xKV-style cross-layer SVD: group consecutive layers, stack their K/V along the
// feature (head) dimension, run a single SVD for K and one for V, then reconstruct
// and split back per-layer.  This mirrors the cloud xKV algorithm inside the
// llama.cpp shadow hook.
bool llama_kv_blocksvd_compress_xkv_group(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        int32_t n_tokens,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx) {
        return fail("ctx is null");
    }
    if (layer_start < 0) {
        return fail("invalid layer_start");
    }
    if (n_tokens <= 0) {
        return true;
    }
    if (k.empty() || k.size() != v.size()) {
        return fail("K/V layer count mismatch");
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return fail("invalid head configuration");
    }

    const int32_t group_size = (int32_t) k.size();
    const int32_t d_k = n_head_kv * head_dim_k;
    const int32_t d_v = n_head_kv * head_dim_v;
    const int32_t combined_k_dim = group_size * d_k;
    const int32_t combined_v_dim = group_size * d_v;

    const int32_t rank = ctx->params.rank;
    const int32_t quant_bits = ctx->params.quant_bits;
    if (rank <= 0) {
        return fail("rank must be > 0");
    }
    if (quant_bits != 8 && quant_bits != 16) {
        return fail("quant_bits must be 8 or 16");
    }

    // Build combined K/V matrices of shape (n_tokens, combined_dim).
    std::vector<float> K(static_cast<size_t>(n_tokens) * combined_k_dim, 0.0f);
    std::vector<float> V(static_cast<size_t>(n_tokens) * combined_v_dim, 0.0f);
    for (int l = 0; l < group_size; ++l) {
        if (!k[l] || !v[l]) {
            return fail("null K/V pointer for layer in group");
        }
        for (int t = 0; t < n_tokens; ++t) {
            const size_t src_off = static_cast<size_t>(t) * d_k;
            const size_t dst_off = static_cast<size_t>(t) * combined_k_dim + static_cast<size_t>(l) * d_k;
            std::memcpy(K.data() + dst_off, k[l] + src_off, static_cast<size_t>(d_k) * sizeof(float));

            const size_t src_off_v = static_cast<size_t>(t) * d_v;
            const size_t dst_off_v = static_cast<size_t>(t) * combined_v_dim + static_cast<size_t>(l) * d_v;
            std::memcpy(V.data() + dst_off_v, v[l] + src_off_v, static_cast<size_t>(d_v) * sizeof(float));
        }
    }

    llama_kv_blocksvd_xkv_factors k_factors;
    llama_kv_blocksvd_xkv_factors v_factors;

    if (!llama_kv_blocksvd_factor_matrix(K, n_tokens, combined_k_dim, rank, quant_bits, k_factors, err)) {
        return false;
    }
    if (!llama_kv_blocksvd_factor_matrix(V, n_tokens, combined_v_dim, rank, quant_bits, v_factors, err)) {
        return false;
    }

    // Split reconstructed combined matrices back into per-layer K/V.
    out_k.resize(group_size);
    out_v.resize(group_size);
    for (int l = 0; l < group_size; ++l) {
        out_k[l].resize(static_cast<size_t>(n_tokens) * d_k);
        out_v[l].resize(static_cast<size_t>(n_tokens) * d_v);
        for (int t = 0; t < n_tokens; ++t) {
            const size_t src_off = static_cast<size_t>(t) * combined_k_dim + static_cast<size_t>(l) * d_k;
            const size_t dst_off = static_cast<size_t>(t) * d_k;
            std::memcpy(out_k[l].data() + dst_off, K.data() + src_off, static_cast<size_t>(d_k) * sizeof(float));

            const size_t src_off_v = static_cast<size_t>(t) * combined_v_dim + static_cast<size_t>(l) * d_v;
            const size_t dst_off_v = static_cast<size_t>(t) * d_v;
            std::memcpy(out_v[l].data() + dst_off_v, V.data() + src_off_v, static_cast<size_t>(d_v) * sizeof(float));
        }
    }

    return true;
}

bool llama_kv_blocksvd_decompress_xkv_chunk(
        const llama_kv_blocksvd_xkv_chunk & chunk,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    const int32_t group_size = chunk.group_size;
    const int32_t n_tokens   = (int32_t) chunk.slots.size();
    if (group_size <= 0 || n_tokens <= 0) {
        return fail("invalid xkv chunk dimensions");
    }

    const int32_t n_head_kv  = chunk.n_head_kv;
    const int32_t head_dim_k = chunk.head_dim_k;
    const int32_t head_dim_v = chunk.head_dim_v;
    const int32_t d_k = n_head_kv * head_dim_k;
    const int32_t d_v = n_head_kv * head_dim_v;
    const int32_t combined_k_dim = group_size * d_k;
    const int32_t combined_v_dim = group_size * d_v;

    const auto reconstruct = [&](const llama_kv_blocksvd_xkv_factors & factors,
                                 int32_t n_dim,
                                 std::vector<float> & M) -> bool {
        const int32_t r = factors.rank;
        const int32_t quant_bits = factors.quant_bits;
        const int32_t dtype_size = llama_kv_blocksvd_dtype_size(quant_bits);
        if (r <= 0) {
            return fail("invalid xkv factor rank");
        }
        if ((int32_t) factors.u_q.size() != n_tokens * r * dtype_size) {
            return fail("xkv U size mismatch");
        }
        if ((int32_t) factors.s_q.size() != r * dtype_size) {
            return fail("xkv S size mismatch");
        }
        if ((int32_t) factors.vh_q.size() != r * n_dim * dtype_size) {
            return fail("xkv Vh size mismatch");
        }

        std::vector<float> u_f(static_cast<size_t>(n_tokens) * r);
        std::vector<float> s_f(r);
        std::vector<float> v_f(static_cast<size_t>(r) * n_dim);
        llama_kv_blocksvd_dequantize_symmetric(factors.u_q.data(), u_f.data(), (int32_t) u_f.size(), quant_bits, factors.u_scale);
        llama_kv_blocksvd_dequantize_symmetric(factors.s_q.data(), s_f.data(), (int32_t) s_f.size(), quant_bits, factors.s_scale);
        llama_kv_blocksvd_dequantize_symmetric(factors.vh_q.data(), v_f.data(), (int32_t) v_f.size(), quant_bits, factors.vh_scale);

        M.assign(static_cast<size_t>(n_tokens) * n_dim, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < r; ++j) {
                const float us = u_f[static_cast<size_t>(i) * r + j] * s_f[j];
                for (int c = 0; c < n_dim; ++c) {
                    M[static_cast<size_t>(i) * n_dim + c] += us * v_f[static_cast<size_t>(j) * n_dim + c];
                }
            }
        }
        return true;
    };

    std::vector<float> K, V;
    if (!reconstruct(chunk.k_factors, combined_k_dim, K)) {
        return false;
    }
    if (!reconstruct(chunk.v_factors, combined_v_dim, V)) {
        return false;
    }

    out_k.resize(group_size);
    out_v.resize(group_size);
    for (int l = 0; l < group_size; ++l) {
        out_k[l].resize(static_cast<size_t>(n_tokens) * d_k);
        out_v[l].resize(static_cast<size_t>(n_tokens) * d_v);
        for (int t = 0; t < n_tokens; ++t) {
            const size_t src_off = static_cast<size_t>(t) * combined_k_dim + static_cast<size_t>(l) * d_k;
            const size_t dst_off = static_cast<size_t>(t) * d_k;
            std::memcpy(out_k[l].data() + dst_off, K.data() + src_off, static_cast<size_t>(d_k) * sizeof(float));

            const size_t src_off_v = static_cast<size_t>(t) * combined_v_dim + static_cast<size_t>(l) * d_v;
            const size_t dst_off_v = static_cast<size_t>(t) * d_v;
            std::memcpy(out_v[l].data() + dst_off_v, V.data() + src_off_v, static_cast<size_t>(d_v) * sizeof(float));
        }
    }

    return true;
}

bool llama_kv_blocksvd_reconstruct_layer(
        const llama_kv_blocksvd_context & bctx,
        const llama_kv_blocksvd_staging_t & staging,
        int32_t il,
        uint32_t n_kv,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        const std::vector<float> & k_staged,
        const std::vector<float> & v_staged,
        std::vector<float> & out_k,
        std::vector<float> & out_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (il < 0) {
        return fail("invalid layer");
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return fail("invalid head configuration");
    }
    if (n_kv == 0) {
        out_k.clear();
        out_v.clear();
        return true;
    }

    const int32_t d_k = n_head_kv * head_dim_k;
    const int32_t d_v = n_head_kv * head_dim_v;
    const uint32_t n_stream = (uint32_t) staging.cell_to_slot.size();

    const size_t k_staged_min = (size_t) n_stream * staging.capacity * d_k;
    const size_t v_staged_min = (size_t) n_stream * staging.capacity * d_v;
    if (k_staged.size() < k_staged_min) {
        return fail("k_staged buffer too small");
    }
    if (v_staged.size() < v_staged_min) {
        return fail("v_staged buffer too small");
    }
    if (staging.slot_to_cell.size() != n_stream) {
        return fail("staging metadata size mismatch");
    }

    out_k.assign((size_t) n_stream * n_kv * d_k, 0.0f);
    out_v.assign((size_t) n_stream * n_kv * d_v, 0.0f);

    // Build a one-time (stream, cell) -> (chunk_index, token_index) index for
    // all chunks that cover the requested layer.
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> chunk_index;
    for (size_t i = 0; i < bctx.xkv_chunks.size(); ++i) {
        const auto & chunk = bctx.xkv_chunks[i];
        if (il < chunk.layer_start || il >= chunk.layer_start + chunk.group_size) {
            continue;
        }
        for (size_t t = 0; t < chunk.slots.size(); ++t) {
            const uint64_t key =
                (static_cast<uint64_t>(chunk.stream) << 32) |
                static_cast<uint64_t>(chunk.slots[t]);
            chunk_index[key] = {i, t};
        }
    }

    // Cache decompressed chunks indexed by their position in bctx.xkv_chunks.
    struct decoded_chunk {
        std::vector<std::vector<float>> k;
        std::vector<std::vector<float>> v;
    };
    std::vector<char> decoded(bctx.xkv_chunks.size(), 0);
    std::vector<decoded_chunk> decoded_data(bctx.xkv_chunks.size());

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & c2s = staging.cell_to_slot[s];
        const auto & s2c = staging.slot_to_cell[s];
        if (c2s.size() < n_kv) {
            return fail("cell_to_slot smaller than n_kv");
        }
        if (s2c.size() < staging.capacity) {
            return fail("slot_to_cell smaller than capacity");
        }

        for (uint32_t c = 0; c < n_kv; ++c) {
            const int32_t slot = c2s[c];
            if (slot >= 0) {
                if ((uint32_t) slot >= staging.capacity) {
                    return fail("staging slot out of range");
                }
                if (s2c[slot] != (int32_t) c) {
                    return fail("staging slot/cell mismatch");
                }

                const size_t src_k = ((size_t) s * staging.capacity + (uint32_t) slot) * d_k;
                const size_t dst_k = ((size_t) s * n_kv + c) * d_k;
                std::memcpy(out_k.data() + dst_k, k_staged.data() + src_k, (size_t) d_k * sizeof(float));

                const size_t src_v = ((size_t) s * staging.capacity + (uint32_t) slot) * d_v;
                const size_t dst_v = ((size_t) s * n_kv + c) * d_v;
                std::memcpy(out_v.data() + dst_v, v_staged.data() + src_v, (size_t) d_v * sizeof(float));
                continue;
            }

            // Find the persistent chunk that contains this cell.
            const uint64_t key =
                (static_cast<uint64_t>(s) << 32) |
                static_cast<uint64_t>(c);
            auto it = chunk_index.find(key);
            if (it == chunk_index.end()) {
                return fail("logical cell " + std::to_string(c) +
                            " on stream " + std::to_string(s) +
                            " is neither staged nor present in any xKV chunk");
            }

            const size_t chunk_idx = it->second.first;
            const size_t token_idx = it->second.second;

            if (!decoded[chunk_idx]) {
                std::string derr;
                if (!llama_kv_blocksvd_decompress_xkv_chunk(
                        bctx.xkv_chunks[chunk_idx],
                        decoded_data[chunk_idx].k,
                        decoded_data[chunk_idx].v,
                        &derr)) {
                    return fail("decompress_xkv_chunk failed: " + derr);
                }
                decoded[chunk_idx] = 1;
            }

            const auto & chunk = bctx.xkv_chunks[chunk_idx];
            const int32_t local_layer = il - chunk.layer_start;
            GGML_ASSERT(local_layer >= 0 && local_layer < chunk.group_size);
            GGML_ASSERT(token_idx < chunk.slots.size());

            const size_t src_k = token_idx * d_k;
            const size_t dst_k = ((size_t) s * n_kv + c) * d_k;
            std::memcpy(out_k.data() + dst_k,
                        decoded_data[chunk_idx].k[local_layer].data() + src_k,
                        (size_t) d_k * sizeof(float));

            const size_t src_v = token_idx * d_v;
            const size_t dst_v = ((size_t) s * n_kv + c) * d_v;
            std::memcpy(out_v.data() + dst_v,
                        decoded_data[chunk_idx].v[local_layer].data() + src_v,
                        (size_t) d_v * sizeof(float));
        }
    }

    return true;
}

bool llama_kv_blocksvd_append_and_compress_xkv_group_store(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        const uint32_t * slots,
        const llama_pos * pos,
        int32_t n_tokens,
        uint32_t stream,
        llama_seq_id seq_id,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        std::vector<uint32_t> & out_compressed_slots,
        std::vector<llama_pos>  & out_compressed_pos,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx) {
        return fail("ctx is null");
    }
    if (layer_start < 0) {
        return fail("invalid layer_start");
    }
    if (n_tokens <= 0) {
        return true;
    }
    if (ctx->params.block_size <= 0) {
        return fail("block_size must be > 0");
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return fail("invalid head configuration");
    }
    if (k.empty() || k.size() != v.size()) {
        return fail("K/V layer count mismatch");
    }
    if (!slots || !pos) {
        return fail("slots and pos pointers must not be null");
    }

    const int32_t group_size = (int32_t) k.size();
    const int32_t d_k = head_dim_k * n_head_kv;
    const int32_t d_v = head_dim_v * n_head_kv;

    // Find or create pending entry for this layer group / stream / seq.
    llama_kv_blocksvd_context::pending_xkv_group * p = nullptr;
    for (auto & pg : ctx->pending_xkv) {
        if (pg.layer_start == layer_start && pg.group_size == group_size &&
            pg.stream == stream && pg.seq_id == seq_id) {
            p = &pg;
            break;
        }
    }
    if (!p) {
        ctx->pending_xkv.push_back({});
        p = &ctx->pending_xkv.back();
        p->layer_start = layer_start;
        p->group_size  = group_size;
        p->stream      = stream;
        p->seq_id      = seq_id;
        p->k.resize(group_size);
        p->v.resize(group_size);
    }

    // Append incoming tokens per layer.
    for (int l = 0; l < group_size; ++l) {
        if (!k[l] || !v[l]) {
            return fail("null K/V pointer for layer in group");
        }
        p->k[l].insert(p->k[l].end(), k[l], k[l] + (size_t) n_tokens * d_k);
        p->v[l].insert(p->v[l].end(), v[l], v[l] + (size_t) n_tokens * d_v);
    }
    p->slots.insert(p->slots.end(), slots, slots + n_tokens);
    p->pos.insert(p->pos.end(), pos, pos + n_tokens);

    out_compressed_slots.clear();
    out_compressed_pos.clear();

    const int32_t block_size = ctx->params.block_size;
    const int32_t rank       = ctx->params.rank;
    const int32_t quant_bits = ctx->params.quant_bits;

    while ((int32_t) p->slots.size() >= block_size) {
        // Build combined K/V matrices of shape (block_size, group_size*d_{k,v}).
        const int32_t combined_k_dim = group_size * d_k;
        const int32_t combined_v_dim = group_size * d_v;
        std::vector<float> K(static_cast<size_t>(block_size) * combined_k_dim, 0.0f);
        std::vector<float> V(static_cast<size_t>(block_size) * combined_v_dim, 0.0f);
        for (int l = 0; l < group_size; ++l) {
            for (int t = 0; t < block_size; ++t) {
                const size_t src_off = static_cast<size_t>(t) * d_k;
                const size_t dst_off = static_cast<size_t>(t) * combined_k_dim + static_cast<size_t>(l) * d_k;
                std::memcpy(K.data() + dst_off, p->k[l].data() + src_off, static_cast<size_t>(d_k) * sizeof(float));

                const size_t src_off_v = static_cast<size_t>(t) * d_v;
                const size_t dst_off_v = static_cast<size_t>(t) * combined_v_dim + static_cast<size_t>(l) * d_v;
                std::memcpy(V.data() + dst_off_v, p->v[l].data() + src_off_v, static_cast<size_t>(d_v) * sizeof(float));
            }
        }

        llama_kv_blocksvd_xkv_factors k_factors;
        llama_kv_blocksvd_xkv_factors v_factors;
        if (!llama_kv_blocksvd_factor_matrix(K, block_size, combined_k_dim, rank, quant_bits, k_factors, err)) {
            return false;
        }
        if (!llama_kv_blocksvd_factor_matrix(V, block_size, combined_v_dim, rank, quant_bits, v_factors, err)) {
            return false;
        }

        llama_kv_blocksvd_xkv_chunk chunk;
        chunk.layer_start = layer_start;
        chunk.group_size  = group_size;
        chunk.stream      = stream;
        chunk.seq_id      = seq_id;
        chunk.slots.assign(p->slots.begin(), p->slots.begin() + block_size);
        chunk.pos.assign(p->pos.begin(), p->pos.begin() + block_size);
        chunk.n_head_kv  = n_head_kv;
        chunk.head_dim_k = head_dim_k;
        chunk.head_dim_v = head_dim_v;
        chunk.k_factors  = std::move(k_factors);
        chunk.v_factors  = std::move(v_factors);

        ctx->xkv_chunks.push_back(std::move(chunk));

        out_compressed_slots.insert(out_compressed_slots.end(), p->slots.begin(), p->slots.begin() + block_size);
        out_compressed_pos.insert(out_compressed_pos.end(), p->pos.begin(), p->pos.begin() + block_size);

        // Remove the processed block from pending.
        for (int l = 0; l < group_size; ++l) {
            p->k[l].erase(p->k[l].begin(), p->k[l].begin() + (size_t) block_size * d_k);
            p->v[l].erase(p->v[l].begin(), p->v[l].begin() + (size_t) block_size * d_v);
        }
        p->slots.erase(p->slots.begin(), p->slots.begin() + block_size);
        p->pos.erase(p->pos.begin(), p->pos.begin() + block_size);
    }

    return true;
}

bool llama_kv_blocksvd_append_and_reconstruct(
        llama_kv_blocksvd_context * ctx,
        int32_t layer,
        const float * k,
        const float * v,
        const uint32_t * slots,
        int32_t n_tokens,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv_start,
        bool v_transposed,
        std::vector<uint32_t> & out_slots,
        std::vector<float> & out_k,
        std::vector<float> & out_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx) {
        return fail("ctx is null");
    }
    if (layer < 0) {
        return fail("invalid layer");
    }
    if (n_tokens <= 0) {
        return true;
    }
    if (ctx->params.block_size <= 0) {
        return fail("block_size must be > 0");
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return fail("invalid head configuration");
    }
    if (!k || !v || !slots) {
        return fail("K/V/slots pointers must not be null");
    }

    if (layer >= (int32_t) ctx->layers.size()) {
        ctx->layers.resize(layer + 1);
    }
    if (layer >= (int32_t) ctx->pending.size()) {
        ctx->pending.resize(layer + 1);
    }

    auto & p = ctx->pending[layer];
    const int32_t d_k = head_dim_k * n_head_kv;
    const int32_t d_v = head_dim_v * n_head_kv;

    // First call for this layer/session: record starting n_kv
    if (p.n_kv_total == 0) {
        p.n_kv_total = n_kv_start;
    }

    // Append incoming tokens
    p.k.insert(p.k.end(), k, k + (size_t) n_tokens * d_k);
    p.v.insert(p.v.end(), v, v + (size_t) n_tokens * d_v);
    p.slots.insert(p.slots.end(), slots, slots + n_tokens);

    out_slots.clear();
    out_k.clear();
    out_v.clear();

    const int32_t block_size = ctx->params.block_size;
    while ((int32_t) p.slots.size() >= block_size) {
        // Compress one block
        if (!llama_kv_blocksvd_compress_chunk(
                ctx, layer,
                p.k.data(), p.v.data(),
                n_head_kv, head_dim_k, head_dim_v,
                block_size,
                p.n_kv_total,
                v_transposed)) {
            return fail("compress_chunk failed");
        }

        const llama_kv_blocksvd_chunk & chunk = ctx->layers[layer].back();

        // Decompress it. decompress_chunk writes at absolute positions [seq_start, seq_end)
        // into arrays sized for chunk.n_kv tokens, so allocate full-size buffers and copy
        // the relevant rows afterwards.
        std::vector<float> k_recon_full((size_t) chunk.n_kv * d_k, 0.0f);
        std::vector<float> v_recon_full((size_t) chunk.n_kv * d_v, 0.0f);
        if (!llama_kv_blocksvd_decompress_chunk(
                chunk,
                k_recon_full.data(), v_recon_full.data(),
                n_head_kv, head_dim_k, head_dim_v,
                v_transposed)) {
            return fail("decompress_chunk failed");
        }

        // Append to output
        out_slots.insert(out_slots.end(), p.slots.begin(), p.slots.begin() + block_size);
        const size_t k_copy_offset = (size_t) chunk.seq_start * d_k;
        const size_t v_copy_offset = (size_t) chunk.seq_start * d_v;
        out_k.insert(out_k.end(),
                     k_recon_full.begin() + k_copy_offset,
                     k_recon_full.begin() + k_copy_offset + (size_t) block_size * d_k);
        out_v.insert(out_v.end(),
                     v_recon_full.begin() + v_copy_offset,
                     v_recon_full.begin() + v_copy_offset + (size_t) block_size * d_v);

        // Remove processed tokens from pending
        const size_t n_k_values = (size_t) block_size * d_k;
        const size_t n_v_values = (size_t) block_size * d_v;
        p.k.erase(p.k.begin(), p.k.begin() + n_k_values);
        p.v.erase(p.v.begin(), p.v.begin() + n_v_values);
        p.slots.erase(p.slots.begin(), p.slots.begin() + block_size);
        p.n_kv_total += block_size;
    }

    return true;
}

bool llama_kv_blocksvd_append_and_reconstruct_xkv_group(
        llama_kv_blocksvd_context * ctx,
        int32_t layer_start,
        const std::vector<const float *> & k,
        const std::vector<const float *> & v,
        const uint32_t * slots,
        int32_t n_tokens,
        uint32_t stream,
        llama_seq_id seq_id,
        int32_t n_head_kv,
        int32_t head_dim_k,
        int32_t head_dim_v,
        int32_t n_kv_start,
        std::vector<uint32_t> & out_slots,
        std::vector<std::vector<float>> & out_k,
        std::vector<std::vector<float>> & out_v,
        std::string * err) {
    auto fail = [err](std::string msg) {
        if (err) {
            *err = std::move(msg);
        }
        return false;
    };

    if (!ctx) {
        return fail("ctx is null");
    }
    if (layer_start < 0) {
        return fail("invalid layer_start");
    }
    if (n_tokens <= 0) {
        return true;
    }
    if (ctx->params.block_size <= 0) {
        return fail("block_size must be > 0");
    }
    if (n_head_kv <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return fail("invalid head configuration");
    }
    if (k.size() != v.size()) {
        return fail("K/V layer count mismatch");
    }
    if (!slots) {
        return fail("slots pointer must not be null");
    }

    const int32_t group_size = (int32_t) k.size();
    if (group_size <= 0) {
        return true;
    }

    const int32_t d_k = head_dim_k * n_head_kv;
    const int32_t d_v = head_dim_v * n_head_kv;

    // Find or create pending entry for this layer group / stream / seq.
    llama_kv_blocksvd_context::pending_xkv_group * p = nullptr;
    for (auto & pg : ctx->pending_xkv) {
        if (pg.layer_start == layer_start && pg.group_size == group_size &&
            pg.stream == stream && pg.seq_id == seq_id) {
            p = &pg;
            break;
        }
    }
    if (!p) {
        ctx->pending_xkv.push_back({});
        p = &ctx->pending_xkv.back();
        p->layer_start = layer_start;
        p->group_size  = group_size;
        p->stream      = stream;
        p->seq_id      = seq_id;
        p->k.resize(group_size);
        p->v.resize(group_size);
    }

    if (p->n_kv_total == 0) {
        p->n_kv_total = n_kv_start;
    }

    // Append incoming tokens per layer.
    for (int l = 0; l < group_size; ++l) {
        if (!k[l] || !v[l]) {
            return fail("null K/V pointer for layer in group");
        }
        p->k[l].insert(p->k[l].end(), k[l], k[l] + (size_t) n_tokens * d_k);
        p->v[l].insert(p->v[l].end(), v[l], v[l] + (size_t) n_tokens * d_v);
    }
    p->slots.insert(p->slots.end(), slots, slots + n_tokens);

    out_slots.clear();
    out_k.clear();
    out_v.clear();

    const int32_t block_size = ctx->params.block_size;
    while ((int32_t) p->slots.size() >= block_size) {
        std::vector<const float *> block_k_ptr(group_size);
        std::vector<const float *> block_v_ptr(group_size);
        for (int l = 0; l < group_size; ++l) {
            block_k_ptr[l] = p->k[l].data();
            block_v_ptr[l] = p->v[l].data();
        }

        std::vector<std::vector<float>> recon_k;
        std::vector<std::vector<float>> recon_v;
        if (!llama_kv_blocksvd_compress_xkv_group(
                ctx, layer_start, block_k_ptr, block_v_ptr,
                block_size, n_head_kv, head_dim_k, head_dim_v,
                recon_k, recon_v, err)) {
            return false;
        }

        // Layout: group_size vectors per processed block.
        out_slots.insert(out_slots.end(), p->slots.begin(), p->slots.begin() + block_size);
        out_k.insert(out_k.end(), std::make_move_iterator(recon_k.begin()), std::make_move_iterator(recon_k.end()));
        out_v.insert(out_v.end(), std::make_move_iterator(recon_v.begin()), std::make_move_iterator(recon_v.end()));

        // Remove the processed block from pending.
        for (int l = 0; l < group_size; ++l) {
            p->k[l].erase(p->k[l].begin(), p->k[l].begin() + (size_t) block_size * d_k);
            p->v[l].erase(p->v[l].begin(), p->v[l].begin() + (size_t) block_size * d_v);
        }
        p->slots.erase(p->slots.begin(), p->slots.begin() + block_size);
        p->n_kv_total += block_size;
    }

    return true;
}

void llama_kv_blocksvd_clear_pending(llama_kv_blocksvd_context * ctx) {
    if (!ctx) {
        return;
    }
    for (auto & p : ctx->pending) {
        p.k.clear();
        p.v.clear();
        p.slots.clear();
        p.n_kv_total = 0;
    }
    for (auto & p : ctx->pending_xkv) {
        for (auto & kv : p.k) {
            kv.clear();
        }
        for (auto & vv : p.v) {
            vv.clear();
        }
        p.slots.clear();
        p.pos.clear();
        p.n_kv_total = 0;
    }
}

void llama_kv_blocksvd_clear(llama_kv_blocksvd_context * ctx) {
    if (!ctx) {
        return;
    }
    llama_kv_blocksvd_clear_pending(ctx);
    ctx->xkv_chunks.clear();
    ctx->layers.clear();
}
