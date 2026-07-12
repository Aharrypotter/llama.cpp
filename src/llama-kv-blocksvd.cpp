#include "llama-kv-blocksvd.h"
#include "ggml.h"
#include "llama-impl.h"

#include <Eigen/SVD>

#include <algorithm>
#include <atomic>
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
    auto * ctx = new llama_kv_blocksvd_context{params, {}, {}, {}, {}, {}, {}};
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
    const int32_t rank_k     = ctx->params.rank;
    const int32_t rank_v     = ctx->params.rank_v > 0 ? ctx->params.rank_v : ctx->params.rank;
    const int32_t quant_bits = ctx->params.quant_bits;

    if (block_size <= 0) {
        LLAMA_LOG_WARN("%s: invalid block_size %d\n", __func__, block_size);
        return false;
    }
    if (rank_k <= 0) {
        LLAMA_LOG_WARN("%s: invalid rank %d\n", __func__, rank_k);
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

    // Process by blocks along the sequence dimension.  The final block may be
    // smaller than block_size; we still compress it with an adaptive rank.
    // Bug history: an earlier version built a joint [K | V] matrix of width
    // n_flat_k + n_flat_v and ran a single SVD with one shared rank. Because K
    // typically dominates V's magnitude by ~15x (RoPE'd K vs V), the top
    // singular directions were essentially K's subspace and V collapsed. Now K
    // and V get independent SVDs with independent ranks (rank_k, rank_v).
    const int32_t n_blocks_full = n_kv / block_size;
    const int32_t n_blocks      = n_blocks_full + (n_kv % block_size > 0 ? 1 : 0);

    for (int b = 0; b < n_blocks; ++b) {
        const int32_t start = b * block_size;
        const int32_t end   = std::min(start + block_size, n_kv);
        const int32_t actual_block_size = end - start;

        // Build K block: (actual_block_size, n_flat_k) row-major.
        std::vector<float> K_block(static_cast<size_t>(actual_block_size) * n_flat_k);
        for (int i = 0; i < actual_block_size; ++i) {
            const int seq = start + i;
            for (int h = 0; h < n_head_kv; ++h) {
                for (int d = 0; d < head_dim_k; ++d) {
                    const int k_idx = d + h * head_dim_k + seq * n_head_kv * head_dim_k;
                    K_block[static_cast<size_t>(i) * n_flat_k + h * head_dim_k + d] = k[k_idx];
                }
            }
        }

        // Build V block: (actual_block_size, n_flat_v) row-major.
        std::vector<float> V_block(static_cast<size_t>(actual_block_size) * n_flat_v);
        for (int i = 0; i < actual_block_size; ++i) {
            const int seq = start + i;
            for (int h = 0; h < n_head_kv; ++h) {
                for (int d = 0; d < head_dim_v; ++d) {
                    int v_idx;
                    if (v_transposed) {
                        // (n_kv, n_head_kv, head_dim_v)
                        v_idx = seq + h * n_kv + d * n_kv * n_head_kv;
                    } else {
                        // (head_dim_v, n_head_kv, n_kv)
                        v_idx = d + h * head_dim_v + seq * n_head_kv * head_dim_v;
                    }
                    V_block[static_cast<size_t>(i) * n_flat_v + h * head_dim_v + d] = v[v_idx];
                }
            }
        }

        llama_kv_blocksvd_chunk chunk;
        chunk.seq_start  = n_kv_start + start;
        chunk.seq_end    = n_kv_start + end;
        chunk.n_kv       = n_kv_start + n_kv;
        chunk.n_flat_k   = n_flat_k;
        chunk.n_flat_v   = n_flat_v;
        chunk.quant_bits = quant_bits;

        std::string factor_err;
        if (!llama_kv_blocksvd_factor_matrix(K_block, actual_block_size, n_flat_k, rank_k, quant_bits, chunk.k_factors, &factor_err)) {
            LLAMA_LOG_WARN("%s: K SVD failed for layer %d block [%d, %d): %s\n",
                           __func__, layer, start, end, factor_err.c_str());
            return false;
        }
        if (!llama_kv_blocksvd_factor_matrix(V_block, actual_block_size, n_flat_v, rank_v, quant_bits, chunk.v_factors, &factor_err)) {
            LLAMA_LOG_WARN("%s: V SVD failed for layer %d block [%d, %d): %s\n",
                           __func__, layer, start, end, factor_err.c_str());
            return false;
        }

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
    const int32_t quant_bits = chunk.quant_bits;
    const int32_t dtype_size = llama_kv_blocksvd_dtype_size(quant_bits);

    if (chunk.n_flat_k != n_flat_k || chunk.n_flat_v != n_flat_v) {
        LLAMA_LOG_WARN("%s: chunk head dims mismatch (chunk: n_flat_k=%d n_flat_v=%d; call: n_flat_k=%d n_flat_v=%d)\n",
                       __func__, chunk.n_flat_k, chunk.n_flat_v, n_flat_k, n_flat_v);
        return false;
    }

    const char * const fname = __func__;
    auto reconstruct = [&](const llama_kv_blocksvd_xkv_factors & f, int32_t n_dim, std::vector<float> & out) -> bool {
        const int32_t r = f.rank;
        if (r <= 0) {
            LLAMA_LOG_WARN("%s: invalid factor rank %d\n", fname, r);
            return false;
        }
        if ((int32_t) f.u_q.size() != block_size * r * dtype_size) {
            LLAMA_LOG_WARN("%s: U buffer size mismatch (expected %d, got %zu)\n",
                           fname, block_size * r * dtype_size, f.u_q.size());
            return false;
        }
        if ((int32_t) f.vh_q.size() != r * n_dim * dtype_size) {
            LLAMA_LOG_WARN("%s: Vh buffer size mismatch (expected %d, got %zu)\n",
                           fname, r * n_dim * dtype_size, f.vh_q.size());
            return false;
        }

        std::vector<float> u_f(static_cast<size_t>(block_size) * r);
        std::vector<float> s_f(r);
        std::vector<float> v_f(static_cast<size_t>(r) * n_dim);
        llama_kv_blocksvd_dequantize_symmetric(f.u_q.data(),  u_f.data(), (int32_t) u_f.size(), quant_bits, f.u_scale);
        llama_kv_blocksvd_dequantize_symmetric(f.s_q.data(),  s_f.data(), (int32_t) s_f.size(), quant_bits, f.s_scale);
        llama_kv_blocksvd_dequantize_symmetric(f.vh_q.data(), v_f.data(), (int32_t) v_f.size(), quant_bits, f.vh_scale);

        out.assign(static_cast<size_t>(block_size) * n_dim, 0.0f);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < r; ++j) {
                const float us = u_f[static_cast<size_t>(i) * r + j] * s_f[j];
                for (int c = 0; c < n_dim; ++c) {
                    out[static_cast<size_t>(i) * n_dim + c] += us * v_f[static_cast<size_t>(j) * n_dim + c];
                }
            }
        }
        return true;
    };

    std::vector<float> K_recon;
    std::vector<float> V_recon;
    if (!reconstruct(chunk.k_factors, n_flat_k, K_recon)) {
        return false;
    }
    if (!reconstruct(chunk.v_factors, n_flat_v, V_recon)) {
        return false;
    }

    // Write back to k/v layout.
    const int32_t n_kv_total = chunk.n_kv;
    for (int i = 0; i < block_size; ++i) {
        const int seq = chunk.seq_start + i;
        if (seq >= n_kv_total) continue;

        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim_k; ++d) {
                const int k_idx = d + h * head_dim_k + seq * n_head_kv * head_dim_k;
                k[k_idx] = K_recon[static_cast<size_t>(i) * n_flat_k + h * head_dim_k + d];
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
                v[v_idx] = V_recon[static_cast<size_t>(i) * n_flat_v + h * head_dim_v + d];
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

    const int32_t rank_k = ctx->params.rank;
    const int32_t rank_v = ctx->params.rank_v > 0 ? ctx->params.rank_v : ctx->params.rank;
    const int32_t quant_bits = ctx->params.quant_bits;
    if (rank_k <= 0) {
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

    if (!llama_kv_blocksvd_factor_matrix(K, n_tokens, combined_k_dim, rank_k, quant_bits, k_factors, err)) {
        return false;
    }
    if (!llama_kv_blocksvd_factor_matrix(V, n_tokens, combined_v_dim, rank_v, quant_bits, v_factors, err)) {
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
    const int32_t rank_k     = ctx->params.rank;
    const int32_t rank_v     = ctx->params.rank_v > 0 ? ctx->params.rank_v : ctx->params.rank;
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
        if (!llama_kv_blocksvd_factor_matrix(K, block_size, combined_k_dim, rank_k, quant_bits, k_factors, err)) {
            return false;
        }
        if (!llama_kv_blocksvd_factor_matrix(V, block_size, combined_v_dim, rank_v, quant_bits, v_factors, err)) {
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
    ctx->decode_cache.clear();
}

void llama_kv_blocksvd_reset_decode_cache(llama_kv_blocksvd_context * ctx) {
    if (!ctx) {
        return;
    }
    ctx->decode_cache.clear();
    ctx->rank_cache.clear();
}

// --- Chunked Attention Compute ---

#include "ggml.h"
#include "llama-impl.h"
#include <atomic>
#include <cmath>
#include <cstring>

void llama_chunked_attn_compute(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    const struct ggml_tensor * a = dst->src[0]; // Q
    const struct ggml_tensor * b = dst->src[1]; // K (staged)
    const struct ggml_tensor * c = dst->src[2]; // V (staged)

    auto * p = (llama_chunked_attn_params *) userdata;
    const auto * bctx    = p->bctx;
    const auto * staging = p->staging;
    const int32_t il         = p->il;
    const int32_t n_head_kv  = p->n_head_kv;
    const int32_t n_head_q   = p->n_head_q;
    const int32_t head_dim_k = p->head_dim_k;
    const int32_t head_dim_v = p->head_dim_v;
    const float   scale      = p->scale;
    const uint32_t n_stream  = p->n_stream;
    const uint32_t cache_size = p->cache_size;

    GGML_ASSERT(n_stream == 1 && "chunked_attn: multi-stream not supported in this milestone");

    const int32_t n_tokens = (int32_t) a->ne[2];
    const int32_t d_k = n_head_kv * head_dim_k;
    const int32_t d_v = n_head_kv * head_dim_v;

    const int32_t n_rep = n_head_q / n_head_kv;

    static std::atomic<bool> logged_once{false};
    if (!logged_once.exchange(true)) {
        LLAMA_LOG_INFO("%s: chunked_attn active (il=%d n_head_q=%d n_head_kv=%d head_dim_k=%d head_dim_v=%d cache_size=%u chunks=%zu)\n",
                __func__, il, n_head_q, n_head_kv, head_dim_k, head_dim_v, cache_size,
                bctx ? bctx->xkv_chunks.size() : (size_t)0);
    }

    float * out = (float *) dst->data;
    const size_t out_size = (size_t) head_dim_v * n_head_q * n_tokens;
    memset(out, 0, out_size * sizeof(float));

    // Causal masking is always on: q_pos is populated for every token by build_attn.
    // A missing q_pos would indicate a caller bug.
    GGML_ASSERT((int32_t) p->q_pos.size() == n_tokens && "chunked_attn: q_pos must cover every token");

    std::vector<float> VKQ_acc((size_t) head_dim_v);

    for (uint32_t s = 0; s < n_stream; ++s) {
        // Build the list of chunks that intersect this layer, populating the
        // per-forward decode cache lazily. Cache lives on bctx and is reset
        // by llama_kv_blocksvd_reset_decode_cache at the start of each ubatch.
        struct chunk_ref {
            const float * k_data;
            const float * v_data;
            int32_t n_tokens;
            const std::vector<llama_pos> * pos;
        };
        std::vector<chunk_ref> chunks;
        chunks.reserve(bctx->xkv_chunks.size());

        auto & decode_cache = bctx->decode_cache;
        if (decode_cache.size() < bctx->xkv_chunks.size()) {
            decode_cache.resize(bctx->xkv_chunks.size());
        }

        for (size_t ci = 0; ci < bctx->xkv_chunks.size(); ++ci) {
            const auto & chunk = bctx->xkv_chunks[ci];
            if (chunk.stream != s) {
                continue;
            }
            if (il < chunk.layer_start || il >= chunk.layer_start + chunk.group_size) {
                continue;
            }

            if (!decode_cache[ci]) {
                auto dc = std::make_unique<llama_kv_blocksvd_context::decoded_chunk>();
                std::string err;
                if (!llama_kv_blocksvd_decompress_xkv_chunk(chunk, dc->k, dc->v, &err)) {
                    LLAMA_LOG_WARN("%s: decompress_xkv_chunk failed for chunk %zu layer_start=%d: %s\n",
                            __func__, ci, chunk.layer_start, err.c_str());
                    continue;
                }
                dc->n_tokens = (int32_t) chunk.slots.size();
                decode_cache[ci] = std::move(dc);
            }

            const int32_t local_layer = il - chunk.layer_start;
            const auto & dc = decode_cache[ci];
            chunk_ref ref;
            ref.k_data   = dc->k[local_layer].data();
            ref.v_data   = dc->v[local_layer].data();
            ref.n_tokens = dc->n_tokens;
            ref.pos      = &chunk.pos;
            chunks.push_back(ref);
        }

        for (int32_t hq = 0; hq < n_head_q; ++hq) {
            const int32_t hkv = hq / n_rep;

            for (int32_t tq = 0; tq < n_tokens; ++tq) {
                const float * q_vec = (const float *)((const char *)a->data +
                    tq * a->nb[2] + hq * a->nb[1]);

                const llama_pos q_position = p->q_pos[tq];

                float M_acc = -INFINITY;
                float S_acc = 0.0f;
                std::fill(VKQ_acc.begin(), VKQ_acc.end(), 0.0f);

                // --- Part 1: Active window (staged cells) ---
                {
                    const auto & s2c = staging->slot_to_cell[s];
                    for (uint32_t slot = 0; slot < cache_size; ++slot) {
                        if (s2c[slot] < 0) {
                            continue;
                        }

                        if ((size_t)slot < p->slot_pos.size() &&
                            p->slot_pos[slot] >= 0 &&
                            p->slot_pos[slot] > q_position) {
                            continue;
                        }

                        const float * k_vec = (const float *)((const char *)b->data +
                            s * b->nb[3] + slot * b->nb[2] + hkv * b->nb[1]);

                        float dot = 0.0f;
                        for (int32_t d = 0; d < head_dim_k; ++d) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        dot *= scale;

                        const float * v_vec = (const float *)((const char *)c->data +
                            s * c->nb[3] + slot * c->nb[2] + hkv * c->nb[1]);

                        float M_new = std::max(M_acc, dot);
                        float exp_old = expf(M_acc - M_new);
                        float exp_new = expf(dot - M_new);
                        float S_new = S_acc * exp_old + exp_new;

                        for (int32_t d = 0; d < head_dim_v; ++d) {
                            VKQ_acc[d] = VKQ_acc[d] * exp_old + v_vec[d] * exp_new;
                        }
                        M_acc = M_new;
                        S_acc = S_new;
                    }
                }

                // --- Part 2: Compressed chunks (lazily decoded via decode_cache) ---
                for (const auto & ref : chunks) {
                    for (int32_t tc = 0; tc < ref.n_tokens; ++tc) {
                        if (ref.pos && !ref.pos->empty() &&
                            (*ref.pos)[tc] > q_position) {
                            continue;
                        }

                        const float * k_vec = ref.k_data + (size_t)tc * d_k + (size_t)hkv * head_dim_k;

                        float dot = 0.0f;
                        for (int32_t d = 0; d < head_dim_k; ++d) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        dot *= scale;

                        const float * v_vec = ref.v_data + (size_t)tc * d_v + (size_t)hkv * head_dim_v;

                        float M_new = std::max(M_acc, dot);
                        float exp_old = expf(M_acc - M_new);
                        float exp_new = expf(dot - M_new);
                        float S_new = S_acc * exp_old + exp_new;

                        for (int32_t d = 0; d < head_dim_v; ++d) {
                            VKQ_acc[d] = VKQ_acc[d] * exp_old + v_vec[d] * exp_new;
                        }
                        M_acc = M_new;
                        S_acc = S_new;
                    }
                }

                // --- Normalize ---
                float * dst_vec = out + (size_t)tq * n_head_q * head_dim_v +
                                        (size_t)hq * head_dim_v;
                if (S_acc > 0.0f) {
                    const float inv_s = 1.0f / S_acc;
                    for (int32_t d = 0; d < head_dim_v; ++d) {
                        dst_vec[d] = VKQ_acc[d] * inv_s;
                    }
                }
            }
        }
    }
}

// Ensure the rank-domain factor cache for xkv_chunk[ci] is populated on bctx.
// Returns pointer to the rank_chunk (never null on success), or nullptr on failure.
// All layers in the group share this cache — call once per (chunk, forward).
static const llama_kv_blocksvd_context::rank_chunk * lowrank_direct_ensure_rank_cache(
        const llama_kv_blocksvd_context * bctx, size_t ci) {
    auto & cache = bctx->rank_cache;
    if (cache.size() < bctx->xkv_chunks.size()) {
        cache.resize(bctx->xkv_chunks.size());
    }
    if (cache[ci]) {
        return cache[ci].get();
    }

    const auto & chunk = bctx->xkv_chunks[ci];
    const int32_t group_size = chunk.group_size;
    const int32_t n_tokens   = (int32_t) chunk.slots.size();
    const int32_t combined_k_dim = group_size * chunk.n_head_kv * chunk.head_dim_k;
    const int32_t combined_v_dim = group_size * chunk.n_head_kv * chunk.head_dim_v;

    const auto dequant_factor = [&](const llama_kv_blocksvd_xkv_factors & f,
                                    int32_t n_dim,
                                    std::vector<float> & us_out,
                                    std::vector<float> & vh_out) -> bool {
        const int32_t r = f.rank;
        if (r <= 0) {
            return false;
        }
        std::vector<float> u_f((size_t) n_tokens * r);
        std::vector<float> s_f(r);
        vh_out.assign((size_t) r * n_dim, 0.0f);
        llama_kv_blocksvd_dequantize_symmetric(f.u_q.data(),  u_f.data(),  (int32_t) u_f.size(),  f.quant_bits, f.u_scale);
        llama_kv_blocksvd_dequantize_symmetric(f.s_q.data(),  s_f.data(),  (int32_t) s_f.size(),  f.quant_bits, f.s_scale);
        llama_kv_blocksvd_dequantize_symmetric(f.vh_q.data(), vh_out.data(),(int32_t) vh_out.size(), f.quant_bits, f.vh_scale);
        // Premultiply US = U * S (row-major, U[t,i] * S[i])
        us_out.assign((size_t) n_tokens * r, 0.0f);
        for (int32_t t = 0; t < n_tokens; ++t) {
            for (int32_t i = 0; i < r; ++i) {
                us_out[(size_t) t * r + i] = u_f[(size_t) t * r + i] * s_f[i];
            }
        }
        return true;
    };

    auto rc = std::make_unique<llama_kv_blocksvd_context::rank_chunk>();
    rc->n_tokens = n_tokens;
    rc->r_k = chunk.k_factors.rank;
    rc->r_v = chunk.v_factors.rank;
    rc->combined_k_dim = combined_k_dim;
    rc->combined_v_dim = combined_v_dim;
    if (!dequant_factor(chunk.k_factors, combined_k_dim, rc->us_k, rc->vh_k)) {
        return nullptr;
    }
    if (!dequant_factor(chunk.v_factors, combined_v_dim, rc->us_v, rc->vh_v)) {
        return nullptr;
    }
    cache[ci] = std::move(rc);
    return cache[ci].get();
}

void llama_kv_lowrank_direct_attn_compute(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    const struct ggml_tensor * a = dst->src[0]; // Q
    const struct ggml_tensor * b = dst->src[1]; // K (staged, active window only)
    const struct ggml_tensor * c = dst->src[2]; // V (staged, active window only)

    auto * p = (llama_chunked_attn_params *) userdata;
    const auto * bctx        = p->bctx;
    const auto * staging     = p->staging;
    const int32_t il         = p->il;
    const int32_t n_head_kv  = p->n_head_kv;
    const int32_t n_head_q   = p->n_head_q;
    const int32_t head_dim_k = p->head_dim_k;
    const int32_t head_dim_v = p->head_dim_v;
    const float   scale      = p->scale;
    const uint32_t n_stream  = p->n_stream;
    const uint32_t cache_size = p->cache_size;

    GGML_ASSERT(n_stream == 1 && "kv_lowrank_direct: multi-stream not supported in this milestone");

    const int32_t n_tokens = (int32_t) a->ne[2];
    const int32_t d_k_flat = n_head_kv * head_dim_k;
    const int32_t d_v_flat = n_head_kv * head_dim_v;
    (void) d_k_flat;
    (void) d_v_flat;

    const int32_t n_rep = n_head_q / n_head_kv;

    static std::atomic<bool> logged_once{false};
    if (!logged_once.exchange(true)) {
        LLAMA_LOG_INFO("%s: kv_lowrank_direct active (il=%d n_head_q=%d n_head_kv=%d head_dim_k=%d head_dim_v=%d cache_size=%u chunks=%zu)\n",
                __func__, il, n_head_q, n_head_kv, head_dim_k, head_dim_v, cache_size,
                bctx ? bctx->xkv_chunks.size() : (size_t) 0);
    }

    float * out = (float *) dst->data;
    const size_t out_size = (size_t) head_dim_v * n_head_q * n_tokens;
    memset(out, 0, out_size * sizeof(float));

    GGML_ASSERT((int32_t) p->q_pos.size() == n_tokens && "kv_lowrank_direct: q_pos must cover every token");

    // Build the per-forward list of (rank-cache, layer_offset) refs for chunks
    // that intersect this layer. layer_offset selects which slice of Vh's
    // combined_dim belongs to this il inside the cross-layer group.
    struct chunk_ref {
        const llama_kv_blocksvd_context::rank_chunk * rc;
        const std::vector<llama_pos> * pos;
        int32_t layer_offset_k;   // local_layer * n_head_kv * head_dim_k
        int32_t layer_offset_v;   // local_layer * n_head_kv * head_dim_v
    };
    std::vector<chunk_ref> refs;
    refs.reserve(bctx->xkv_chunks.size());

    for (uint32_t s = 0; s < n_stream; ++s) {
        for (size_t ci = 0; ci < bctx->xkv_chunks.size(); ++ci) {
            const auto & chunk = bctx->xkv_chunks[ci];
            if (chunk.stream != s) {
                continue;
            }
            if (il < chunk.layer_start || il >= chunk.layer_start + chunk.group_size) {
                continue;
            }
            const auto * rc = lowrank_direct_ensure_rank_cache(bctx, ci);
            if (!rc) {
                LLAMA_LOG_WARN("%s: rank-cache build failed for chunk %zu\n", __func__, ci);
                continue;
            }
            const int32_t local_layer = il - chunk.layer_start;
            chunk_ref ref;
            ref.rc = rc;
            ref.pos = &chunk.pos;
            ref.layer_offset_k = local_layer * n_head_kv * head_dim_k;
            ref.layer_offset_v = local_layer * n_head_kv * head_dim_v;
            refs.push_back(ref);
        }

        std::vector<float> VKQ_acc((size_t) head_dim_v);
        // Rank-domain scratch, reused across (hq, tq, chunk) iterations.
        // Sized to the max rank we may encounter in refs; grown lazily below.
        std::vector<float> qVh_k;
        std::vector<float> v_val((size_t) head_dim_v);

        for (int32_t hq = 0; hq < n_head_q; ++hq) {
            const int32_t hkv = hq / n_rep;

            for (int32_t tq = 0; tq < n_tokens; ++tq) {
                const float * q_vec = (const float *)((const char *) a->data +
                    tq * a->nb[2] + hq * a->nb[1]);

                const llama_pos q_position = p->q_pos[tq];

                float M_acc = -INFINITY;
                float S_acc = 0.0f;
                std::fill(VKQ_acc.begin(), VKQ_acc.end(), 0.0f);

                // --- Part 1: Active window (staged dense cells) ---
                // Identical to llama_chunked_attn_compute.
                {
                    const auto & s2c = staging->slot_to_cell[s];
                    for (uint32_t slot = 0; slot < cache_size; ++slot) {
                        if (s2c[slot] < 0) {
                            continue;
                        }

                        if ((size_t) slot < p->slot_pos.size() &&
                            p->slot_pos[slot] >= 0 &&
                            p->slot_pos[slot] > q_position) {
                            continue;
                        }

                        const float * k_vec = (const float *)((const char *) b->data +
                            s * b->nb[3] + slot * b->nb[2] + hkv * b->nb[1]);

                        float dot = 0.0f;
                        for (int32_t d = 0; d < head_dim_k; ++d) {
                            dot += q_vec[d] * k_vec[d];
                        }
                        dot *= scale;

                        const float * v_vec = (const float *)((const char *) c->data +
                            s * c->nb[3] + slot * c->nb[2] + hkv * c->nb[1]);

                        float M_new = std::max(M_acc, dot);
                        float exp_old = expf(M_acc - M_new);
                        float exp_new = expf(dot - M_new);
                        float S_new = S_acc * exp_old + exp_new;

                        for (int32_t d = 0; d < head_dim_v; ++d) {
                            VKQ_acc[d] = VKQ_acc[d] * exp_old + v_vec[d] * exp_new;
                        }
                        M_acc = M_new;
                        S_acc = S_new;
                    }
                }

                // --- Part 2: Compressed chunks (rank-domain) ---
                for (const auto & ref : refs) {
                    const auto * rc = ref.rc;
                    const int32_t r_k = rc->r_k;
                    const int32_t r_v = rc->r_v;
                    const int32_t combined_k_dim = rc->combined_k_dim;
                    const int32_t combined_v_dim = rc->combined_v_dim;

                    // qVh_k[i] = Σ_d q_vec[d] * vh_k[i, layer_offset_k + hkv*head_dim_k + d]
                    if ((int32_t) qVh_k.size() < r_k) {
                        qVh_k.resize(r_k);
                    }
                    const int32_t vh_k_head_off = ref.layer_offset_k + hkv * head_dim_k;
                    for (int32_t i = 0; i < r_k; ++i) {
                        const float * vh_row = rc->vh_k.data() + (size_t) i * combined_k_dim + vh_k_head_off;
                        float acc = 0.0f;
                        for (int32_t d = 0; d < head_dim_k; ++d) {
                            acc += q_vec[d] * vh_row[d];
                        }
                        qVh_k[i] = acc;
                    }

                    const int32_t vh_v_head_off = ref.layer_offset_v + hkv * head_dim_v;

                    for (int32_t tc = 0; tc < rc->n_tokens; ++tc) {
                        if (ref.pos && !ref.pos->empty() &&
                            (*ref.pos)[tc] > q_position) {
                            continue;
                        }

                        // logit = (qVh_k · US_k[tc, :]) * scale
                        float logit = 0.0f;
                        const float * us_k_row = rc->us_k.data() + (size_t) tc * r_k;
                        for (int32_t i = 0; i < r_k; ++i) {
                            logit += qVh_k[i] * us_k_row[i];
                        }
                        logit *= scale;

                        // v_val[d] = Σ_i us_v[tc, i] * vh_v[i, layer_offset_v + hkv*head_dim_v + d]
                        // On-the-fly V reconstruct — rank-r inner product per output dim.
                        const float * us_v_row = rc->us_v.data() + (size_t) tc * r_v;
                        for (int32_t d = 0; d < head_dim_v; ++d) {
                            float vd = 0.0f;
                            for (int32_t i = 0; i < r_v; ++i) {
                                const float vhv = rc->vh_v[(size_t) i * combined_v_dim + vh_v_head_off + d];
                                vd += us_v_row[i] * vhv;
                            }
                            v_val[d] = vd;
                        }

                        float M_new = std::max(M_acc, logit);
                        float exp_old = expf(M_acc - M_new);
                        float exp_new = expf(logit - M_new);
                        float S_new = S_acc * exp_old + exp_new;

                        for (int32_t d = 0; d < head_dim_v; ++d) {
                            VKQ_acc[d] = VKQ_acc[d] * exp_old + v_val[d] * exp_new;
                        }
                        M_acc = M_new;
                        S_acc = S_new;
                    }
                }

                // --- Normalize ---
                float * dst_vec = out + (size_t) tq * n_head_q * head_dim_v +
                                        (size_t) hq * head_dim_v;
                if (S_acc > 0.0f) {
                    const float inv_s = 1.0f / S_acc;
                    for (int32_t d = 0; d < head_dim_v; ++d) {
                        dst_vec[d] = VKQ_acc[d] * inv_s;
                    }
                }
            }
        }

        refs.clear();
    }
}
