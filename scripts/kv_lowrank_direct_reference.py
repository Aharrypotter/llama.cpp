#!/usr/bin/env python3
"""Reference NumPy implementation of the direct low-rank attention consumer.

For BlockSVD-compressed KV with independent K/V SVDs (post-788b24de3), verifies
elementwise agreement between:

  reconstruct: attn = softmax(Q @ K^T / sqrt(d_k)) @ V   where K = U_k @ diag(S_k) @ Vh_k
  direct    : attn = softmax((Q @ Vh_k^T) @ (U_k @ diag(S_k))^T / sqrt(d_k)) @ V
              with V-side on-the-fly rank-r reconstruct: v_val = (U_v[t] * S_v) @ Vh_v[hkv]

Passing this file (fp32, no quant) validates the algebra used by the P3 kernel.
Storage layout matches llama-kv-blocksvd.h:16-58 (merged-heads Vh).
"""

import numpy as np
import sys

RTOL_FP32 = 1e-4
ATOL_FP32 = 1e-5


def gen_factors(block_size, r, n_head_kv, d_k, d_v, rng):
    """Random SVD-like factors matching BlockSVD storage layout (merged-heads)."""
    U_k = rng.standard_normal((block_size, r)).astype(np.float32)
    S_k = np.abs(rng.standard_normal(r).astype(np.float32)) + 0.1
    Vh_k = rng.standard_normal((r, n_head_kv * d_k)).astype(np.float32)

    U_v = rng.standard_normal((block_size, r)).astype(np.float32)
    S_v = np.abs(rng.standard_normal(r).astype(np.float32)) + 0.1
    Vh_v = rng.standard_normal((r, n_head_kv * d_v)).astype(np.float32)

    return U_k, S_k, Vh_k, U_v, S_v, Vh_v


def reconstruct_attn(Q, factors_per_chunk, chunk_positions, q_positions,
                     n_head_q, n_head_kv, d_k, d_v):
    """Baseline: reconstruct dense K/V per block, then standard softmax attention.

    Q shape: [n_tokens_q, n_head_q, d_k]
    factors_per_chunk: list of (U_k, S_k, Vh_k, U_v, S_v, Vh_v)
    chunk_positions[ci]: [block_size] absolute pos of each cell in chunk ci
    q_positions: [n_tokens_q] absolute pos of each query token
    """
    n_tokens_q = Q.shape[0]
    n_rep = n_head_q // n_head_kv
    inv_sqrt_dk = 1.0 / np.sqrt(d_k)
    out = np.zeros((n_tokens_q, n_head_q, d_v), dtype=np.float32)

    # Reconstruct all K/V per chunk once.
    K_chunks = []  # [ci] -> [block_size, n_head_kv, d_k]
    V_chunks = []
    for (U_k, S_k, Vh_k, U_v, S_v, Vh_v) in factors_per_chunk:
        K_flat = (U_k * S_k) @ Vh_k  # [block_size, n_head_kv * d_k]
        V_flat = (U_v * S_v) @ Vh_v
        K_chunks.append(K_flat.reshape(-1, n_head_kv, d_k))
        V_chunks.append(V_flat.reshape(-1, n_head_kv, d_v))

    for tq in range(n_tokens_q):
        q_pos = q_positions[tq]
        for hq in range(n_head_q):
            hkv = hq // n_rep
            q = Q[tq, hq]  # [d_k]

            # collect all valid (K, V) pairs across chunks under causal mask
            keys, vals = [], []
            for ci, K_blk in enumerate(K_chunks):
                V_blk = V_chunks[ci]
                block_size = K_blk.shape[0]
                for t_k in range(block_size):
                    if chunk_positions[ci][t_k] > q_pos:
                        break
                    keys.append(K_blk[t_k, hkv])
                    vals.append(V_blk[t_k, hkv])
            if not keys:
                continue
            K = np.stack(keys)  # [T, d_k]
            V = np.stack(vals)  # [T, d_v]
            logits = (K @ q) * inv_sqrt_dk  # [T]
            m = logits.max()
            w = np.exp(logits - m)
            p = w / w.sum()
            out[tq, hq] = p @ V

    return out


def direct_attn(Q, factors_per_chunk, chunk_positions, q_positions,
                n_head_q, n_head_kv, d_k, d_v):
    """Direct consumer: rank-domain qVh dot + on-the-fly V-side reconstruct."""
    n_tokens_q = Q.shape[0]
    n_rep = n_head_q // n_head_kv
    inv_sqrt_dk = 1.0 / np.sqrt(d_k)
    out = np.zeros((n_tokens_q, n_head_q, d_v), dtype=np.float32)

    # Pre-compute US_k, US_v per chunk (fp32 rank-vector premultiply).
    prep = []
    for (U_k, S_k, Vh_k, U_v, S_v, Vh_v) in factors_per_chunk:
        US_k = (U_k * S_k).astype(np.float32)  # [block_size, r]
        US_v = (U_v * S_v).astype(np.float32)
        prep.append((US_k, Vh_k, US_v, Vh_v))

    for tq in range(n_tokens_q):
        q_pos = q_positions[tq]
        for hq in range(n_head_q):
            hkv = hq // n_rep
            q = Q[tq, hq]  # [d_k]

            m = -1e30
            l = 0.0
            o = np.zeros(d_v, dtype=np.float32)

            for ci, (US_k, Vh_k, US_v, Vh_v) in enumerate(prep):
                block_size = US_k.shape[0]
                # Vh_k head slice: [r, d_k]  — merged-heads pointer arithmetic
                Vh_k_head = Vh_k.reshape(-1, n_head_kv, d_k)[:, hkv, :]
                Vh_v_head = Vh_v.reshape(-1, n_head_kv, d_v)[:, hkv, :]
                qVh_k = Vh_k_head @ q  # [r]

                for t_k in range(block_size):
                    if chunk_positions[ci][t_k] > q_pos:
                        break
                    logit = (qVh_k @ US_k[t_k]) * inv_sqrt_dk
                    v_val = US_v[t_k] @ Vh_v_head  # [d_v] — rank-r on the fly

                    m_new = max(m, logit)
                    w = np.exp(logit - m_new)
                    scale_old = np.exp(m - m_new)
                    o = o * scale_old + w * v_val
                    l = l * scale_old + w
                    m = m_new

            if l > 0:
                out[tq, hq] = o / l

    return out


def run_case(name, block_size, r, n_head_kv, n_head_q, d_k, d_v, n_chunks, n_tokens_q, seed):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n_tokens_q, n_head_q, d_k)).astype(np.float32) * 0.1

    factors = [gen_factors(block_size, r, n_head_kv, d_k, d_v, rng) for _ in range(n_chunks)]
    chunk_positions = [np.arange(ci * block_size, (ci + 1) * block_size) for ci in range(n_chunks)]
    total_kv = n_chunks * block_size
    q_positions = np.linspace(total_kv // 2, total_kv - 1, n_tokens_q).astype(int)

    a = reconstruct_attn(Q, factors, chunk_positions, q_positions,
                         n_head_q, n_head_kv, d_k, d_v)
    b = direct_attn(Q, factors, chunk_positions, q_positions,
                    n_head_q, n_head_kv, d_k, d_v)

    max_abs = np.max(np.abs(a - b))
    max_rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-8))
    passed = np.allclose(a, b, rtol=RTOL_FP32, atol=ATOL_FP32)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name} seed={seed}  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
    return passed


def main():
    cases = [
        ("tiny",       dict(block_size=8,  r=4,  n_head_kv=1, n_head_q=1,  d_k=8,   d_v=8,   n_chunks=2, n_tokens_q=3)),
        ("mid",        dict(block_size=64, r=16, n_head_kv=8, n_head_q=16, d_k=64,  d_v=64,  n_chunks=3, n_tokens_q=4)),
        ("qwen3-shape", dict(block_size=64, r=32, n_head_kv=8, n_head_q=16, d_k=128, d_v=128, n_chunks=2, n_tokens_q=2)),
    ]
    seeds = [0, 1, 2, 3, 42, 100, 2026, 7, 12, 99]

    all_pass = True
    for name, kw in cases:
        print(f"case: {name}  {kw}")
        for s in seeds:
            ok = run_case(name, seed=s, **kw)
            all_pass = all_pass and ok
        print()

    if all_pass:
        print("ALL PASS")
        sys.exit(0)
    else:
        print("SOME FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    main()
