#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_hf_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_int(config: Dict[str, Any], keys: list[str], fallback: Optional[int]) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return int(value)
    return fallback


def infer_head_dim(config: Dict[str, Any], fallback: Optional[int]) -> Optional[int]:
    value = pick_int(config, ["head_dim", "kv_channels"], fallback)
    if value is not None:
        return value

    hidden_size = config.get("hidden_size")
    n_head = config.get("num_attention_heads")
    if hidden_size is not None and n_head is not None:
        return int(hidden_size) // int(n_head)

    return None


def dtype_size(dtype: str) -> int:
    if dtype == "f16":
        return 2
    if dtype == "f32":
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def pack_value(dtype: str, value: float) -> bytes:
    if dtype == "f16":
        return struct.pack("<e", value)
    if dtype == "f32":
        return struct.pack("<f", value)
    raise ValueError(f"unsupported dtype: {dtype}")


def write_dummy_basis(path: Path, rank: int, d_kv: int, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    zero = pack_value(dtype, 0.0)
    one = pack_value(dtype, 1.0)

    with path.open("wb") as f:
        for r in range(rank):
            hot_col = r % d_kv
            for c in range(d_kv):
                f.write(one if c == hot_col else zero)


def write_array_basis(path: Path, basis: Any, dtype: str) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.dtype("<f2" if dtype == "f16" else "<f4")
    np.asarray(basis, dtype=np_dtype).tofile(path)


def sample_key(layer: int, kind: str) -> Tuple[str, ...]:
    return (
        f"layer_{layer:03d}.{kind}",
        f"layer_{layer:03d}_{kind}",
        f"layer_{layer}.{kind}",
        f"layer_{layer}_{kind}",
        f"{kind}_{layer:03d}",
        f"{kind}_{layer}",
    )


def load_sample_array(samples: Any, layer: int, kind: str) -> Any:
    for key in sample_key(layer, kind):
        if key in samples:
            return samples[key]

    candidates = ", ".join(sample_key(layer, kind))
    raise KeyError(f"missing {kind.upper()} samples for layer {layer}; tried: {candidates}")


def compute_uncentered_svd_basis(samples: Any, rank: int, d_kv: int) -> Any:
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        raise ValueError("sample array is empty")
    if x.shape[-1] != d_kv:
        raise ValueError(f"sample d_kv mismatch: got {x.shape[-1]}, expected {d_kv}")

    x = x.reshape(-1, d_kv)
    if x.shape[0] == 0:
        raise ValueError("sample array has zero rows after reshape")

    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = np.zeros((rank, d_kv), dtype=np.float32)
    n_copy = min(rank, vt.shape[0])
    basis[:n_copy, :] = vt[:n_copy, :]
    return basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a WHLR-KV sidecar manifest and low-rank KV basis files.",
    )
    parser.add_argument("--hf-config", type=Path, help="optional Hugging Face config.json path")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--rank", type=int, default=32, help="default low-rank basis rank")
    parser.add_argument("--rank-k", type=int, help="low-rank basis rank for K (default: --rank)")
    parser.add_argument("--rank-v", type=int, help="low-rank basis rank for V (default: --rank)")
    parser.add_argument("--n-layer", type=int, help="number of transformer layers")
    parser.add_argument("--head-dim", type=int, help="per-head K/V dimension")
    parser.add_argument("--n-head-kv", type=int, help="number of K/V heads")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="basis dtype")
    parser.add_argument("--manifest-name", default="whlr_kv_basis.json", help="manifest filename")
    parser.add_argument(
        "--samples-npz",
        type=Path,
        help=(
            "optional calibration samples in npz format; keys may be "
            "layer_000.k/layer_000.v, layer_000_k/layer_000_v, k_0/v_0, etc."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hf_config(args.hf_config)

    n_layer = pick_int(config, ["num_hidden_layers", "n_layer", "num_layers"], args.n_layer)
    head_dim = infer_head_dim(config, args.head_dim)
    n_head_kv = pick_int(config, ["num_key_value_heads", "n_head_kv", "num_attention_heads"], args.n_head_kv)

    if n_layer is None or n_layer <= 0:
        raise SystemExit("error: unable to infer n_layer; pass --n-layer or --hf-config")
    if head_dim is None or head_dim <= 0:
        raise SystemExit("error: unable to infer head_dim; pass --head-dim or --hf-config")
    if n_head_kv is None or n_head_kv <= 0:
        raise SystemExit("error: unable to infer n_head_kv; pass --n-head-kv or --hf-config")
    rank_k = args.rank_k if args.rank_k is not None else args.rank
    rank_v = args.rank_v if args.rank_v is not None else args.rank
    if args.rank <= 0 or rank_k <= 0 or rank_v <= 0:
        raise SystemExit("error: --rank and optional --rank-k/--rank-v must be positive")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d_kv = head_dim * n_head_kv
    expected_k_size = rank_k * d_kv * dtype_size(args.dtype)
    expected_v_size = rank_v * d_kv * dtype_size(args.dtype)
    samples = None
    if args.samples_npz is not None:
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit("error: --samples-npz requires numpy") from exc

        samples = np.load(args.samples_npz)

    layers = []
    for layer in range(n_layer):
        k_name = f"layer_{layer:03d}.bk.{args.dtype}.bin"
        v_name = f"layer_{layer:03d}.bv.{args.dtype}.bin"
        if samples is None:
            write_dummy_basis(out_dir / k_name, rank_k, d_kv, args.dtype)
            write_dummy_basis(out_dir / v_name, rank_v, d_kv, args.dtype)
        else:
            try:
                k_basis = compute_uncentered_svd_basis(load_sample_array(samples, layer, "k"), rank_k, d_kv)
                v_basis = compute_uncentered_svd_basis(load_sample_array(samples, layer, "v"), rank_v, d_kv)
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"error: layer {layer}: {exc}") from exc

            write_array_basis(out_dir / k_name, k_basis, args.dtype)
            write_array_basis(out_dir / v_name, v_basis, args.dtype)

        layers.append({"layer": layer, "k": k_name, "v": v_name})

    manifest = {
        "format": "whlr-kv-basis",
        "version": 1,
        "dtype": args.dtype,
        "layout": "row-major",
        "rank": max(rank_k, rank_v),
        "rank_k": rank_k,
        "rank_v": rank_v,
        "n_layer": n_layer,
        "head_dim": head_dim,
        "n_head_kv": n_head_kv,
        "layers": layers,
    }

    manifest_path = out_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    mode = "uncentered-svd" if samples is not None else "dummy-identity"
    print(f"wrote manifest: {manifest_path}")
    print(f"mode: {mode}")
    print(
        f"layers: {n_layer}, rank_k: {rank_k}, rank_v: {rank_v}, d_kv: {d_kv}, "
        f"K basis bytes/file: {expected_k_size}, V basis bytes/file: {expected_v_size}"
    )


if __name__ == "__main__":
    main()
