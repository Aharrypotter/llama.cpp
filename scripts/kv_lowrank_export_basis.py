#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a WHLR-KV sidecar manifest and dummy low-rank KV basis files.",
    )
    parser.add_argument("--hf-config", type=Path, help="optional Hugging Face config.json path")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--rank", type=int, default=32, help="low-rank basis rank")
    parser.add_argument("--n-layer", type=int, help="number of transformer layers")
    parser.add_argument("--head-dim", type=int, help="per-head K/V dimension")
    parser.add_argument("--n-head-kv", type=int, help="number of K/V heads")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="basis dtype")
    parser.add_argument("--manifest-name", default="whlr_kv_basis.json", help="manifest filename")
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
    if args.rank <= 0:
        raise SystemExit("error: --rank must be positive")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d_kv = head_dim * n_head_kv
    expected_size = args.rank * d_kv * dtype_size(args.dtype)

    layers = []
    for layer in range(n_layer):
        k_name = f"layer_{layer:03d}.bk.{args.dtype}.bin"
        v_name = f"layer_{layer:03d}.bv.{args.dtype}.bin"
        write_dummy_basis(out_dir / k_name, args.rank, d_kv, args.dtype)
        write_dummy_basis(out_dir / v_name, args.rank, d_kv, args.dtype)
        layers.append({"layer": layer, "k": k_name, "v": v_name})

    manifest = {
        "format": "whlr-kv-basis",
        "version": 1,
        "dtype": args.dtype,
        "layout": "row-major",
        "rank": args.rank,
        "n_layer": n_layer,
        "head_dim": head_dim,
        "n_head_kv": n_head_kv,
        "layers": layers,
    }

    manifest_path = out_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"wrote manifest: {manifest_path}")
    print(f"layers: {n_layer}, rank: {args.rank}, d_kv: {d_kv}, basis bytes/file: {expected_size}")


if __name__ == "__main__":
    main()

