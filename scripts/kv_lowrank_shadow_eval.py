#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


RECON_RE = re.compile(
    r"kv_lowrank_shadow_project_current: .*?"
    r"projected_tokens=(?P<projected_tokens>\d+) .*?"
    r"projected_chunks=(?P<projected_chunks>\d+) .*?"
    r"history_bytes=(?P<history_bytes>\d+) .*?"
    r"recon_err_k_max=(?P<k_max>[0-9.eE+-]+) "
    r"recon_err_k_mean=(?P<k_mean>[0-9.eE+-]+) "
    r"recon_err_v_max=(?P<v_max>[0-9.eE+-]+) "
    r"recon_err_v_mean=(?P<v_mean>[0-9.eE+-]+)"
)


def run_cmd(cmd: List[str], log_path: Path | None = None) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout


def parse_ranks(value: str) -> List[int]:
    ranks = [int(item) for item in value.split(",") if item.strip()]
    if not ranks or any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be a comma-separated list of positive integers")
    return ranks


def parse_rank_pairs(value: str) -> List[Tuple[int, int]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("rank pairs must use K:V entries, for example 32:64,32:128")
        k_rank, v_rank = (int(part) for part in item.split(":", 1))
        pairs.append((k_rank, v_rank))
    if not pairs or any(k_rank <= 0 or v_rank <= 0 for k_rank, v_rank in pairs):
        raise argparse.ArgumentTypeError("rank pairs must contain positive integers")
    return pairs


def parse_recon_log(text: str) -> Dict[str, Any]:
    records = []
    for match in RECON_RE.finditer(text):
        item = {
            "projected_tokens": int(match.group("projected_tokens")),
            "projected_chunks": int(match.group("projected_chunks")),
            "history_bytes": int(match.group("history_bytes")),
            "k_max": float(match.group("k_max")),
            "k_mean": float(match.group("k_mean")),
            "v_max": float(match.group("v_max")),
            "v_mean": float(match.group("v_mean")),
        }
        if item["projected_tokens"] > 0:
            records.append(item)

    if not records:
        return {
            "steps": 0,
            "projected_tokens": 0,
            "projected_chunks": 0,
            "history_bytes": 0,
            "k_max": 0.0,
            "k_mean_max": 0.0,
            "v_max": 0.0,
            "v_mean_max": 0.0,
        }

    return {
        "steps": len(records),
        "projected_tokens": sum(item["projected_tokens"] for item in records),
        "projected_chunks": sum(item["projected_chunks"] for item in records),
        "history_bytes": max(item["history_bytes"] for item in records),
        "k_max": max(item["k_max"] for item in records),
        "k_mean_max": max(item["k_mean"] for item in records),
        "v_max": max(item["v_max"] for item in records),
        "v_mean_max": max(item["v_mean"] for item in records),
    }


def llama_cmd(args: argparse.Namespace, basis_path: Path, rank_k: int, rank_v: int, extra: List[str]) -> List[str]:
    return [
        str(args.llama_cli),
        "-m", str(args.model),
        "-p", args.prompt,
        "-n", str(args.n_predict),
        "-c", str(args.ctx_size),
        "--single-turn",
        "--no-warmup",
        "--simple-io",
        "--no-display-prompt",
        "--verbose",
        "--kv-lowrank",
        "--kv-lowrank-basis-path", str(basis_path),
        "--kv-lowrank-rank", str(max(rank_k, rank_v)),
        "--kv-lowrank-rank-k", str(rank_k),
        "--kv-lowrank-rank-v", str(rank_v),
        "--kv-lowrank-window", str(args.window),
        "--kv-lowrank-chunk", str(args.chunk),
        *args.llama_arg,
        *extra,
    ]


def export_basis_cmd(args: argparse.Namespace, out_dir: Path, rank_k: int, rank_v: int, samples_npz: Path | None) -> List[str]:
    cmd = [
        "python3",
        str(args.exporter),
        "--out-dir", str(out_dir),
        "--rank", str(max(rank_k, rank_v)),
        "--rank-k", str(rank_k),
        "--rank-v", str(rank_v),
        "--n-layer", str(args.n_layer),
        "--head-dim", str(args.head_dim),
        "--n-head-kv", str(args.n_head_kv),
        "--dtype", args.dtype,
    ]
    if samples_npz is not None:
        cmd.extend(["--samples-npz", str(samples_npz)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect WHLR-KV dense samples, export SVD bases, and run shadow reconstruction eval.",
    )
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, default=Path("scripts/kv_lowrank_export_basis.py"))
    parser.add_argument("--ranks", type=parse_ranks, default=[4])
    parser.add_argument("--rank-pairs", type=parse_rank_pairs, help="comma-separated asymmetric K:V ranks, e.g. 32:64,32:128")
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--n-head-kv", type=int, required=True)
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--sample-max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="hi")
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument(
        "--llama-arg",
        action="append",
        default=[],
        help="extra argument passed through to llama-cli; repeat for multi-token options",
    )
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rank_pairs = args.rank_pairs if args.rank_pairs is not None else [(rank, rank) for rank in args.ranks]

    dummy_dir = args.work_dir / "dummy_basis"
    run_cmd(export_basis_cmd(args, dummy_dir, rank_pairs[0][0], rank_pairs[0][1], None), args.work_dir / "collect_basis.log")
    dummy_manifest = dummy_dir / "whlr_kv_basis.json"

    samples_npz = args.work_dir / "kv_samples.npz"
    collect_log = args.work_dir / "collect_samples.log"
    run_cmd(
        llama_cmd(
            args,
            dummy_manifest,
            rank_pairs[0][0],
            rank_pairs[0][1],
            [
                "--kv-lowrank-samples-out", str(samples_npz),
                "--kv-lowrank-sample-max-tokens", str(args.sample_max_tokens),
            ],
        ),
        collect_log,
    )

    results = {
        "samples_npz": str(samples_npz),
        "rank_pairs": [],
    }

    for rank_k, rank_v in rank_pairs:
        rank_dir = args.work_dir / f"basis_k{rank_k}_v{rank_v}"
        run_cmd(export_basis_cmd(args, rank_dir, rank_k, rank_v, samples_npz), args.work_dir / f"export_k{rank_k}_v{rank_v}.log")

        eval_log = args.work_dir / f"eval_k{rank_k}_v{rank_v}.log"
        text = run_cmd(llama_cmd(args, rank_dir / "whlr_kv_basis.json", rank_k, rank_v, []), eval_log)
        metrics = parse_recon_log(text)
        metrics["rank"] = max(rank_k, rank_v)
        metrics["rank_k"] = rank_k
        metrics["rank_v"] = rank_v
        metrics["basis_manifest"] = str(rank_dir / "whlr_kv_basis.json")
        metrics["eval_log"] = str(eval_log)
        results["rank_pairs"].append(metrics)

    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
