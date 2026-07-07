#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


PPL_RE = re.compile(r"Final estimate: PPL = (?P<ppl>[0-9.eE+-]+) \+/- (?P<unc>[0-9.eE+-]+)")
WHLR_RE = re.compile(
    r"kv_lowrank_shadow_project_current: .*?"
    r"projected_tokens=(?P<projected_tokens>\d+) .*?"
    r"projected_chunks=(?P<projected_chunks>\d+) .*?"
    r"projected_layers=(?P<projected_layers>\d+) .*?"
    r"reconstructed_layers=(?P<reconstructed_layers>\d+) .*?"
    r"history_bytes=(?P<history_bytes>\d+) .*?"
    r"recon_err_k_max=(?P<k_max>[0-9.eE+-]+) "
    r"recon_err_k_mean=(?P<k_mean>[0-9.eE+-]+) "
    r"recon_err_v_max=(?P<v_max>[0-9.eE+-]+) "
    r"recon_err_v_mean=(?P<v_mean>[0-9.eE+-]+)"
)


def run_cmd(cmd: List[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout


def parse_ppl(text: str) -> Dict[str, Any]:
    matches = list(PPL_RE.finditer(text))
    if not matches:
        return {"found": False, "ppl": None, "uncertainty": None}

    match = matches[-1]
    return {
        "found": True,
        "ppl": float(match.group("ppl")),
        "uncertainty": float(match.group("unc")),
    }


def parse_whlr(text: str) -> Dict[str, Any]:
    records = []
    for match in WHLR_RE.finditer(text):
        records.append({
            "projected_tokens": int(match.group("projected_tokens")),
            "projected_chunks": int(match.group("projected_chunks")),
            "projected_layers": int(match.group("projected_layers")),
            "reconstructed_layers": int(match.group("reconstructed_layers")),
            "history_bytes": int(match.group("history_bytes")),
            "k_max": float(match.group("k_max")),
            "k_mean": float(match.group("k_mean")),
            "v_max": float(match.group("v_max")),
            "v_mean": float(match.group("v_mean")),
        })

    projected = [item for item in records if item["projected_tokens"] > 0]
    return {
        "steps": len(records),
        "projected_steps": len(projected),
        "projected_tokens": sum(item["projected_tokens"] for item in projected),
        "projected_chunks": sum(item["projected_chunks"] for item in projected),
        "projected_layers_max": max((item["projected_layers"] for item in projected), default=0),
        "reconstructed_layers_max": max((item["reconstructed_layers"] for item in projected), default=0),
        "history_bytes": max((item["history_bytes"] for item in records), default=0),
        "k_max": max((item["k_max"] for item in projected), default=0.0),
        "k_mean_max": max((item["k_mean"] for item in projected), default=0.0),
        "v_max": max((item["v_max"] for item in projected), default=0.0),
        "v_mean_max": max((item["v_mean"] for item in projected), default=0.0),
    }


def ppl_cmd(args: argparse.Namespace, mode: str) -> List[str]:
    cmd = [
        str(args.llama_perplexity),
        "-m", str(args.model),
        "-f", str(args.file),
        "-c", str(args.ctx_size),
        "--chunks", str(args.chunks),
        "--no-warmup",
        *args.llama_arg,
    ]
    if args.ppl_stride > 0:
        cmd.extend(["--ppl-stride", str(args.ppl_stride)])
    if mode == "no_lowrank":
        return cmd

    cmd.extend([
        "--kv-lowrank",
        "--kv-lowrank-basis-path", str(args.basis_path),
        "--kv-lowrank-rank", str(max(args.rank_k, args.rank_v)),
        "--kv-lowrank-rank-k", str(args.rank_k),
        "--kv-lowrank-rank-v", str(args.rank_v),
        "--kv-lowrank-window", str(args.window),
        "--kv-lowrank-chunk", str(args.chunk),
    ])
    if mode == "reconstruct_cache":
        cmd.append("--kv-lowrank-reconstruct-cache")
    return cmd


def summarize_run(name: str, text: str, log_path: Path) -> Dict[str, Any]:
    summary = {
        "name": name,
        "log": str(log_path),
        "ppl": parse_ppl(text),
        "whlr": parse_whlr(text),
    }
    if summary["ppl"]["found"]:
        summary["ppl_value"] = summary["ppl"]["ppl"]
        summary["ppl_uncertainty"] = summary["ppl"]["uncertainty"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare WHLR-KV dense-shadow and reconstruct-cache perplexity on the same corpus.",
    )
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rank-k", type=int, required=True)
    parser.add_argument("--rank-v", type=int, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--ppl-stride", type=int, default=0)
    parser.add_argument("--include-no-lowrank", action="store_true")
    parser.add_argument("--llama-arg", action="append", default=[])
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    no_lowrank = None
    if args.include_no_lowrank:
        no_lowrank_log = args.work_dir / "no_lowrank_ppl.log"
        no_lowrank_text = run_cmd(ppl_cmd(args, mode="no_lowrank"), no_lowrank_log)
        no_lowrank = summarize_run("no_lowrank", no_lowrank_text, no_lowrank_log)

    dense_log = args.work_dir / "dense_shadow_ppl.log"
    dense_text = run_cmd(ppl_cmd(args, mode="dense_shadow"), dense_log)

    reconstruct_log = args.work_dir / "reconstruct_cache_ppl.log"
    reconstruct_text = run_cmd(ppl_cmd(args, mode="reconstruct_cache"), reconstruct_log)

    dense = summarize_run("dense_shadow", dense_text, dense_log)
    reconstruct = summarize_run("reconstruct_cache", reconstruct_text, reconstruct_log)

    ppl_delta = None
    ppl_ratio = None
    if dense["ppl"]["found"] and reconstruct["ppl"]["found"]:
        ppl_delta = reconstruct["ppl_value"] - dense["ppl_value"]
        ppl_ratio = reconstruct["ppl_value"] / dense["ppl_value"] if dense["ppl_value"] != 0.0 else None

    summary = {
        "dense_shadow": dense,
        "reconstruct_cache": reconstruct,
        "ppl_delta_reconstruct_minus_dense": ppl_delta,
        "ppl_ratio_reconstruct_over_dense": ppl_ratio,
    }
    if no_lowrank is not None:
        summary["no_lowrank"] = no_lowrank
        if no_lowrank["ppl"]["found"] and dense["ppl"]["found"]:
            summary["ppl_delta_dense_minus_no_lowrank"] = dense["ppl_value"] - no_lowrank["ppl_value"]

    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
