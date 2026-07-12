#!/usr/bin/env python3
"""
Phase 5 PPL parity gate for --kv-lowrank-direct.

Runs tools/perplexity in three modes on the same corpus/model:
  - dense: no BlockSVD, full-precision KV
  - reconstruct: --kv-blocksvd-memory-reduction (chunked_attn against dequant'd K/V)
  - direct: --kv-blocksvd-memory-reduction --kv-lowrank-direct (rank-domain kernel)

Sweeps the paper §5.5 sanity config: rank=32, block=64, ctx in {2048, 4096}.
Emits CSV + JSON. Exit criteria:
  |PPL(direct) - PPL(reconstruct)| <= 0.05 at ctx=2048
  |PPL(direct) - PPL(reconstruct)| <= 0.15 at ctx=4096
Exit 0 if all cells pass, 1 otherwise.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PPL_RE = re.compile(r"Final estimate: PPL = (?P<ppl>[0-9.eE+-]+) \+/- (?P<unc>[0-9.eE+-]+)")
COMPRESS_RE = re.compile(r"cross-layer stored n_tokens=(\d+) layers=(\d+)")

# Thresholds per plan §Phase 5
DEFAULT_THRESHOLDS = {2048: 0.05, 4096: 0.15}


def build_cmd(args: argparse.Namespace, mode: str, ctx: int) -> List[str]:
    cmd = [
        str(args.llama_perplexity),
        "-m", str(args.model),
        "-f", str(args.file),
        "-c", str(ctx),
        "-b", str(ctx),
        "-ngl", str(args.gpu_layers),
        "-t", str(args.threads),
        "--kv-unified",
        "--chunks", str(args.chunks),
        "--no-warmup",
    ]
    if mode == "dense":
        pass
    elif mode in ("reconstruct", "direct"):
        cmd += [
            "--kv-blocksvd",
            "--kv-blocksvd-backend",
            "--kv-blocksvd-cross-layer",
            "--kv-blocksvd-memory-reduction",
            "--kv-blocksvd-block-size", str(args.block),
            "--kv-blocksvd-rank", str(args.rank),
        ]
        if mode == "direct":
            cmd.append("--kv-lowrank-direct")
    else:
        raise ValueError(f"unknown mode: {mode}")
    cmd += args.extra_arg
    return cmd


def parse_ppl(text: str) -> Optional[Dict[str, float]]:
    matches = list(PPL_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {"ppl": float(m.group("ppl")), "unc": float(m.group("unc"))}


def parse_compress_events(text: str) -> Dict[str, int]:
    events = list(COMPRESS_RE.finditer(text))
    return {
        "events": len(events),
        "tokens_max": max((int(m.group(1)) for m in events), default=0),
        "layers_max": max((int(m.group(2)) for m in events), default=0),
    }


def parse_markers(text: str) -> Dict[str, bool]:
    return {
        "chunked_attn_active": "chunked_attn active" in text,
        "kv_lowrank_direct_active": "kv_lowrank_direct active" in text,
    }


def run_one(cmd: List[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")
    out: Dict[str, Any] = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "log": str(log_path),
    }
    ppl = parse_ppl(proc.stdout)
    if ppl is None:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        out["error"] = f"PPL not found (rc={proc.returncode})\n{tail}"
        return out
    out.update(ppl)
    out["markers"] = parse_markers(proc.stdout)
    out["compress"] = parse_compress_events(proc.stdout)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ctx", action="append", type=int, default=None,
                        help="context sizes to sweep (repeatable; default: 2048 and 4096)")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=4,
                        help="perplexity --chunks value (time-bound the sweep)")
    parser.add_argument("--modes", default="dense,reconstruct,direct",
                        help="comma-separated modes to run")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="extra args passed through to llama-perplexity")
    args = parser.parse_args()

    if args.ctx is None:
        args.ctx = [2048, 4096]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for ctx in args.ctx:
        cell: Dict[str, Any] = {"ctx": ctx, "rank": args.rank, "block": args.block, "runs": {}}
        for mode in modes:
            cmd = build_cmd(args, mode, ctx)
            log = args.work_dir / f"ctx{ctx}_{mode}.log"
            print(f"[ctx={ctx} mode={mode}] running... ({args.chunks} chunks)")
            run = run_one(cmd, log)
            cell["runs"][mode] = run
            if "error" in run:
                print(f"  ! ERROR: {run['error'].splitlines()[0]}")
            else:
                mk = run["markers"]
                comp = run["compress"]
                print(f"  PPL = {run['ppl']:.4f} +/- {run['unc']:.4f}  "
                      f"(elapsed={run['elapsed_sec']:.1f}s, compress_events={comp['events']}, "
                      f"chunked={'Y' if mk['chunked_attn_active'] else 'N'}, "
                      f"direct={'Y' if mk['kv_lowrank_direct_active'] else 'N'})")

        recon = cell["runs"].get("reconstruct")
        direct = cell["runs"].get("direct")
        dense = cell["runs"].get("dense")
        if recon and direct and "ppl" in recon and "ppl" in direct:
            delta = abs(direct["ppl"] - recon["ppl"])
            threshold = DEFAULT_THRESHOLDS.get(ctx, 0.15)
            cell["delta_direct_vs_reconstruct"] = round(delta, 6)
            cell["threshold"] = threshold
            cell["pass"] = delta <= threshold
            print(f"  |direct - reconstruct| = {delta:.6f}  (threshold {threshold})  "
                  f"=> {'PASS' if cell['pass'] else 'FAIL'}")
        if dense and recon and "ppl" in dense and "ppl" in recon:
            cell["delta_reconstruct_vs_dense"] = round(recon["ppl"] - dense["ppl"], 6)
        results.append(cell)

    summary_json = args.work_dir / "summary.json"
    summary_json.write_text(json.dumps({
        "config": {"rank": args.rank, "block": args.block, "chunks": args.chunks,
                   "model": str(args.model), "file": str(args.file),
                   "gpu_layers": args.gpu_layers, "threads": args.threads},
        "results": results,
    }, indent=2), encoding="utf-8")

    csv_path = args.work_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ctx", "rank", "block", "mode", "ppl", "uncertainty",
                    "compress_events", "chunked_marker", "direct_marker", "elapsed_sec"])
        for cell in results:
            for mode, run in cell["runs"].items():
                if "ppl" not in run:
                    w.writerow([cell["ctx"], cell["rank"], cell["block"], mode,
                                "ERR", "ERR", "", "", "", run.get("elapsed_sec", "")])
                    continue
                mk = run["markers"]
                w.writerow([cell["ctx"], cell["rank"], cell["block"], mode,
                            run["ppl"], run["unc"], run["compress"]["events"],
                            int(mk["chunked_attn_active"]), int(mk["kv_lowrank_direct_active"]),
                            run["elapsed_sec"]])

    all_pass = all(cell.get("pass", True) for cell in results if "delta_direct_vs_reconstruct" in cell)
    any_gate = any("delta_direct_vs_reconstruct" in cell for cell in results)

    print(f"\nwrote {summary_json}")
    print(f"wrote {csv_path}")
    if any_gate:
        print(f"GATE: {'PASS' if all_pass else 'FAIL'}")
        return 0 if all_pass else 1
    print("GATE: no reconstruct+direct pair produced — nothing gated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
