#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def run_cmd(cmd: List[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout


def parse_ints(value: str) -> List[int]:
    items = [int(item) for item in value.split(",") if item.strip()]
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return items


def parse_rank_pairs(value: str) -> List[Tuple[int, int]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            k_rank, v_rank = (int(part) for part in item.split(":", 1))
        else:
            k_rank = v_rank = int(item)
        pairs.append((k_rank, v_rank))
    if not pairs or any(k_rank <= 0 or v_rank <= 0 for k_rank, v_rank in pairs):
        raise argparse.ArgumentTypeError("rank pairs must be positive, e.g. 64:128,128:128")
    return pairs


def basis_from_template(template: str, rank_k: int, rank_v: int) -> Path:
    return Path(template.format(rank=max(rank_k, rank_v), rank_k=rank_k, rank_v=rank_v))


def export_basis_cmd(args: argparse.Namespace, out_dir: Path, rank_k: int, rank_v: int) -> List[str]:
    return [
        "python3",
        str(args.exporter),
        "--samples-npz", str(args.samples_npz),
        "--out-dir", str(out_dir),
        "--rank", str(max(rank_k, rank_v)),
        "--rank-k", str(rank_k),
        "--rank-v", str(rank_v),
        "--n-layer", str(args.n_layer),
        "--head-dim", str(args.head_dim),
        "--n-head-kv", str(args.n_head_kv),
        "--dtype", args.dtype,
    ]


def ensure_basis(args: argparse.Namespace, rank_k: int, rank_v: int) -> Path:
    if args.samples_npz is not None:
        basis_dir = args.work_dir / "basis" / f"k{rank_k}_v{rank_v}"
        manifest = basis_dir / "whlr_kv_basis.json"
        if not manifest.exists() or args.force_export:
            log_path = args.work_dir / "logs" / f"export_k{rank_k}_v{rank_v}.log"
            run_cmd(export_basis_cmd(args, basis_dir, rank_k, rank_v), log_path)
        return manifest

    if args.basis_template:
        return basis_from_template(args.basis_template, rank_k, rank_v)

    if args.basis_path is None:
        raise SystemExit("provide --samples-npz, --basis-template, or --basis-path")
    if len(args.rank_pairs) != 1:
        raise SystemExit("--basis-path can only be used with one rank pair; use --basis-template for sweeps")
    return args.basis_path


def validate_basis(path: Path, rank_k: int, rank_v: int) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"basis manifest does not exist for rank_k={rank_k} rank_v={rank_v}: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_rank = int(manifest.get("rank", 0))
    manifest_rank_k = int(manifest.get("rank_k", manifest_rank))
    manifest_rank_v = int(manifest.get("rank_v", manifest_rank))
    if manifest_rank_k != rank_k or manifest_rank_v != rank_v:
        raise SystemExit(
            "basis manifest rank mismatch for "
            f"{path}: requested rank_k={rank_k} rank_v={rank_v}, "
            f"manifest rank_k={manifest_rank_k} rank_v={manifest_rank_v}"
        )
    return {
        "rank": int(manifest.get("rank", max(rank_k, rank_v))),
        "rank_k": manifest_rank_k,
        "rank_v": manifest_rank_v,
        "n_layer": int(manifest.get("n_layer", 0)),
        "head_dim": int(manifest.get("head_dim", 0)),
        "n_head_kv": int(manifest.get("n_head_kv", 0)),
    }


def ppl_cmd(args: argparse.Namespace, basis_path: Path, rank_k: int, rank_v: int, window: int, chunk: int, case_dir: Path) -> List[str]:
    cmd = [
        "python3",
        str(args.ppl_harness),
        "--llama-perplexity", str(args.llama_perplexity),
        "--model", str(args.model),
        "--file", str(args.file),
        "--basis-path", str(basis_path),
        "--work-dir", str(case_dir),
        "--rank-k", str(rank_k),
        "--rank-v", str(rank_v),
        "--window", str(window),
        "--chunk", str(chunk),
        "--ctx-size", str(args.ctx_size),
        "--chunks", str(args.chunks),
    ]
    if args.ppl_stride > 0:
        cmd.extend(["--ppl-stride", str(args.ppl_stride)])
    if args.include_no_lowrank:
        cmd.append("--include-no-lowrank")
    for item in args.llama_arg:
        cmd.append(f"--llama-arg={item}")
    return cmd


def load_case_summary(case_dir: Path) -> Dict[str, Any]:
    return json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))


def flatten_case(
        rank_k: int,
        rank_v: int,
        window: int,
        chunk: int,
        basis_path: Path,
        basis_meta: Dict[str, Any],
        case_dir: Path,
        summary: Dict[str, Any]) -> Dict[str, Any]:
    dense = summary["dense_shadow"]
    reconstruct = summary["reconstruct_cache"]
    no_lowrank = summary.get("no_lowrank")
    whlr = reconstruct["whlr"]
    d_kv = basis_meta["head_dim"] * basis_meta["n_head_kv"]
    n_layer = basis_meta["n_layer"]
    dense_f32_history_bytes = whlr["projected_tokens"] * n_layer * 2 * d_kv * 4 if d_kv > 0 and n_layer > 0 else None
    lowrank_history_bytes = whlr["history_bytes"]
    lowrank_history_ratio = (
        lowrank_history_bytes / dense_f32_history_bytes
        if dense_f32_history_bytes not in (None, 0)
        else None
    )
    return {
        "rank_k": rank_k,
        "rank_v": rank_v,
        "window": window,
        "chunk": chunk,
        "basis_path": str(basis_path),
        "case_dir": str(case_dir),
        "no_lowrank_ppl": no_lowrank.get("ppl_value") if no_lowrank else None,
        "dense_ppl": dense.get("ppl_value"),
        "reconstruct_ppl": reconstruct.get("ppl_value"),
        "ppl_delta_reconstruct_minus_dense": summary.get("ppl_delta_reconstruct_minus_dense"),
        "ppl_ratio_reconstruct_over_dense": summary.get("ppl_ratio_reconstruct_over_dense"),
        "ppl_delta_dense_minus_no_lowrank": summary.get("ppl_delta_dense_minus_no_lowrank"),
        "projected_tokens": whlr["projected_tokens"],
        "projected_chunks": whlr["projected_chunks"],
        "projected_layers_max": whlr["projected_layers_max"],
        "reconstructed_layers_max": whlr["reconstructed_layers_max"],
        "history_bytes": lowrank_history_bytes,
        "dense_f32_history_bytes_est": dense_f32_history_bytes,
        "lowrank_history_ratio_est": lowrank_history_ratio,
        "k_mean_max": whlr["k_mean_max"],
        "v_mean_max": whlr["v_mean_max"],
        "k_max": whlr["k_max"],
        "v_max": whlr["v_max"],
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row: Dict[str, Any]) -> Tuple[float, float, int, int, int, int]:
    ratio = row.get("ppl_ratio_reconstruct_over_dense")
    delta = row.get("ppl_delta_reconstruct_minus_dense")
    return (
        float(ratio) if ratio is not None else float("inf"),
        float(delta) if delta is not None else float("inf"),
        int(row["rank_k"]),
        int(row["rank_v"]),
        int(row["window"]),
        int(row["chunk"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep WHLR-KV reconstruct-cache PPL across rank/window/chunk settings.",
    )
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rank-pairs", type=parse_rank_pairs, required=True)
    parser.add_argument("--windows", type=parse_ints, required=True)
    parser.add_argument("--chunks-list", type=parse_ints, required=True)
    parser.add_argument("--basis-path", type=Path)
    parser.add_argument("--basis-template")
    parser.add_argument("--samples-npz", type=Path)
    parser.add_argument("--exporter", type=Path, default=Path("scripts/kv_lowrank_export_basis.py"))
    parser.add_argument("--ppl-harness", type=Path, default=Path("scripts/kv_lowrank_reconstruct_ppl.py"))
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--n-head-kv", type=int)
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse an existing case summary.json when present")
    parser.add_argument("--max-cases", type=int, default=0, help="optional cap on cases to run after invalid windows are skipped")
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--ppl-stride", type=int, default=0)
    parser.add_argument("--include-no-lowrank", action="store_true")
    parser.add_argument("--llama-arg", action="append", default=[])
    args = parser.parse_args()

    if args.samples_npz is not None and (args.n_layer is None or args.head_dim is None or args.n_head_kv is None):
        raise SystemExit("--samples-npz export mode requires --n-layer, --head-dim, and --n-head-kv")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = []
    n_run = 0
    for rank_k, rank_v in args.rank_pairs:
        basis_path = ensure_basis(args, rank_k, rank_v)
        basis_meta = validate_basis(basis_path, rank_k, rank_v)
        for window in args.windows:
            for chunk in args.chunks_list:
                if window < chunk:
                    continue
                if args.max_cases > 0 and n_run >= args.max_cases:
                    break
                case_name = f"k{rank_k}_v{rank_v}_w{window}_c{chunk}"
                case_dir = args.work_dir / "cases" / case_name
                log_path = args.work_dir / "logs" / f"{case_name}.log"
                if not args.resume or not (case_dir / "summary.json").exists():
                    run_cmd(ppl_cmd(args, basis_path, rank_k, rank_v, window, chunk, case_dir), log_path)
                summary = load_case_summary(case_dir)
                row = flatten_case(rank_k, rank_v, window, chunk, basis_path, basis_meta, case_dir, summary)
                rows.append(row)
                cases.append({"case": case_name, "summary": summary, "flat": row})
                print(json.dumps(row, ensure_ascii=False))
                n_run += 1
            if args.max_cases > 0 and n_run >= args.max_cases:
                break
        if args.max_cases > 0 and n_run >= args.max_cases:
            break

    result = {
        "cases": cases,
        "rows": rows,
        "best_rows": sorted(rows, key=sort_key),
    }
    summary_path = args.work_dir / "summary.json"
    csv_path = args.work_dir / "summary.csv"
    best_csv_path = args.work_dir / "best.csv"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_csv(best_csv_path, result["best_rows"])
    print(f"wrote summary: {summary_path}")
    print(f"wrote csv: {csv_path}")
    print(f"wrote best csv: {best_csv_path}")


if __name__ == "__main__":
    main()
