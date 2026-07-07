#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        return list(data["rows"])
    if isinstance(data, list):
        return data
    raise SystemExit(f"unsupported report input shape: {path}")


def f64(row: Dict[str, Any], key: str, fallback: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return fallback
    return float(value)


def i64(row: Dict[str, Any], key: str, fallback: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return fallback
    return int(float(value))


def best_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> tuple[float, float, int, int, int, int]:
        ratio = f64(row, "ppl_ratio_reconstruct_over_dense", float("inf"))
        delta = f64(row, "ppl_delta_reconstruct_minus_dense", float("inf"))
        return (
            ratio if ratio is not None else float("inf"),
            delta if delta is not None else float("inf"),
            i64(row, "rank_k"),
            i64(row, "rank_v"),
            i64(row, "window"),
            i64(row, "chunk"),
        )

    return sorted(rows, key=key)[:limit]


def dense_parity(rows: List[Dict[str, Any]], tolerance: float, required: bool) -> Dict[str, Any]:
    checked = []
    for row in rows:
        dense = f64(row, "dense_ppl")
        no_lowrank = f64(row, "no_lowrank_ppl")
        if dense is None or no_lowrank is None:
            continue
        checked.append(abs(dense - no_lowrank))

    return {
        "checked_cases": len(checked),
        "max_abs_delta": max(checked, default=None),
        "required": required,
        "pass": (not required and not checked) or (bool(checked) and max(checked, default=float("inf")) <= tolerance),
    }


def reconstruct_active(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    active = [
        row for row in rows
        if i64(row, "projected_tokens") > 0 and i64(row, "reconstructed_layers_max") > 0
    ]
    return {
        "active_cases": len(active),
        "total_cases": len(rows),
        "pass": len(active) > 0,
    }


def compression_target(rows: List[Dict[str, Any]], max_ratio: float) -> Dict[str, Any]:
    qualifying = []
    for row in rows:
        ratio = f64(row, "lowrank_history_ratio_est")
        if ratio is not None and ratio <= max_ratio and i64(row, "projected_tokens") > 0:
            qualifying.append(row)
    best = min((f64(row, "lowrank_history_ratio_est") for row in qualifying), default=None)
    return {
        "max_allowed_ratio": max_ratio,
        "best_ratio": best,
        "qualifying_cases": len(qualifying),
        "pass": len(qualifying) > 0,
    }


def quality_target(rows: List[Dict[str, Any]], max_ppl_ratio: float, max_ppl_delta: float | None) -> Dict[str, Any]:
    qualifying = []
    for row in rows:
        ratio = f64(row, "ppl_ratio_reconstruct_over_dense")
        delta = f64(row, "ppl_delta_reconstruct_minus_dense")
        if ratio is None:
            continue
        if ratio > max_ppl_ratio:
            continue
        if max_ppl_delta is not None and delta is not None and delta > max_ppl_delta:
            continue
        if i64(row, "projected_tokens") <= 0:
            continue
        qualifying.append(row)

    best = best_rows(qualifying, 1)
    return {
        "max_allowed_ppl_ratio": max_ppl_ratio,
        "max_allowed_ppl_delta": max_ppl_delta,
        "qualifying_cases": len(qualifying),
        "best_case": best[0] if best else None,
        "pass": len(qualifying) > 0,
    }


def window_sensitivity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[tuple[int, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((i64(row, "rank_k"), i64(row, "rank_v"), i64(row, "chunk")), []).append(row)

    checked = 0
    non_increasing = 0
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: i64(row, "window"))
        ratios = [f64(row, "ppl_ratio_reconstruct_over_dense") for row in ordered]
        if len(ratios) < 2 or any(ratio is None for ratio in ratios):
            continue
        checked += 1
        if all(float(ratios[i]) <= float(ratios[i - 1]) for i in range(1, len(ratios))):
            non_increasing += 1

    return {
        "checked_groups": checked,
        "non_increasing_groups": non_increasing,
        "pass": checked > 0 and checked == non_increasing,
    }


def build_report(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = {
        "dense_parity": dense_parity(rows, args.dense_parity_tol, args.require_dense_parity),
        "reconstruct_active": reconstruct_active(rows),
        "compression_target": compression_target(rows, args.max_history_ratio),
        "quality_target": quality_target(rows, args.max_ppl_ratio, args.max_ppl_delta),
        "window_sensitivity": window_sensitivity(rows),
    }
    return {
        "input": str(args.input),
        "n_cases": len(rows),
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
        "best_rows": best_rows(rows, args.top_k),
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# WHLR-KV Reproduction Report",
        "",
        f"- input: `{report['input']}`",
        f"- cases: {report['n_cases']}",
        f"- overall: {'PASS' if report['pass'] else 'FAIL'}",
        "",
        "## Checks",
        "",
    ]
    for name, check in report["checks"].items():
        lines.append(f"- {name}: {'PASS' if check['pass'] else 'FAIL'}")
        for key, value in check.items():
            if key != "pass":
                lines.append(f"  - {key}: {value}")
    lines.extend(["", "## Best Rows", ""])
    for row in report["best_rows"]:
        lines.append(
            "- "
            f"k={row.get('rank_k')} v={row.get('rank_v')} "
            f"window={row.get('window')} chunk={row.get('chunk')} "
            f"ppl_ratio={row.get('ppl_ratio_reconstruct_over_dense')} "
            f"history_ratio={row.get('lowrank_history_ratio_est')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether WHLR-KV sweep results satisfy reproduction criteria.",
    )
    parser.add_argument("--input", type=Path, required=True, help="sweep summary.json, summary.csv, or best.csv")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--dense-parity-tol", type=float, default=1e-6)
    parser.add_argument("--require-dense-parity", action="store_true")
    parser.add_argument("--max-history-ratio", type=float, default=0.25)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.10)
    parser.add_argument("--max-ppl-delta", type=float)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = build_report(args, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
