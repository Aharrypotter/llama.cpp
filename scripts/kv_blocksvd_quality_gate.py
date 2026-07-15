#!/usr/bin/env python3
"""Measure BlockSVD quality against frozen dense log-probabilities.

For every corpus/context pair, this harness first runs dense KV with
``--save-all-logits``. It then evaluates reconstruct, portable-direct, and
packed-direct modes with llama-perplexity's built-in KL-divergence oracle.

The production direct profile requires token-at-a-time decode, so batch and
ubatch are intentionally fixed to one. The harness records PPL ratios, KLD
percentiles, token-probability drift, execution markers, file hashes, commands,
and environment overrides in JSON and CSV. Quality thresholds are optional;
without them the result is MEASURED rather than an invented pass/fail gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODES = ("dense", "reconstruct", "portable", "packed")
DIRECT_ENV_KEYS = (
    "LLAMA_KV_BLOCKSVD_EDGEKV_DIRECT",
    "LLAMA_KV_BLOCKSVD_EDGEKV_RECONSTRUCT",
    "LLAMA_KV_BLOCKSVD_DIRECT_DIAG_FOLDED",
    "LLAMA_KV_BLOCKSVD_DIRECT_DIAG_COMPRESSED_FIRST",
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PLUS_MINUS = r"(?:\+/-|\u00b1)"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PPL_RE = re.compile(
    rf"Final estimate: PPL = (?P<value>{NUMBER}) {PLUS_MINUS} (?P<uncertainty>{NUMBER})"
)
SHAPE_RE = re.compile(
    r"(?:calculating perplexity|computing) over (?P<chunks>\d+) chunks, "
    r"n_ctx=(?P<ctx>\d+), batch_size=(?P<batch>\d+), n_seq=(?P<seq>\d+)"
)
COMPRESS_RE = re.compile(r"cross-layer stored n_tokens=(?P<tokens>\d+) layers=(?P<layers>\d+)")


@dataclass(frozen=True)
class QualityConfig:
    llama_perplexity: Path
    model: Path
    threads: int
    gpu_layers: int
    chunks: int
    block_size: int
    rank_k: int
    rank_v: int
    group_size: int
    quant_bits: int
    extra_args: Tuple[str, ...] = ()


def parse_modes(value: str) -> List[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in modes if item not in MODES]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown modes: {', '.join(invalid)}")
    if not modes or modes[0] != "dense" or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("modes must be unique and start with dense")
    return modes


def parse_corpus(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(raw_path).stem
    label = label.strip()
    if not label or not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise argparse.ArgumentTypeError("corpus label must match [A-Za-z0-9._-]+")
    if not raw_path:
        raise argparse.ArgumentTypeError("corpus path is empty")
    return label, Path(raw_path)


def blocksvd_args(config: QualityConfig) -> List[str]:
    return [
        "--kv-blocksvd",
        "--kv-blocksvd-backend",
        "--kv-blocksvd-cross-layer",
        "--kv-blocksvd-memory-reduction",
        "--kv-blocksvd-block-size",
        str(config.block_size),
        "--kv-blocksvd-rank",
        str(config.rank_k),
        "--kv-blocksvd-rank-v",
        str(config.rank_v),
        "--kv-blocksvd-layer-group-size",
        str(config.group_size),
        "--kv-blocksvd-quant-bits",
        str(config.quant_bits),
    ]


def build_command(
    config: QualityConfig,
    mode: str,
    corpus: Path,
    ctx: int,
    dense_logits: Path,
) -> List[str]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")

    command = [
        str(config.llama_perplexity),
        "-m",
        str(config.model),
        "-f",
        str(corpus),
        "-c",
        str(ctx),
        "-b",
        "1",
        "-ub",
        "1",
        "-ngl",
        str(config.gpu_layers),
        "-t",
        str(config.threads),
        "--kv-unified",
        "--chunks",
        str(config.chunks),
        "--no-warmup",
    ]
    if mode == "dense":
        command.extend(["--save-all-logits", str(dense_logits)])
    else:
        command.extend(["--kl-divergence", "--kl-divergence-base", str(dense_logits)])
        command.extend(blocksvd_args(config))
        if mode in ("portable", "packed"):
            command.append("--kv-lowrank-direct")
    command.extend(config.extra_args)
    return command


def environment_overrides(mode: str) -> Dict[str, Optional[str]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    overrides: Dict[str, Optional[str]] = {key: None for key in DIRECT_ENV_KEYS}
    if mode == "packed":
        overrides["LLAMA_KV_BLOCKSVD_EDGEKV_DIRECT"] = "1"
    return overrides


def apply_environment(overrides: Mapping[str, Optional[str]]) -> Dict[str, str]:
    environment = os.environ.copy()
    for key, value in overrides.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def clean_output(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_pair(text: str, label: str) -> Optional[Tuple[float, float]]:
    match = re.search(
        rf"{label}\s*:\s*(?P<value>{NUMBER})\s*{PLUS_MINUS}\s*(?P<uncertainty>{NUMBER})",
        text,
    )
    if not match:
        return None
    return float(match.group("value")), float(match.group("uncertainty"))


def parse_scalar(text: str, label: str) -> Optional[float]:
    match = re.search(rf"{label}\s*:\s*(?P<value>{NUMBER})", text)
    return float(match.group("value")) if match else None


def parse_final_ppl(text: str) -> Dict[str, float]:
    matches = list(PPL_RE.finditer(clean_output(text)))
    if not matches:
        return {}
    match = matches[-1]
    return {
        "ppl": float(match.group("value")),
        "ppl_uncertainty": float(match.group("uncertainty")),
    }


def parse_kl_metrics(text: str) -> Dict[str, float]:
    text = clean_output(text)
    metrics: Dict[str, float] = {}
    pair_patterns = {
        "mean_ppl_q": r"Mean PPL\(Q\)",
        "mean_ppl_base": r"Mean PPL\(base\)",
        "mean_log_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)",
        "mean_ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)",
        "mean_ppl_difference": r"Mean PPL\(Q\)-PPL\(base\)",
        "mean_kld": r"Mean\s+KLD",
        "rms_delta_p_percent": r"RMS \u0394p",
        "same_top_percent": r"Same top p",
    }
    for key, pattern in pair_patterns.items():
        pair = parse_pair(text, pattern)
        if pair is not None:
            metrics[key] = pair[0]
            metrics[f"{key}_uncertainty"] = pair[1]

    scalar_patterns = {
        "maximum_kld": r"Maximum KLD",
        "p99_9_kld": r"99\.9%\s+KLD",
        "p99_kld": r"99\.0%\s+KLD",
        "p95_kld": r"95\.0%\s+KLD",
        "p90_kld": r"90\.0%\s+KLD",
        "median_kld": r"Median\s+KLD",
    }
    for key, pattern in scalar_patterns.items():
        value = parse_scalar(text, pattern)
        if value is not None:
            metrics[key] = value
    return metrics


def parse_execution(text: str) -> Dict[str, Any]:
    text = clean_output(text)
    shape_matches = list(SHAPE_RE.finditer(text))
    shape: Dict[str, int] = {}
    if shape_matches:
        match = shape_matches[-1]
        shape = {key: int(match.group(key)) for key in ("chunks", "ctx", "batch", "seq")}
        shape["sample_count"] = shape["chunks"] * (shape["ctx"] - 1 - shape["ctx"] // 2)

    compress_matches = list(COMPRESS_RE.finditer(text))
    return {
        "shape": shape,
        "markers": {
            "chunked_attn_active": "chunked_attn active" in text,
            "kv_lowrank_direct_active": "kv_lowrank_direct active" in text,
            "edgekv_direct_decode_active": "EdgeKV direct decode active" in text,
        },
        "compression": {
            "events": len(compress_matches),
            "tokens_max": max((int(match.group("tokens")) for match in compress_matches), default=0),
            "layers_max": max((int(match.group("layers")) for match in compress_matches), default=0),
        },
    }


def marker_errors(mode: str, execution: Mapping[str, Any]) -> List[str]:
    markers = execution["markers"]
    compression = execution["compression"]
    errors: List[str] = []
    if mode == "dense":
        if any(markers.values()):
            errors.append("dense run unexpectedly selected a BlockSVD attention path")
        return errors

    if compression["events"] == 0:
        errors.append("no cross-layer compression event was observed")
    if mode == "reconstruct":
        if not markers["chunked_attn_active"]:
            errors.append("reconstruct run did not select chunked_attn")
        if markers["kv_lowrank_direct_active"] or markers["edgekv_direct_decode_active"]:
            errors.append("reconstruct run selected a direct attention path")
    elif mode == "portable":
        if not markers["kv_lowrank_direct_active"]:
            errors.append("portable run did not select kv_lowrank_direct")
        if markers["edgekv_direct_decode_active"]:
            errors.append("portable run unexpectedly selected the packed operation")
    elif mode == "packed" and not markers["edgekv_direct_decode_active"]:
        errors.append("packed run did not emit EdgeKV direct decode")
    return errors


def required_metric_errors(mode: str, metrics: Mapping[str, float]) -> List[str]:
    required = ("ppl",) if mode == "dense" else (
        "mean_ppl_q",
        "mean_ppl_base",
        "mean_log_ppl_ratio",
        "mean_ppl_ratio",
        "mean_kld",
        "maximum_kld",
        "p99_kld",
        "rms_delta_p_percent",
        "same_top_percent",
    )
    missing = [name for name in required if name not in metrics]
    return [f"missing metrics: {', '.join(missing)}"] if missing else []


def run_one(
    mode: str,
    command: Sequence[str],
    overrides: Mapping[str, Optional[str]],
    log_path: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.run(
            list(command),
            env=apply_environment(overrides),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        output = process.stdout
        returncode: Optional[int] = process.returncode
        timeout = False
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        output = partial.decode() if isinstance(partial, bytes) else partial
        returncode = None
        timeout = True

    elapsed = time.monotonic() - started
    log_path.write_text(output, encoding="utf-8")
    execution = parse_execution(output)
    metrics = parse_final_ppl(output) if mode == "dense" else parse_kl_metrics(output)
    errors: List[str] = []
    if timeout:
        errors.append(f"timed out after {timeout_sec} seconds")
    elif returncode != 0:
        errors.append(f"process exited with status {returncode}")
    errors.extend(required_metric_errors(mode, metrics))
    errors.extend(marker_errors(mode, execution))
    return {
        "mode": mode,
        "status": "ok" if not errors else "error",
        "errors": errors,
        "command": list(command),
        "environment_overrides": dict(overrides),
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "log": str(log_path),
        "metrics": metrics,
        "execution": execution,
    }


def result_signature(command: Sequence[str], overrides: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    return {"command": list(command), "environment_overrides": dict(overrides)}


def load_resumable_result(
    path: Path,
    command: Sequence[str],
    overrides: Mapping[str, Optional[str]],
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") != "ok":
        return None
    signature = result_signature(command, overrides)
    if any(result.get(key) != value for key, value in signature.items()):
        return None
    result["resumed"] = True
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, include_hash: bool = True) -> Dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result = {"path": str(resolved), "size_bytes": stat.st_size}
    if include_hash:
        result["sha256"] = sha256_file(resolved)
    return result


def git_provenance(path: Path) -> Dict[str, Any]:
    directory = path.resolve().parent
    root = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if root.returncode != 0:
        return {}
    repo = Path(root.stdout.strip())
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    return {"root": str(repo), "revision": revision, "dirty": bool(status), "dirty_entry_count": len(status)}


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for result in results:
        if result["mode"] == "dense" or result["status"] != "ok":
            continue
        grouped.setdefault(str(result["mode"]), []).append(result)

    aggregates: Dict[str, Any] = {}
    for mode, mode_results in grouped.items():
        weighted: List[Tuple[int, Mapping[str, float]]] = []
        for result in mode_results:
            shape = result["execution"].get("shape", {})
            count = int(shape.get("sample_count", 0))
            if count > 0:
                weighted.append((count, result["metrics"]))
        total = sum(count for count, _ in weighted)
        if total == 0:
            continue
        mean_log_ratio = sum(count * metrics["mean_log_ppl_ratio"] for count, metrics in weighted) / total
        mean_kld = sum(count * metrics["mean_kld"] for count, metrics in weighted) / total
        same_top = sum(count * metrics["same_top_percent"] for count, metrics in weighted) / total
        rms_delta_p = math.sqrt(
            sum(count * metrics["rms_delta_p_percent"] ** 2 for count, metrics in weighted) / total
        )
        aggregates[mode] = {
            "case_count": len(weighted),
            "sample_count": total,
            "weighted_mean_log_ppl_ratio": mean_log_ratio,
            "weighted_ppl_ratio": math.exp(mean_log_ratio),
            "weighted_mean_kld": mean_kld,
            "weighted_rms_delta_p_percent": rms_delta_p,
            "weighted_same_top_percent": same_top,
            "worst_mean_kld": max(metrics["mean_kld"] for _, metrics in weighted),
            "maximum_kld": max(metrics["maximum_kld"] for _, metrics in weighted),
            "worst_ppl_ratio": max(metrics["mean_ppl_ratio"] for _, metrics in weighted),
            "worst_same_top_percent": min(metrics["same_top_percent"] for _, metrics in weighted),
        }
    return aggregates


def evaluate_gate(
    results: Sequence[Mapping[str, Any]],
    max_mean_kld: Optional[float],
    max_ppl_ratio: Optional[float],
    min_same_top_percent: Optional[float],
) -> Dict[str, Any]:
    errors = [
        f"{result.get('corpus', '?')}/ctx{result.get('ctx', '?')}/{result['mode']}: "
        + "; ".join(result.get("errors", []))
        for result in results
        if result["status"] != "ok"
    ]
    thresholds = {
        "max_mean_kld": max_mean_kld,
        "max_ppl_ratio": max_ppl_ratio,
        "min_same_top_percent": min_same_top_percent,
    }
    configured = any(value is not None for value in thresholds.values())
    violations: List[str] = []
    for result in results:
        if result["mode"] == "dense" or result["status"] != "ok":
            continue
        prefix = f"{result['corpus']}/ctx{result['ctx']}/{result['mode']}"
        metrics = result["metrics"]
        if max_mean_kld is not None and metrics["mean_kld"] > max_mean_kld:
            violations.append(f"{prefix}: mean_kld={metrics['mean_kld']} > {max_mean_kld}")
        if max_ppl_ratio is not None and metrics["mean_ppl_ratio"] > max_ppl_ratio:
            violations.append(f"{prefix}: ppl_ratio={metrics['mean_ppl_ratio']} > {max_ppl_ratio}")
        if min_same_top_percent is not None and metrics["same_top_percent"] < min_same_top_percent:
            violations.append(
                f"{prefix}: same_top_percent={metrics['same_top_percent']} < {min_same_top_percent}"
            )

    if errors or violations:
        status = "FAIL"
    elif configured:
        status = "PASS"
    else:
        status = "MEASURED"
    return {
        "status": status,
        "thresholds": thresholds,
        "execution_errors": errors,
        "quality_violations": violations,
    }


def csv_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {})
        execution = result.get("execution", {})
        shape = execution.get("shape", {})
        markers = execution.get("markers", {})
        rows.append(
            {
                "corpus": result.get("corpus"),
                "ctx": result.get("ctx"),
                "mode": result.get("mode"),
                "status": result.get("status"),
                "sample_count": shape.get("sample_count"),
                "ppl": metrics.get("ppl", metrics.get("mean_ppl_q")),
                "ppl_uncertainty": metrics.get("ppl_uncertainty", metrics.get("mean_ppl_q_uncertainty")),
                "ppl_ratio": metrics.get("mean_ppl_ratio"),
                "mean_kld": metrics.get("mean_kld"),
                "p99_kld": metrics.get("p99_kld"),
                "maximum_kld": metrics.get("maximum_kld"),
                "rms_delta_p_percent": metrics.get("rms_delta_p_percent"),
                "same_top_percent": metrics.get("same_top_percent"),
                "compression_events": execution.get("compression", {}).get("events"),
                "chunked_marker": int(bool(markers.get("chunked_attn_active"))),
                "direct_marker": int(bool(markers.get("kv_lowrank_direct_active"))),
                "packed_marker": int(bool(markers.get("edgekv_direct_decode_active"))),
                "elapsed_sec": result.get("elapsed_sec"),
                "log": result.get("log"),
                "errors": "; ".join(result.get("errors", [])),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: Mapping[str, Any]) -> None:
    prefix = f"[{result['corpus']} ctx={result['ctx']} mode={result['mode']}]"
    if result["status"] != "ok":
        print(f"{prefix} ERROR: {'; '.join(result['errors'])}")
        return
    metrics = result["metrics"]
    resumed = " resumed" if result.get("resumed") else ""
    if result["mode"] == "dense":
        print(f"{prefix} PPL={metrics['ppl']:.6f}{resumed}")
    else:
        print(
            f"{prefix} PPL-ratio={metrics['mean_ppl_ratio']:.6f} "
            f"mean-KLD={metrics['mean_kld']:.6f} p99-KLD={metrics['p99_kld']:.6f} "
            f"same-top={metrics['same_top_percent']:.3f}%{resumed}"
        )


def validate_args(args: argparse.Namespace) -> None:
    for path, label in ((args.llama_perplexity, "llama-perplexity"), (args.model, "model")):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    labels = set()
    for label, path in args.corpus:
        if label in labels:
            raise SystemExit(f"duplicate corpus label: {label}")
        labels.add(label)
        if not path.is_file():
            raise SystemExit(f"corpus does not exist: {path}")
    if any(ctx < 256 for ctx in args.ctx):
        raise SystemExit("every --ctx must be >= 256 so KL statistics cover at least 100 tokens")
    positive = (args.threads, args.chunks, args.block_size, args.rank_k, args.rank_v, args.group_size)
    if any(value <= 0 for value in positive):
        raise SystemExit("threads/chunks/block/ranks/group-size must be positive")
    if args.quant_bits not in (8, 16):
        raise SystemExit("--quant-bits must be 8 or 16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        action="append",
        type=parse_corpus,
        required=True,
        metavar="LABEL=PATH",
        help="repeatable fixed corpus window source",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ctx", action="append", type=int, default=None, help="repeatable; default: 256")
    parser.add_argument("--modes", type=parse_modes, default=parse_modes(",".join(MODES)))
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rank-k", type=int, default=32)
    parser.add_argument("--rank-v", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-base-logits", action="store_true")
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--max-mean-kld", type=float)
    parser.add_argument("--max-ppl-ratio", type=float)
    parser.add_argument("--min-same-top-percent", type=float)
    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = [256]
    validate_args(args)

    config = QualityConfig(
        llama_perplexity=args.llama_perplexity.resolve(),
        model=args.model.resolve(),
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        chunks=args.chunks,
        block_size=args.block_size,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        group_size=args.group_size,
        quant_bits=args.quant_bits,
        extra_args=tuple(args.extra_arg),
    )
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "binary": file_metadata(config.llama_perplexity),
        "model": file_metadata(config.model, include_hash=not args.skip_model_hash),
        "source": git_provenance(config.llama_perplexity),
        "corpora": {
            label: file_metadata(path.resolve())
            for label, path in args.corpus
        },
    }

    all_results: List[Dict[str, Any]] = []
    for corpus_label, corpus_path_raw in args.corpus:
        corpus_path = corpus_path_raw.resolve()
        for ctx in args.ctx:
            case_dir = work_dir / corpus_label / f"ctx{ctx}"
            case_dir.mkdir(parents=True, exist_ok=True)
            dense_logits = case_dir / "dense-base.logits"

            commands = {
                mode: build_command(config, mode, corpus_path, ctx, dense_logits)
                for mode in args.modes
            }
            overrides = {mode: environment_overrides(mode) for mode in args.modes}
            resumed: Dict[str, Dict[str, Any]] = {}
            if args.resume:
                for mode in args.modes:
                    result = load_resumable_result(
                        case_dir / f"{mode}.json",
                        commands[mode],
                        overrides[mode],
                    )
                    if result is not None:
                        resumed[mode] = result

            pending_candidates = [mode for mode in args.modes if mode != "dense" and mode not in resumed]
            dense_result = resumed.get("dense")
            if dense_result is None or (pending_candidates and not dense_logits.is_file()):
                dense_result = run_one(
                    "dense",
                    commands["dense"],
                    overrides["dense"],
                    case_dir / "dense.log",
                    args.timeout_sec,
                )
                write_json(case_dir / "dense.json", dense_result)
            dense_result.update({"corpus": corpus_label, "ctx": ctx})
            all_results.append(dense_result)
            print_result(dense_result)

            case_results = [dense_result]
            for mode in args.modes:
                if mode == "dense":
                    continue
                result = resumed.get(mode)
                if result is None:
                    if dense_result["status"] == "ok" and dense_logits.is_file():
                        result = run_one(
                            mode,
                            commands[mode],
                            overrides[mode],
                            case_dir / f"{mode}.log",
                            args.timeout_sec,
                        )
                        write_json(case_dir / f"{mode}.json", result)
                    else:
                        result = {
                            "mode": mode,
                            "status": "error",
                            "errors": ["dense baseline or logits file is unavailable"],
                            "metrics": {},
                            "execution": {},
                        }
                result.update({"corpus": corpus_label, "ctx": ctx})
                all_results.append(result)
                case_results.append(result)
                print_result(result)

            if (
                dense_logits.exists()
                and not args.keep_base_logits
                and all(result["status"] == "ok" for result in case_results)
            ):
                dense_logits.unlink()
            write_json(case_dir / "summary.json", {"results": case_results})

    aggregates = aggregate_results(all_results)
    gate = evaluate_gate(
        all_results,
        args.max_mean_kld,
        args.max_ppl_ratio,
        args.min_same_top_percent,
    )
    summary = {
        "schema_version": 1,
        "config": {
            **asdict(config),
            "llama_perplexity": str(config.llama_perplexity),
            "model": str(config.model),
            "extra_args": list(config.extra_args),
            "contexts": args.ctx,
            "modes": args.modes,
        },
        "provenance": provenance,
        "results": all_results,
        "aggregates": aggregates,
        "gate": gate,
    }
    write_json(work_dir / "summary.json", summary)
    write_csv(work_dir / "summary.csv", csv_rows(all_results))

    print(f"wrote {work_dir / 'summary.json'}")
    print(f"wrote {work_dir / 'summary.csv'}")
    print(f"GATE: {gate['status']}")
    for error in gate["execution_errors"]:
        print(f"  execution: {error}")
    for violation in gate["quality_violations"]:
        print(f"  quality: {violation}")
    return 1 if gate["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
