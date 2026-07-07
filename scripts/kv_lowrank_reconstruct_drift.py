#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


PARSED_RE = re.compile(r"Parsed message: (?P<message>\{.*\})")
WHLR_RE = re.compile(
    r"kv_lowrank_shadow_project_current: .*?"
    r"projected_tokens=(?P<projected_tokens>\d+) .*?"
    r"projected_chunks=(?P<projected_chunks>\d+) .*?"
    r"projected_layers=(?P<projected_layers>\d+) .*?"
    r"reconstructed_layers=(?P<reconstructed_layers>\d+) .*?"
    r"history_bytes=(?P<history_bytes>\d+) .*?"
    r"recon_err_k_mean=(?P<k_mean>[0-9.eE+-]+) .*?"
    r"recon_err_v_mean=(?P<v_mean>[0-9.eE+-]+)"
)


def run_cmd(cmd: List[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-60:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout


def parse_message(text: str) -> Dict[str, Any]:
    matches = list(PARSED_RE.finditer(text))
    if not matches:
        return {"content": "", "reasoning_content": "", "found": False}

    raw = matches[-1].group("message")
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return {"content": "", "reasoning_content": "", "found": False, "raw": raw}

    return {
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "found": True,
    }


def parse_whlr(text: str) -> Dict[str, Any]:
    records = []
    for match in WHLR_RE.finditer(text):
        item = {
            "projected_tokens": int(match.group("projected_tokens")),
            "projected_chunks": int(match.group("projected_chunks")),
            "projected_layers": int(match.group("projected_layers")),
            "reconstructed_layers": int(match.group("reconstructed_layers")),
            "history_bytes": int(match.group("history_bytes")),
            "k_mean": float(match.group("k_mean")),
            "v_mean": float(match.group("v_mean")),
        }
        records.append(item)

    projected = [item for item in records if item["projected_tokens"] > 0]
    return {
        "steps": len(records),
        "projected_steps": len(projected),
        "projected_tokens": sum(item["projected_tokens"] for item in projected),
        "projected_chunks": sum(item["projected_chunks"] for item in projected),
        "projected_layers_max": max((item["projected_layers"] for item in projected), default=0),
        "reconstructed_layers_max": max((item["reconstructed_layers"] for item in projected), default=0),
        "history_bytes": max((item["history_bytes"] for item in records), default=0),
        "k_mean_max": max((item["k_mean"] for item in projected), default=0.0),
        "v_mean_max": max((item["v_mean"] for item in projected), default=0.0),
    }


def output_digest(message: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "content": message.get("content", ""),
            "reasoning_content": message.get("reasoning_content", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def llama_cmd(args: argparse.Namespace, prompt: str, reconstruct_cache: bool) -> List[str]:
    cmd = [
        str(args.llama_cli),
        "-m", str(args.model),
        "-p", prompt,
        "-n", str(args.n_predict),
        "-c", str(args.ctx_size),
        "--single-turn",
        "--no-warmup",
        "--simple-io",
        "--no-display-prompt",
        "--verbose",
        "--kv-lowrank",
        "--kv-lowrank-basis-path", str(args.basis_path),
        "--kv-lowrank-rank", str(max(args.rank_k, args.rank_v)),
        "--kv-lowrank-rank-k", str(args.rank_k),
        "--kv-lowrank-rank-v", str(args.rank_v),
        "--kv-lowrank-window", str(args.window),
        "--kv-lowrank-chunk", str(args.chunk),
        *args.llama_arg,
    ]
    if reconstruct_cache:
        cmd.append("--kv-lowrank-reconstruct-cache")
    return cmd


def summarize_run(name: str, text: str, log_path: Path) -> Dict[str, Any]:
    message = parse_message(text)
    return {
        "name": name,
        "log": str(log_path),
        "message_found": message.get("found", False),
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "output_sha256": output_digest(message),
        "whlr": parse_whlr(text),
    }


def read_prompts(args: argparse.Namespace) -> List[str]:
    prompts = list(args.prompt or [])
    for path in args.prompt_file or []:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            data = json.loads(text)
            if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
                raise SystemExit(f"prompt JSON must be a list of strings: {path}")
            prompts.extend(data)
        else:
            prompts.extend(line.strip() for line in text.splitlines() if line.strip())

    if not prompts:
        raise SystemExit("at least one --prompt or --prompt-file entry is required")
    return prompts


def summarize_case(args: argparse.Namespace, case_index: int, prompt: str) -> Dict[str, Any]:
    case_dir = args.work_dir if case_index == 0 and len(args.prompts) == 1 else args.work_dir / f"case_{case_index:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    dense_log = case_dir / "dense_shadow.log"
    dense_text = run_cmd(llama_cmd(args, prompt, reconstruct_cache=False), dense_log)

    reconstruct_log = case_dir / "reconstruct_cache.log"
    reconstruct_text = run_cmd(llama_cmd(args, prompt, reconstruct_cache=True), reconstruct_log)

    dense = summarize_run("dense_shadow", dense_text, dense_log)
    reconstruct = summarize_run("reconstruct_cache", reconstruct_text, reconstruct_log)

    return {
        "index": case_index,
        "prompt": prompt,
        "dense_shadow": dense,
        "reconstruct_cache": reconstruct,
        "same_output_digest": dense["output_sha256"] == reconstruct["output_sha256"],
        "same_content": dense["content"] == reconstruct["content"],
        "same_reasoning_content": dense["reasoning_content"] == reconstruct["reasoning_content"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare WHLR-KV shadow-only output against reconstruct-cache output on a fixed prompt.",
    )
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rank-k", type=int, required=True)
    parser.add_argument("--rank-v", type=int, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path, action="append", default=[])
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--llama-arg", action="append", default=[])
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.prompts = read_prompts(args)

    cases = [summarize_case(args, i, prompt) for i, prompt in enumerate(args.prompts)]
    if len(cases) == 1:
        summary = dict(cases[0])
        summary.pop("index", None)
        summary.pop("prompt", None)
    else:
        summary = {
            "cases": cases,
            "n_cases": len(cases),
            "n_same_output_digest": sum(1 for case in cases if case["same_output_digest"]),
            "n_same_content": sum(1 for case in cases if case["same_content"]),
            "n_same_reasoning_content": sum(1 for case in cases if case["same_reasoning_content"]),
        }

    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
