#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import struct
from collections import defaultdict

def load_manifest(dump_dir: str):
    path = os.path.join(dump_dir, "manifest.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["dump_dir"] = dump_dir
            rows.append(row)
    return rows


def row_base_key(row):
    return (
        row["node_name"],
        row["op"],
        row["role"],
        row["tensor_name"],
        row["type"],
        row["ne0"],
        row["ne1"],
        row["ne2"],
        row["ne3"],
        row["nbytes"],
    )


def with_occurrence_key(rows):
    out = {}
    counter = defaultdict(int)
    for row in rows:
        b = row_base_key(row)
        idx = counter[b]
        counter[b] += 1
        out[b + (idx,)] = row
    return out


def read_tensor(path: str, ggml_type: str, nbytes: int):
    ggml_type = ggml_type.lower()
    with open(path, "rb") as f:
        data = f.read()

    if len(data) != nbytes:
        raise RuntimeError(f"size mismatch: {path}, expected={nbytes}, actual={len(data)}")

    def unpack_all(fmt, step):
        if len(data) % step != 0:
            raise RuntimeError(f"unaligned data for {ggml_type}: {path}")
        return [x[0] for x in struct.iter_unpack(fmt, data)]

    if ggml_type == "f32":
        return unpack_all("<f", 4)
    if ggml_type == "f16":
        return unpack_all("<e", 2)
    if ggml_type == "bf16":
        if len(data) % 2 != 0:
            raise RuntimeError(f"unaligned data for bf16: {path}")
        out = []
        for (u16,) in struct.iter_unpack("<H", data):
            u32 = u16 << 16
            out.append(struct.unpack("<f", struct.pack("<I", u32))[0])
        return out
    if ggml_type == "i8":
        return [float(v[0]) for v in struct.iter_unpack("<b", data)]
    if ggml_type == "i16":
        return [float(v[0]) for v in struct.iter_unpack("<h", data)]
    if ggml_type == "i32":
        return [float(v[0]) for v in struct.iter_unpack("<i", data)]
    if ggml_type == "i64":
        return [float(v[0]) for v in struct.iter_unpack("<q", data)]
    return None


def compare_rows(row_a, row_b):
    nbytes = int(row_a["nbytes"])
    pa = os.path.join(row_a["dump_dir"], row_a["file"])
    pb = os.path.join(row_b["dump_dir"], row_b["file"])
    ta = read_tensor(pa, row_a["type"], nbytes)
    tb = read_tensor(pb, row_b["type"], nbytes)

    if ta is None or tb is None:
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            ba = fa.read()
            bb = fb.read()
        same = ba == bb
        return {
            "max_abs": 0.0 if same else float("inf"),
            "mean_abs": 0.0 if same else float("inf"),
            "rmse": 0.0 if same else float("inf"),
            "same": same,
        }

    if len(ta) != len(tb):
        raise RuntimeError(f"shape mismatch in tensor bytes: {pa} vs {pb}")

    n = len(ta)
    if n == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0, "same": True}

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    same = True
    for a, b in zip(ta, tb):
        d = float(a) - float(b)
        ad = abs(d)
        if ad != 0.0:
            same = False
        if ad > max_abs:
            max_abs = ad
        sum_abs += ad
        sum_sq += d * d

    mean_abs = sum_abs / n
    rmse = math.sqrt(sum_sq / n)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "same": same,
    }


def parse_block_point(row):
    role = row["role"]
    if role not in ("blk_in", "attn_out", "ffn_out", "blk_out", "head_norm", "head_out"):
        return None, None

    if role.startswith("head_"):
        return -1, role

    m = re.match(r".*-(\d+)$", row["node_name"])
    if not m:
        return None, None
    return int(m.group(1)), role


def print_block_summary(results, threshold):
    grouped = defaultdict(list)
    for r in results:
        block_idx, point = parse_block_point(r)
        if block_idx is None:
            continue
        grouped[(block_idx, point)].append(r)

    if not grouped:
        print("\nNo block-keypoint rows found. Set LLAMA_DUMP_BLOCK_KEYPOINTS_ONLY=1 when dumping.")
        return

    order = {
        "blk_in": 0,
        "attn_out": 1,
        "ffn_out": 2,
        "blk_out": 3,
        "head_norm": 4,
        "head_out": 5,
    }

    summary = []
    for (block_idx, point), rows in grouped.items():
        best = max(rows, key=lambda x: x["max_abs"])
        summary.append({
            "block_idx": block_idx,
            "point": point,
            "max_abs": best["max_abs"],
            "mean_abs": best["mean_abs"],
            "rmse": best["rmse"],
            "tensor_name": best["tensor_name"],
            "shape": best["shape"],
            "count": len(rows),
        })

    summary.sort(key=lambda x: (x["block_idx"], order.get(x["point"], 99)))

    print("\nBlock keypoint summary (max diff per point):")
    for s in summary:
        block_text = "head" if s["block_idx"] < 0 else f"blk.{s['block_idx']}"
        print(
            f"{block_text:>8}\t{s['point']:<9}\t"
            f"max={s['max_abs']:.6g}\tmean={s['mean_abs']:.6g}\trmse={s['rmse']:.6g}\t"
            f"n={s['count']}\t{s['tensor_name']}\t{s['shape']}"
        )

    block_overall = defaultdict(float)
    for s in summary:
        if s["block_idx"] >= 0:
            block_overall[s["block_idx"]] = max(block_overall[s["block_idx"]], s["max_abs"])

    first_diverged = None
    for i in sorted(block_overall.keys()):
        if block_overall[i] > threshold:
            first_diverged = i
            break

    print("\nDiagnosis:")
    if first_diverged is None:
        print(f"- blocks all <= threshold ({threshold:g}); check head_norm/head_out for late divergence.")
    elif first_diverged == 0:
        print(f"- diverges from blk.0 (threshold={threshold:g}), likely early path issue (matmul/rope/attn).")
    else:
        print(f"- first diverged block: blk.{first_diverged} (threshold={threshold:g}).")


def main():
    ap = argparse.ArgumentParser(description="Compare llama layer I/O dumps from two repos/runs.")
    ap.add_argument("--a", required=True, help="dump dir A (contains manifest.tsv)")
    ap.add_argument("--b", required=True, help="dump dir B (contains manifest.tsv)")
    ap.add_argument("--top", type=int, default=50, help="show top N largest diffs")
    ap.add_argument("--out", default="", help="optional output TSV path")
    ap.add_argument("--block-summary", action="store_true", help="summarize by block keypoints (blk_in/attn_out/ffn_out/blk_out)")
    ap.add_argument("--threshold", type=float, default=1e-4, help="divergence threshold used by --block-summary")
    args = ap.parse_args()

    rows_a = load_manifest(args.a)
    rows_b = load_manifest(args.b)
    map_a = with_occurrence_key(rows_a)
    map_b = with_occurrence_key(rows_b)

    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"rows_a={len(rows_a)} rows_b={len(rows_b)} common={len(common)} only_a={len(only_a)} only_b={len(only_b)}")

    results = []
    for k in common:
        ra = map_a[k]
        rb = map_b[k]
        metrics = compare_rows(ra, rb)
        result = {
            "node_idx_a": ra["node_idx"],
            "node_idx_b": rb["node_idx"],
            "node_name": ra["node_name"],
            "op": ra["op"],
            "role": ra["role"],
            "tensor_name": ra["tensor_name"],
            "type": ra["type"],
            "shape": f"[{ra['ne0']},{ra['ne1']},{ra['ne2']},{ra['ne3']}]",
            "nbytes": ra["nbytes"],
            **metrics,
        }
        results.append(result)

    results.sort(key=lambda x: (x["max_abs"], x["mean_abs"], x["rmse"]), reverse=True)
    print("\nTop diffs:")
    for r in results[: args.top]:
        print(
            f"{r['max_abs']:.6g}\t{r['mean_abs']:.6g}\t{r['rmse']:.6g}\t"
            f"{r['op']}\t{r['role']}\t{r['tensor_name']}\t{r['shape']}\t"
            f"idxA={r['node_idx_a']} idxB={r['node_idx_b']}"
        )

    if args.out:
        fieldnames = [
            "node_idx_a", "node_idx_b", "node_name", "op", "role", "tensor_name",
            "type", "shape", "nbytes", "max_abs", "mean_abs", "rmse", "same"
        ]
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote comparison TSV: {args.out}")

    if args.block_summary:
        print_block_summary(results, args.threshold)


if __name__ == "__main__":
    main()
