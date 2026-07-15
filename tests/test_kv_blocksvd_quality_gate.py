#!/usr/bin/env python3

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kv_blocksvd_quality_gate.py"
SPEC = importlib.util.spec_from_file_location("kv_blocksvd_quality_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


class QualityGateTest(unittest.TestCase):
    def test_parse_dense_and_kl_metrics(self):
        dense = "Final estimate: PPL = 26.3719 +/- 21.58328\n"
        self.assertEqual(
            QUALITY.parse_final_ppl(dense),
            {"ppl": 26.3719, "ppl_uncertainty": 21.58328},
        )

        plus_minus = "\u00b1"
        delta = "\u0394"
        kl = f"""
Mean PPL(Q)                   :  42.000000 {plus_minus}   1.250000
Mean PPL(base)                :  21.000000 {plus_minus}   0.750000
Mean ln(PPL(Q)/PPL(base))     :   0.693147 {plus_minus}   0.010000
Mean PPL(Q)/PPL(base)         :   2.000000 {plus_minus}   0.020000
Mean PPL(Q)-PPL(base)         :  21.000000 {plus_minus}   1.000000
Mean    KLD:   0.125000 {plus_minus}   0.005000
Maximum KLD:   1.250000
99.9%   KLD:   1.000000
99.0%   KLD:   0.750000
95.0%   KLD:   0.500000
90.0%   KLD:   0.400000
Median  KLD:   0.100000
RMS {delta}p    :  1.250 {plus_minus} 0.125 %
Same top p: 80.000 {plus_minus} 2.000 %
"""
        metrics = QUALITY.parse_kl_metrics(kl)
        self.assertEqual(metrics["mean_ppl_ratio"], 2.0)
        self.assertEqual(metrics["mean_kld"], 0.125)
        self.assertEqual(metrics["p99_kld"], 0.75)
        self.assertEqual(metrics["rms_delta_p_percent"], 1.25)
        self.assertEqual(metrics["same_top_percent"], 80.0)

    def test_build_command_and_environment_are_mode_specific(self):
        config = QUALITY.QualityConfig(
            llama_perplexity=Path("/tmp/llama-perplexity"),
            model=Path("/tmp/model.gguf"),
            threads=4,
            gpu_layers=0,
            chunks=2,
            block_size=64,
            rank_k=32,
            rank_v=64,
            group_size=4,
            quant_bits=8,
        )
        dense = QUALITY.build_command(config, "dense", Path("code.txt"), 256, Path("base.logits"))
        packed = QUALITY.build_command(config, "packed", Path("code.txt"), 256, Path("base.logits"))
        self.assertIn("--save-all-logits", dense)
        self.assertNotIn("--kv-blocksvd", dense)
        self.assertIn("--kl-divergence", packed)
        self.assertIn("--kv-lowrank-direct", packed)
        self.assertEqual(packed[packed.index("-b") + 1], "1")
        self.assertEqual(packed[packed.index("-ub") + 1], "1")
        self.assertEqual(
            QUALITY.environment_overrides("packed")["LLAMA_KV_BLOCKSVD_EDGEKV_DIRECT"],
            "1",
        )
        self.assertIsNone(
            QUALITY.environment_overrides("portable")["LLAMA_KV_BLOCKSVD_EDGEKV_DIRECT"]
        )

    def test_validate_args_accepts_q16_and_rejects_q4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama-perplexity"
            model = root / "model.gguf"
            corpus = root / "corpus.txt"
            for path in (binary, model, corpus):
                path.write_bytes(b"fixture")
            args = argparse.Namespace(
                llama_perplexity=binary,
                model=model,
                corpus=[("text", corpus)],
                ctx=[256],
                threads=4,
                chunks=1,
                block_size=64,
                rank_k=32,
                rank_v=32,
                group_size=4,
                quant_bits=16,
            )
            QUALITY.validate_args(args)
            args.quant_bits = 4
            with self.assertRaises(SystemExit):
                QUALITY.validate_args(args)

    def test_execution_markers_and_sample_count(self):
        output = """
kl_divergence: computing over 2 chunks, n_ctx=256, batch_size=1, n_seq=1
llama_kv_lowrank_direct_attn_compute: kv_lowrank_direct active
build_attn: EdgeKV direct decode active
kv_blocksvd_shadow_compress_current: cross-layer stored n_tokens=64 layers=28
"""
        execution = QUALITY.parse_execution(output)
        self.assertEqual(execution["shape"]["sample_count"], 254)
        self.assertEqual(execution["compression"]["events"], 1)
        self.assertEqual(QUALITY.marker_errors("packed", execution), [])
        self.assertNotEqual(QUALITY.marker_errors("portable", execution), [])

    def test_weighted_aggregate_and_threshold_gate(self):
        results = []
        for corpus, count, kld, log_ratio, same_top, rms in (
            ("code", 100, 0.1, math.log(1.2), 80.0, 2.0),
            ("text", 300, 0.3, math.log(1.4), 60.0, 4.0),
        ):
            results.append(
                {
                    "corpus": corpus,
                    "ctx": 256,
                    "mode": "portable",
                    "status": "ok",
                    "errors": [],
                    "execution": {"shape": {"sample_count": count}},
                    "metrics": {
                        "mean_kld": kld,
                        "maximum_kld": kld * 10,
                        "mean_log_ppl_ratio": log_ratio,
                        "mean_ppl_ratio": math.exp(log_ratio),
                        "same_top_percent": same_top,
                        "rms_delta_p_percent": rms,
                    },
                }
            )
        aggregate = QUALITY.aggregate_results(results)["portable"]
        self.assertAlmostEqual(aggregate["weighted_mean_kld"], 0.25)
        self.assertAlmostEqual(aggregate["weighted_same_top_percent"], 65.0)
        self.assertAlmostEqual(aggregate["weighted_rms_delta_p_percent"], math.sqrt(13.0))

        measured = QUALITY.evaluate_gate(results, None, None, None)
        self.assertEqual(measured["status"], "MEASURED")
        failed = QUALITY.evaluate_gate(results, 0.2, 2.0, 50.0)
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(len(failed["quality_violations"]), 1)

    def test_resume_requires_matching_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "portable.json"
            command = ["llama-perplexity", "-c", "256"]
            overrides = QUALITY.environment_overrides("portable")
            QUALITY.write_json(
                result_path,
                {
                    "status": "ok",
                    "command": command,
                    "environment_overrides": overrides,
                },
            )
            self.assertIsNotNone(QUALITY.load_resumable_result(result_path, command, overrides))
            self.assertIsNone(
                QUALITY.load_resumable_result(result_path, command + ["--chunks", "2"], overrides)
            )

    def test_resume_keeps_completed_candidate_without_dense_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary = root / "llama-perplexity"
            model = root / "model.gguf"
            corpus = root / "corpus.txt"
            work_dir = root / "work"
            for path in (binary, model, corpus):
                path.write_bytes(b"fixture")

            config = QUALITY.QualityConfig(
                llama_perplexity=binary,
                model=model,
                threads=4,
                gpu_layers=0,
                chunks=1,
                block_size=64,
                rank_k=32,
                rank_v=32,
                group_size=4,
                quant_bits=8,
            )
            case_dir = work_dir / "text" / "ctx256"
            dense_logits = case_dir / "dense-base.logits"
            shape = {"chunks": 1, "ctx": 256, "batch": 1, "seq": 1, "sample_count": 127}
            dense_command = QUALITY.build_command(config, "dense", corpus, 256, dense_logits)
            portable_command = QUALITY.build_command(config, "portable", corpus, 256, dense_logits)
            QUALITY.write_json(
                case_dir / "dense.json",
                {
                    "mode": "dense",
                    "status": "ok",
                    "errors": [],
                    "command": dense_command,
                    "environment_overrides": QUALITY.environment_overrides("dense"),
                    "metrics": {"ppl": 2.0, "ppl_uncertainty": 0.1},
                    "execution": {"shape": shape, "markers": {}, "compression": {}},
                },
            )
            QUALITY.write_json(
                case_dir / "portable.json",
                {
                    "mode": "portable",
                    "status": "ok",
                    "errors": [],
                    "command": portable_command,
                    "environment_overrides": QUALITY.environment_overrides("portable"),
                    "metrics": {
                        "mean_ppl_q": 2.2,
                        "mean_ppl_ratio": 1.1,
                        "mean_log_ppl_ratio": math.log(1.1),
                        "mean_kld": 0.01,
                        "maximum_kld": 0.1,
                        "p99_kld": 0.08,
                        "rms_delta_p_percent": 1.0,
                        "same_top_percent": 90.0,
                    },
                    "execution": {"shape": shape, "markers": {}, "compression": {}},
                },
            )

            argv = [
                str(SCRIPT),
                "--llama-perplexity",
                str(binary),
                "--model",
                str(model),
                "--corpus",
                f"text={corpus}",
                "--work-dir",
                str(work_dir),
                "--ctx",
                "256",
                "--modes",
                "dense,portable",
                "--resume",
                "--skip-model-hash",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(QUALITY.main(), 0)

            summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([result["mode"] for result in summary["results"]], ["dense", "portable"])
            self.assertTrue(all(result.get("resumed") for result in summary["results"]))


if __name__ == "__main__":
    unittest.main()
