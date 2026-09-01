#!/usr/bin/env python3
"""Compare an EXP-0147 FP16 device capture with independent references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHYSICAL_M = 64
HIDDEN = 2048
KV_HEADS = 8
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    actual_f32 = actual.astype(np.float32)
    reference_f32 = reference.astype(np.float32)
    difference = actual_f32 - reference_f32
    denominator = np.linalg.norm(actual_f32.ravel()) * np.linalg.norm(
        reference_f32.ravel()
    )
    cosine = 1.0 if denominator == 0.0 else float(
        np.dot(actual_f32.ravel(), reference_f32.ravel()) / denominator
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": cosine,
        "fp16_bit_mismatches": int(np.count_nonzero(
            actual.view(np.uint16) != reference.view(np.uint16)
        )),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(actual_f32))),
    }


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    capture = args.capture.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    scan = manifest["shape_scan"]
    logical_m = int(scan["logical_m"])
    physical_rows = int(scan["physical_chunks"]) * PHYSICAL_M
    output_rows = 1 if scan["mode"] == "decode" else logical_m
    capacity = int(scan["kv_cache_capacity"])

    actual_output = np.fromfile(
        capture / "actual_block_output_f16.bin", dtype="<f2"
    ).reshape(physical_rows, HIDDEN)[:output_rows]
    independent_output_name = f"reference_{args.recipe}_block_output_f16.bin"
    if (package / f"independent_{args.recipe}_block_output_f16.bin").is_file():
        independent_output_name = f"independent_{args.recipe}_block_output_f16.bin"
    reference_output = np.fromfile(
        package / independent_output_name, dtype="<f2"
    ).reshape(physical_rows, HIDDEN)[:output_rows]

    cache_reports: dict[str, dict[str, float | int]] = {}
    for kind in ("k", "v"):
        actual = np.fromfile(
            capture / f"actual_kv_cache_{kind}_f16.bin", dtype="<f2"
        ).reshape(KV_HEADS, capacity, HEAD_DIM)
        reference_name = f"reference_kv_cache_{kind}_f16.bin"
        if (package / f"independent_kv_cache_{kind}_f16.bin").is_file():
            reference_name = f"independent_kv_cache_{kind}_f16.bin"
        reference = np.fromfile(
            package / reference_name, dtype="<f2"
        ).reshape(KV_HEADS, capacity, HEAD_DIM)
        cache_reports[kind] = metrics(actual, reference)

    output_report = metrics(actual_output, reference_output)
    all_metrics = [output_report, *cache_reports.values()]
    gate_pass = all(
        metric["nonfinite_count"] == 0
        and metric["max_abs"] <= 0.0625
        and metric["cosine"] >= 0.99999
        for metric in all_metrics
    )
    report = {
        "experiment": "EXP-0147",
        "cell": scan["cell"],
        "recipe": args.recipe,
        "tolerance": {"max_abs": 0.0625, "minimum_cosine": 0.99999},
        "output": output_report,
        "cache_k": cache_reports["k"],
        "cache_v": cache_reports["v"],
        "gate_pass": gate_pass,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
