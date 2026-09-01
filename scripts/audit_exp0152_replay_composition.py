#!/usr/bin/env python3
"""Audit the EXP-0152 formal replay under the approved FP16 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


LAYERS = 28
PREFILL_ROWS = 64
DECODE_STEPS = 8
HIDDEN = 2048
KV_HEADS = 8
HEAD_DIM = 128
CACHE_CAPACITY = 72
ATOL = 0.0625
RTOL = 0.002
MIN_COSINE = 0.99999
MAX_CACHE_VIOLATION_FRACTION = 0.01
MAX_COMPOSED_NRMSE = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    actual64 = actual.astype(np.float64, copy=False).reshape(-1)
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    if actual64.shape != reference64.shape:
        raise ValueError(
            f"shape mismatch: actual={actual64.shape} reference={reference64.shape}"
        )
    finite = np.isfinite(actual64) & np.isfinite(reference64)
    nonfinite = int(actual64.size - np.count_nonzero(finite))
    if not np.all(finite):
        return {
            "elements": int(actual64.size),
            "nonfinite_count": nonfinite,
            "max_abs": math.inf,
            "rmse": math.inf,
            "nrmse": math.inf,
            "cosine": 0.0,
            "mixed_tolerance_violations": int(actual64.size),
            "mixed_tolerance_violation_fraction": 1.0,
        }
    delta = np.abs(actual64 - reference64)
    tolerance = ATOL + RTOL * np.abs(reference64)
    squared_error = float(np.dot(delta, delta))
    reference_energy = float(np.dot(reference64, reference64))
    actual_energy = float(np.dot(actual64, actual64))
    denominator = math.sqrt(actual_energy * reference_energy)
    violations = int(np.count_nonzero(delta > tolerance))
    return {
        "elements": int(actual64.size),
        "nonfinite_count": 0,
        "max_abs": float(delta.max(initial=0.0)),
        "rmse": float(math.sqrt(squared_error / actual64.size)),
        "nrmse": (
            float(math.sqrt(squared_error / reference_energy))
            if reference_energy > 0.0 else
            (0.0 if squared_error == 0.0 else math.inf)
        ),
        "cosine": (
            float(np.dot(actual64, reference64) / denominator)
            if denominator > 0.0 else
            (1.0 if actual_energy == 0.0 and reference_energy == 0.0 else 0.0)
        ),
        "mixed_tolerance_violations": violations,
        "mixed_tolerance_violation_fraction": violations / actual64.size,
    }


def main() -> None:
    args = parse_args()
    output_records: dict[str, dict[str, object]] = {}
    for step in range(DECODE_STEPS + 1):
        rows = PREFILL_ROWS if step == 0 else 1
        actual = np.fromfile(
            args.capture / f"actual_replay_output_{step:02d}_f16.bin",
            dtype="<f2",
        ).reshape(PREFILL_ROWS, HIDDEN)[:rows]
        reference_name = (
            f"reference_{args.recipe}_block_output_f16.bin"
            if step == 0 else
            f"replay_decode_reference_{step - 1:02d}_f16.bin"
        )
        reference = np.fromfile(
            args.package / reference_name, dtype="<f2"
        ).reshape(PREFILL_ROWS, HIDDEN)[:rows]
        record = metrics(actual, reference)
        record["gate_pass"] = (
            record["nonfinite_count"] == 0 and
            record["cosine"] >= MIN_COSINE and
            record["nrmse"] <= MAX_COMPOSED_NRMSE
        )
        output_records[f"step_{step:02d}"] = record

    cache_records: dict[str, dict[str, object]] = {}
    expected_cache_elements = KV_HEADS * CACHE_CAPACITY * HEAD_DIM
    for layer in range(LAYERS):
        for kind in ("k", "v"):
            actual = np.fromfile(
                args.capture / f"actual_layer{layer}_replay_{kind}_cache.bin",
                dtype="<f2",
            )
            reference = np.fromfile(
                args.package / f"layer{layer}" /
                f"reference_kv_cache_{kind}_f16.bin",
                dtype="<f2",
            )
            if actual.size != expected_cache_elements:
                raise ValueError(
                    f"layer {layer} {kind} actual elements={actual.size}"
                )
            record = metrics(actual, reference)
            record["gate_pass"] = (
                record["nonfinite_count"] == 0 and
                record["cosine"] >= MIN_COSINE
            )
            record["legacy_local_bound_pass"] = (
                record["mixed_tolerance_violation_fraction"] <=
                MAX_CACHE_VIOLATION_FRACTION
            )
            cache_records[f"layer{layer}_{kind}"] = record

    output_failures = [
        name for name, record in output_records.items()
        if not record["gate_pass"]
    ]
    cache_failures = [
        name for name, record in cache_records.items()
        if not record["gate_pass"]
    ]
    cache_legacy_local_bound_failures = [
        name for name, record in cache_records.items()
        if not record["legacy_local_bound_pass"]
    ]
    worst_caches = sorted(
        (
            {"tensor": name, **record}
            for name, record in cache_records.items()
        ),
        key=lambda item: (
            item["mixed_tolerance_violation_fraction"],
            -item["cosine"],
        ),
        reverse=True,
    )
    report = {
        "experiment": "EXP-0152",
        "recipe": args.recipe,
        "gate_version": "composition_v2",
        "thresholds": {
            "fp16_atol": ATOL,
            "fp16_rtol": RTOL,
            "minimum_cosine": MIN_COSINE,
            "maximum_cache_mixed_tolerance_violation_fraction": (
                MAX_CACHE_VIOLATION_FRACTION
            ),
            "formal_composed_cache_gate": (
                "no_nonfinite_and_cosine_only; mixed fraction is local "
                "conditional diagnostic"
            ),
            "maximum_composed_output_nrmse": MAX_COMPOSED_NRMSE,
        },
        "output_failures": output_failures,
        "cache_failures": cache_failures,
        "cache_legacy_local_bound_failures": (
            cache_legacy_local_bound_failures
        ),
        "pass": not output_failures and not cache_failures,
        "worst_caches": worst_caches[:10],
        "outputs": output_records,
        "caches": cache_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.output),
        "pass": report["pass"],
        "output_failures": output_failures,
        "cache_failures": cache_failures,
        "cache_legacy_local_bound_failures": (
            cache_legacy_local_bound_failures
        ),
        "worst_caches": worst_caches[:10],
    }, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
