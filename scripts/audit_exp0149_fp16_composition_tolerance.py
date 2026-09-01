#!/usr/bin/env python3
"""Audit a composition-aware FP16 tolerance candidate for EXP-0149.

The current authoritative gate remains unchanged.  This script quantifies why
the inherited single-layer absolute threshold fails after three layers and
tests a candidate mixed tolerance without promoting or accepting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREFILL_ROWS = 64
DECODE_STEPS = 8
HIDDEN = 2048
LAYERS = (13, 14, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0625)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--minimum-cosine", type=float, default=0.99999)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cosine(actual: np.ndarray, reference: np.ndarray) -> float:
    actual64 = actual.astype(np.float64).ravel()
    reference64 = reference.astype(np.float64).ravel()
    denominator = np.linalg.norm(actual64) * np.linalg.norm(reference64)
    return 1.0 if denominator == 0.0 else float(
        np.dot(actual64, reference64) / denominator
    )


def ordered_fp16(bits: np.ndarray) -> np.ndarray:
    """Map IEEE-754 binary16 bits to monotonically ordered integers."""
    sign = (bits & np.uint16(0x8000)) != 0
    magnitude = (bits & np.uint16(0x7fff)).astype(np.int32)
    return np.where(sign, 0x8000 - magnitude, 0x8000 + magnitude)


def metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
    minimum_cosine: float,
) -> dict[str, object]:
    actual32 = actual.astype(np.float32)
    reference32 = reference.astype(np.float32)
    difference = np.abs(actual32 - reference32)
    allowance = np.float32(atol) + np.float32(rtol) * np.abs(reference32)
    denominator = np.maximum(np.abs(reference32), np.float32(1.0e-30))
    required_rtol = np.maximum(difference - np.float32(atol), 0.0) / denominator
    absolute_violation_indices = np.flatnonzero(difference > atol)
    channels = np.unique(absolute_violation_indices % HIDDEN).tolist()
    ulp_distance = np.abs(
        ordered_fp16(actual.view(np.uint16)) -
        ordered_fp16(reference.view(np.uint16))
    )
    absolute_violation_ulp = ulp_distance.ravel()[absolute_violation_indices]
    value_cosine = cosine(actual, reference)
    nonfinite = int(np.count_nonzero(~np.isfinite(actual32)))
    mixed_violations = int(np.count_nonzero(difference > allowance))
    return {
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": value_cosine,
        "nonfinite_count": nonfinite,
        "absolute_only_violation_count": int(absolute_violation_indices.size),
        "absolute_only_violation_channels": channels,
        "absolute_only_max_fp16_ulp_distance": (
            int(np.max(absolute_violation_ulp))
            if absolute_violation_ulp.size != 0 else 0
        ),
        "max_fp16_ulp_distance": int(np.max(ulp_distance)),
        "mixed_tolerance_violation_count": mixed_violations,
        "max_required_rtol_after_atol": float(np.max(required_rtol)),
        "candidate_pass": (
            nonfinite == 0 and mixed_violations == 0 and
            value_cosine >= minimum_cosine
        ),
    }


def main() -> None:
    args = parse_args()
    package = (args.package_root.resolve() / args.recipe)
    capture = (args.capture_root.resolve() / f"capture_{args.recipe}")
    tensors: dict[str, dict[str, object]] = {}

    for step in range(DECODE_STEPS + 1):
        rows = PREFILL_ROWS if step == 0 else 1
        actual = np.fromfile(
            capture / f"actual_replay_output_{step:02d}_f16.bin", dtype="<f2"
        ).reshape(PREFILL_ROWS, HIDDEN)[:rows]
        reference_name = (
            f"reference_{args.recipe}_block_output_f16.bin"
            if step == 0
            else f"replay_decode_reference_{step - 1:02d}_f16.bin"
        )
        reference = np.fromfile(package / reference_name, dtype="<f2").reshape(
            PREFILL_ROWS, HIDDEN
        )[:rows]
        tensors[f"output_step_{step:02d}"] = metrics(
            actual, reference, args.atol, args.rtol, args.minimum_cosine
        )

    for layer in LAYERS:
        for kind in ("k", "v"):
            actual = np.fromfile(
                capture / f"actual_layer{layer}_replay_{kind}_cache.bin",
                dtype="<f2",
            )
            reference = np.fromfile(
                package / f"layer{layer}" /
                    f"reference_kv_cache_{kind}_f16.bin",
                dtype="<f2",
            )
            tensors[f"layer{layer}_cache_{kind}"] = metrics(
                actual, reference, args.atol, args.rtol,
                args.minimum_cosine
            )

    report = {
        "experiment": "EXP-0149",
        "recipe": args.recipe,
        "status": "candidate_gate_not_authoritative",
        "candidate_tolerance": {
            "absolute": args.atol,
            "relative": args.rtol,
            "minimum_cosine": args.minimum_cosine,
            "formula": "abs(actual-reference) <= atol + rtol*abs(reference)",
        },
        "old_absolute_gate_pass": all(
            value["absolute_only_violation_count"] == 0
            for value in tensors.values()
        ),
        "candidate_gate_pass": all(
            bool(value["candidate_pass"]) for value in tensors.values()
        ),
        "maximum_required_rtol_after_atol": max(
            float(value["max_required_rtol_after_atol"])
            for value in tensors.values()
        ),
        "absolute_only_violation_channels": sorted({
            int(channel)
            for value in tensors.values()
            for channel in value["absolute_only_violation_channels"]
        }),
        "tensors": tensors,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["candidate_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
