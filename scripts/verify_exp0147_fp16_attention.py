#!/usr/bin/env python3
"""Verify the EXP-0147 dynamic FP16 Attention boundary independently.

The diagnostic capture contains the physical Q row consumed by QK, the
post-append head-major K/V cache, and the row-major Attention output.  This
script recomputes QK -> standard causal Softmax -> AV without using the DSP
implementation or its intermediate buffers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHYSICAL_M = 64
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    difference = actual.astype(np.float32) - reference.astype(np.float32)
    actual_flat = actual.astype(np.float32).ravel()
    reference_flat = reference.astype(np.float32).ravel()
    denominator = np.linalg.norm(actual_flat) * np.linalg.norm(reference_flat)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(actual_flat, reference_flat) / denominator),
        "fp16_bit_mismatches": int(np.count_nonzero(
            actual.view(np.uint16) != reference.view(np.uint16)
        )),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(actual))),
    }


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    capture = args.capture.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    scan = manifest["shape_scan"]
    logical_rows = int(scan["logical_m"])
    if scan["mode"] == "decode":
        logical_rows = 1
        past_tokens = int(scan["initial_kv_length"])
    elif logical_rows > PHYSICAL_M:
        # The dynamic path is executed by the second physical chunk.
        logical_rows = PHYSICAL_M
        past_tokens = PHYSICAL_M
    else:
        past_tokens = 0
    total_tokens = past_tokens + logical_rows
    capacity = int(scan["kv_cache_capacity"])

    q = np.fromfile(capture / "actual_scan_q_f16.bin", dtype="<f2").reshape(
        PHYSICAL_M, HEADS, HEAD_DIM
    )
    k = np.fromfile(capture / "actual_kv_cache_k_f16.bin", dtype="<f2").reshape(
        KV_HEADS, capacity, HEAD_DIM
    )
    v = np.fromfile(capture / "actual_kv_cache_v_f16.bin", dtype="<f2").reshape(
        KV_HEADS, capacity, HEAD_DIM
    )
    actual = np.fromfile(
        capture / "actual_scan_attention_f16.bin", dtype="<f2"
    ).reshape(PHYSICAL_M, HEADS, HEAD_DIM)[:logical_rows]
    reference = np.zeros_like(actual)
    scale = np.float32(HEAD_DIM ** -0.5)

    for group in range(KV_HEADS):
        for local_head in range(Q_HEADS_PER_GROUP):
            head = group * Q_HEADS_PER_GROUP + local_head
            scores = (
                q[:logical_rows, head].astype(np.float32)
                @ k[group, :total_tokens].astype(np.float32).T
            ).astype(np.float16)
            for row in range(logical_rows):
                valid = past_tokens + row + 1
                scaled = scores[row, :valid].astype(np.float32) * scale
                exponent = np.exp(scaled - np.max(scaled))
                # Mirror the FP16 probability carrier used by the DSP path.
                probability = (
                    exponent.astype(np.float16).astype(np.float32)
                    / np.sum(exponent)
                ).astype(np.float16)
                reference[row, head] = (
                    probability.astype(np.float32)
                    @ v[group, :valid].astype(np.float32)
                ).astype(np.float16)

    report = {
        "experiment": "EXP-0147",
        "cell": scan["cell"],
        "recipe": scan["recipe"],
        "reference": (
            "captured physical Q and post-append K/V; FP32 accumulation; "
            "FP16 QK/probability/AV boundaries"
        ),
        "logical_rows": logical_rows,
        "total_kv_tokens": total_tokens,
        **metrics(actual, reference),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
