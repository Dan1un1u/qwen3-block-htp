#!/usr/bin/env python3
"""Diagnose local versus composed FP16 drift across the EXP-0152 stack.

The device capture is intentionally diagnostic evidence: every layer output was
copied from VTCM to a dedicated DDR region.  The formal zero-intermediate-DDR
runtime is not changed and this script never treats the capture as performance
evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import export_exp0022_block as base  # noqa: E402


LAYERS = 28
M = 64
HIDDEN = 2048
ATOL = 0.0625
RTOL = 0.002
COSINE_GATE = 0.99999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--package", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0152/f16f16"
        ),
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def projection_keys(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}"
    return {
        "q": f"{prefix}.self_attn.q_proj.weight",
        "k": f"{prefix}.self_attn.k_proj.weight",
        "v": f"{prefix}.self_attn.v_proj.weight",
        "o": f"{prefix}.self_attn.o_proj.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
    }


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    actual_f = actual.astype(np.float64, copy=False).reshape(-1)
    reference_f = reference.astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(actual_f) & np.isfinite(reference_f)
    nonfinite = int(actual_f.size - np.count_nonzero(finite))
    if np.any(finite):
        delta = np.abs(actual_f[finite] - reference_f[finite])
        tolerance = ATOL + RTOL * np.abs(reference_f[finite])
        tolerance_ratio = delta / tolerance
        violations = int(np.count_nonzero(delta > tolerance))
        max_index_finite = int(np.argmax(delta))
        max_ratio_index_finite = int(np.argmax(tolerance_ratio))
        finite_indices = np.flatnonzero(finite)
        max_index = int(finite_indices[max_index_finite])
        max_ratio_index = int(finite_indices[max_ratio_index_finite])
        max_abs = float(delta[max_index_finite])
        mean_abs = float(delta.mean())
        rmse = float(np.sqrt(np.mean(delta * delta)))
        rms_reference = float(
            np.sqrt(np.mean(reference_f[finite] * reference_f[finite]))
        )
        nrmse = rmse / rms_reference if rms_reference > 0.0 else 0.0
        denominator = math.sqrt(
            float(np.dot(actual_f[finite], actual_f[finite])) *
            float(np.dot(reference_f[finite], reference_f[finite]))
        )
        cosine = (
            float(np.dot(actual_f[finite], reference_f[finite])) /
            denominator if denominator > 0.0 else 1.0
        )
        percentiles = np.percentile(delta, [50.0, 90.0, 99.0, 99.9])
        actual_half = actual.reshape(-1).astype("<f2", copy=False)
        reference_half = reference.reshape(-1).astype("<f2", copy=False)
        actual_bits = actual_half.view("<u2").astype(np.int32)
        reference_bits = reference_half.view("<u2").astype(np.int32)
        actual_ordered = np.where(
            (actual_bits & 0x8000) != 0,
            0x8000 - (actual_bits & 0x7fff),
            0x8000 + actual_bits,
        )
        reference_ordered = np.where(
            (reference_bits & 0x8000) != 0,
            0x8000 - (reference_bits & 0x7fff),
            0x8000 + reference_bits,
        )
        ulp_distance = np.abs(actual_ordered - reference_ordered)
        max_ulp = int(ulp_distance[finite].max(initial=0))
        max_abs_ulp = int(ulp_distance[max_index])
        max_ratio_ulp = int(ulp_distance[max_ratio_index])
    else:
        violations = actual_f.size
        max_abs = math.inf
        mean_abs = math.inf
        rmse = math.inf
        nrmse = math.inf
        cosine = 0.0
        percentiles = np.full(4, math.inf)
        max_index = -1
        max_ratio_index = -1
        tolerance_ratio = np.asarray([math.inf])
        max_ulp = 65535
        max_abs_ulp = 65535
        max_ratio_ulp = 65535
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "nrmse": nrmse,
        "cosine": cosine,
        "mixed_tolerance_violations": violations,
        "mixed_tolerance_violation_fraction": (
            violations / actual_f.size
        ),
        "nonfinite_count": nonfinite,
        "max_abs_flat_index": max_index,
        "max_abs_row": max_index // HIDDEN if max_index >= 0 else -1,
        "max_abs_channel": max_index % HIDDEN if max_index >= 0 else -1,
        "max_abs_actual": (
            float(actual_f[max_index]) if max_index >= 0 else math.nan
        ),
        "max_abs_reference": (
            float(reference_f[max_index]) if max_index >= 0 else math.nan
        ),
        "max_abs_fp16_ulp_distance": max_abs_ulp,
        "max_tolerance_ratio": float(tolerance_ratio.max()),
        "max_tolerance_ratio_flat_index": max_ratio_index,
        "max_tolerance_ratio_row": (
            max_ratio_index // HIDDEN if max_ratio_index >= 0 else -1
        ),
        "max_tolerance_ratio_channel": (
            max_ratio_index % HIDDEN if max_ratio_index >= 0 else -1
        ),
        "max_tolerance_ratio_actual": (
            float(actual_f[max_ratio_index])
            if max_ratio_index >= 0 else math.nan
        ),
        "max_tolerance_ratio_reference": (
            float(reference_f[max_ratio_index])
            if max_ratio_index >= 0 else math.nan
        ),
        "max_tolerance_ratio_abs": (
            float(abs(actual_f[max_ratio_index] -
                      reference_f[max_ratio_index]))
            if max_ratio_index >= 0 else math.nan
        ),
        "max_tolerance_ratio_fp16_ulp_distance": max_ratio_ulp,
        "max_fp16_ulp_distance": max_ulp,
        "abs_p50": float(percentiles[0]),
        "abs_p90": float(percentiles[1]),
        "abs_p99": float(percentiles[2]),
        "abs_p99_9": float(percentiles[3]),
        "local_gate": (
            nonfinite == 0 and violations == 0 and cosine >= COSINE_GATE
        ),
    }


def main() -> None:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    base.M = M
    initial_np = np.fromfile(
        args.package / "block_input_f16.bin", dtype="<f2"
    )
    actual_np = np.fromfile(args.capture, dtype="<f2")
    if initial_np.size != M * HIDDEN:
        raise ValueError(f"unexpected input elements: {initial_np.size}")
    if actual_np.size != LAYERS * M * HIDDEN:
        raise ValueError(f"unexpected capture elements: {actual_np.size}")
    initial_np = initial_np.reshape(M, HIDDEN)
    actual_np = actual_np.reshape(LAYERS, M, HIDDEN)

    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    required: list[str] = []
    for layer in range(LAYERS):
        required.extend(projection_keys(layer).values())
        prefix = f"model.layers.{layer}"
        required.extend([
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.k_norm.weight",
        ])
    shards = {index["weight_map"][name] for name in required}
    if len(shards) != 1:
        raise ValueError(f"required weights span shards: {sorted(shards)}")
    shard = args.model / next(iter(shards))

    cos, sin = base.rope_tables(torch.float16)
    unconditional = torch.from_numpy(initial_np.copy()).reshape(1, M, HIDDEN)
    records: list[dict[str, object]] = []
    with torch.inference_mode(), safe_open(
        shard, framework="pt", device="cpu"
    ) as source:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}"
            weights = {
                name: source.get_tensor(key).to(torch.float16)
                for name, key in projection_keys(layer).items()
            }
            norms = {
                "input": source.get_tensor(
                    f"{prefix}.input_layernorm.weight"
                ).to(torch.float16),
                "post": source.get_tensor(
                    f"{prefix}.post_attention_layernorm.weight"
                ).to(torch.float16),
                "q": source.get_tensor(
                    f"{prefix}.self_attn.q_norm.weight"
                ).to(torch.float16),
                "k": source.get_tensor(
                    f"{prefix}.self_attn.k_norm.weight"
                ).to(torch.float16),
            }
            unconditional, _ = base.layer_forward_f16(
                unconditional, weights, norms, cos, sin
            )
            if layer == 0:
                conditional = unconditional
                conditional_input_np = initial_np
            else:
                conditional_input_np = actual_np[layer - 1]
                conditional_input = torch.from_numpy(
                    conditional_input_np.copy()
                ).reshape(1, M, HIDDEN)
                conditional, _ = base.layer_forward_f16(
                    conditional_input, weights, norms, cos, sin
                )
            actual = actual_np[layer]
            unconditional_np = (
                unconditional.detach().cpu().numpy().astype("<f2", copy=False)
                .reshape(M, HIDDEN)
            )
            conditional_np = (
                conditional.detach().cpu().numpy().astype("<f2", copy=False)
                .reshape(M, HIDDEN)
            )
            record = {
                "layer": layer,
                "conditional_input": {
                    "max_abs": float(
                        np.abs(conditional_input_np.astype(np.float64)).max()
                    ),
                    "rms": float(np.sqrt(np.mean(
                        conditional_input_np.astype(np.float64) ** 2
                    ))),
                },
                "conditional_local": metrics(actual, conditional_np),
                "unconditional_composed": metrics(
                    actual, unconditional_np
                ),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del weights, norms, conditional

    package_final = np.fromfile(
        args.package / "reference_f16f16_block_output_f16.bin",
        dtype="<f2",
    ).reshape(M, HIDDEN)
    cpu_final = (
        unconditional.detach().cpu().numpy().astype("<f2", copy=False)
        .reshape(M, HIDDEN)
    )
    conditional_failures = [
        int(record["layer"]) for record in records
        if not bool(record["conditional_local"]["local_gate"])
    ]
    summary = {
        "experiment": "EXP-0152",
        "diagnostic": "full_stack_hidden_trajectory",
        "formal_physical_evidence": False,
        "layers": LAYERS,
        "rows": M,
        "conditional_gate": {
            "atol": ATOL,
            "rtol": RTOL,
            "cosine_minimum": COSINE_GATE,
            "failure_layers": conditional_failures,
            "pass": not conditional_failures,
        },
        "package_final_reference_reproduction": metrics(
            cpu_final, package_final
        ),
        "final_unconditional_composed": records[-1][
            "unconditional_composed"
        ],
        "final_conditional_local": records[-1]["conditional_local"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "summary": str(args.output),
        "conditional_gate_pass": not conditional_failures,
        "conditional_failure_layers": conditional_failures,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
