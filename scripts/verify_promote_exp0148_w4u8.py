#!/usr/bin/env python3
"""Verify W4U8 replay by boundaries and publish its formal package.

For each decode step, actual Q plus the persistent K/V cache are the declared
integer-Attention input boundary.  This script independently recomputes QK,
log2 Softmax, AV, and then the O/MLP tail.  Only a byte-exact result is written
as the formal decode reference.  Captured K/V remains explicitly labelled a
physical cache golden rather than an independent high-level reference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import export_exp0022_block as base  # noqa: E402
from export_exp0148_replay import load_qparams, sha256  # noqa: E402
from prepare_exp0042_attention import CONFIG, HMX_CENTER, centered_hmx_requant  # noqa: E402
from prepare_exp0042_block import unpack_w4_weight  # noqa: E402
from prepare_exp0147_decode import (  # noqa: E402
    DIVISION_NAMES,
    log2_softmax_decode,
    recenter_v,
)


PHYSICAL_M = 64
PREFILL_M = 64
DECODE_STEPS = 8
CAPACITY = 72
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
Q_BYTES = PHYSICAL_M * HIDDEN
KV_BYTES = PHYSICAL_M * KV_HEADS * HEAD_DIM
SCORE_BYTES = HEADS * PHYSICAL_M * PHYSICAL_M
AV_OFFSET = Q_BYTES + 2 * KV_BYTES + 2 * SCORE_BYTES
CORE_AUDIT_BYTES = AV_OFFSET + Q_BYTES
O_OFFSET = CORE_AUDIT_BYTES
POST_RESIDUAL_OFFSET = O_OFFSET + Q_BYTES
POST_NORM_OFFSET = POST_RESIDUAL_OFFSET + Q_BYTES
MIDDLE_OFFSET = POST_NORM_OFFSET + Q_BYTES
DOWN_OFFSET = MIDDLE_OFFSET + PHYSICAL_M * INTERMEDIATE
FINAL_OFFSET = DOWN_OFFSET + Q_BYTES
AUDIT_BYTES = FINAL_OFFSET + Q_BYTES
PROJECTION_SHAPES = {
    "o": (HIDDEN, HIDDEN),
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staging-root", type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0148"),
    )
    return parser.parse_args()


def unpack_feature(value: np.ndarray, heads: int) -> np.ndarray:
    expected = heads * PHYSICAL_M * HEAD_DIM
    if value.size != expected:
        raise ValueError(f"feature bytes {value.size}, expected {expected}")
    physical = value.reshape(heads, HEAD_DIM // 32, PHYSICAL_M, 32)
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(PHYSICAL_M, heads, HEAD_DIM)
    )


def decode_av(
    q_row: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
    valid: int, config_bytes: bytes,
) -> np.ndarray:
    output = np.empty((HEADS, HEAD_DIM), dtype=np.uint8)
    for group in range(KV_HEADS):
        fields = CONFIG.unpack_from(config_bytes, group * CONFIG.size)
        (
            abi, config_group, fraction_bits, division_mode,
            q_zero_point, k_zero_point, v_zero_point,
            _probability_zero_point, output_zero_point,
            v_numerator, v_denominator, score_shift, score_multiplier,
            av_shift, av_multiplier,
        ) = fields
        if abi != 1 or config_group != group or division_mode not in DIVISION_NAMES:
            raise ValueError(f"invalid Attention config group {group}")
        first_head = group * Q_HEADS_PER_GROUP
        q_centered = (
            q_row[first_head:first_head + Q_HEADS_PER_GROUP].astype(np.int32)
            - q_zero_point
        )
        k_centered = np.clip(
            k_cache[group, :valid].astype(np.int32) - k_zero_point,
            -128, 127,
        ).astype(np.int8)
        score_accumulator = np.matmul(
            q_centered, k_centered.astype(np.int32).T
        )
        score, _ = centered_hmx_requant(
            score_accumulator, score_multiplier, score_shift, HMX_CENTER
        )
        probability = log2_softmax_decode(
            score, fraction_bits, DIVISION_NAMES[division_mode]
        )
        centered_v = v_cache[group, :valid].astype(np.int32) - v_zero_point
        signed_v = recenter_v(centered_v, v_numerator, v_denominator)
        av_accumulator = np.matmul(
            probability.astype(np.int32), signed_v.astype(np.int32)
        )
        group_output, _ = centered_hmx_requant(
            av_accumulator, av_multiplier, av_shift, output_zero_point
        )
        output[first_head:first_head + Q_HEADS_PER_GROUP] = group_output
    return output


def padded_rows(row: np.ndarray, zero_point: int) -> np.ndarray:
    output = np.full((PHYSICAL_M, row.size), zero_point, dtype=np.uint8)
    output[0] = row
    return output


def tile_major_carrier(row: np.ndarray, zero_point: int) -> np.ndarray:
    logical = padded_rows(row, zero_point)
    return np.ascontiguousarray(
        logical.reshape(PHYSICAL_M, row.size // 32, 32).transpose(1, 0, 2)
    ).reshape(-1)


def first_tile_major_row(value: np.ndarray, channels: int) -> np.ndarray:
    expected = PHYSICAL_M * channels
    if value.size != expected:
        raise ValueError(f"tile-major bytes {value.size}, expected {expected}")
    return np.ascontiguousarray(
        value.reshape(channels // 32, PHYSICAL_M, 32)[:, 0, :]
    ).reshape(channels)


def tail_output(
    input_row: np.ndarray, av: np.ndarray,
    weights: dict[str, torch.Tensor], post_weight: torch.Tensor,
    qparams: dict[str, dict[str, object]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden_encoded = torch.from_numpy(input_row.reshape(1, 1, HIDDEN).copy())
    hidden = base.dequantize_u8(
        hidden_encoded, qparams["block_input"]
    ).to(torch.float16)
    av_encoded = torch.from_numpy(av.reshape(1, 1, HIDDEN).copy())
    attention = base.dequantize_u8(
        av_encoded, qparams["attention_concat"]
    ).to(torch.float16)

    def boundary(
        name: str, value: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor]:
        encoded = base.quantize_u8(value, qparams[name])
        logical = encoded.cpu().numpy().reshape(-1).astype(np.uint8, copy=False)
        decoded = base.dequantize_u8(encoded, qparams[name]).to(torch.float16)
        return logical, decoded

    projected_u8, projected = boundary(
        "attention_projection", base.linear_half(attention, weights["o"])
    )
    residual_u8, residual = boundary("post_attention_residual", hidden + projected)
    normalized_u8, normalized = boundary(
        "post_attention_norm", base.rms_norm(residual, post_weight)
    )
    _gate_u8, gate = boundary(
        "gate", base.linear_half(normalized, weights["gate"])
    )
    _up_u8, up = boundary("up", base.linear_half(normalized, weights["up"]))
    middle_u8, middle = boundary("middle", F.silu(gate.float()) * up.float())
    down_u8, down = boundary("down", base.linear_half(middle, weights["down"]))
    final = base.quantize_u8(
        residual + down, qparams["block_output"]
    ).cpu().numpy().reshape(HIDDEN)
    boundaries = {
        "o": tile_major_carrier(
            projected_u8, int(qparams["attention_projection"]["zero_point"])
        ),
        "post_residual": padded_rows(
            residual_u8, int(qparams["post_attention_residual"]["zero_point"])
        ).reshape(-1),
        "post_norm": tile_major_carrier(
            normalized_u8, int(qparams["post_attention_norm"]["zero_point"])
        ),
        "middle": tile_major_carrier(
            middle_u8, int(qparams["middle"]["zero_point"])
        ),
        "down": tile_major_carrier(
            down_u8, int(qparams["down"]["zero_point"])
        ),
        "final": padded_rows(
            final, int(qparams["block_output"]["zero_point"])
        ).reshape(-1),
    }
    return final, boundaries


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    capture = args.capture.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    qparams = load_qparams(source)
    config_bytes = (source / "attention_config_all_groups.bin").read_bytes()
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError("unexpected Attention config size")
    k_cache = np.fromfile(
        capture / "actual_replay_k_cache.bin", dtype=np.uint8
    ).reshape(KV_HEADS, CAPACITY, HEAD_DIM)
    v_cache = np.fromfile(
        capture / "actual_replay_v_cache.bin", dtype=np.uint8
    ).reshape(KV_HEADS, CAPACITY, HEAD_DIM)
    weights = {
        name: unpack_w4_weight(source, name, *shape)
        for name, shape in PROJECTION_SHAPES.items()
    }
    post_weight = torch.from_numpy(
        np.fromfile(source / "post_norm_weight_f16.bin", dtype="<f2").copy()
    )

    independent_rows: list[np.ndarray] = []
    step_reports = []
    prefill_actual = np.fromfile(
        capture / "actual_replay_output_00_u8.bin", dtype=np.uint8
    ).reshape(PHYSICAL_M, HIDDEN)[0]
    prefill_reference = np.fromfile(
        source / "reference_w4u8_integer_attention_block_output_u8.bin",
        dtype=np.uint8,
    ).reshape(PHYSICAL_M, HIDDEN)[0]
    if np.any(prefill_actual != prefill_reference):
        raise ValueError("prefill output is not byte-exact")

    for index in range(DECODE_STEPS):
        step = index + 1
        audit = np.fromfile(
            capture / f"actual_replay_attention_audit_{step:02d}.bin",
            dtype=np.uint8,
        )
        if audit.size != AUDIT_BYTES:
            raise ValueError(f"step {step}: audit bytes {audit.size}")
        q = unpack_feature(audit[:Q_BYTES], HEADS)
        actual_av = unpack_feature(
            audit[AV_OFFSET:CORE_AUDIT_BYTES], HEADS
        )[0]
        valid = PREFILL_M + step
        expected_av = decode_av(q[0], k_cache, v_cache, valid, config_bytes)
        av_mismatches = int(np.count_nonzero(actual_av != expected_av))
        input_row = np.fromfile(
            source / f"replay_decode_input_{index:02d}_u8.bin",
            dtype=np.uint8,
        ).reshape(PHYSICAL_M, HIDDEN)[0]
        expected_output, expected_boundaries = tail_output(
            input_row, expected_av, weights, post_weight, qparams
        )
        actual_output = np.fromfile(
            capture / f"actual_replay_output_{step:02d}_u8.bin",
            dtype=np.uint8,
        ).reshape(PHYSICAL_M, HIDDEN)[0]
        output_mismatches = int(np.count_nonzero(actual_output != expected_output))
        actual_boundaries = {
            "o": audit[O_OFFSET:POST_RESIDUAL_OFFSET],
            "post_residual": audit[POST_RESIDUAL_OFFSET:POST_NORM_OFFSET],
            "post_norm": audit[POST_NORM_OFFSET:MIDDLE_OFFSET],
            "middle": audit[MIDDLE_OFFSET:DOWN_OFFSET],
            "down": audit[DOWN_OFFSET:FINAL_OFFSET],
            "final": audit[FINAL_OFFSET:AUDIT_BYTES],
        }
        physical_boundary_mismatches = {
            name: int(np.count_nonzero(actual_boundaries[name] != expected))
            for name, expected in expected_boundaries.items()
        }
        actual_active = {
            "o": first_tile_major_row(actual_boundaries["o"], HIDDEN),
            "post_residual": actual_boundaries["post_residual"][:HIDDEN],
            "post_norm": first_tile_major_row(
                actual_boundaries["post_norm"], HIDDEN
            ),
            "middle": first_tile_major_row(
                actual_boundaries["middle"], INTERMEDIATE
            ),
            "down": first_tile_major_row(actual_boundaries["down"], HIDDEN),
            "final": actual_boundaries["final"][:HIDDEN],
        }
        expected_active = {
            "o": first_tile_major_row(expected_boundaries["o"], HIDDEN),
            "post_residual": expected_boundaries["post_residual"][:HIDDEN],
            "post_norm": first_tile_major_row(
                expected_boundaries["post_norm"], HIDDEN
            ),
            "middle": first_tile_major_row(
                expected_boundaries["middle"], INTERMEDIATE
            ),
            "down": first_tile_major_row(expected_boundaries["down"], HIDDEN),
            "final": expected_boundaries["final"][:HIDDEN],
        }
        boundary_mismatches = {
            name: int(np.count_nonzero(actual_active[name] != expected))
            for name, expected in expected_active.items()
        }
        step_reports.append(
            {
                "step": step,
                "position": PREFILL_M + index,
                "valid_length": valid,
                "attention_av_mismatches": av_mismatches,
                "teacher_internal_boundary_mismatches_diagnostic":
                    boundary_mismatches,
                "inactive_physical_boundary_mismatches_diagnostic": {
                    name: physical_boundary_mismatches[name] - mismatches
                    for name, mismatches in boundary_mismatches.items()
                },
                "tail_output_mismatches": output_mismatches,
                "q_sha256": hashlib_array(q[0]),
                "expected_av_sha256": hashlib_array(expected_av),
                "expected_output_sha256": hashlib_array(expected_output),
            }
        )
        if av_mismatches != 0 or output_mismatches != 0:
            raise ValueError(f"decode step {step} failed: {step_reports[-1]}")
        independent_rows.append(expected_output)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="w4u8-formal-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        k_cache.tofile(staging / "reference_kv_cache_k_u8.bin")
        v_cache.tofile(staging / "reference_kv_cache_v_u8.bin")
        output_zero = int(qparams["block_output"]["zero_point"])
        for index, row in enumerate(independent_rows):
            padded = np.full((PHYSICAL_M, HIDDEN), output_zero, dtype=np.uint8)
            padded[0] = row
            padded.tofile(staging / f"replay_decode_reference_{index:02d}_u8.bin")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest["w4u8_boundary_verification"] = {
            "method": "actual_Q_and_persistent_KV_are_declared_inputs_then_independent_QK_log2Softmax_AV_and_O_MLP_tail",
            "capture": str(capture),
            "cache_reference_kind": "device_physical_boundary_golden",
            "cache_is_independent_math_reference": False,
            "prefill_output_mismatches": 0,
            "decode_steps": step_reports,
        }
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(json.dumps({"output": str(output), "steps": step_reports}, indent=2))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


def hashlib_array(value: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
