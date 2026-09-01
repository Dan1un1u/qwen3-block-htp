#!/usr/bin/env python3
"""Prepare one-token W4U8 decode packages for EXP-0147.

The persistent cache is synthesized by repeating the accepted M=64 device
Attention-boundary audit.  The current token is audit row zero, so the CPU
reference can independently reproduce the generalized integer QK, log2
Softmax, AV, and the remainder of the Qwen3 block without adding diagnostic
DDR traffic to the measured device run.
"""

from __future__ import annotations

import argparse
import hashlib
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

from export_exp0022_block import (  # noqa: E402
    dequantize_u8,
    linear_half,
    quantize_u8,
    rms_norm,
)
from prepare_exp0042_attention import (  # noqa: E402
    CONFIG,
    EXP_FRAC_BITS,
    HMX_CENTER,
    centered_hmx_requant,
    divide_probability,
)
from prepare_exp0042_block import unpack_w4_weight  # noqa: E402


PHYSICAL_M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
PROJECTION_SHAPES = {
    "o": (HIDDEN, HIDDEN),
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}
DIVISION_NAMES = {1: "exact", 2: "sole", 3: "endpoint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--past-length", type=int, choices=(64, 256, 1024, 4096), required=True
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/"
            "block_package_layer14_m64_integer_attention_parallel"
        ),
    )
    parser.add_argument(
        "--reference-source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
        ),
    )
    parser.add_argument(
        "--actual-audit",
        type=Path,
        default=Path(
            "/mnt/c/Users/35961/AppData/Local/Temp/exp0147-audit-m64"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0147"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_native_feature_tiles(path: Path, heads: int) -> np.ndarray:
    value = np.fromfile(path, dtype=np.uint8)
    expected = heads * PHYSICAL_M * HEAD_DIM
    if value.size != expected:
        raise ValueError(f"{path}: got {value.size}, expected {expected}")
    physical = value.reshape(heads, HEAD_DIM // 32, PHYSICAL_M, 32)
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(PHYSICAL_M, heads, HEAD_DIM)
    )


def recenter_v(values: np.ndarray, numerator: int, denominator: int) -> np.ndarray:
    values_i32 = values.astype(np.int32)
    positive = (values_i32 * numerator + denominator // 2) // denominator
    negative = -(
        ((-values_i32) * numerator + denominator // 2) // denominator
    )
    return np.clip(np.where(values_i32 >= 0, positive, negative), -128, 127).astype(
        np.int8
    )


def log2_softmax_decode(
    scores: np.ndarray, fraction_bits: int, division: str
) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(scores.shape)
    probability = np.zeros_like(scores, dtype=np.uint8)
    rounding = 1 << (fraction_bits - 1)
    valid_count = scores.shape[1]
    for head in range(scores.shape[0]):
        maximum = int(scores[head].max())
        delta = maximum - scores[head].astype(np.int32)
        exponents = np.minimum(15, (delta + rounding) >> fraction_bits)
        total = sum(1 << (EXP_FRAC_BITS - int(code)) for code in exponents)
        for token, exponent in enumerate(exponents):
            probability[head, token] = divide_probability(
                int(exponent), total, division, valid_count
            )
    return probability


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    reference_source = args.reference_source.resolve()
    actual_audit = args.actual_audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing package: {output}")
    for required in (
        source / "manifest.json",
        reference_source / "manifest.json",
        actual_audit / "actual_q_tiles_u8.bin",
        actual_audit / "actual_k_tiles_u8.bin",
        actual_audit / "actual_v_tiles_u8.bin",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    reference_manifest = json.loads(
        (reference_source / "manifest.json").read_text(encoding="utf-8")
    )
    qparams = reference_manifest["u8_qparams"]
    block_input = np.fromfile(
        source / "reference_w4u8_block_input_u8.bin", dtype=np.uint8
    ).reshape(PHYSICAL_M, HIDDEN)
    padded_input = np.full_like(
        block_input, int(qparams["block_input"]["zero_point"])
    )
    padded_input[0] = block_input[0]

    q_rows = unpack_native_feature_tiles(
        actual_audit / "actual_q_tiles_u8.bin", HEADS
    )
    k_rows = unpack_native_feature_tiles(
        actual_audit / "actual_k_tiles_u8.bin", KV_HEADS
    )
    v_rows = unpack_native_feature_tiles(
        actual_audit / "actual_v_tiles_u8.bin", KV_HEADS
    )
    capacity = args.past_length + 1
    cache_indices = np.arange(args.past_length) % PHYSICAL_M
    k_cache_initial = np.ascontiguousarray(
        k_rows[cache_indices].transpose(1, 0, 2)
    )
    v_cache_initial = np.ascontiguousarray(
        v_rows[cache_indices].transpose(1, 0, 2)
    )
    k_cache = np.empty((KV_HEADS, capacity, HEAD_DIM), dtype=np.uint8)
    v_cache = np.empty_like(k_cache)
    k_cache[:, : args.past_length] = k_cache_initial
    v_cache[:, : args.past_length] = v_cache_initial
    k_cache[:, args.past_length] = k_rows[0]
    v_cache[:, args.past_length] = v_rows[0]

    initial_k_file = np.zeros_like(k_cache)
    initial_v_file = np.zeros_like(v_cache)
    initial_k_file[:, : args.past_length] = k_cache_initial
    initial_v_file[:, : args.past_length] = v_cache_initial

    config_bytes = (source / "attention_config_all_groups.bin").read_bytes()
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError("unexpected Attention config size")
    attention_output = np.empty((HEADS, HEAD_DIM), dtype=np.uint8)
    for group in range(KV_HEADS):
        fields = CONFIG.unpack_from(config_bytes, group * CONFIG.size)
        (
            abi,
            config_group,
            fraction_bits,
            division_mode,
            q_zero_point,
            k_zero_point,
            v_zero_point,
            _probability_zero_point,
            output_zero_point,
            v_numerator,
            v_denominator,
            score_shift,
            score_multiplier,
            av_shift,
            av_multiplier,
        ) = fields
        if abi != 1 or config_group != group or division_mode not in DIVISION_NAMES:
            raise ValueError(f"invalid Attention config for group {group}")
        first_head = group * Q_HEADS_PER_GROUP
        q_centered = (
            q_rows[0, first_head : first_head + Q_HEADS_PER_GROUP]
            .astype(np.int32)
            - q_zero_point
        )
        k_centered = np.clip(
            k_cache[group].astype(np.int32) - k_zero_point, -128, 127
        ).astype(np.int8)
        score_accumulator = np.matmul(q_centered, k_centered.astype(np.int32).T)
        score, _ = centered_hmx_requant(
            score_accumulator, score_multiplier, score_shift, HMX_CENTER
        )
        probability = log2_softmax_decode(
            score, fraction_bits, DIVISION_NAMES[division_mode]
        )
        v_centered = v_cache[group].astype(np.int32) - v_zero_point
        v_signed = recenter_v(v_centered, v_numerator, v_denominator)
        av_accumulator = np.matmul(
            probability.astype(np.int32), v_signed.astype(np.int32)
        )
        group_output, _ = centered_hmx_requant(
            av_accumulator, av_multiplier, av_shift, output_zero_point
        )
        attention_output[first_head : first_head + Q_HEADS_PER_GROUP] = group_output

    weights = {
        name: unpack_w4_weight(reference_source, name, *shape)
        for name, shape in PROJECTION_SHAPES.items()
    }
    hidden_encoded = torch.from_numpy(padded_input[:1].reshape(1, 1, HIDDEN))
    hidden = dequantize_u8(hidden_encoded, qparams["block_input"]).to(torch.float16)
    attention_encoded = torch.from_numpy(
        attention_output.reshape(1, 1, HIDDEN)
    )
    attention_half = dequantize_u8(
        attention_encoded, qparams["attention_concat"]
    ).to(torch.float16)
    post_weight = torch.from_numpy(
        np.fromfile(reference_source / "post_norm_weight_f16.bin", dtype="<f2").copy()
    )

    def boundary(name: str, value: torch.Tensor) -> torch.Tensor:
        return dequantize_u8(quantize_u8(value, qparams[name]), qparams[name]).to(
            torch.float16
        )

    projected = boundary("attention_projection", linear_half(attention_half, weights["o"]))
    post_attention_residual = boundary(
        "post_attention_residual", hidden + projected
    )
    post_normalized = boundary(
        "post_attention_norm", rms_norm(post_attention_residual, post_weight)
    )
    gate = boundary("gate", linear_half(post_normalized, weights["gate"]))
    up = boundary("up", linear_half(post_normalized, weights["up"]))
    middle = boundary("middle", F.silu(gate.float()) * up.float())
    down = boundary("down", linear_half(middle, weights["down"]))
    output_row = quantize_u8(
        post_attention_residual + down, qparams["block_output"]
    ).cpu().numpy().reshape(HIDDEN)
    output_reference = np.full(
        (PHYSICAL_M, HIDDEN),
        int(qparams["block_output"]["zero_point"]),
        dtype=np.uint8,
    )
    output_reference[0] = output_row

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="decode-package-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        padded_input.tofile(staging / "reference_w4u8_block_input_u8.bin")
        output_reference.tofile(
            staging / "reference_w4u8_integer_attention_block_output_u8.bin"
        )
        attention_output.tofile(
            staging / "reference_exp0147_attention_heads_u8.bin"
        )
        initial_k_file.tofile(staging / "kv_cache_k_u8.bin")
        initial_v_file.tofile(staging / "kv_cache_v_u8.bin")
        k_cache.tofile(staging / "reference_kv_cache_k_u8.bin")
        v_cache.tofile(staging / "reference_kv_cache_v_u8.bin")

        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "experiment": "EXP-0147",
                "execution_unit": "qwen3_layer14_decode_kv_scan",
                "source": str(source),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "reference_source": str(reference_source),
                "reference_manifest_sha256": sha256(
                    reference_source / "manifest.json"
                ),
                "actual_audit_source": str(actual_audit),
                "shape_scan": {
                    "mode": "decode",
                    "logical_m": 1,
                    "physical_m": PHYSICAL_M,
                    "initial_kv_length": args.past_length,
                    "kv_cache_capacity": capacity,
                    "cache_layout": "head_major_[8,capacity,128]",
                    "cache_fixture": "accepted_M64_device_boundary_repeated",
                    "rope_position": 0,
                },
            }
        )
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if publish.exists():
            raise FileExistsError(publish)
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "past_length": args.past_length,
                    "capacity": capacity,
                    "output_row_sha256": hashlib.sha256(output_row.tobytes()).hexdigest(),
                    "k_cache_sha256": sha256(output / "reference_kv_cache_k_u8.bin"),
                    "v_cache_sha256": sha256(output / "reference_kv_cache_v_u8.bin"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()
