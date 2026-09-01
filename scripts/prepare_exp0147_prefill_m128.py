#!/usr/bin/env python3
"""Prepare the two-chunk W4U8 M=128 prefill cell for EXP-0147."""

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

from export_exp0022_block import dequantize_u8, linear_half, quantize_u8, rms_norm  # noqa: E402
from prepare_exp0042_attention import (  # noqa: E402
    CONFIG,
    EXP_FRAC_BITS,
    HMX_CENTER,
    centered_hmx_requant,
    divide_probability,
)
from prepare_exp0042_block import unpack_w4_weight  # noqa: E402
from prepare_exp0147_decode import (  # noqa: E402
    DIVISION_NAMES,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    PHYSICAL_M,
    PROJECTION_SHAPES,
    Q_HEADS_PER_GROUP,
    recenter_v,
    sha256,
    unpack_native_feature_tiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def log2_softmax_chunk(
    scores: np.ndarray, fraction_bits: int, division: str, past: int
) -> np.ndarray:
    probability = np.zeros_like(scores, dtype=np.uint8)
    rounding = 1 << (fraction_bits - 1)
    for head in range(scores.shape[0]):
        for row in range(PHYSICAL_M):
            valid_count = past + row + 1
            maximum = int(scores[head, row, :valid_count].max())
            delta = maximum - scores[head, row, :valid_count].astype(np.int32)
            exponents = np.minimum(15, (delta + rounding) >> fraction_bits)
            total = sum(1 << (EXP_FRAC_BITS - int(code)) for code in exponents)
            for token, exponent in enumerate(exponents):
                probability[head, row, token] = divide_probability(
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
    reference_manifest = json.loads(
        (reference_source / "manifest.json").read_text(encoding="utf-8")
    )
    qparams = reference_manifest["u8_qparams"]
    base_input = np.fromfile(
        source / "reference_w4u8_block_input_u8.bin", dtype=np.uint8
    ).reshape(PHYSICAL_M, HIDDEN)
    block_input = np.ascontiguousarray(np.concatenate((base_input, base_input), axis=0))
    q_rows = unpack_native_feature_tiles(actual_audit / "actual_q_tiles_u8.bin", HEADS)
    k_rows = unpack_native_feature_tiles(actual_audit / "actual_k_tiles_u8.bin", KV_HEADS)
    v_rows = unpack_native_feature_tiles(actual_audit / "actual_v_tiles_u8.bin", KV_HEADS)
    k_cache = np.ascontiguousarray(
        np.concatenate((k_rows, k_rows), axis=0).transpose(1, 0, 2)
    )
    v_cache = np.ascontiguousarray(
        np.concatenate((v_rows, v_rows), axis=0).transpose(1, 0, 2)
    )
    config_bytes = (source / "attention_config_all_groups.bin").read_bytes()
    attention_output = np.empty((HEADS, PHYSICAL_M, HEAD_DIM), dtype=np.uint8)
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
            q_rows[:, first_head : first_head + Q_HEADS_PER_GROUP]
            .transpose(1, 0, 2)
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
        probability = log2_softmax_chunk(
            score, fraction_bits, DIVISION_NAMES[division_mode], PHYSICAL_M
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
    hidden_encoded = torch.from_numpy(base_input.reshape(1, PHYSICAL_M, HIDDEN))
    hidden = dequantize_u8(hidden_encoded, qparams["block_input"]).to(torch.float16)
    attention_encoded = torch.from_numpy(
        np.ascontiguousarray(attention_output.transpose(1, 0, 2)).reshape(
            1, PHYSICAL_M, HIDDEN
        )
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
    post_residual = boundary("post_attention_residual", hidden + projected)
    post_norm = boundary("post_attention_norm", rms_norm(post_residual, post_weight))
    gate = boundary("gate", linear_half(post_norm, weights["gate"]))
    up = boundary("up", linear_half(post_norm, weights["up"]))
    middle = boundary("middle", F.silu(gate.float()) * up.float())
    down = boundary("down", linear_half(middle, weights["down"]))
    second_output = quantize_u8(
        post_residual + down, qparams["block_output"]
    ).cpu().numpy().reshape(PHYSICAL_M, HIDDEN)
    first_output = np.fromfile(
        source / "reference_w4u8_integer_attention_block_output_u8.bin",
        dtype=np.uint8,
    ).reshape(PHYSICAL_M, HIDDEN)
    output_reference = np.ascontiguousarray(
        np.concatenate((first_output, second_output), axis=0)
    )

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="prefill128-package-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        block_input.tofile(staging / "reference_w4u8_block_input_u8.bin")
        output_reference.tofile(
            staging / "reference_w4u8_integer_attention_block_output_u8.bin"
        )
        np.concatenate((
            np.fromfile(source / "rope_cos_f16.bin", dtype=np.uint8),
            np.fromfile(source / "rope_cos_f16.bin", dtype=np.uint8),
        )).tofile(staging / "rope_cos_f16.bin")
        np.concatenate((
            np.fromfile(source / "rope_sin_f16.bin", dtype=np.uint8),
            np.fromfile(source / "rope_sin_f16.bin", dtype=np.uint8),
        )).tofile(staging / "rope_sin_f16.bin")
        np.zeros_like(k_cache).tofile(staging / "kv_cache_k_u8.bin")
        np.zeros_like(v_cache).tofile(staging / "kv_cache_v_u8.bin")
        k_cache.tofile(staging / "reference_kv_cache_k_u8.bin")
        v_cache.tofile(staging / "reference_kv_cache_v_u8.bin")
        attention_output.tofile(staging / "reference_exp0147_chunk1_attention_heads_u8.bin")

        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "experiment": "EXP-0147",
                "execution_unit": "qwen3_layer14_prefill_shape_scan",
                "source": str(source),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "reference_source": str(reference_source),
                "reference_manifest_sha256": sha256(reference_source / "manifest.json"),
                "actual_audit_source": str(actual_audit),
                "shape_scan": {
                    "mode": "prefill",
                    "logical_m": 128,
                    "physical_chunks": 2,
                    "chunk_fixture": "accepted_M64_fixture_repeated",
                    "initial_kv_length": 0,
                    "kv_cache_capacity": 128,
                    "cache_layout": "head_major_[8,128,128]",
                },
            }
        )
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if publish.exists():
            raise FileExistsError(publish)
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(json.dumps({"output": str(output), "logical_m": 128}, indent=2))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()
