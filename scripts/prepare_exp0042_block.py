#!/usr/bin/env python3
"""Publish the EXP-0042 full-block package with integer GQA Attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export_exp0022_block import (
    dequantize_u8,
    linear_half,
    quantize_u8,
    rms_norm,
)
from prepare_exp0042_attention import (
    ABI_VERSION,
    CONFIG,
    DIVISION_MODES,
    EXP_FRAC_BITS,
    HEAD_DIM,
    HEADS,
    HMX_CENTER,
    KV_HEADS,
    M,
    Q_HEADS_PER_GROUP,
    centered_hmx_requant,
    choose_carrier,
    log2_softmax,
)


HIDDEN = 2048
INTERMEDIATE = 6144
PROJECTION_SHAPES = {
    "o": (HIDDEN, HIDDEN),
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/"
            "block_package_layer14_m64_stage_b"
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
    parser.add_argument("--fraction-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--division", choices=tuple(DIVISION_MODES), default="sole"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/"
            "block_package_layer14_m64_integer_attention_parallel"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0042"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_w4_weight(root: Path, name: str, n: int, k: int) -> torch.Tensor:
    packed = np.fromfile(
        root / f"{name}_weight_w4_hmx.bin", dtype=np.uint8
    ).reshape(n // 32, k // 32, 512)
    flat = np.empty((n // 32, k // 32, 1024), dtype=np.uint8)
    flat[..., 0::2] = packed & 0xF
    flat[..., 1::2] = packed >> 4
    signed = flat.astype(np.int8)
    signed[signed >= 8] -= 16
    physical = signed.reshape(n // 32, k // 32, 8, 32, 4)
    logical = (
        physical.transpose(0, 1, 2, 4, 3)
        .reshape(n // 32, k // 32, 32, 32)
        .transpose(0, 3, 1, 2)
        .reshape(n, k)
    )
    scales = np.fromfile(
        root / f"{name}_weight_w4_scale_f32.bin", dtype="<f4"
    )
    if scales.shape != (n,):
        raise ValueError(f"{name} scale shape {scales.shape}, expected {(n,)}")
    dequantized = (
        logical.astype(np.float32) * scales[:, None]
    ).astype(np.float16)
    return torch.from_numpy(dequantized)


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = left.astype(np.float64).reshape(-1)
    b = right.astype(np.float64).reshape(-1)
    difference = a - b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(a, b) / denominator)
        if denominator > 0.0
        else 0.0,
    }


def tile_last_dimension(value: np.ndarray) -> np.ndarray:
    """Serialize [head, row, channel] as native 64x32 HMX tiles."""
    if value.ndim != 3 or value.shape[1] != M or value.shape[2] % 32:
        raise ValueError(f"cannot tile shape {value.shape}")
    return np.ascontiguousarray(
        value.reshape(value.shape[0], M, value.shape[2] // 32, 32)
        .transpose(0, 2, 1, 3)
    )


def main() -> None:
    args = parse_args()
    source = args.block_source.resolve()
    reference = args.reference_source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing package: {output}")
    for path in (source / "manifest.json", reference / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_manifest = json.loads(
        (reference / "manifest.json").read_text(encoding="utf-8")
    )
    qparams = reference_manifest["u8_qparams"]
    q_all = np.fromfile(
        reference / "reference_w4u8_q_rope_u8.bin", dtype=np.uint8
    ).reshape(M, HEADS, HEAD_DIM)
    k_all = np.fromfile(
        reference / "reference_w4u8_k_rope_u8.bin", dtype=np.uint8
    ).reshape(M, KV_HEADS, HEAD_DIM)
    v_all = np.fromfile(
        reference / "reference_w4u8_v_u8.bin", dtype=np.uint8
    ).reshape(M, KV_HEADS, HEAD_DIM)
    q_qp = qparams["q_rope"]
    k_qp = qparams["k_rope"]
    v_qp = qparams["v"]
    p_qp = qparams["attention_probability"]
    y_qp = qparams["attention_concat"]
    attention_output = np.empty((HEADS, M, HEAD_DIM), dtype=np.uint8)
    score_reference = np.empty((HEADS, M, M), dtype=np.uint8)
    probability_reference = np.empty((HEADS, M, M), dtype=np.uint8)
    config_records: list[bytes] = []
    group_manifests: list[dict[str, object]] = []

    score_step = math.log(2.0) / (1 << args.fraction_bits)
    score_ratio = (
        float(q_qp["scale"])
        * float(k_qp["scale"])
        / math.sqrt(HEAD_DIM)
        / score_step
    )
    score_multiplier, score_shift, score_effective = choose_carrier(
        score_ratio
    )
    for group in range(KV_HEADS):
        first_q_head = group * Q_HEADS_PER_GROUP
        q = np.ascontiguousarray(
            q_all[:, first_q_head : first_q_head + Q_HEADS_PER_GROUP, :]
            .transpose(1, 0, 2)
        )
        k = np.ascontiguousarray(k_all[:, group, :])
        v = np.ascontiguousarray(v_all[:, group, :])
        q_centered = q.astype(np.int32) - int(q_qp["zero_point"])
        k_centered = k.astype(np.int16) - int(k_qp["zero_point"])
        if int(k_centered.min()) < -128 or int(k_centered.max()) > 127:
            raise ValueError(f"group {group}: K does not fit S8")
        v_centered = v.astype(np.int32) - int(v_qp["zero_point"])
        v_denominator = max(
            1, int(np.max(np.abs(v_centered.astype(np.int64))))
        )
        v_numerator = 127
        v_signed = np.where(
            v_centered >= 0,
            (v_centered * v_numerator + v_denominator // 2)
            // v_denominator,
            -((-v_centered * v_numerator + v_denominator // 2)
              // v_denominator),
        ).astype(np.int8)
        v_s8_scale = (
            float(v_qp["scale"]) * v_denominator / v_numerator
        )
        av_ratio = (
            float(p_qp["scale"]) * v_s8_scale / float(y_qp["scale"])
        )
        av_multiplier, av_shift, av_effective = choose_carrier(av_ratio)

        score_accumulator = np.matmul(
            q_centered, k_centered.astype(np.int32).T
        )
        score, score_saturations = centered_hmx_requant(
            score_accumulator,
            score_multiplier,
            score_shift,
            HMX_CENTER,
        )
        probability, _ = log2_softmax(
            score, args.fraction_bits, args.division
        )
        av_accumulator = np.matmul(
            probability.astype(np.int32), v_signed.astype(np.int32)
        )
        group_output, av_saturations = centered_hmx_requant(
            av_accumulator,
            av_multiplier,
            av_shift,
            int(y_qp["zero_point"]),
        )
        attention_output[
            first_q_head : first_q_head + Q_HEADS_PER_GROUP
        ] = group_output
        score_reference[
            first_q_head : first_q_head + Q_HEADS_PER_GROUP
        ] = score
        probability_reference[
            first_q_head : first_q_head + Q_HEADS_PER_GROUP
        ] = probability
        config_records.append(
            CONFIG.pack(
                ABI_VERSION,
                group,
                args.fraction_bits,
                DIVISION_MODES[args.division],
                int(q_qp["zero_point"]),
                int(k_qp["zero_point"]),
                int(v_qp["zero_point"]),
                int(p_qp["zero_point"]),
                int(y_qp["zero_point"]),
                v_numerator,
                v_denominator,
                score_shift,
                score_multiplier,
                av_shift,
                av_multiplier,
            )
        )
        group_manifests.append(
            {
                "group": group,
                "q_heads": [first_q_head, first_q_head + 1],
                "v_recenter": [v_numerator, v_denominator],
                "score_carrier": {
                    "target": score_ratio,
                    "multiplier": score_multiplier,
                    "shift": score_shift,
                    "effective": score_effective,
                    "saturations": score_saturations,
                },
                "av_carrier": {
                    "target": av_ratio,
                    "multiplier": av_multiplier,
                    "shift": av_shift,
                    "effective": av_effective,
                    "saturations": av_saturations,
                },
            }
        )

    weights = {
        name: unpack_w4_weight(reference, name, *shape)
        for name, shape in PROJECTION_SHAPES.items()
    }
    input_encoded = torch.from_numpy(
        np.fromfile(
            reference / "reference_w4u8_block_input_u8.bin",
            dtype=np.uint8,
        ).reshape(1, M, HIDDEN)
    )
    hidden = dequantize_u8(input_encoded, qparams["block_input"]).to(
        torch.float16
    )
    attention_encoded = torch.from_numpy(
        np.ascontiguousarray(
            attention_output.transpose(1, 0, 2).reshape(1, M, HIDDEN)
        )
    )
    attention_half = dequantize_u8(
        attention_encoded, qparams["attention_concat"]
    ).to(torch.float16)
    post_weight = torch.from_numpy(
        np.fromfile(
            reference / "post_norm_weight_f16.bin", dtype="<f2"
        ).copy()
    )

    def boundary(name: str, value: torch.Tensor) -> torch.Tensor:
        return dequantize_u8(
            quantize_u8(value, qparams[name]), qparams[name]
        ).to(torch.float16)

    projected = boundary(
        "attention_projection", linear_half(attention_half, weights["o"])
    )
    post_attention_residual = boundary(
        "post_attention_residual", hidden + projected
    )
    post_normalized = boundary(
        "post_attention_norm",
        rms_norm(post_attention_residual, post_weight),
    )
    gate = boundary("gate", linear_half(post_normalized, weights["gate"]))
    up = boundary("up", linear_half(post_normalized, weights["up"]))
    middle = boundary("middle", F.silu(gate.float()) * up.float())
    down = boundary("down", linear_half(middle, weights["down"]))
    output_encoded = quantize_u8(
        post_attention_residual + down, qparams["block_output"]
    ).cpu().numpy().astype(np.uint8, copy=False)

    old_output = np.fromfile(
        source / "reference_w4u8_block_output_u8.bin", dtype=np.uint8
    ).reshape(output_encoded.shape)
    w4a16_output = np.fromfile(
        source / "reference_w4f16_block_output_f16.bin", dtype="<f2"
    ).astype(np.float32)
    output_real = (
        output_encoded.astype(np.float32)
        - int(qparams["block_output"]["zero_point"])
    ) * float(qparams["block_output"]["scale"])

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="block-package-", dir=args.staging_root)
    )
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        (staging / "attention_config_all_groups.bin").write_bytes(
            b"".join(config_records)
        )
        np.ascontiguousarray(attention_output).tofile(
            staging / "reference_w4u8_integer_attention_heads_u8.bin"
        )
        np.ascontiguousarray(score_reference).tofile(
            staging / "reference_w4u8_integer_attention_scores_u8.bin"
        )
        np.ascontiguousarray(probability_reference).tofile(
            staging / "reference_w4u8_integer_attention_probability_u8.bin"
        )
        tile_last_dimension(score_reference).tofile(
            staging
            / "reference_w4u8_integer_attention_score_tiles_u8.bin"
        )
        tile_last_dimension(probability_reference).tofile(
            staging
            / "reference_w4u8_integer_attention_probability_tiles_u8.bin"
        )
        tile_last_dimension(attention_output).tofile(
            staging
            / "reference_w4u8_integer_attention_av_tiles_u8.bin"
        )
        np.ascontiguousarray(output_encoded).tofile(
            staging / "reference_w4u8_integer_attention_block_output_u8.bin"
        )
        files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "experiment": "EXP-0042",
            "execution_unit": "qwen3_layer14_complete_block_m64",
            "source": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "reference_source": str(reference),
            "reference_manifest_sha256": sha256(
                reference / "manifest.json"
            ),
            "candidate_contract": {
                "attention": "integer QK -> log2 U8 Softmax -> integer AV",
                "qkv_input_layout": "native HMX U8 tiles",
                "attention_output_layout": "native HMX U8 tiles consumed by O",
                "intermediate_ddr_allowed": False,
                "vtcm_request_bytes": 8 * 1024 * 1024,
                "fraction_bits": args.fraction_bits,
                "division": args.division,
            },
            "groups": group_manifests,
            "diagnostic_metrics": {
                "candidate_vs_retained_w4u8_u8_codes": metrics(
                    output_encoded, old_output
                ),
                "candidate_vs_w4a16_real": metrics(
                    output_real, w4a16_output
                ),
            },
            "files": files,
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
                    "config_bytes": CONFIG.size * KV_HEADS,
                    "output_sha256": sha256(
                        output
                        / "reference_w4u8_integer_attention_block_output_u8.bin"
                    ),
                    "diagnostic_metrics": manifest[
                        "diagnostic_metrics"
                    ],
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
