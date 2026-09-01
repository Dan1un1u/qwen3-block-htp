#!/usr/bin/env python3
"""Byte-exact 28-layer EXP-0152 W4U8 prefill reference and audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_exp0042_attention import CONFIG, log2_softmax
from reference_w4u8_hmx import (
    HmxU8Converter,
    exact_attention_prefill,
    exact_qk_norm_rope_u8,
    exact_residual_add_u8,
    exact_rms_norm_u8,
    load_qparams_bin,
    project_w4u8,
)


LAYERS = 28
M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
CACHE_CAPACITY = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path)
    return parser.parse_args()


def read_u8(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype=np.uint8)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"{path}: got {value.size} bytes, expected {expected}")
    return value.reshape(shape)


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    if actual.shape != expected.shape:
        return {
            "shape_mismatch": True,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "mismatches": max(actual.size, expected.size),
            "max_lsb": 255,
        }
    delta = actual.astype(np.int16) - expected.astype(np.int16)
    mismatch = np.flatnonzero(delta)
    result: dict[str, object] = {
        "elements": int(actual.size),
        "mismatches": int(mismatch.size),
        "max_lsb": int(np.max(np.abs(delta))) if delta.size else 0,
    }
    if mismatch.size:
        flat = int(mismatch[0])
        result["first_index"] = [
            int(index) for index in np.unravel_index(flat, actual.shape)
        ]
        result["first_actual"] = int(actual.flat[flat])
        result["first_expected"] = int(expected.flat[flat])
    return result


def run_layer_prefill(
    input_u8: np.ndarray,
    package: Path,
    cosine: np.ndarray,
    sine: np.ndarray,
    converter: HmxU8Converter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qparams = load_qparams_bin(package / "qparams_u8.bin")
    input_norm = exact_rms_norm_u8(
        input_u8,
        qparams["block_input"],
        np.fromfile(package / "input_norm_weight_f16.bin", dtype="<f2"),
        qparams["input_norm"],
    )
    q_projection = project_w4u8(
        input_norm, package, "q", HIDDEN, HIDDEN,
        qparams["input_norm"], qparams["q_projection"], converter,
    )
    k_projection = project_w4u8(
        input_norm, package, "k", KV_HEADS * HEAD_DIM, HIDDEN,
        qparams["input_norm"], qparams["k_projection"], converter,
    )
    v_projection = project_w4u8(
        input_norm, package, "v", KV_HEADS * HEAD_DIM, HIDDEN,
        qparams["input_norm"], qparams["v"], converter,
    )
    q_rope = exact_qk_norm_rope_u8(
        q_projection, HEADS,
        qparams["q_projection"], qparams["q_rope"],
        np.fromfile(package / "q_norm_weight_f16.bin", dtype="<f2"),
        cosine, sine,
    ).reshape(M, HEADS, HEAD_DIM)
    k_rope = exact_qk_norm_rope_u8(
        k_projection, KV_HEADS,
        qparams["k_projection"], qparams["k_rope"],
        np.fromfile(package / "k_norm_weight_f16.bin", dtype="<f2"),
        cosine, sine,
    ).reshape(M, KV_HEADS, HEAD_DIM)
    v_heads = v_projection.reshape(M, KV_HEADS, HEAD_DIM)

    config_bytes = (package / "attention_config_all_groups.bin").read_bytes()
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError(f"invalid Attention config size in {package}")
    configs = [
        CONFIG.unpack_from(config_bytes, group * CONFIG.size)
        for group in range(KV_HEADS)
    ]
    attention, _, _ = exact_attention_prefill(
        q_rope, k_rope, v_heads, configs, converter, log2_softmax
    )
    o_projection = project_w4u8(
        attention.reshape(M, HIDDEN), package, "o", HIDDEN, HIDDEN,
        qparams["attention_concat"],
        qparams["attention_projection"], converter,
    )
    post_residual = exact_residual_add_u8(
        input_u8, qparams["block_input"],
        o_projection, qparams["attention_projection"],
        qparams["post_attention_residual"],
    )
    post_norm = exact_rms_norm_u8(
        post_residual,
        qparams["post_attention_residual"],
        np.fromfile(package / "post_norm_weight_f16.bin", dtype="<f2"),
        qparams["post_attention_norm"],
    )
    gate = project_w4u8(
        post_norm, package, "gate", INTERMEDIATE, HIDDEN,
        qparams["post_attention_norm"], qparams["gate"], converter,
    )
    up = project_w4u8(
        post_norm, package, "up", INTERMEDIATE, HIDDEN,
        qparams["post_attention_norm"], qparams["up"], converter,
    )
    lut = np.fromfile(
        package / "silu_up_lut_u16.bin", dtype="<u2"
    ).reshape(256, 256)
    middle = np.clip(lut[gate, up], 0, 255).astype(np.uint8)
    down = project_w4u8(
        middle, package, "down", HIDDEN, INTERMEDIATE,
        qparams["middle"], qparams["down"], converter,
    )
    output = exact_residual_add_u8(
        post_residual, qparams["post_attention_residual"],
        down, qparams["down"], qparams["block_output"],
    )
    return output, k_rope, v_heads


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    capture = args.capture.resolve()
    converter = HmxU8Converter(args.converter)
    cosine = np.fromfile(package / "rope_cos_f16.bin", dtype="<f2")
    sine = np.fromfile(package / "rope_sin_f16.bin", dtype="<f2")
    hidden = read_u8(
        package / "reference_w4u8_block_input_u8.bin", (M, HIDDEN)
    )
    layer_records: list[dict[str, object]] = []
    if args.reference_cache_dir:
        args.reference_cache_dir.mkdir(parents=True, exist_ok=True)

    for layer in range(LAYERS):
        layer_package = package / f"layer{layer}"
        hidden, k_rope, v_heads = run_layer_prefill(
            hidden, layer_package, cosine, sine, converter
        )
        layer_records.append({
            "layer": layer,
            "output_sha256": sha256_array(hidden),
            "k_sha256": sha256_array(k_rope),
            "v_sha256": sha256_array(v_heads),
        })
        if args.reference_cache_dir:
            for name, values in (("k", k_rope), ("v", v_heads)):
                cache = read_u8(
                    layer_package / f"kv_cache_{name}_u8.bin",
                    (KV_HEADS, CACHE_CAPACITY, HEAD_DIM),
                ).copy()
                cache[:, :M, :] = values.transpose(1, 0, 2)
                cache.tofile(
                    args.reference_cache_dir /
                    f"reference_layer{layer}_{name}_prefill_cache_u8.bin"
                )

    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    hidden.tofile(args.reference_output)
    actual = read_u8(
        capture / "actual_replay_output_00_u8.bin", (M, HIDDEN)
    )
    comparison = difference(actual, hidden)
    passed = comparison["mismatches"] == 0
    report = {
        "experiment": "EXP-0152",
        "contract": "byte_exact_W4U8_28_layer_prefill",
        "layers": LAYERS,
        "passed": passed,
        "comparison": comparison,
        "actual_sha256": sha256_array(actual),
        "reference_sha256": sha256_array(hidden),
        "layer_records": layer_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.output),
        "passed": passed,
        "comparison": comparison,
        "actual_sha256": report["actual_sha256"],
        "reference_sha256": report["reference_sha256"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
