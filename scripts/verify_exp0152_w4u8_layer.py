#!/usr/bin/env python3
"""Byte-exact audit of one EXP-0152 W4U8 layer from device boundaries."""

from __future__ import annotations

import argparse
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
    pack_u8_hmx_activation,
    project_w4u8,
)


M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rope-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_u8(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype=np.uint8)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"{path}: got {value.size} bytes, expected {expected}")
    return value.reshape(shape)


def unpack_feature(path: Path, heads: int) -> np.ndarray:
    physical = read_u8(path, (heads, HEAD_DIM // 32, M, 32))
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(M, heads, HEAD_DIM)
    )


def unpack_score(path: Path) -> np.ndarray:
    physical = read_u8(path, (HEADS, M // 32, M, 32))
    return np.ascontiguousarray(
        physical.transpose(0, 2, 1, 3).reshape(HEADS, M, M)
    )


def unpack_av(path: Path) -> np.ndarray:
    physical = read_u8(path, (HEADS, HEAD_DIM // 32, M, 32))
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(M, HEADS, HEAD_DIM)
    )


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
            int(value) for value in np.unravel_index(flat, actual.shape)
        ]
        result["first_actual"] = int(actual.flat[flat])
        result["first_expected"] = int(expected.flat[flat])
    return result


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    dump = args.dump.resolve()
    rope_root = (
        args.rope_root.resolve() if args.rope_root else package.parent
    )
    input_path = (
        args.input.resolve() if args.input else
        package.parent / "reference_w4u8_block_input_u8.bin"
    )
    qparams = load_qparams_bin(package / "qparams_u8.bin")
    converter = HmxU8Converter(args.converter)
    input_u8 = read_u8(input_path, (M, HIDDEN))
    cosine = np.fromfile(rope_root / "rope_cos_f16.bin", dtype="<f2")
    sine = np.fromfile(rope_root / "rope_sin_f16.bin", dtype="<f2")

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
        q_projection, HEADS, qparams["q_projection"], qparams["q_rope"],
        np.fromfile(package / "q_norm_weight_f16.bin", dtype="<f2"),
        cosine, sine,
    ).reshape(M, HEADS, HEAD_DIM)
    k_rope = exact_qk_norm_rope_u8(
        k_projection, KV_HEADS, qparams["k_projection"], qparams["k_rope"],
        np.fromfile(package / "k_norm_weight_f16.bin", dtype="<f2"),
        cosine, sine,
    ).reshape(M, KV_HEADS, HEAD_DIM)
    v_heads = v_projection.reshape(M, KV_HEADS, HEAD_DIM)

    config_bytes = (package / "attention_config_all_groups.bin").read_bytes()
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError("invalid Attention config size")
    configs = [
        CONFIG.unpack_from(config_bytes, group * CONFIG.size)
        for group in range(KV_HEADS)
    ]
    attention, score, probability = exact_attention_prefill(
        q_rope, k_rope, v_heads, configs, converter, log2_softmax
    )
    attention_flat = attention.reshape(M, HIDDEN)
    o_projection = project_w4u8(
        attention_flat, package, "o", HIDDEN, HIDDEN,
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
    final = exact_residual_add_u8(
        post_residual, qparams["post_attention_residual"],
        down, qparams["down"], qparams["block_output"],
    )

    expected = {
        "input_norm": pack_u8_hmx_activation(input_norm),
        "q": q_rope,
        "k": k_rope,
        "v": v_heads,
        "score": score,
        "probability": probability,
        "av": attention,
        "o": pack_u8_hmx_activation(o_projection),
        "post_residual": post_residual,
        "post_norm": pack_u8_hmx_activation(post_norm),
        "middle": pack_u8_hmx_activation(middle),
        "down": pack_u8_hmx_activation(down),
        "final": final,
    }
    actual = {
        "input_norm": np.fromfile(
            dump / "actual_input_norm_tiles_u8.bin", dtype=np.uint8
        ),
        "q": unpack_feature(dump / "actual_q_tiles_u8.bin", HEADS),
        "k": unpack_feature(dump / "actual_k_tiles_u8.bin", KV_HEADS),
        "v": unpack_feature(dump / "actual_v_tiles_u8.bin", KV_HEADS),
        "score": unpack_score(dump / "actual_score_tiles_u8.bin"),
        "probability": unpack_score(
            dump / "actual_probability_tiles_u8.bin"
        ),
        "av": unpack_av(dump / "actual_av_tiles_u8.bin"),
        "o": np.fromfile(dump / "actual_o_tiles_u8.bin", dtype=np.uint8),
        "post_residual": read_u8(
            dump / "actual_post_residual_u8.bin", (M, HIDDEN)
        ),
        "post_norm": np.fromfile(
            dump / "actual_post_norm_tiles_u8.bin", dtype=np.uint8
        ),
        "middle": np.fromfile(
            dump / "actual_middle_tiles_u8.bin", dtype=np.uint8
        ),
        "down": np.fromfile(
            dump / "actual_down_tiles_u8.bin", dtype=np.uint8
        ),
        "final": read_u8(dump / "actual_final_u8.bin", (M, HIDDEN)),
    }
    comparisons = {
        name: difference(actual[name], expected[name]) for name in expected
    }
    if dump.joinpath("actual_output_u8.bin").is_file():
        comparisons["host_output"] = difference(
            read_u8(dump / "actual_output_u8.bin", (M, HIDDEN)), final
        )
    passed = all(item["mismatches"] == 0 for item in comparisons.values())
    report = {
        "experiment": "EXP-0152",
        "contract": "byte_exact_W4U8_layer_device_vs_independent_reference",
        "passed": passed,
        "comparisons": comparisons,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
