#!/usr/bin/env python3
"""Byte-exact EXP-0152 prefill-to-decode replay and KV-cache audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from prepare_exp0042_attention import CONFIG, divide_probability
from reference_w4u8_hmx import (
    HmxU8Converter,
    exact_attention_dynamic,
    exact_qk_norm_rope_u8,
    exact_residual_add_u8,
    exact_rms_norm_u8,
    load_qparams_bin,
    project_w4u8,
)
from verify_exp0152_w4u8_prefill import (
    CACHE_CAPACITY,
    HEADS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    LAYERS,
    M,
    difference,
    read_u8,
    run_layer_prefill,
    sha256_array,
)


DECODE_STEPS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    return parser.parse_args()


def load_attention_configs(package: Path) -> list[tuple[int, ...]]:
    payload = (package / "attention_config_all_groups.bin").read_bytes()
    if len(payload) != KV_HEADS * CONFIG.size:
        raise ValueError(f"invalid Attention config size in {package}")
    return [
        CONFIG.unpack_from(payload, group * CONFIG.size)
        for group in range(KV_HEADS)
    ]


def run_layer_decode(
    input_u8: np.ndarray,
    package: Path,
    cosine: np.ndarray,
    sine: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    past_tokens: int,
    converter: HmxU8Converter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute one logical decode row and append this layer's cache."""
    if input_u8.shape != (1, HIDDEN):
        raise ValueError(f"invalid decode input {input_u8.shape}")
    if (
        k_cache.ndim != 3
        or k_cache.shape[0] != KV_HEADS
        or k_cache.shape[2] != HEAD_DIM
        or v_cache.shape != k_cache.shape
        or past_tokens >= k_cache.shape[1]
    ):
        raise ValueError(
            f"invalid replay cache k={k_cache.shape} v={v_cache.shape}"
        )
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
    ).reshape(1, HEADS, HEAD_DIM)
    k_rope = exact_qk_norm_rope_u8(
        k_projection, KV_HEADS,
        qparams["k_projection"], qparams["k_rope"],
        np.fromfile(package / "k_norm_weight_f16.bin", dtype="<f2"),
        cosine, sine,
    ).reshape(1, KV_HEADS, HEAD_DIM)
    v_heads = v_projection.reshape(1, KV_HEADS, HEAD_DIM)

    k_cache[:, past_tokens, :] = k_rope[0]
    v_cache[:, past_tokens, :] = v_heads[0]
    valid_tokens = past_tokens + 1
    attention, _, _ = exact_attention_dynamic(
        q_rope,
        k_cache[:, :valid_tokens, :],
        v_cache[:, :valid_tokens, :],
        past_tokens,
        load_attention_configs(package),
        converter,
        divide_probability,
    )
    o_projection = project_w4u8(
        attention.reshape(1, HIDDEN), package, "o", HIDDEN, HIDDEN,
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
    output = args.output.resolve()
    reference_dir = args.reference_dir.resolve()
    if reference_dir.exists():
        raise FileExistsError(f"refusing to overwrite {reference_dir}")
    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{reference_dir.name}.staging-",
        dir=reference_dir.parent,
    ))
    converter = HmxU8Converter(args.converter)
    layer_packages = [package / f"layer{layer}" for layer in range(LAYERS)]
    k_caches = [
        read_u8(
            layer_package / "kv_cache_k_u8.bin",
            (KV_HEADS, CACHE_CAPACITY, HEAD_DIM),
        ).copy()
        for layer_package in layer_packages
    ]
    v_caches = [
        read_u8(
            layer_package / "kv_cache_v_u8.bin",
            (KV_HEADS, CACHE_CAPACITY, HEAD_DIM),
        ).copy()
        for layer_package in layer_packages
    ]
    records: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    try:
        hidden = read_u8(
            package / "reference_w4u8_block_input_u8.bin", (M, HIDDEN)
        )
        cosine = np.fromfile(package / "rope_cos_f16.bin", dtype="<f2")
        sine = np.fromfile(package / "rope_sin_f16.bin", dtype="<f2")
        prefill_layers: list[dict[str, object]] = []
        for layer, layer_package in enumerate(layer_packages):
            hidden, k_rope, v_heads = run_layer_prefill(
                hidden, layer_package, cosine, sine, converter
            )
            k_caches[layer][:, :M, :] = k_rope.transpose(1, 0, 2)
            v_caches[layer][:, :M, :] = v_heads.transpose(1, 0, 2)
            prefill_layers.append({
                "layer": layer,
                "output_sha256": sha256_array(hidden),
                "k_append_sha256": sha256_array(k_rope),
                "v_append_sha256": sha256_array(v_heads),
            })
            print(f"prefill layer {layer + 1}/{LAYERS}", flush=True)
        actual_prefill = read_u8(
            capture / "actual_replay_output_00_u8.bin", (M, HIDDEN)
        )
        prefill_comparison = difference(actual_prefill, hidden)
        hidden.tofile(staging / "reference_replay_output_00_u8.bin")
        records.append({
            "step": 0,
            "mode": "prefill",
            "logical_rows": M,
            "comparison": prefill_comparison,
            "actual_sha256": sha256_array(actual_prefill),
            "reference_sha256": sha256_array(hidden),
            "layers": prefill_layers,
        })
        comparisons.append(prefill_comparison)

        for decode_index in range(DECODE_STEPS):
            hidden = read_u8(
                package / f"replay_decode_input_{decode_index:02d}_u8.bin",
                (M, HIDDEN),
            )[:1].copy()
            cosine = np.fromfile(
                package / f"replay_decode_rope_cos_{decode_index:02d}_f16.bin",
                dtype="<f2",
            )
            sine = np.fromfile(
                package / f"replay_decode_rope_sin_{decode_index:02d}_f16.bin",
                dtype="<f2",
            )
            decode_layers: list[dict[str, object]] = []
            past_tokens = M + decode_index
            for layer, layer_package in enumerate(layer_packages):
                hidden, k_rope, v_heads = run_layer_decode(
                    hidden,
                    layer_package,
                    cosine,
                    sine,
                    k_caches[layer],
                    v_caches[layer],
                    past_tokens,
                    converter,
                )
                decode_layers.append({
                    "layer": layer,
                    "output_sha256": sha256_array(hidden),
                    "k_append_sha256": sha256_array(k_rope),
                    "v_append_sha256": sha256_array(v_heads),
                })
                print(
                    f"decode {decode_index + 1}/{DECODE_STEPS} "
                    f"layer {layer + 1}/{LAYERS}",
                    flush=True,
                )
            actual_physical = read_u8(
                capture / f"actual_replay_output_{decode_index + 1:02d}_u8.bin",
                (M, HIDDEN),
            )
            actual = actual_physical[:1]
            comparison = difference(actual, hidden)
            hidden.tofile(
                staging /
                f"reference_replay_decode_row_{decode_index:02d}_u8.bin"
            )
            records.append({
                "step": decode_index + 1,
                "mode": "decode",
                "position": past_tokens,
                "logical_rows": 1,
                "comparison": comparison,
                "actual_sha256": sha256_array(actual),
                "reference_sha256": sha256_array(hidden),
                "layers": decode_layers,
            })
            comparisons.append(comparison)

        cache_records: list[dict[str, object]] = []
        cache_comparisons: list[dict[str, object]] = []
        for layer in range(LAYERS):
            for kind, reference in (
                ("k", k_caches[layer]), ("v", v_caches[layer])
            ):
                actual = read_u8(
                    capture / f"actual_layer{layer}_replay_{kind}_cache.bin",
                    (KV_HEADS, CACHE_CAPACITY, HEAD_DIM),
                )
                comparison = difference(actual, reference)
                reference.tofile(
                    staging /
                    f"reference_layer{layer}_replay_{kind}_cache_u8.bin"
                )
                cache_records.append({
                    "layer": layer,
                    "kind": kind,
                    "comparison": comparison,
                    "actual_sha256": sha256_array(actual),
                    "reference_sha256": sha256_array(reference),
                })
                cache_comparisons.append(comparison)

        passed = all(
            int(comparison["mismatches"]) == 0
            for comparison in comparisons + cache_comparisons
        )
        report = {
            "experiment": "EXP-0152",
            "contract": "byte_exact_W4U8_28_layer_prefill_decode_replay",
            "layers": LAYERS,
            "prefill_rows": M,
            "decode_steps": DECODE_STEPS,
            "cache_capacity": CACHE_CAPACITY,
            "passed": passed,
            "steps": records,
            "caches": cache_records,
            "summary": {
                "step_mismatches": int(sum(
                    int(item["mismatches"]) for item in comparisons
                )),
                "cache_mismatches": int(sum(
                    int(item["mismatches"]) for item in cache_comparisons
                )),
                "maximum_step_lsb": max(
                    int(item["max_lsb"]) for item in comparisons
                ),
                "maximum_cache_lsb": max(
                    int(item["max_lsb"]) for item in cache_comparisons
                ),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, reference_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "report": str(output),
            "reference_dir": str(reference_dir),
            "passed": passed,
            "summary": report["summary"],
        }, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
