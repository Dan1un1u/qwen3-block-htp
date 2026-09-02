#!/usr/bin/env python3
"""Build the EXP-0155 layer-14 package and independent HMX-cache references."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np


HEADS = 8
CAPACITY = 72
PADDED = 96
HEAD_DIM = 128
HMX_IN = 32
HMX_OUT = 32
WEIGHT_BYTES = 1024
BIAS_BYTES = 256
HEAD_DIM_TILES = 4
K_TILES = PADDED // HMX_IN


def half_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float16).view(np.uint16))


def round_div_signed(numerator: int, denominator: int) -> int:
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def load_configs(path: Path) -> list[tuple[int, ...]]:
    data = path.read_bytes()
    if len(data) != HEADS * 60:
        raise ValueError(f"unexpected attention config size: {len(data)}")
    return [struct.unpack_from("<IIIIiiiiiIIIIII", data, i * 60)
            for i in range(HEADS)]


def pack_k(rows: np.ndarray, valid: int, config: tuple[int, ...]) -> bytes:
    q_zero_point = config[4]
    k_zero_point = config[5]
    score_shift = config[11]
    divisor = 1 << score_shift
    rounding = 0 if score_shift == 0 else divisor // 2
    conversion = half_bits(512.0 / divisor)
    weights = np.zeros((K_TILES, HEAD_DIM_TILES, WEIGHT_BYTES), dtype=np.uint8)
    bias = np.zeros((K_TILES, BIAS_BYTES // 4), dtype="<u4")

    for n_tile in range(K_TILES):
        sums = np.zeros(HMX_OUT, dtype=np.int32)
        for k_tile in range(HEAD_DIM_TILES):
            destination = weights[n_tile, k_tile]
            for input_group in range(HMX_IN // 4):
                for output in range(HMX_OUT):
                    token = n_tile * HMX_OUT + output
                    word = 0
                    for lane in range(4):
                        centered = 0
                        if token < valid:
                            channel = k_tile * HMX_IN + input_group * 4 + lane
                            centered = int(rows[token, channel]) - k_zero_point
                        centered = max(-128, min(127, centered))
                        sums[output] += centered
                        word |= (centered & 0xFF) << (lane * 8)
                    start = input_group * 128 + output * 4
                    destination[start:start + 4] = np.frombuffer(
                        struct.pack("<I", word), dtype=np.uint8)
        bias[n_tile, :HMX_OUT] = conversion
        for output in range(HMX_OUT):
            upper = (-q_zero_point * int(sums[output]) +
                     128 * divisor + rounding)
            bias[n_tile, HMX_OUT + output] = np.uint32(upper & 0xFFFFFFFF)
    return weights.tobytes(order="C") + bias.tobytes(order="C")


def pack_v(rows: np.ndarray, valid: int, config: tuple[int, ...]) -> bytes:
    v_zero_point = config[6]
    output_zero_point = config[8]
    numerator = config[9]
    denominator = config[10]
    av_shift = config[13]
    av_multiplier = config[14]
    divisor = 1 << av_shift
    rounding = 0 if av_shift == 0 else divisor // 2
    conversion = half_bits(512.0 / divisor)
    hmx_output_zero_point = output_zero_point if av_multiplier == 1 else 128
    weights = np.zeros((HEAD_DIM_TILES, K_TILES, WEIGHT_BYTES), dtype=np.uint8)
    bias = np.zeros((HEAD_DIM_TILES, BIAS_BYTES // 4), dtype="<u4")

    for n_tile in range(HEAD_DIM_TILES):
        for k_tile in range(K_TILES):
            destination = weights[n_tile, k_tile]
            for input_group in range(HMX_IN // 4):
                for output in range(HMX_OUT):
                    channel = n_tile * HMX_OUT + output
                    word = 0
                    for lane in range(4):
                        token = k_tile * HMX_IN + input_group * 4 + lane
                        requantized = 0
                        if token < valid:
                            centered = int(rows[token, channel]) - v_zero_point
                            requantized = round_div_signed(
                                centered * numerator, denominator)
                            requantized = max(-128, min(127, requantized))
                        word |= (requantized & 0xFF) << (lane * 8)
                    start = input_group * 128 + output * 4
                    destination[start:start + 4] = np.frombuffer(
                        struct.pack("<I", word), dtype=np.uint8)
        bias[n_tile, :HMX_OUT] = conversion
        bias[n_tile, HMX_OUT:] = np.uint32(
            (hmx_output_zero_point * divisor + rounding) & 0xFFFFFFFF)
    return weights.tobytes(order="C") + bias.tobytes(order="C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/mnt/d/llm_exp/models/qwen3-block-htp/exp0148/w4u8_formal")
    parser.add_argument(
        "--output",
        default="/mnt/d/llm_exp/models/qwen3-block-htp/exp0155/layer14_hmx_cache_v1")
    parser.add_argument(
        "--reference-dir",
        default=("/mnt/d/llm_exp/models/qwen3-block-htp/exp0155/"
                 "layer14_logical_reference_v1"))
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    reference_dir = Path(args.reference_dir)
    layer = output / "layer14"
    output.mkdir(parents=True, exist_ok=True)
    layer.mkdir(parents=True, exist_ok=True)

    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)

    layer_files = [
        "qparams_u8.bin", "input_norm_weight_f16.bin",
        "post_norm_weight_f16.bin", "q_norm_weight_f16.bin",
        "k_norm_weight_f16.bin", "attention_config_all_groups.bin",
        "silu_up_lut_u16.bin", "kv_cache_k_u8.bin",
        "kv_cache_v_u8.bin", "reference_kv_cache_k_u8.bin",
        "reference_kv_cache_v_u8.bin",
    ]
    for projection in ("q", "k", "v", "o", "gate", "up", "down"):
        layer_files.extend([
            f"{projection}_weight_w4_hmx.bin",
            f"{projection}_weight_w4_scale_f32.bin",
        ])
    for name in layer_files:
        shutil.copy2(source / name, layer / name)

    configs = load_configs(source / "attention_config_all_groups.bin")
    k_bytes = HEADS * (PADDED * HEAD_DIM + K_TILES * BIAS_BYTES)
    v_bytes = HEADS * (PADDED * HEAD_DIM + HEAD_DIM_TILES * BIAS_BYTES)
    (layer / "kv_cache_k_hmx_u8.bin").write_bytes(bytes(k_bytes))
    (layer / "kv_cache_v_hmx_u8.bin").write_bytes(bytes(v_bytes))
    for step in range(9):
        valid = 64 + step
        k_reference = reference_dir / (
            f"reference_kv_cache_k_u8_step{step:02d}.bin")
        v_reference = reference_dir / (
            f"reference_kv_cache_v_u8_step{step:02d}.bin")
        k_rows = np.fromfile(k_reference, dtype=np.uint8).reshape(
            HEADS, CAPACITY, HEAD_DIM)
        v_rows = np.fromfile(v_reference, dtype=np.uint8).reshape(
            HEADS, CAPACITY, HEAD_DIM)
        packed_k = b"".join(
            pack_k(k_rows[head], valid, configs[head])
            for head in range(HEADS))
        packed_v = b"".join(
            pack_v(v_rows[head], valid, configs[head])
            for head in range(HEADS))
        if len(packed_k) != k_bytes or len(packed_v) != v_bytes:
            raise AssertionError("carrier byte contract mismatch")
        (layer / f"reference_kv_cache_k_hmx_u8_step{step:02d}.bin").write_bytes(
            packed_k)
        (layer / f"reference_kv_cache_v_hmx_u8_step{step:02d}.bin").write_bytes(
            packed_v)

    final_k = reference_dir / "reference_kv_cache_k_u8_step08.bin"
    final_v = reference_dir / "reference_kv_cache_v_u8_step08.bin"
    for destination in (output, layer):
        shutil.copy2(final_k, destination / "reference_kv_cache_k_u8.bin")
        shutil.copy2(final_v, destination / "reference_kv_cache_v_u8.bin")

    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest_exp0155.json":
            files[str(path.relative_to(output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "experiment": "EXP-0155",
        "execution_unit": "qwen3_real_layer14_M64_prefill_decode64_71",
        "source_package": str(source),
        "reference_source": {
            "kind": "independent exact W4U8 logical-cache replay",
            "directory": str(reference_dir),
            "manifest_sha256": sha256(reference_dir / "manifest.json"),
            "final_k_cache_sha256": sha256(final_k),
            "final_v_cache_sha256": sha256(final_v),
        },
        "cache_abi": {
            "version": 1,
            "logical_capacity": CAPACITY,
            "physical_padded_capacity": PADDED,
            "k_format": "hmx_u8_k_weight_v1",
            "v_format": "hmx_u8_v_weight_v1",
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "reference_steps": list(range(9)),
        },
        "files": files,
    }
    (output / "manifest_exp0155.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "reference_steps": 9,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
