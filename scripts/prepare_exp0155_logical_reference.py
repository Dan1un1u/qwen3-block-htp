#!/usr/bin/env python3
"""Generate an independent exact logical-cache reference for EXP-0155."""

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

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from reference_w4u8_hmx import HmxU8Converter  # noqa: E402
from verify_exp0152_w4u8_prefill import (  # noqa: E402
    CACHE_CAPACITY,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    M,
    difference,
    read_u8,
    run_layer_prefill,
    sha256_array,
)
from verify_exp0152_w4u8_replay import run_layer_decode  # noqa: E402


DECODE_STEPS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0148/w4u8_formal"))
    parser.add_argument(
        "--converter", type=Path,
        default=Path(
            "/home/daniuniu/work/qwen3-block-htp/build/reference/"
            "qbh_hmx_u8_reference.so"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0155/"
            "layer14_logical_reference_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    converter_path = args.converter.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable reference {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent))
    converter = HmxU8Converter(converter_path)
    records: list[dict[str, object]] = []
    try:
        hidden = read_u8(
            source / "reference_w4u8_block_input_u8.bin", (M, HIDDEN))
        cosine = np.fromfile(source / "rope_cos_f16.bin", dtype="<f2")
        sine = np.fromfile(source / "rope_sin_f16.bin", dtype="<f2")
        output_u8, k_rope, v_heads = run_layer_prefill(
            hidden, source, cosine, sine, converter)
        output_reference = read_u8(
            source / "reference_w4u8_block_output_u8.bin", (M, HIDDEN))
        comparison = difference(output_reference, output_u8)

        cache_shape = (KV_HEADS, CACHE_CAPACITY, HEAD_DIM)
        k_cache = read_u8(source / "kv_cache_k_u8.bin", cache_shape).copy()
        v_cache = read_u8(source / "kv_cache_v_u8.bin", cache_shape).copy()
        k_cache[:, :M, :] = k_rope.transpose(1, 0, 2)
        v_cache[:, :M, :] = v_heads.transpose(1, 0, 2)
        output_u8.tofile(staging / "reference_replay_output_00_u8.bin")
        k_cache.tofile(staging / "reference_kv_cache_k_u8_step00.bin")
        v_cache.tofile(staging / "reference_kv_cache_v_u8_step00.bin")
        records.append({
            "step": 0,
            "mode": "prefill",
            "valid_length": M,
            "output_comparison": comparison,
            "output_sha256": sha256_array(output_u8),
            "k_cache_sha256": sha256_array(k_cache),
            "v_cache_sha256": sha256_array(v_cache),
        })

        for decode_index in range(DECODE_STEPS):
            hidden = read_u8(
                source / f"replay_decode_input_{decode_index:02d}_u8.bin",
                (M, HIDDEN),
            )[:1].copy()
            cosine = np.fromfile(
                source / f"replay_decode_rope_cos_{decode_index:02d}_f16.bin",
                dtype="<f2",
            )
            sine = np.fromfile(
                source / f"replay_decode_rope_sin_{decode_index:02d}_f16.bin",
                dtype="<f2",
            )
            output_u8, _, _ = run_layer_decode(
                hidden, source, cosine, sine, k_cache, v_cache,
                M + decode_index, converter)
            output_reference = read_u8(
                source / f"replay_decode_reference_{decode_index:02d}_u8.bin",
                (M, HIDDEN),
            )[:1]
            comparison = difference(output_reference, output_u8)
            step = decode_index + 1
            output_u8.tofile(
                staging / f"reference_replay_output_{step:02d}_u8.bin")
            k_cache.tofile(
                staging / f"reference_kv_cache_k_u8_step{step:02d}.bin")
            v_cache.tofile(
                staging / f"reference_kv_cache_v_u8_step{step:02d}.bin")
            records.append({
                "step": step,
                "mode": "decode",
                "position": M + decode_index,
                "valid_length": M + step,
                "output_comparison": comparison,
                "output_sha256": sha256_array(output_u8),
                "k_cache_sha256": sha256_array(k_cache),
                "v_cache_sha256": sha256_array(v_cache),
            })

        passed = all(
            int(record["output_comparison"]["mismatches"]) == 0
            for record in records)
        manifest = {
            "experiment": "EXP-0155",
            "kind": "independent_exact_single_layer14_W4U8_logical_cache",
            "source_package": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "converter": str(converter_path),
            "converter_sha256": sha256(converter_path),
            "cache_shape": list(cache_shape),
            "passed": passed,
            "steps": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.rename(staging, output)
        print(json.dumps({
            "output": str(output),
            "passed": passed,
            "steps": len(records),
            "maximum_output_lsb": max(
                int(record["output_comparison"]["max_lsb"])
                for record in records),
        }, sort_keys=True))
        if not passed:
            raise SystemExit(1)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
