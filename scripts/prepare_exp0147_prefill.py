#!/usr/bin/env python3
"""Prepare EXP-0147 W4U8 prefill shape-scan packages.

The M=16/32/64 cells reuse the accepted M=64 calibration, weights, and
integer-Attention reference.  Only the logical row count changes.  Rows above
logical M are padded with the block-input zero code, while the host compares
only logical rows.  The persistent K/V cache is serialized head-major as
[kv_head, capacity, head_dim].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


PHYSICAL_M = 64
HIDDEN = 2048
KV_HEADS = 8
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-m", type=int, choices=(16, 32, 64), required=True)
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
        help=(
            "accepted M=64 device audit containing native K/V tiles; the "
            "script independently converts those tiles to the cache ABI"
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


def head_major_cache(row_major: np.ndarray, logical_m: int) -> np.ndarray:
    if row_major.shape != (PHYSICAL_M, KV_HEADS, HEAD_DIM):
        raise ValueError(f"unexpected K/V shape: {row_major.shape}")
    return np.ascontiguousarray(row_major[:logical_m].transpose(1, 0, 2))


def unpack_native_feature_tiles(path: Path) -> np.ndarray:
    physical = np.fromfile(path, dtype=np.uint8)
    expected = KV_HEADS * PHYSICAL_M * HEAD_DIM
    if physical.size != expected:
        raise ValueError(f"{path}: got {physical.size} bytes, expected {expected}")
    physical = physical.reshape(KV_HEADS, HEAD_DIM // 32, PHYSICAL_M, 32)
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(
            PHYSICAL_M, KV_HEADS, HEAD_DIM
        )
    )


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
        reference_source / "reference_w4u8_k_rope_u8.bin",
        reference_source / "reference_w4u8_v_u8.bin",
        actual_audit / "actual_k_tiles_u8.bin",
        actual_audit / "actual_v_tiles_u8.bin",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    reference_manifest = json.loads(
        (reference_source / "manifest.json").read_text(encoding="utf-8")
    )
    block_input_zero = int(
        reference_manifest["u8_qparams"]["block_input"]["zero_point"]
    )
    block_input = np.fromfile(
        source / "reference_w4u8_block_input_u8.bin", dtype=np.uint8
    ).reshape(PHYSICAL_M, HIDDEN)
    padded_input = np.full_like(block_input, block_input_zero)
    padded_input[: args.logical_m] = block_input[: args.logical_m]

    k_rows = unpack_native_feature_tiles(actual_audit / "actual_k_tiles_u8.bin")
    v_rows = unpack_native_feature_tiles(actual_audit / "actual_v_tiles_u8.bin")
    k_reference = head_major_cache(k_rows, args.logical_m)
    v_reference = head_major_cache(v_rows, args.logical_m)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="prefill-package-", dir=args.staging_root)
    )
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        padded_input.tofile(staging / "reference_w4u8_block_input_u8.bin")
        np.zeros_like(k_reference).tofile(staging / "kv_cache_k_u8.bin")
        np.zeros_like(v_reference).tofile(staging / "kv_cache_v_u8.bin")
        k_reference.tofile(staging / "reference_kv_cache_k_u8.bin")
        v_reference.tofile(staging / "reference_kv_cache_v_u8.bin")

        source_manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        source_manifest.update(
            {
                "experiment": "EXP-0147",
                "execution_unit": "qwen3_layer14_prefill_shape_scan",
                "source": str(source),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "reference_source": str(reference_source),
                "reference_manifest_sha256": sha256(
                    reference_source / "manifest.json"
                ),
                "actual_audit_source": str(actual_audit),
                "actual_k_tiles_sha256": sha256(
                    actual_audit / "actual_k_tiles_u8.bin"
                ),
                "actual_v_tiles_sha256": sha256(
                    actual_audit / "actual_v_tiles_u8.bin"
                ),
                "shape_scan": {
                    "mode": "prefill",
                    "logical_m": args.logical_m,
                    "physical_m": PHYSICAL_M,
                    "initial_kv_length": 0,
                    "kv_cache_capacity": args.logical_m,
                    "cache_layout": "head_major_[8,capacity,128]",
                },
            }
        )
        source_manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
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
                    "logical_m": args.logical_m,
                    "input_sha256": sha256(
                        output / "reference_w4u8_block_input_u8.bin"
                    ),
                    "k_cache_sha256": sha256(
                        output / "reference_kv_cache_k_u8.bin"
                    ),
                    "v_cache_sha256": sha256(
                        output / "reference_kv_cache_v_u8.bin"
                    ),
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
    main()
