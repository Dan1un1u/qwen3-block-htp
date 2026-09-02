#!/usr/bin/env python3
"""Create EXP-0161 Phase-B output-major segmented W4U8 cache packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from prepare_exp0155_hmx_cache import (
    BIAS_BYTES,
    HEADS,
    HEAD_DIM,
    HEAD_DIM_TILES,
    PADDED,
    WEIGHT_BYTES,
    load_configs,
    pack_k,
    pack_v,
)


SEGMENT_TOKENS = 32
SEGMENT_WEIGHT_BYTES = SEGMENT_TOKENS * HEAD_DIM
SEGMENT_K_BYTES = SEGMENT_WEIGHT_BYTES + BIAS_BYTES
TAIL_BYTES = SEGMENT_TOKENS * HEAD_DIM
V_SEGMENT_BLOCK_SEGMENTS = 32
SUPPORTED_LENGTHS = (64, 256, 1024, 4096)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_package(root: Path, past_length: int) -> Path:
    suffix = "_v2" if past_length == 64 else ""
    return root / f"decode_l{past_length}_w4u8{suffix}"


def pack_k_segment(rows: np.ndarray, config: tuple[int, ...]) -> bytes:
    packed = pack_k(rows, SEGMENT_TOKENS, config)
    bias_offset = PADDED * HEAD_DIM
    result = packed[:SEGMENT_WEIGHT_BYTES] + packed[
        bias_offset:bias_offset + BIAS_BYTES
    ]
    if len(result) != SEGMENT_K_BYTES:
        raise AssertionError("K segment size mismatch")
    return result


def pack_v_segment(rows: np.ndarray, config: tuple[int, ...]) -> tuple[bytes, bytes]:
    packed = np.frombuffer(pack_v(rows, SEGMENT_TOKENS, config), dtype=np.uint8)
    weight_bytes = PADDED * HEAD_DIM
    weight = (
        packed[:weight_bytes]
        .reshape(HEAD_DIM_TILES, PADDED // SEGMENT_TOKENS, WEIGHT_BYTES)
        [:, 0]
        .reshape(-1)
        .tobytes()
    )
    bias = packed[weight_bytes:weight_bytes + HEAD_DIM_TILES * BIAS_BYTES].tobytes()
    if len(weight) != SEGMENT_WEIGHT_BYTES or len(bias) != HEAD_DIM_TILES * BIAS_BYTES:
        raise AssertionError("V segment size mismatch")
    return weight, bias


def pack_cache(
    rows: np.ndarray,
    reference_rows: np.ndarray,
    kind: str,
    past_length: int,
    configs: list[tuple[int, ...]],
) -> tuple[bytes, bytes]:
    if past_length % SEGMENT_TOKENS != 0:
        raise ValueError("Phase-B fixtures require a sealed 32-token prefix")
    segments = past_length // SEGMENT_TOKENS
    initial_heads: list[bytes] = []
    reference_heads: list[bytes] = []
    for head in range(HEADS):
        sealed: list[bytes] = []
        v_bias = b""
        for segment in range(segments):
            first = segment * SEGMENT_TOKENS
            segment_rows = rows[head, first:first + SEGMENT_TOKENS]
            if kind == "k":
                sealed.append(pack_k_segment(segment_rows, configs[head]))
            else:
                weight, bias = pack_v_segment(segment_rows, configs[head])
                sealed.append(weight)
                if not v_bias:
                    v_bias = bias
        tail_initial = bytes(TAIL_BYTES)
        tail_reference = np.zeros((SEGMENT_TOKENS, HEAD_DIM), dtype=np.uint8)
        tail_reference[0] = reference_rows[head, past_length]
        if kind == "k":
            initial_heads.append(b"".join(sealed) + tail_initial)
            reference_heads.append(b"".join(sealed) + tail_reference.tobytes())
        else:
            output_major = b"".join(
                packed_segment[
                    output_tile * WEIGHT_BYTES:
                    (output_tile + 1) * WEIGHT_BYTES
                ]
                for block_first in range(
                    0, segments, V_SEGMENT_BLOCK_SEGMENTS
                )
                for output_tile in range(HEAD_DIM_TILES)
                for packed_segment in sealed[
                    block_first:block_first + V_SEGMENT_BLOCK_SEGMENTS
                ]
            )
            if len(output_major) != segments * SEGMENT_WEIGHT_BYTES:
                raise AssertionError("output-major V segment size mismatch")
            initial_heads.append(output_major + v_bias + tail_initial)
            reference_heads.append(
                output_major + v_bias + tail_reference.tobytes()
            )
    return b"".join(initial_heads), b"".join(reference_heads)


def publish_one(source: Path, output: Path, past_length: int) -> dict[str, object]:
    capacity = past_length + 1
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    shutil.rmtree(staging)
    try:
        try:
            shutil.copytree(source, staging, copy_function=os.link)
            clone_mode = "hardlink"
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source, staging)
            clone_mode = "copy"

        configs = load_configs(source / "attention_config_all_groups.bin")
        generated: list[Path] = []
        sizes: dict[str, int] = {}
        for kind in ("k", "v"):
            initial_rows = np.fromfile(
                source / f"kv_cache_{kind}_u8.bin", dtype=np.uint8
            ).reshape(HEADS, capacity, HEAD_DIM)
            reference_rows = np.fromfile(
                source / f"reference_kv_cache_{kind}_u8.bin", dtype=np.uint8
            ).reshape(HEADS, capacity, HEAD_DIM)
            initial, reference = pack_cache(
                initial_rows, reference_rows, kind, past_length, configs
            )
            initial_path = staging / f"kv_cache_{kind}_hmx_u8_segmented.bin"
            reference_path = staging / f"reference_kv_cache_{kind}_hmx_u8_segmented.bin"
            initial_path.write_bytes(initial)
            reference_path.write_bytes(reference)
            generated.extend((initial_path, reference_path))
            sizes[kind] = len(initial)

        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        for path in generated:
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest.update({
            "experiment": "EXP-0161",
            "phase": "B",
            "source_package": str(source),
            "source_manifest_sha256": sha256(source_manifest),
            "clone_mode": clone_mode,
            "cache_abi": {
                "format_version": 4,
                "logical_capacity": capacity,
                "segment_tokens": SEGMENT_TOKENS,
                "v_segment_block_segments": V_SEGMENT_BLOCK_SEGMENTS,
                "sealed_segments": past_length // SEGMENT_TOKENS,
                "active_tail_capacity": SEGMENT_TOKENS,
                "initial_valid_tokens": past_length,
                "k_format": "hmx_u8_k_segmented_v4",
                "v_format": "hmx_u8_v_segmented_v4_blocked_output_major",
                "k_bytes": sizes["k"],
                "v_bytes": sizes["v"],
                "reference_generation": (
                    "EXP0147 logical cache independently repacked into immutable "
                    "32-token HMX segments plus one mutable logical tail; "
                    "V uses 32-segment output-tile-major blocks"
                ),
            },
        })
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return {
            "past_length": past_length,
            "capacity": capacity,
            "output": str(output),
            "clone_mode": clone_mode,
            "sealed_segments": past_length // SEGMENT_TOKENS,
            "k_bytes": sizes["k"],
            "v_bytes": sizes["v"],
            "manifest_sha256": sha256(output / "manifest.json"),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/exact_logical_v1"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0161"),
    )
    parser.add_argument("--length", type=int, action="append", choices=SUPPORTED_LENGTHS)
    args = parser.parse_args()
    results = []
    for past_length in tuple(args.length or SUPPORTED_LENGTHS):
        result = publish_one(
            source_package(args.source_root.resolve(), past_length),
            args.output_root.resolve() /
                f"decode_l{past_length}_w4u8_hmx_segmented_v4c_exact",
            past_length,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"experiment": "EXP-0161", "phase": "B", "packages": results}, indent=2))


if __name__ == "__main__":
    main()
