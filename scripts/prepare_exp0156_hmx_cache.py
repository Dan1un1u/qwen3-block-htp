#!/usr/bin/env python3
"""Publish the 28-layer EXP-0156 W4U8 HMX-native cache package."""

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
    CAPACITY,
    HEADS,
    HEAD_DIM,
    K_TILES,
    BIAS_BYTES,
    PADDED,
    load_configs,
    pack_k,
    pack_v,
)


LAYERS = 28


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0152/"
            "w4u8_exact_reference_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0156/"
            "w4u8_hmx_native_v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest_path = source / "manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    ))
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

        k_bytes = HEADS * (PADDED * HEAD_DIM + K_TILES * BIAS_BYTES)
        v_bytes = HEADS * (
            PADDED * HEAD_DIM + (HEAD_DIM // 32) * BIAS_BYTES
        )
        generated: list[Path] = []
        for layer_index in range(LAYERS):
            source_layer = source / f"layer{layer_index}"
            layer = staging / f"layer{layer_index}"
            configs = load_configs(
                source_layer / "attention_config_all_groups.bin"
            )
            final_k = np.fromfile(
                source_layer / "reference_kv_cache_k_u8.bin",
                dtype=np.uint8,
            ).reshape(HEADS, CAPACITY, HEAD_DIM)
            final_v = np.fromfile(
                source_layer / "reference_kv_cache_v_u8.bin",
                dtype=np.uint8,
            ).reshape(HEADS, CAPACITY, HEAD_DIM)
            initial_k = np.fromfile(
                source_layer / "kv_cache_k_u8.bin", dtype=np.uint8
            )
            initial_v = np.fromfile(
                source_layer / "kv_cache_v_u8.bin", dtype=np.uint8
            )
            if np.any(initial_k) or np.any(initial_v):
                raise ValueError(
                    f"layer {layer_index} initial logical cache is not zero"
                )

            k_cache = layer / "kv_cache_k_hmx_u8.bin"
            v_cache = layer / "kv_cache_v_hmx_u8.bin"
            k_cache.write_bytes(bytes(k_bytes))
            v_cache.write_bytes(bytes(v_bytes))
            generated.extend((k_cache, v_cache))

            for step in range(9):
                valid = 64 + step
                packed_k = b"".join(
                    pack_k(final_k[head], valid, configs[head])
                    for head in range(HEADS)
                )
                packed_v = b"".join(
                    pack_v(final_v[head], valid, configs[head])
                    for head in range(HEADS)
                )
                if len(packed_k) != k_bytes or len(packed_v) != v_bytes:
                    raise AssertionError("HMX carrier byte contract mismatch")
                k_reference = layer / (
                    f"reference_kv_cache_k_hmx_u8_step{step:02d}.bin"
                )
                v_reference = layer / (
                    f"reference_kv_cache_v_hmx_u8_step{step:02d}.bin"
                )
                k_reference.write_bytes(packed_k)
                v_reference.write_bytes(packed_v)
                generated.extend((k_reference, v_reference))
            print(f"packed layer {layer_index + 1}/{LAYERS}", flush=True)

        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        for path in generated:
            relative = str(path.relative_to(staging))
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest["experiment"] = "EXP-0156"
        manifest["source_package"] = str(source)
        manifest["source_manifest_sha256"] = sha256(source_manifest_path)
        manifest["clone_mode"] = clone_mode
        manifest["cache_abi"] = {
            "decode_session_abi_version": 3,
            "active_layers": list(range(LAYERS)),
            "logical_capacity": CAPACITY,
            "physical_padded_capacity": PADDED,
            "k_format": "hmx_u8_k_weight_v1",
            "v_format": "hmx_u8_v_weight_v1",
            "k_bytes_per_layer": k_bytes,
            "v_bytes_per_layer": v_bytes,
            "reference_steps": list(range(9)),
            "logical_reference_authority": (
                "EXP-0152 independent exact W4U8 full-stack replay"
            ),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        print(json.dumps({
            "experiment": "EXP-0156",
            "output": str(output),
            "clone_mode": clone_mode,
            "layers": LAYERS,
            "generated_cache_files": len(generated),
            "manifest_sha256": sha256(output / "manifest.json"),
        }, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
