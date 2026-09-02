#!/usr/bin/env python3
"""Publish the EXP-0159 W4U8 immutable-carrier + delta-journal package."""

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
    CAPACITY,
    HEADS,
    HEAD_DIM,
    K_TILES,
    PADDED,
    load_configs,
    pack_k,
    pack_v,
)


LAYERS = 28
PREFILL = 64
DECODE_STEPS = CAPACITY - PREFILL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_delta_cache(
    rows: np.ndarray, kind: str, valid: int, config: tuple[int, ...]
) -> bytes:
    if kind == "k":
        full = pack_k(rows, PREFILL, config)
        full_weight_bytes = PADDED * HEAD_DIM
        compact_weight_bytes = PREFILL * HEAD_DIM
        compact_bias_bytes = PREFILL // 32 * BIAS_BYTES
        base = (
            full[:compact_weight_bytes]
            + full[full_weight_bytes:
                   full_weight_bytes + compact_bias_bytes]
        )
    else:
        full = np.frombuffer(pack_v(rows, PREFILL, config), dtype=np.uint8)
        full_weight_bytes = PADDED * HEAD_DIM
        compact_weight = (
            full[:full_weight_bytes]
            .reshape(HEAD_DIM // 32, PADDED // 32, 1024)[:, :PREFILL // 32]
            .reshape(-1)
        )
        base = compact_weight.tobytes() + full[full_weight_bytes:].tobytes()
    journal = np.zeros((DECODE_STEPS, HEAD_DIM), dtype=np.uint8)
    if valid > PREFILL:
        journal[: valid - PREFILL] = rows[PREFILL:valid]
    return base + journal.tobytes(order="C")


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
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0159/"
            "w4u8_hmx_delta_v2_compact"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest = source / "manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

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

        k_base_bytes = (
            PREFILL * HEAD_DIM + PREFILL // 32 * BIAS_BYTES
        )
        v_base_bytes = (
            PREFILL * HEAD_DIM + (HEAD_DIM // 32) * BIAS_BYTES
        )
        journal_bytes = DECODE_STEPS * HEAD_DIM
        k_bytes = HEADS * (k_base_bytes + journal_bytes)
        v_bytes = HEADS * (v_base_bytes + journal_bytes)
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

            k_initial = layer / "kv_cache_k_hmx_u8_delta.bin"
            v_initial = layer / "kv_cache_v_hmx_u8_delta.bin"
            k_initial.write_bytes(bytes(k_bytes))
            v_initial.write_bytes(bytes(v_bytes))
            generated.extend((k_initial, v_initial))

            for step in range(DECODE_STEPS + 1):
                valid = PREFILL + step
                packed_k = b"".join(
                    pack_delta_cache(
                        final_k[head], "k", valid, configs[head]
                    )
                    for head in range(HEADS)
                )
                packed_v = b"".join(
                    pack_delta_cache(
                        final_v[head], "v", valid, configs[head]
                    )
                    for head in range(HEADS)
                )
                if len(packed_k) != k_bytes or len(packed_v) != v_bytes:
                    raise AssertionError("delta cache byte contract mismatch")
                k_reference = layer / (
                    "reference_kv_cache_k_hmx_u8_delta_"
                    f"step{step:02d}.bin"
                )
                v_reference = layer / (
                    "reference_kv_cache_v_hmx_u8_delta_"
                    f"step{step:02d}.bin"
                )
                k_reference.write_bytes(packed_k)
                v_reference.write_bytes(packed_v)
                generated.extend((k_reference, v_reference))
            print(f"packed layer {layer_index + 1}/{LAYERS}", flush=True)

        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        for path in generated:
            files[str(path.relative_to(staging))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest["experiment"] = "EXP-0159"
        manifest["source_package"] = str(source)
        manifest["source_manifest_sha256"] = sha256(source_manifest)
        manifest["clone_mode"] = clone_mode
        manifest["cache_abi"] = {
            "decode_session_abi_version": 3,
            "active_layers": list(range(LAYERS)),
            "logical_capacity": CAPACITY,
            "physical_padded_capacity": PADDED,
            "prefill_tokens": PREFILL,
            "decode_steps": DECODE_STEPS,
            "k_format": "hmx_u8_k_weight_delta_v2",
            "v_format": "hmx_u8_v_weight_delta_v2",
            "k_base_bytes_per_head": k_base_bytes,
            "v_base_bytes_per_head": v_base_bytes,
            "journal_bytes_per_head": journal_bytes,
            "k_bytes_per_layer": k_bytes,
            "v_bytes_per_layer": v_bytes,
            "reference_steps": list(range(DECODE_STEPS + 1)),
            "reference_generation": (
                "immutable exact M64 HMX carrier plus logical U8 tail rows"
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
            "experiment": "EXP-0159",
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
