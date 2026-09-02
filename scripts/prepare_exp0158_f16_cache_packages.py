#!/usr/bin/env python3
"""Add exact FP16 HMX-native KV-cache carriers to EXP-0158 packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HEADS = 8
CAPACITY = 72
PADDED_CAPACITY = 96
HEAD_DIM = 128
PREFILL = 64
DECODE_STEPS = 8


def pack_tile(tile: np.ndarray) -> np.ndarray:
    assert tile.shape == (32, 32)
    return tile.reshape(16, 2, 32).transpose(0, 2, 1).reshape(1024)


def pack_k(rows: np.ndarray, valid: int) -> np.ndarray:
    logical = np.zeros((HEAD_DIM, PREFILL), dtype=np.uint16)
    logical[:, :valid] = rows[:valid].T
    tiles = []
    for n0 in range(0, PREFILL, 32):
        for k0 in range(0, HEAD_DIM, 32):
            tiles.append(pack_tile(logical[k0 : k0 + 32, n0 : n0 + 32]))
    return np.concatenate(tiles)


def pack_v(rows: np.ndarray, valid: int) -> np.ndarray:
    logical = np.zeros((PREFILL, HEAD_DIM), dtype=np.uint16)
    logical[:valid] = rows[:valid]
    tiles = []
    for n0 in range(0, HEAD_DIM, 32):
        for k0 in range(0, PREFILL, 32):
            tiles.append(pack_tile(logical[k0 : k0 + 32, n0 : n0 + 32]))
    return np.concatenate(tiles)


def pack_cache(row_major: np.ndarray, kind: str, valid: int) -> bytes:
    packed = []
    for head in range(HEADS):
        carrier = (
            pack_k(row_major[head], min(valid, PREFILL))
            if kind == "k"
            else pack_v(row_major[head], min(valid, PREFILL))
        )
        delta = np.zeros((DECODE_STEPS, HEAD_DIM), dtype=np.uint16)
        if valid > PREFILL:
            delta[: valid - PREFILL] = row_major[head, PREFILL:valid]
        packed.append(np.concatenate((carrier, delta.reshape(-1))))
    return np.concatenate(packed).astype("<u2", copy=False).tobytes()


def write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def prepare_package(package: Path) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated: list[Path] = []

    for layer in range(28):
        layer_root = package / f"layer{layer}"
        for kind in ("k", "v"):
            initial = np.fromfile(
                layer_root / f"kv_cache_{kind}_f16.bin", dtype="<u2"
            ).reshape(HEADS, CAPACITY, HEAD_DIM)
            reference = np.fromfile(
                layer_root / f"reference_kv_cache_{kind}_f16.bin", dtype="<u2"
            ).reshape(HEADS, CAPACITY, HEAD_DIM)

            native_initial = layer_root / f"kv_cache_{kind}_hmx_f16.bin"
            write_bytes(native_initial, pack_cache(initial, kind, CAPACITY))
            generated.append(native_initial)
            for step in range(DECODE_STEPS + 1):
                valid = PREFILL + step
                native_reference = layer_root / (
                    f"reference_kv_cache_{kind}_hmx_f16_step{step:02d}.bin"
                )
                write_bytes(
                    native_reference, pack_cache(reference, kind, valid)
                )
                generated.append(native_reference)

    files = manifest.setdefault("files", {})
    for path in generated:
        files[str(path.relative_to(package))] = file_record(path)
    manifest["exp0158_cache_native_overlay"] = {
        "format": "HMX_F16_K_WEIGHT_V1/HMX_F16_V_WEIGHT_V1",
        "capacity": CAPACITY,
        "padded_capacity": PADDED_CAPACITY,
        "prefill_tokens": PREFILL,
        "decode_steps": DECODE_STEPS,
        "reference_generation": "exact_bitwise_pack_from_row_major_f16",
        "decode_update": "exact_m64_hmx_carrier_plus_contiguous_delta_journal",
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_bytes(manifest_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    for package in args.packages:
        prepare_package(package.resolve())
        print(f"PREPARED={package.resolve()}")


if __name__ == "__main__":
    main()
