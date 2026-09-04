#!/usr/bin/env python3
"""Build the real layer-14 direct-n projection gate package for EXP-0187."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from reference_w4u8_hmx import HmxU8Converter, load_qparams_bin, project_w4u8


TILE_K = 32
TILE_N = 32
PACKED_TILE_BYTES = 512
METADATA_ALIGNMENT = 256
BIAS_BYTES = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_nibble(data: memoryview, offset: int) -> int:
    byte = int(data[offset // 2])
    return (byte >> (4 if offset & 1 else 0)) & 0x0F


def write_nibble(data: bytearray, byte_offset: int, high: int, value: int) -> None:
    if high:
        data[byte_offset] |= (value & 0x0F) << 4
    else:
        data[byte_offset] |= value & 0x0F


def convert_bundles(source: Path, destination: Path, k: int, n: int) -> dict[str, int]:
    k_tiles = k // TILE_K
    n_tiles = n // TILE_N
    packed_bytes = k_tiles * PACKED_TILE_BYTES
    scale_offset = packed_bytes
    bias_offset = align_up(scale_offset + TILE_N, METADATA_ALIGNMENT)
    bundle_bytes = bias_offset + BIAS_BYTES
    raw = source.read_bytes()
    expected = n_tiles * bundle_bytes
    if len(raw) != expected:
        raise ValueError(f"{source}: got {len(raw)} bytes, expected {expected}")

    converted = bytearray(raw)
    for n_tile in range(n_tiles):
        base = n_tile * bundle_bytes
        converted[base : base + packed_bytes] = bytes(packed_bytes)
        source_bundle = memoryview(raw)[base : base + bundle_bytes]
        for k_tile in range(k_tiles):
            source_tile_nibble = k_tile * PACKED_TILE_BYTES * 2
            destination_tile = base + k_tile * PACKED_TILE_BYTES
            for input_channel in range(TILE_K):
                nibble_lane = (input_channel % 4) * 2 + (input_channel % 8) // 4
                for output_channel in range(TILE_N):
                    old_physical = (
                        (input_channel // 4) * TILE_N + output_channel
                    ) * 4 + input_channel % 4
                    value = read_nibble(
                        source_bundle, source_tile_nibble + old_physical
                    )
                    destination_byte = destination_tile + (
                        (input_channel // 8) * TILE_N + output_channel
                    ) * 4 + nibble_lane // 2
                    write_nibble(
                        converted, destination_byte, nibble_lane & 1, value
                    )

    destination.write_bytes(converted)
    return {
        "k": k,
        "n": n,
        "k_tiles": k_tiles,
        "n_tiles": n_tiles,
        "packed_bytes_per_bundle": packed_bytes,
        "bias_offset": bias_offset,
        "bundle_bytes": bundle_bytes,
        "total_bytes": len(converted),
    }


def interleave_gate_up(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    if gate.shape != (64, 6144) or up.shape != (64, 6144):
        raise ValueError(f"invalid Gate/Up shapes: {gate.shape}, {up.shape}")
    gate_tiles = gate.reshape(64, 6144 // TILE_N, TILE_N)
    up_tiles = up.reshape(64, 6144 // TILE_N, TILE_N)
    interleaved = np.empty((64, 2 * (6144 // TILE_N), TILE_N), dtype=np.uint8)
    interleaved[:, 0::2, :] = gate_tiles
    interleaved[:, 1::2, :] = up_tiles
    return interleaved.reshape(64, 12288)


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, int]:
    delta = actual.astype(np.int16) - expected.astype(np.int16)
    return {
        "elements": int(delta.size),
        "mismatches": int(np.count_nonzero(delta)),
        "max_lsb": int(np.max(np.abs(delta))) if delta.size else 0,
    }


def build_hmx_exact_references(
    source: Path, block_source: Path, converter_path: Path, staging: Path
) -> dict[str, dict[str, int]]:
    qparams = load_qparams_bin(block_source / "qparams_u8.bin")
    converter = HmxU8Converter(converter_path)
    gate_input = np.fromfile(
        source / "reference_w4u8_post_attention_norm_u8.bin", dtype=np.uint8
    ).reshape(64, 2048)
    down_input = np.fromfile(
        source / "reference_w4u8_middle_u8.bin", dtype=np.uint8
    ).reshape(64, 6144)
    gate_row = project_w4u8(
        gate_input[:1], block_source, "gate", 6144, 2048,
        qparams["post_attention_norm"], qparams["gate"], converter,
    )
    up_row = project_w4u8(
        gate_input[:1], block_source, "up", 6144, 2048,
        qparams["post_attention_norm"], qparams["up"], converter,
    )
    down_row = project_w4u8(
        down_input[:1], block_source, "down", 2048, 6144,
        qparams["middle"], qparams["down"], converter,
    )

    gate_full = np.full((64, 6144), int(qparams["gate"]["zero_point"]), dtype=np.uint8)
    up_full = np.full((64, 6144), int(qparams["up"]["zero_point"]), dtype=np.uint8)
    down_full = np.full((64, 2048), int(qparams["down"]["zero_point"]), dtype=np.uint8)
    gate_full[0] = gate_row[0]
    up_full[0] = up_row[0]
    down_full[0] = down_row[0]
    interleave_gate_up(gate_full, up_full).tofile(
        staging / "reference_w4u8_gate_up_interleaved_u8.bin"
    )
    down_full.tofile(staging / "reference_w4u8_down_u8.bin")

    old_gate = np.fromfile(source / "reference_w4u8_gate_u8.bin", dtype=np.uint8).reshape(64, 6144)
    old_up = np.fromfile(source / "reference_w4u8_up_u8.bin", dtype=np.uint8).reshape(64, 6144)
    old_down = np.fromfile(source / "reference_w4u8_down_u8.bin", dtype=np.uint8).reshape(64, 2048)
    return {
        "gate_vs_export_reference_row0": difference(gate_row, old_gate[:1]),
        "up_vs_export_reference_row0": difference(up_row, old_up[:1]),
        "down_vs_export_reference_row0": difference(down_row, old_down[:1]),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied = [
            "reference_w4u8_post_attention_norm_u8.bin",
            "reference_w4u8_middle_u8.bin",
            "reference_w4u8_down_u8.bin",
            "gate_up_packed_w4_bundles.bin",
            "down_packed_w4_bundles.bin",
            "gate_up_expanded_s8_bundles.bin",
            "down_expanded_s8_bundles.bin",
        ]
        for name in copied:
            shutil.copyfile(source / name, staging / name)
        source_manifest = json.loads((source / "manifest.json").read_text())
        block_source = Path(source_manifest["source_package"]).resolve()
        reference_differences = build_hmx_exact_references(
            source, block_source, args.converter.resolve(), staging
        )
        layouts = {
            "gate_up": convert_bundles(
                source / "gate_up_packed_w4_bundles.bin",
                staging / "gate_up_direct_n_bundles.bin",
                2048,
                12288,
            ),
            "down": convert_bundles(
                source / "down_packed_w4_bundles.bin",
                staging / "down_direct_n_bundles.bin",
                6144,
                2048,
            ),
        }
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "experiment": "EXP-0187",
            "source_package": str(source),
            "source_block_package": str(block_source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "input_zero_points": {"gate_up": 127, "down": 102},
            "contract": {
                "math": "unchanged real layer-14 W4U8 projection",
                "only_change": "W4 carrier reordered for HMX weight.n",
                "gate_up_tile_order": "gate_n32,up_n32",
                "logical_m_gate": 1,
            },
            "reference_contract": {
                "authority": "independent integer accumulation plus native HMX conversion emulator",
                "valid_rows": 1,
                "export_reference_difference": reference_differences,
            },
            "layouts": layouts,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"refusing to replace existing package: {output}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
