#!/usr/bin/env python3
"""Prepare the real layer-14 EXP-0040 U8 MLP A/B package.

Each real per-output-channel ratio is approximated as an unsigned integer
carrier multiplier times a power-of-two HMX postscale.  This keeps the S8
carrier in range while making the integer-HMX conversion independently and
byte-exactly reproducible.  The packed and expanded artifacts use identical
S8 carrier values, bias words, requantization words and tile order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HMX_K = 32
HMX_N = 32
PACKED_TILE_BYTES = 512
S8_TILE_BYTES = 1024
BIAS_BYTES = 256
ALIGNMENT = 256
MAX_INTEGER_MULTIPLIER = 18
MAX_POWER2_SHIFT = 15
HMX_INTERMEDIATE_ZERO_POINT = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/"
            "mlp_package_layer14_m64"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0040"),
    )
    return parser.parse_args()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_nibble(values: np.ndarray) -> np.ndarray:
    signed = values.astype(np.int8)
    return np.where(values >= 8, signed - np.int8(16), signed).astype(np.int8)


def expand_physical_w4(packed: np.ndarray) -> np.ndarray:
    flat = packed.reshape(-1)
    expanded = np.empty(flat.size * 2, dtype=np.int8)
    expanded[0::2] = decode_nibble(flat & np.uint8(0x0F))
    expanded[1::2] = decode_nibble(flat >> np.uint8(4))
    return expanded


def unpack_logical_w4(packed: np.ndarray) -> np.ndarray:
    n_tiles, k_tiles, _ = packed.shape
    physical = expand_physical_w4(packed).reshape(
        n_tiles, k_tiles, 8, HMX_N, 4
    )
    return (
        physical.transpose(0, 1, 2, 4, 3)
        .reshape(n_tiles, k_tiles, HMX_K, HMX_N)
        .transpose(0, 3, 1, 2)
        .reshape(n_tiles * HMX_N, k_tiles * HMX_K)
    )


def output_weight_sums(packed_tiles: np.ndarray) -> np.ndarray:
    k_tiles = packed_tiles.shape[0]
    sums = np.zeros(HMX_N, dtype=np.int64)
    for k_tile in range(k_tiles):
        tile = packed_tiles[k_tile].reshape(-1)
        for input_lane in range(HMX_K):
            for output_lane in range(HMX_N):
                physical = ((input_lane // 4) * HMX_N + output_lane) * 4
                physical += input_lane % 4
                byte = tile[physical // 2]
                nibble = int(byte >> 4) if physical & 1 else int(byte & 0x0F)
                sums[output_lane] += nibble - 16 if nibble >= 8 else nibble
    return sums


def select_integer_carrier(
    weight_scales: np.ndarray,
    input_scale: float,
    output_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ratios = (
        np.float64(input_scale)
        * weight_scales.astype(np.float64)
        / np.float64(output_scale)
    )
    if not np.all(np.isfinite(ratios)) or not np.all(ratios > 0):
        raise ValueError("invalid projection requantization ratio")

    best_error = np.full(ratios.shape, np.inf, dtype=np.float64)
    best_multiplier = np.zeros(ratios.shape, dtype=np.int64)
    best_shift = np.zeros(ratios.shape, dtype=np.int64)
    for shift in range(MAX_POWER2_SHIFT + 1):
        quantum = np.ldexp(np.float64(1.0), -shift)
        multiplier = np.rint(ratios / quantum).astype(np.int64)
        multiplier = np.clip(multiplier, 1, MAX_INTEGER_MULTIPLIER)
        effective = multiplier.astype(np.float64) * quantum
        error = np.abs(effective - ratios)
        update = error < best_error
        best_error[update] = error[update]
        best_multiplier[update] = multiplier[update]
        best_shift[update] = shift

    effective_ratios = np.ldexp(
        best_multiplier.astype(np.float64), -best_shift
    )
    return (
        best_multiplier.astype(np.uint8),
        best_shift.astype(np.uint8),
        ratios,
        effective_ratios,
    )


def bias_words(
    packed_tiles: np.ndarray,
    shifts: np.ndarray,
    input_zero_point: int,
) -> np.ndarray:
    sums = output_weight_sums(packed_tiles)
    conversion = np.ldexp(
        np.float32(512.0), -shifts.astype(np.int32)
    ).astype(np.float16).view(np.uint16)
    shifts_i64 = shifts.astype(np.int64)
    divisors = np.left_shift(np.int64(1), shifts_i64)
    rounding = np.where(
        shifts_i64 > 0,
        np.right_shift(divisors, np.int64(1)),
        np.int64(0),
    )
    offsets = (
        -np.int64(input_zero_point)
        * sums.astype(np.int64)
        + np.int64(HMX_INTERMEDIATE_ZERO_POINT)
        * divisors
        + rounding
    )
    if np.any(offsets < np.iinfo(np.int32).min) or np.any(
        offsets > np.iinfo(np.int32).max
    ):
        raise ValueError("HMX offset is outside S32")
    words = np.zeros(64, dtype="<u4")
    words[:32] = conversion.astype(np.uint32)
    words[32:] = offsets.astype(np.int32).view(np.uint32)
    return words


def integer_hmx_projection(
    activation: np.ndarray,
    packed: np.ndarray,
    weight_scales: np.ndarray,
    input_qparam: dict[str, object],
    output_qparam: dict[str, object],
) -> np.ndarray:
    logical_w4 = unpack_logical_w4(packed)
    multipliers, shifts, _, _ = select_integer_carrier(
        weight_scales,
        float(input_qparam["scale"]),
        float(output_qparam["scale"]),
    )
    sums = logical_w4.astype(np.int32).sum(axis=1, dtype=np.int32)
    shifts_i64 = shifts.astype(np.int64)
    divisors = np.left_shift(np.int64(1), shifts_i64)
    rounding = np.where(
        shifts_i64 > 0,
        np.right_shift(divisors, np.int64(1)),
        np.int64(0),
    )
    offsets = (
        -np.int64(input_qparam["zero_point"])
        * sums.astype(np.int64)
        + np.int64(HMX_INTERMEDIATE_ZERO_POINT)
        * divisors
        + rounding
    )
    accumulators = (
        activation.astype(np.int32) @ logical_w4.astype(np.int32).T
    )
    accumulators = accumulators.astype(np.int64)
    accumulators += offsets[None, :]
    intermediate = np.floor_divide(
        accumulators, divisors[None, :]
    ).clip(0, 255).astype(np.int32)
    output = (
        (intermediate - HMX_INTERMEDIATE_ZERO_POINT)
        * multipliers.astype(np.int32)[None, :]
        + int(output_qparam["zero_point"])
    )
    return output.clip(0, 255).astype(np.uint8)


def build_projection(
    sources: list[tuple[np.ndarray, np.ndarray]],
    input_qparam: dict[str, object],
    output_qparams: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_tiles_each = [source.shape[0] for source, _ in sources]
    if len(set(n_tiles_each)) != 1:
        raise ValueError("paired projections require identical N tile counts")
    k_tiles = sources[0][0].shape[1]
    if any(source.shape[1] != k_tiles for source, _ in sources):
        raise ValueError("paired projections require identical K tile counts")
    output_tiles = n_tiles_each[0] * len(sources)
    packed_chunk = k_tiles * PACKED_TILE_BYTES
    packed_bias_offset = align_up(packed_chunk + 32)
    packed_bundle = packed_bias_offset + BIAS_BYTES
    expanded_chunk = k_tiles * S8_TILE_BYTES
    expanded_bundle = expanded_chunk + BIAS_BYTES
    packed_output = np.zeros((output_tiles, packed_bundle), dtype=np.uint8)
    expanded_output = np.zeros((output_tiles, expanded_bundle), dtype=np.uint8)
    output_multipliers = np.zeros((output_tiles, HMX_N), dtype=np.uint8)

    approximation_records: list[dict[str, object]] = []
    for source_index, ((_, scales), output_qparam) in enumerate(
        zip(sources, output_qparams)
    ):
        multipliers, shifts, ratios, effective = select_integer_carrier(
            scales,
            float(input_qparam["scale"]),
            float(output_qparam["scale"]),
        )
        relative = np.abs(effective - ratios) / ratios
        approximation_records.append(
            {
                "source_index": source_index,
                "channels": int(scales.size),
                "multiplier_min": int(multipliers.min()),
                "multiplier_max": int(multipliers.max()),
                "shift_min": int(shifts.min()),
                "shift_max": int(shifts.max()),
                "maximum_absolute_ratio_error": float(
                    np.max(np.abs(effective - ratios))
                ),
                "maximum_relative_ratio_error": float(np.max(relative)),
                "mean_relative_ratio_error": float(np.mean(relative)),
            }
        )

    for logical_tile in range(n_tiles_each[0]):
        for source_index, ((source, scales), output_qparam) in enumerate(
            zip(sources, output_qparams)
        ):
            output_tile = logical_tile * len(sources) + source_index
            source_tiles = source[logical_tile]
            scale_slice = scales[
                logical_tile * HMX_N : (logical_tile + 1) * HMX_N
            ]
            multipliers, shifts, _, _ = select_integer_carrier(
                scale_slice,
                float(input_qparam["scale"]),
                float(output_qparam["scale"]),
            )
            bias = bias_words(
                source_tiles,
                shifts,
                int(input_qparam["zero_point"]),
            )
            packed_output[output_tile, :packed_chunk] = source_tiles.reshape(-1)
            packed_output[
                output_tile, packed_bias_offset : packed_bias_offset + BIAS_BYTES
            ] = bias.view(np.uint8)
            expanded_output[output_tile, :expanded_chunk] = (
                expand_physical_w4(source_tiles).view(np.uint8)
            )
            expanded_output[output_tile, expanded_chunk:] = bias.view(np.uint8)
            output_multipliers[output_tile] = multipliers

    return packed_output, expanded_output, output_multipliers, {
        "k_tiles": k_tiles,
        "n_tiles": output_tiles,
        "packed_bundle_bytes": packed_bundle,
        "expanded_bundle_bytes": expanded_bundle,
        "bias_bytes": BIAS_BYTES,
        "scale_approximation": approximation_records,
    }


def write_array(root: Path, name: str, array: np.ndarray) -> dict[str, object]:
    path = root / name
    contiguous = np.ascontiguousarray(array)
    path.write_bytes(contiguous.tobytes(order="C"))
    return {
        "file": name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    qparams = source_manifest["u8_qparams"]
    args.staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="package-", dir=args.staging_root))
    tensors: dict[str, dict[str, object]] = {}

    try:
        def packed(name: str, n: int, k: int) -> np.ndarray:
            return np.fromfile(
                args.source / f"{name}_weight_w4_hmx.bin", dtype=np.uint8
            ).reshape(n // HMX_N, k // HMX_K, PACKED_TILE_BYTES)

        def scales(name: str) -> np.ndarray:
            return np.fromfile(
                args.source / f"{name}_weight_w4_scale_f32.bin", dtype="<f4"
            )

        gate_carrier = packed("gate", INTERMEDIATE, HIDDEN)
        up_carrier = packed("up", INTERMEDIATE, HIDDEN)
        down_carrier = packed("down", HIDDEN, INTERMEDIATE)
        gate_scales = scales("gate")
        up_scales = scales("up")
        down_scales = scales("down")
        (
            gate_up_packed,
            gate_up_expanded,
            gate_up_multipliers,
            gate_layout,
        ) = build_projection(
            [
                (gate_carrier, gate_scales),
                (up_carrier, up_scales),
            ],
            qparams["post_attention_norm"],
            [qparams["gate"], qparams["up"]],
        )
        (
            down_packed,
            down_expanded,
            down_multipliers,
            down_layout,
        ) = build_projection(
            [(down_carrier, down_scales)],
            qparams["middle"],
            [qparams["down"]],
        )

        gate_values = np.arange(256, dtype=np.float32)[:, None]
        up_values = np.arange(256, dtype=np.float32)[None, :]
        gate_real = (
            gate_values - np.float32(qparams["gate"]["zero_point"])
        ) * np.float32(qparams["gate"]["scale"])
        up_real = (
            up_values - np.float32(qparams["up"]["zero_point"])
        ) * np.float32(qparams["up"]["scale"])
        middle = gate_real / (
            np.float32(1.0) + np.exp(-gate_real, dtype=np.float32)
        ) * up_real
        lut_u8 = np.rint(
            middle / np.float32(qparams["middle"]["scale"])
            + np.float32(qparams["middle"]["zero_point"])
        ).clip(0, 255).astype(np.uint8)
        lut_u16 = lut_u8.astype("<u2")

        actual_gate = np.fromfile(
            args.source / "reference_w4u8_gate_u8.bin", dtype=np.uint8
        )
        actual_up = np.fromfile(
            args.source / "reference_w4u8_up_u8.bin", dtype=np.uint8
        )
        actual_middle = np.fromfile(
            args.source / "reference_w4u8_middle_u8.bin", dtype=np.uint8
        )
        lut_middle = lut_u8[actual_gate, actual_up]
        mismatch_count = int(np.count_nonzero(lut_middle != actual_middle))
        if mismatch_count != 0:
            raise ValueError(f"real Middle LUT mismatch count: {mismatch_count}")

        real_input = np.fromfile(
            args.source / "reference_w4u8_post_attention_norm_u8.bin",
            dtype=np.uint8,
        ).reshape(M, HIDDEN)
        integer_gate = integer_hmx_projection(
            real_input,
            gate_carrier,
            gate_scales,
            qparams["post_attention_norm"],
            qparams["gate"],
        )
        integer_up = integer_hmx_projection(
            real_input,
            up_carrier,
            up_scales,
            qparams["post_attention_norm"],
            qparams["up"],
        )
        integer_middle = lut_u8[integer_gate, integer_up]
        integer_down = integer_hmx_projection(
            integer_middle,
            down_carrier,
            down_scales,
            qparams["middle"],
            qparams["down"],
        )

        tensors["gate_up_packed"] = write_array(
            staging, "gate_up_packed_w4_bundles.bin", gate_up_packed
        )
        tensors["gate_up_expanded"] = write_array(
            staging, "gate_up_expanded_s8_bundles.bin", gate_up_expanded
        )
        tensors["down_packed"] = write_array(
            staging, "down_packed_w4_bundles.bin", down_packed
        )
        tensors["down_expanded"] = write_array(
            staging, "down_expanded_s8_bundles.bin", down_expanded
        )
        tensors["gate_up_output_multipliers"] = write_array(
            staging,
            "gate_up_output_multipliers_u8.bin",
            gate_up_multipliers,
        )
        tensors["down_output_multipliers"] = write_array(
            staging,
            "down_output_multipliers_u8.bin",
            down_multipliers,
        )
        tensors["activation_lut"] = write_array(
            staging, "silu_up_lut_u16.bin", lut_u16
        )
        tensors["integer_reference_gate"] = write_array(
            staging, "reference_integer_hmx_gate_u8.bin", integer_gate
        )
        tensors["integer_reference_up"] = write_array(
            staging, "reference_integer_hmx_up_u8.bin", integer_up
        )
        tensors["integer_reference_middle"] = write_array(
            staging, "reference_integer_hmx_middle_u8.bin", integer_middle
        )
        tensors["integer_reference_output"] = write_array(
            staging, "reference_integer_hmx_down_u8.bin", integer_down
        )

        copies = {
            "input": "reference_w4u8_post_attention_norm_u8.bin",
            "reference_gate": "reference_w4u8_gate_u8.bin",
            "reference_up": "reference_w4u8_up_u8.bin",
            "reference_middle": "reference_w4u8_middle_u8.bin",
            "reference_output": "reference_w4u8_down_u8.bin",
        }
        for key, filename in copies.items():
            shutil.copy2(args.source / filename, staging / filename)
            path = staging / filename
            tensors[key] = {
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        manifest = {
            "experiment": "EXP-0040",
            "source_package": str(args.source),
            "source_manifest_sha256": sha256(args.source / "manifest.json"),
            "shape": {"m": M, "hidden": HIDDEN, "intermediate": INTERMEDIATE},
            "weight_contract": {
                "logical_values": "same signed W4 [-7,7] carrier in both variants",
                "packed_candidate": "no persistent pre-expanded carrier",
                "expanded_control": "persistent raw signed-W4 S8 carrier",
                "per_channel_scale_factorization": (
                    "power-of-two integer-HMX intermediate postscale "
                    "followed by an output-channel integer multiplier"
                ),
                "hmx_intermediate_zero_point": HMX_INTERMEDIATE_ZERO_POINT,
                "bias_and_requant_tables_identical": True,
                "gate_up_interleave": "gate_tile,up_tile for each logical N32 tile",
            },
            "activation_contract": {
                "dtype": "asymmetric_u8",
                "input": qparams["post_attention_norm"],
                "gate": qparams["gate"],
                "up": qparams["up"],
                "middle": qparams["middle"],
                "down": qparams["down"],
                "lut_real_capture_mismatches": mismatch_count,
                "implementation_reference": (
                    "S32 integer accumulation plus exact power-of-two "
                    "integer-HMX rounded right shift, saturated U8 intermediate "
                    "store, and exact per-channel integer output requant"
                ),
            },
            "gate_up_layout": gate_layout,
            "down_layout": down_layout,
            "tensors": tensors,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        if args.output.exists():
            raise FileExistsError(f"refusing to replace existing package: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, args.output)
        print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
