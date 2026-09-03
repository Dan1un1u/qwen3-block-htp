#!/usr/bin/env python3
"""Publish byte-exact 28-layer M64 + 40-token W4U8 replay references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from reference_w4u8_hmx import HmxU8Converter
from verify_exp0152_w4u8_prefill import (
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    M,
    run_layer_prefill,
)
from verify_exp0152_w4u8_replay import run_layer_decode


CAPACITY = 104
DECODE_STEPS = CAPACITY - M


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"{path}: got {value.size} elements, expected {expected}")
    return value.reshape(shape)


def replace_array(path: Path, value: np.ndarray) -> None:
    path.unlink(missing_ok=True)
    np.ascontiguousarray(value).tofile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    prior = args.prior.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(source / "manifest.json")
    if not (prior / "manifest.json").is_file():
        raise FileNotFoundError(prior / "manifest.json")

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

        immutable_names = (
            "qparams_u8.bin",
            "input_norm_weight_f16.bin",
            "post_norm_weight_f16.bin",
            "q_norm_weight_f16.bin",
            "k_norm_weight_f16.bin",
            "attention_config_all_groups.bin",
            "silu_up_lut_u16.bin",
            *(f"{name}_weight_w4_hmx.bin" for name in
              ("q", "k", "v", "o", "gate", "up", "down")),
            *(f"{name}_weight_w4_scale_f32.bin" for name in
              ("q", "k", "v", "o", "gate", "up", "down")),
        )
        for layer in range(LAYERS):
            for name in immutable_names:
                if sha256(source / f"layer{layer}" / name) != sha256(
                    prior / f"layer{layer}" / name
                ):
                    raise ValueError(
                        f"immutable layer payload changed: layer{layer}/{name}"
                    )

        for name in (
            "reference_w4u8_block_input_u8.bin",
            *(f"replay_decode_input_{index:02d}_u8.bin"
              for index in range(8)),
            *(f"replay_decode_rope_cos_{index:02d}_f16.bin"
              for index in range(8)),
            *(f"replay_decode_rope_sin_{index:02d}_f16.bin"
              for index in range(8)),
        ):
            if sha256(source / name) != sha256(prior / name):
                raise ValueError(f"retained replay prefix changed: {name}")

        converter = HmxU8Converter(args.converter)
        layer_packages = [staging / f"layer{layer}" for layer in range(LAYERS)]
        k_caches = [
            np.zeros((KV_HEADS, CAPACITY, HEAD_DIM), dtype=np.uint8)
            for _ in range(LAYERS)
        ]
        v_caches = [np.zeros_like(k_caches[0]) for _ in range(LAYERS)]
        changed: list[Path] = []
        step_hashes: list[str] = []

        hidden = read_exact(
            staging / "reference_w4u8_block_input_u8.bin",
            np.uint8, (M, HIDDEN),
        )
        cosine = read_exact(
            staging / "rope_cos_f16.bin", np.dtype("<f2"), (M, HEAD_DIM)
        )
        sine = read_exact(
            staging / "rope_sin_f16.bin", np.dtype("<f2"), (M, HEAD_DIM)
        )
        for layer, layer_package in enumerate(layer_packages):
            hidden, k_rope, v_heads = run_layer_prefill(
                hidden, layer_package, cosine, sine, converter
            )
            k_caches[layer][:, :M] = k_rope.transpose(1, 0, 2)
            v_caches[layer][:, :M] = v_heads.transpose(1, 0, 2)
            print(f"exact prefill layer {layer + 1}/{LAYERS}", flush=True)
        prefill_path = (
            staging / "reference_w4u8_integer_attention_block_output_u8.bin"
        )
        replace_array(prefill_path, hidden)
        changed.append(prefill_path)
        if sha256(prefill_path) != sha256(
            prior / "reference_w4u8_integer_attention_block_output_u8.bin"
        ):
            raise ValueError("exact prefill output changed from EXP-0152 authority")

        for decode_index in range(DECODE_STEPS):
            hidden = read_exact(
                staging / f"replay_decode_input_{decode_index:02d}_u8.bin",
                np.uint8, (M, HIDDEN),
            )[:1].copy()
            cosine = read_exact(
                staging / f"replay_decode_rope_cos_{decode_index:02d}_f16.bin",
                np.dtype("<f2"), (M, HEAD_DIM),
            )
            sine = read_exact(
                staging / f"replay_decode_rope_sin_{decode_index:02d}_f16.bin",
                np.dtype("<f2"), (M, HEAD_DIM),
            )
            past_tokens = M + decode_index
            for layer, layer_package in enumerate(layer_packages):
                hidden, k_rope, v_heads = run_layer_decode(
                    hidden, layer_package, cosine, sine,
                    k_caches[layer], v_caches[layer], past_tokens,
                    converter,
                )
            # Preserve the accepted replay carrier contract: only row zero is
            # logical during decode and every inactive physical row is zero.
            physical = np.zeros((M, HIDDEN), dtype=np.uint8)
            physical[0] = hidden[0]
            reference_path = staging / (
                f"replay_decode_reference_{decode_index:02d}_u8.bin"
            )
            replace_array(reference_path, physical)
            changed.append(reference_path)
            step_hashes.append(sha256(reference_path))
            if decode_index < 8 and sha256(reference_path) != sha256(
                prior / f"replay_decode_reference_{decode_index:02d}_u8.bin"
            ):
                raise ValueError(
                    f"exact retained decode output changed at {decode_index}"
                )
            print(
                f"exact decode {decode_index + 1}/{DECODE_STEPS}", flush=True
            )

        for layer in range(LAYERS):
            for kind, cache in (("k", k_caches[layer]),
                                ("v", v_caches[layer])):
                path = staging / f"layer{layer}/reference_kv_cache_{kind}_u8.bin"
                replace_array(path, cache)
                changed.append(path)
                prior_cache = read_exact(
                    prior / f"layer{layer}/reference_kv_cache_{kind}_u8.bin",
                    np.uint8, (KV_HEADS, 72, HEAD_DIM),
                )
                if not np.array_equal(cache[:, :72], prior_cache):
                    raise ValueError(
                        f"exact retained cache prefix changed: layer{layer}/{kind}"
                    )

        source_manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        files = source_manifest.setdefault("files", {})
        for path in changed:
            relative = str(path.relative_to(staging))
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        exact_record = {
            "kind": "independent_exact_W4U8_28_layer_M64_plus_40_decode",
            "converter": str(args.converter.resolve()),
            "capacity": CAPACITY,
            "decode_steps": DECODE_STEPS,
            "retained_exp0152_prefix_bytes_exact": True,
            "step_reference_sha256": step_hashes,
        }
        source_manifest["experiment"] = "EXP-0162"
        source_manifest["exact_reference_revision"] = exact_record
        source_manifest["clone_mode"] = clone_mode
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        print(json.dumps({
            "experiment": "EXP-0162",
            "output": str(output),
            "clone_mode": clone_mode,
            "decode_steps": DECODE_STEPS,
            "retained_prefix_gate": "PASS",
            "manifest_sha256": sha256(output / "manifest.json"),
        }, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
