#!/usr/bin/env python3
"""Publish EXP-0162 control and dynamic segmented-cache packages."""

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
    load_configs,
    pack_k,
    pack_v,
)
from prepare_exp0161_segmented_cache import (
    SEGMENT_K_BYTES,
    SEGMENT_TOKENS,
    SEGMENT_WEIGHT_BYTES,
    TAIL_BYTES,
    V_SEGMENT_BLOCK_SEGMENTS,
    pack_k_segment,
    pack_v_segment,
)


LAYERS = 28
EXPERIMENT = "EXP-0162"
PREFILL = 64
CAPACITY = 104
DECODE_STEPS = CAPACITY - PREFILL
DELTA_ROWS = CAPACITY - PREFILL
MAX_SEGMENTS = (CAPACITY - 1) // SEGMENT_TOKENS
V_BIAS_BYTES = HEAD_DIM_TILES * BIAS_BYTES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delta_cache(
    rows: np.ndarray, kind: str, valid: int, config: tuple[int, ...]
) -> bytes:
    if kind == "k":
        packed = pack_k(rows, PREFILL, config)
        full_weight_bytes = 96 * HEAD_DIM
        base = (
            packed[:PREFILL * HEAD_DIM]
            + packed[full_weight_bytes:
                     full_weight_bytes + PREFILL // 32 * BIAS_BYTES]
        )
    else:
        packed = np.frombuffer(
            pack_v(rows, PREFILL, config), dtype=np.uint8
        )
        full_weight_bytes = 96 * HEAD_DIM
        base_weight = (
            packed[:full_weight_bytes]
            .reshape(HEAD_DIM_TILES, 3, 1024)[:, :2]
            .reshape(-1)
            .tobytes()
        )
        base = base_weight + packed[
            full_weight_bytes:full_weight_bytes + V_BIAS_BYTES
        ].tobytes()
    journal = np.zeros((DELTA_ROWS, HEAD_DIM), dtype=np.uint8)
    if valid > PREFILL:
        journal[:valid - PREFILL] = rows[PREFILL:valid]
    return base + journal.tobytes()


def segmented_cache(
    rows: np.ndarray, kind: str, valid: int, config: tuple[int, ...],
    tail: np.ndarray,
    prepacked: tuple[list[bytes], bytes] | None = None,
) -> bytes:
    sealed_segments = valid // SEGMENT_TOKENS
    if prepacked is None:
        prepacked = prepack_segments(rows, kind, config)
    sealed_all, v_bias = prepacked
    sealed = sealed_all[:sealed_segments]

    if kind == "k":
        prefix = b"".join(sealed)
        prefix += bytes((MAX_SEGMENTS - sealed_segments) * SEGMENT_K_BYTES)
        return prefix + tail.tobytes()

    blocks: list[bytes] = []
    for block_first in range(0, MAX_SEGMENTS, V_SEGMENT_BLOCK_SEGMENTS):
        block_count = min(
            V_SEGMENT_BLOCK_SEGMENTS, MAX_SEGMENTS - block_first
        )
        for output_tile in range(HEAD_DIM_TILES):
            for segment in range(block_first, block_first + block_count):
                if segment < sealed_segments:
                    blocks.append(sealed[segment][
                        output_tile * 1024:(output_tile + 1) * 1024
                    ])
                else:
                    blocks.append(bytes(1024))
    return b"".join(blocks) + v_bias + tail.tobytes()


def prepack_segments(
    rows: np.ndarray, kind: str, config: tuple[int, ...]
) -> tuple[list[bytes], bytes]:
    sealed: list[bytes] = []
    v_bias = b""
    for segment in range(MAX_SEGMENTS):
        first = segment * SEGMENT_TOKENS
        segment_rows = rows[first:first + SEGMENT_TOKENS]
        if kind == "k":
            sealed.append(pack_k_segment(segment_rows, config))
        else:
            weight, bias = pack_v_segment(segment_rows, config)
            sealed.append(weight)
            if not v_bias:
                v_bias = bias
    if kind == "v" and not v_bias:
        _, v_bias = pack_v_segment(
            np.zeros((SEGMENT_TOKENS, HEAD_DIM), dtype=np.uint8), config
        )
    return sealed, v_bias


def publish(source: Path, output: Path, mode: str) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
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

        generated: list[Path] = []
        if mode == "control":
            k_head_bytes = PREFILL * HEAD_DIM + 2 * BIAS_BYTES + \
                DELTA_ROWS * HEAD_DIM
            v_head_bytes = PREFILL * HEAD_DIM + V_BIAS_BYTES + \
                DELTA_ROWS * HEAD_DIM
            suffix = "u8_delta"
        else:
            k_head_bytes = MAX_SEGMENTS * SEGMENT_K_BYTES + TAIL_BYTES
            v_head_bytes = MAX_SEGMENTS * SEGMENT_WEIGHT_BYTES + \
                V_BIAS_BYTES + TAIL_BYTES
            suffix = "u8_segmented"

        for layer_index in range(LAYERS):
            layer = staging / f"layer{layer_index}"
            configs = load_configs(
                layer / "attention_config_all_groups.bin"
            )
            logical = {
                kind: np.fromfile(
                    layer / f"reference_kv_cache_{kind}_u8.bin",
                    dtype=np.uint8,
                ).reshape(HEADS, CAPACITY, HEAD_DIM)
                for kind in ("k", "v")
            }
            for kind in ("k", "v"):
                head_bytes = k_head_bytes if kind == "k" else v_head_bytes
                prepacked = (
                    [prepack_segments(
                        logical[kind][head], kind, configs[head]
                    ) for head in range(HEADS)]
                    if mode == "candidate" else None
                )
                initial = layer / f"kv_cache_{kind}_hmx_{suffix}.bin"
                initial.write_bytes(bytes(HEADS * head_bytes))
                generated.append(initial)
                tails = np.zeros(
                    (HEADS, SEGMENT_TOKENS, HEAD_DIM), dtype=np.uint8
                )
                for step in range(DECODE_STEPS + 1):
                    valid = PREFILL + step
                    if step:
                        token = valid - 1
                        tails[:, token % SEGMENT_TOKENS] = logical[kind][:, token]
                    payload = b"".join(
                        (delta_cache(
                            logical[kind][head], kind, valid, configs[head]
                        ) if mode == "control" else segmented_cache(
                            logical[kind][head], kind, valid, configs[head],
                            tails[head],
                            prepacked[head],
                        ))
                        for head in range(HEADS)
                    )
                    if len(payload) != HEADS * head_bytes:
                        raise AssertionError(
                            f"{mode} {kind} cache byte mismatch"
                        )
                    reference = layer / (
                        f"reference_kv_cache_{kind}_hmx_{suffix}_"
                        f"step{step:02d}.bin"
                    )
                    reference.write_bytes(payload)
                    generated.append(reference)
            print(
                f"{mode} cache layer {layer_index + 1}/{LAYERS}",
                flush=True,
            )

        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        for path in generated:
            files[str(path.relative_to(staging))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest.update({
            "experiment": EXPERIMENT,
            "source_package": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "clone_mode": clone_mode,
            "cache_abi": {
                "decode_session_abi_version": 4,
                "mode": mode,
                "active_layers": list(range(LAYERS)),
                "logical_capacity": CAPACITY,
                "prefill_tokens": PREFILL,
                "decode_steps": DECODE_STEPS,
                "segment_tokens": SEGMENT_TOKENS if mode == "candidate" else None,
                "max_segment_slots": MAX_SEGMENTS if mode == "candidate" else None,
                "k_format": (
                    "hmx_u8_k_segmented_v4_dynamic" if mode == "candidate"
                    else "hmx_u8_k_weight_delta_v2"
                ),
                "v_format": (
                    "hmx_u8_v_segmented_v4_dynamic" if mode == "candidate"
                    else "hmx_u8_v_weight_delta_v2"
                ),
                "k_bytes_per_layer": HEADS * k_head_bytes,
                "v_bytes_per_layer": HEADS * v_head_bytes,
                "reference_steps": list(range(DECODE_STEPS + 1)),
                "reference_generation": (
                    "independent exact logical cache repacked at every state; "
                    "dynamic tail bytes retain stale rows across a seal exactly "
                    "as the runtime ABI does"
                ),
            },
        })
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        return {
            "mode": mode,
            "output": str(output),
            "clone_mode": clone_mode,
            "k_bytes_per_layer": HEADS * k_head_bytes,
            "v_bytes_per_layer": HEADS * v_head_bytes,
            "manifest_sha256": sha256(output / "manifest.json"),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    global CAPACITY, DECODE_STEPS, DELTA_ROWS, MAX_SEGMENTS, EXPERIMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--experiment", default="EXP-0162")
    parser.add_argument("--control-name", default="control_delta_capacity104")
    parser.add_argument(
        "--candidate-name", default="candidate_segmented_capacity104"
    )
    parser.add_argument(
        "--mode", choices=("control", "candidate"), action="append"
    )
    args = parser.parse_args()
    if args.capacity <= PREFILL:
        raise ValueError("capacity must exceed the M64 prefill")
    if args.decode_steps < 1 or PREFILL + args.decode_steps > args.capacity:
        raise ValueError("decode steps exceed the declared cache capacity")
    CAPACITY = args.capacity
    DECODE_STEPS = args.decode_steps
    DELTA_ROWS = CAPACITY - PREFILL
    MAX_SEGMENTS = (CAPACITY - 1) // SEGMENT_TOKENS
    EXPERIMENT = args.experiment
    source = args.source.resolve()
    root = args.output_root.resolve()
    modes = tuple(args.mode or ("control", "candidate"))
    names = {
        "control": args.control_name,
        "candidate": args.candidate_name,
    }
    results = [publish(source, root / names[mode], mode) for mode in modes]
    print(json.dumps({"experiment": EXPERIMENT, "packages": results},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
