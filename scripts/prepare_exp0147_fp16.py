#!/usr/bin/env python3
"""Prepare FP16-boundary shape/KV packages for EXP-0147.

The fixture deliberately preserves the accepted M=64 block input, weights,
and recipe-specific output files.  Smaller prefill cells zero-pad physical
rows, M128 repeats the accepted M64 fixture, and decode repeats the accepted
post-RoPE K/V reference to construct a deterministic persistent cache.
Dynamic-cell block outputs are placeholders until the device boundary audit
and independent FP16 comparison are captured.
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
KV_HIDDEN = 1024
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument(
        "--cell",
        choices=(
            "prefill_m16", "prefill_m32", "prefill_m64", "prefill_m128",
            "decode_l64", "decode_l256", "decode_l1024", "decode_l4096",
        ),
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
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


def cell_shape(cell: str) -> tuple[str, int, int, int]:
    if cell.startswith("prefill_m"):
        logical_m = int(cell.removeprefix("prefill_m"))
        return "prefill", logical_m, 0, logical_m
    past = int(cell.removeprefix("decode_l"))
    return "decode", 1, past, past + 1


def head_major(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        rows.reshape(rows.shape[0], KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    mode, logical_m, past, capacity = cell_shape(args.cell)
    chunks = (logical_m + PHYSICAL_M - 1) // PHYSICAL_M
    if output.exists():
        raise FileExistsError(f"refusing to replace existing package: {output}")

    input_base = np.fromfile(source / "block_input_f16.bin", dtype="<f2").reshape(
        PHYSICAL_M, HIDDEN
    )
    output_base = np.fromfile(
        source / f"reference_{args.recipe}_block_output_f16.bin", dtype="<f2"
    ).reshape(PHYSICAL_M, HIDDEN)
    k_base = np.fromfile(
        source / f"reference_{args.recipe}_k_rope_f16.bin", dtype="<f2"
    ).reshape(PHYSICAL_M, KV_HIDDEN)
    v_base = np.fromfile(
        source / f"reference_{args.recipe}_v_f16.bin", dtype="<f2"
    ).reshape(PHYSICAL_M, KV_HIDDEN)

    if mode == "prefill" and logical_m <= PHYSICAL_M:
        block_input = np.zeros_like(input_base)
        block_input[:logical_m] = input_base[:logical_m]
        block_reference = output_base.copy()
        cache_k_reference = head_major(k_base[:logical_m])
        cache_v_reference = head_major(v_base[:logical_m])
        cache_k_initial = np.zeros_like(cache_k_reference)
        cache_v_initial = np.zeros_like(cache_v_reference)
    elif mode == "prefill":
        block_input = np.ascontiguousarray(
            np.concatenate((input_base, input_base), axis=0)
        )
        block_reference = np.ascontiguousarray(
            np.concatenate((output_base, output_base), axis=0)
        )
        cache_k_reference = head_major(
            np.concatenate((k_base, k_base), axis=0)
        )
        cache_v_reference = head_major(
            np.concatenate((v_base, v_base), axis=0)
        )
        cache_k_initial = np.zeros_like(cache_k_reference)
        cache_v_initial = np.zeros_like(cache_v_reference)
    else:
        block_input = np.zeros_like(input_base)
        block_input[0] = input_base[0]
        block_reference = np.zeros_like(output_base)
        block_reference[0] = output_base[0]
        indices = np.arange(past) % PHYSICAL_M
        k_initial_rows = k_base[indices]
        v_initial_rows = v_base[indices]
        cache_k_initial = np.zeros((KV_HEADS, capacity, HEAD_DIM), dtype="<f2")
        cache_v_initial = np.zeros_like(cache_k_initial)
        cache_k_initial[:, :past] = head_major(k_initial_rows)
        cache_v_initial[:, :past] = head_major(v_initial_rows)
        cache_k_reference = cache_k_initial.copy()
        cache_v_reference = cache_v_initial.copy()
        cache_k_reference[:, past] = head_major(k_base[:1])[:, 0]
        cache_v_reference[:, past] = head_major(v_base[:1])[:, 0]

    rope_cos = np.fromfile(source / "rope_cos_f16.bin", dtype=np.uint8)
    rope_sin = np.fromfile(source / "rope_sin_f16.bin", dtype=np.uint8)
    if chunks == 2:
        rope_cos = np.concatenate((rope_cos, rope_cos))
        rope_sin = np.concatenate((rope_sin, rope_sin))

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="fp16-package-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        block_input.tofile(staging / "block_input_f16.bin")
        block_reference.tofile(
            staging / f"reference_{args.recipe}_block_output_f16.bin"
        )
        rope_cos.tofile(staging / "rope_cos_f16.bin")
        rope_sin.tofile(staging / "rope_sin_f16.bin")
        cache_k_initial.tofile(staging / "kv_cache_k_f16.bin")
        cache_v_initial.tofile(staging / "kv_cache_v_f16.bin")
        cache_k_reference.tofile(staging / "reference_kv_cache_k_f16.bin")
        cache_v_reference.tofile(staging / "reference_kv_cache_v_f16.bin")

        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "experiment": "EXP-0147",
                "execution_unit": "qwen3_layer14_shape_kv_scan",
                "source": str(source),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "shape_scan": {
                    "cell": args.cell,
                    "recipe": args.recipe,
                    "mode": mode,
                    "logical_m": logical_m,
                    "physical_chunks": chunks,
                    "initial_kv_length": past,
                    "kv_cache_capacity": capacity,
                    "cache_layout": "head_major_[8,capacity,128]",
                    "fixture": "accepted_M64_fixture_repeated",
                    "dynamic_output_reference":
                        mode == "decode" or logical_m > PHYSICAL_M,
                },
            }
        )
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(json.dumps({"output": str(output), **manifest["shape_scan"]}, indent=2))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()

