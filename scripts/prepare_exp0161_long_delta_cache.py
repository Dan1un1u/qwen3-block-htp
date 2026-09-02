#!/usr/bin/env python3
"""Create EXP-0161 long-context W4U8 delta-cache scan packages.

The independent block and cache references come from EXP-0147.  This script
changes only the persistent cache carrier: an immutable M64 HMX-native base is
followed by contiguous U8 rows for tokens 64..capacity-1.
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

from prepare_exp0155_hmx_cache import (
    BIAS_BYTES,
    HEADS,
    HEAD_DIM,
    PADDED,
    load_configs,
    pack_k,
    pack_v,
)


BASE_TOKENS = 64
WEIGHT_BYTES = 1024
HEAD_DIM_TILES = HEAD_DIM // 32
BASE_K_TILES = BASE_TOKENS // 32
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


def pack_base(rows: np.ndarray, kind: str, config: tuple[int, ...]) -> bytes:
    if kind == "k":
        full = pack_k(rows, BASE_TOKENS, config)
        full_weight_bytes = PADDED * HEAD_DIM
        return (
            full[: BASE_TOKENS * HEAD_DIM]
            + full[full_weight_bytes:
                   full_weight_bytes + BASE_K_TILES * BIAS_BYTES]
        )

    full = np.frombuffer(pack_v(rows, BASE_TOKENS, config), dtype=np.uint8)
    full_weight_bytes = PADDED * HEAD_DIM
    compact_weight = (
        full[:full_weight_bytes]
        .reshape(HEAD_DIM_TILES, PADDED // 32, WEIGHT_BYTES)
        [:, :BASE_K_TILES]
        .reshape(-1)
    )
    return compact_weight.tobytes() + full[full_weight_bytes:].tobytes()


def pack_delta(
    rows: np.ndarray,
    kind: str,
    valid_length: int,
    config: tuple[int, ...],
) -> bytes:
    capacity = rows.shape[0]
    if capacity <= BASE_TOKENS or not BASE_TOKENS <= valid_length <= capacity:
        raise ValueError((capacity, valid_length))
    journal = np.zeros((capacity - BASE_TOKENS, HEAD_DIM), dtype=np.uint8)
    journal[: valid_length - BASE_TOKENS] = rows[
        BASE_TOKENS:valid_length
    ]
    return pack_base(rows, kind, config) + journal.tobytes(order="C")


def publish_one(source: Path, output: Path, past_length: int) -> dict[str, object]:
    capacity = past_length + 1
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
            initial = b"".join(
                pack_delta(initial_rows[head], kind, past_length, configs[head])
                for head in range(HEADS)
            )
            reference = b"".join(
                pack_delta(reference_rows[head], kind, capacity, configs[head])
                for head in range(HEADS)
            )
            if len(initial) != len(reference):
                raise AssertionError(f"{kind} cache/reference size mismatch")
            initial_path = staging / f"kv_cache_{kind}_hmx_u8_delta.bin"
            reference_path = (
                staging / f"reference_kv_cache_{kind}_hmx_u8_delta.bin"
            )
            initial_path.write_bytes(initial)
            reference_path.write_bytes(reference)
            generated.extend((initial_path, reference_path))
            sizes[kind] = len(initial)

        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        for path in generated:
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest.update({
            "experiment": "EXP-0161",
            "source_package": str(source),
            "source_manifest_sha256": sha256(source_manifest_path),
            "clone_mode": clone_mode,
            "cache_abi": {
                "format_version": 2,
                "logical_capacity": capacity,
                "physical_padded_capacity": ((capacity + 31) // 32) * 32,
                "base_tokens": BASE_TOKENS,
                "initial_valid_tokens": past_length,
                "k_format": "hmx_u8_k_weight_delta_v2",
                "v_format": "hmx_u8_v_weight_delta_v2",
                "k_bytes": sizes["k"],
                "v_bytes": sizes["v"],
                "reference_generation": (
                    "EXP0147 independent logical cache repacked as immutable "
                    "M64 HMX base plus contiguous U8 journal"
                ),
            },
        })
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        return {
            "past_length": past_length,
            "capacity": capacity,
            "output": str(output),
            "clone_mode": clone_mode,
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
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0147"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0161"),
    )
    parser.add_argument(
        "--length", type=int, action="append", choices=SUPPORTED_LENGTHS,
    )
    args = parser.parse_args()
    lengths = tuple(args.length or SUPPORTED_LENGTHS)
    results = []
    for past_length in lengths:
        source = source_package(args.source_root.resolve(), past_length)
        output = args.output_root.resolve() / (
            f"decode_l{past_length}_w4u8_hmx_delta_v2"
        )
        result = publish_one(source, output, past_length)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"experiment": "EXP-0161", "packages": results}, indent=2))


if __name__ == "__main__":
    main()
