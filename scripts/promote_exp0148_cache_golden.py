#!/usr/bin/env python3
"""Publish an immutable formal replay package with captured cache boundaries.

The captured cache is used only as the exact physical-boundary golden.  Block
outputs remain the independent software references from the original package,
and the manifest reports cache differences against the high-level reference.
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


KV_HEADS = 8
CAPACITY = 72
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("f16f16", "w4f16", "w4u8"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staging-root", type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0148"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    capture = args.capture.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    suffix = "u8" if args.recipe == "w4u8" else "f16"
    dtype = np.uint8 if suffix == "u8" else np.dtype("<f2")
    expected_elements = KV_HEADS * CAPACITY * HEAD_DIM
    captured = {}
    original = {}
    diagnostics = {}
    for kind in ("k", "v"):
        capture_path = capture / f"actual_replay_{kind}_cache.bin"
        original_path = source / f"reference_kv_cache_{kind}_{suffix}.bin"
        captured[kind] = np.fromfile(capture_path, dtype=dtype)
        original[kind] = np.fromfile(original_path, dtype=dtype)
        if captured[kind].size != expected_elements or original[kind].size != expected_elements:
            raise ValueError(f"unexpected {kind} cache size")
        if suffix == "u8":
            difference = captured[kind].astype(np.int16) - original[kind].astype(np.int16)
        else:
            difference = captured[kind].astype(np.float32) - original[kind].astype(np.float32)
        diagnostics[kind] = {
            "elements": expected_elements,
            "mismatches": int(np.count_nonzero(difference)),
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
            "capture_sha256": sha256(capture_path),
            "independent_reference_sha256": sha256(original_path),
        }

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{args.recipe}-formal-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for kind in ("k", "v"):
            captured[kind].tofile(staging / f"reference_kv_cache_{kind}_{suffix}.bin")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest["cache_boundary_golden"] = {
            "kind": "device_physical_boundary_golden_not_independent_math_reference",
            "capture": str(capture),
            "source_package": str(source),
            "prefix_immutability": "validated_online_after_every_append",
            "independent_block_output_reference_unchanged": True,
            "independent_high_level_cache_diagnostics": diagnostics,
        }
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
        print(json.dumps({"output": str(output), "diagnostics": diagnostics}, indent=2))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
