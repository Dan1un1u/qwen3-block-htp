#!/usr/bin/env python3
"""Publish a formal EXP-0149 package with physical cache-boundary goldens.

The cache files are byte-exact device carrier goldens only.  The independent
three-layer final-output references remain unchanged and authoritative; the
manifest records numerical differences between the captured cache carrier and
the retained high-level cache reference.
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


LAYERS = (13, 14, 15)
KV_HEADS = 8
CAPACITY = 72
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        choices=("f16f16", "w4f16", "w4u8"),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0149"),
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
    diagnostics: dict[str, object] = {}
    captured: dict[tuple[int, str], np.ndarray] = {}
    for layer in LAYERS:
        diagnostics[str(layer)] = {}
        for kind in ("k", "v"):
            actual_path = (
                capture /
                f"actual_layer{layer}_replay_{kind}_cache.bin"
            )
            reference_path = (
                source / f"layer{layer}" /
                f"reference_kv_cache_{kind}_{suffix}.bin"
            )
            actual = np.fromfile(actual_path, dtype=dtype)
            reference = np.fromfile(reference_path, dtype=dtype)
            if (
                actual.size != expected_elements
                or reference.size != expected_elements
            ):
                raise ValueError(
                    f"unexpected layer {layer} {kind} cache size"
                )
            difference = (
                actual.astype(np.int16) - reference.astype(np.int16)
                if suffix == "u8"
                else actual.astype(np.float32) -
                    reference.astype(np.float32)
            )
            captured[(layer, kind)] = actual
            diagnostics[str(layer)][kind] = {
                "elements": expected_elements,
                "mismatches": int(np.count_nonzero(difference)),
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "capture_sha256": sha256(actual_path),
                "independent_reference_sha256":
                    sha256(reference_path),
            }

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"{args.recipe}-formal-", dir=args.staging_root
        )
    )
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for layer in LAYERS:
            for kind in ("k", "v"):
                captured[(layer, kind)].tofile(
                    staging / f"layer{layer}" /
                    f"reference_kv_cache_{kind}_{suffix}.bin"
                )
        manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["cache_boundary_golden"] = {
            "kind":
                "device_physical_boundary_golden_not_independent_math_reference",
            "capture": str(capture),
            "source_package": str(source),
            "layers": list(LAYERS),
            "prefix_immutability":
                "validated_online_after_every_append_for_all_layers",
            "independent_three_layer_output_reference_unchanged": True,
            "independent_high_level_cache_diagnostics": diagnostics,
        }
        manifest["files"] = {
            str(path.relative_to(staging)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(
            json.dumps(
                {"output": str(output), "diagnostics": diagnostics},
                indent=2,
            )
        )
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
