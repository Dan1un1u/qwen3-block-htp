#!/usr/bin/env python3
"""Publish an isolated EXP-0152 W4U8 package with independent references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


LAYERS = 28
M = 64
HIDDEN = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_file(destination: Path, source: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copyfile(source, destination)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    reference_dir = args.reference_dir.resolve()
    output = args.output.resolve()
    source_manifest_path = source / "manifest.json"
    reference_manifest_path = reference_dir / "manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not source_manifest_path.is_file() or not reference_manifest_path.is_file():
        raise FileNotFoundError("source or independent reference manifest missing")
    reference_manifest = json.loads(reference_manifest_path.read_text())
    if not reference_manifest.get("passed", False):
        raise ValueError("independent exact replay gate did not pass")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    ))
    try:
        shutil.rmtree(staging)
        try:
            shutil.copytree(source, staging, copy_function=os.link)
            clone_mode = "hardlink"
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source, staging)
            clone_mode = "copy"

        replace_file(
            staging / "reference_w4u8_integer_attention_block_output_u8.bin",
            reference_dir / "reference_replay_output_00_u8.bin",
        )
        for index in range(8):
            logical = np.fromfile(
                reference_dir /
                f"reference_replay_decode_row_{index:02d}_u8.bin",
                dtype=np.uint8,
            )
            if logical.shape != (HIDDEN,):
                raise ValueError(
                    f"decode reference {index} has shape {logical.shape}"
                )
            physical = np.zeros((M, HIDDEN), dtype=np.uint8)
            physical[0] = logical
            destination = staging / f"replay_decode_reference_{index:02d}_u8.bin"
            destination.unlink(missing_ok=True)
            physical.tofile(destination)
        for layer in range(LAYERS):
            for kind in ("k", "v"):
                replace_file(
                    staging / f"layer{layer}" /
                    f"reference_kv_cache_{kind}_u8.bin",
                    reference_dir /
                    f"reference_layer{layer}_replay_{kind}_cache_u8.bin",
                )

        reference_copy = staging / "independent_exact_replay_reference.json"
        reference_copy.write_bytes(reference_manifest_path.read_bytes())
        manifest = json.loads(source_manifest_path.read_text())
        files = manifest["files"]
        changed = [
            "reference_w4u8_integer_attention_block_output_u8.bin",
            *(f"replay_decode_reference_{index:02d}_u8.bin"
              for index in range(8)),
            *(f"layer{layer}/reference_kv_cache_{kind}_u8.bin"
              for layer in range(LAYERS) for kind in ("k", "v")),
            "independent_exact_replay_reference.json",
        ]
        for name in changed:
            path = staging / name
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest["reference_revision"] = {
            "kind": "independent_exact_W4U8_prefill_decode_replay",
            "source_package": str(source),
            "source_manifest_sha256": sha256(source_manifest_path),
            "reference_manifest_sha256": sha256(reference_manifest_path),
            "clone_mode": clone_mode,
            "gate": {
                "step_mismatches": 0,
                "cache_mismatches": 0,
                "maximum_step_lsb": 0,
                "maximum_cache_lsb": 0,
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        print(json.dumps({
            "output": str(output),
            "clone_mode": clone_mode,
            "manifest_sha256": sha256(output / "manifest.json"),
            "changed_reference_files": len(changed),
        }, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
