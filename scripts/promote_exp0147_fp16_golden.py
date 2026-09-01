#!/usr/bin/env python3
"""Atomically install captured FP16 EXP-0147 cache/output goldens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument("--promote-output", action="store_true")
    args = parser.parse_args()

    package = args.package.resolve()
    capture = args.capture.resolve()
    parent = package.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{package.name}-golden-", dir=parent))
    try:
        shutil.copytree(package, staging, dirs_exist_ok=True)
        replacements = {
            "actual_kv_cache_k_f16.bin": "reference_kv_cache_k_f16.bin",
            "actual_kv_cache_v_f16.bin": "reference_kv_cache_v_f16.bin",
        }
        if args.promote_output:
            replacements["actual_block_output_f16.bin"] = (
                f"reference_{args.recipe}_block_output_f16.bin"
            )
        for source_name, destination_name in replacements.items():
            source = capture / source_name
            destination = staging / destination_name
            if not source.is_file() or source.stat().st_size != destination.stat().st_size:
                raise ValueError(f"capture mismatch: {source} -> {destination}")
            shutil.copy2(source, destination)

        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["device_golden"] = {
            "capture": str(capture),
            "cache_promoted": True,
            "output_promoted": args.promote_output,
            "role": (
                "two_run_reproducibility_golden_for_dynamic_cell; "
                "independent_fp16_boundary_error_is_reported_separately"
            ),
        }
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        backup = parent / f".{package.name}.before-golden-{os.getpid()}"
        os.rename(package, backup)
        try:
            os.rename(staging, package)
        except Exception:
            os.rename(backup, package)
            raise
        shutil.rmtree(backup)
        print(json.dumps({"package": str(package), **manifest["device_golden"]}, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()

