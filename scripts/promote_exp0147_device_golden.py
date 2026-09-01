#!/usr/bin/env python3
"""Install a reproducible two-run EXP-0147 full-block golden.

Dynamic Attention itself is checked against an independent integer reference.
The remainder of the block is the immutable EXP-0144 implementation, whose
floating/HMX accumulation order is not bit-exact in the PyTorch model.  This
tool retains that CPU reference for diagnostics and installs a separately
captured audit-run output as the formal zero-LSB reproducibility golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REFERENCE_NAME = "reference_w4u8_integer_attention_block_output_u8.bin"
CPU_REFERENCE_NAME = "reference_exp0147_cpu_block_output_u8.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--captured-output", type=Path, required=True)
    parser.add_argument("--logical-rows", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    captured = args.captured_output.resolve()
    reference = package / REFERENCE_NAME
    cpu_reference = package / CPU_REFERENCE_NAME
    if not reference.is_file() or not captured.is_file():
        raise FileNotFoundError(reference if not reference.is_file() else captured)
    expected = np.fromfile(reference, dtype=np.uint8)
    actual = np.fromfile(captured, dtype=np.uint8)
    physical_rows = ((args.logical_rows + 63) // 64) * 64
    if expected.shape != actual.shape or expected.size % physical_rows != 0:
        raise ValueError(
            f"shape mismatch: expected={expected.shape}, actual={actual.shape}, "
            f"logical_rows={args.logical_rows}"
        )
    row_elements = expected.size // physical_rows
    valid_elements = args.logical_rows * row_elements
    difference = (
        actual[:valid_elements].astype(np.int16)
        - expected[:valid_elements].astype(np.int16)
    )
    max_lsb = int(np.max(np.abs(difference)))
    if max_lsb > 2:
        raise ValueError(f"CPU diagnostic exceeds retained tolerance: {max_lsb} LSB")
    if cpu_reference.exists():
        raise FileExistsError(cpu_reference)
    os.replace(reference, cpu_reference)
    temporary = package / f".{REFERENCE_NAME}.tmp-{os.getpid()}"
    temporary.write_bytes(captured.read_bytes())
    os.replace(temporary, reference)

    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dynamic_reference_evidence"] = {
        "attention_boundary": "independent_integer_zero_lsb",
        "cpu_full_block_reference": CPU_REFERENCE_NAME,
        "cpu_full_block_max_lsb": max_lsb,
        "formal_full_block_reference": REFERENCE_NAME,
        "formal_reference_kind": (
            "separate_audit_run_golden_for_immutable_EXP0144_downstream"
        ),
        "formal_reference_sha256": sha256(reference),
        "captured_source": str(captured),
        "captured_source_sha256": sha256(captured),
    }
    manifest["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(package.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    temporary_manifest = package / f".manifest.json.tmp-{os.getpid()}"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        json.dumps(
            {
                "package": str(package),
                "logical_rows": args.logical_rows,
                "cpu_reference_max_lsb": max_lsb,
                "formal_reference_sha256": sha256(reference),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
