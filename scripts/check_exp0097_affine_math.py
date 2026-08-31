#!/usr/bin/env python3
"""Exhaustively validate the compact EXP-0097 affine representation."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("lut", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.read_text()
    match = re.search(
        r"qbh_mlp_exact_affine_coefficients\[256\].*?=\s*\{(.*?)\};",
        source, re.S,
    )
    if match is None:
        raise SystemExit("coefficient table not found")
    coefficients = [
        int(value) for value in re.findall(r"UINT16_C\((\d+)\)", match.group(1))
    ]
    if len(coefficients) != 256 or max(coefficients) > 65_535:
        raise SystemExit("invalid coefficient table")
    raw = args.lut.read_bytes()
    if len(raw) != 131_072:
        raise SystemExit("unexpected formal LUT size")
    mismatch_count = 0
    max_lsb = 0
    for gate, magnitude in enumerate(coefficients):
        shift = 15 if gate <= 121 else 14
        multiplier = -magnitude if gate < 125 else magnitude
        for up in range(256):
            actual = (
                multiplier * (up - 110) + (1 << (shift - 1))
            ) // (1 << shift) + 102
            actual = max(0, min(255, actual))
            expected = struct.unpack_from(
                "<H", raw, 2 * (gate * 256 + up)
            )[0]
            error = abs(actual - expected)
            mismatch_count += error != 0
            max_lsb = max(max_lsb, error)
    result = {
        "experiment": "EXP-0097",
        "coefficient_entries": len(coefficients),
        "coefficient_bytes": 2 * len(coefficients),
        "q15_gate_code_max": 121,
        "q14_gate_code_min": 122,
        "gate_zero_point": 125,
        "up_zero_point": 110,
        "middle_zero_point": 102,
        "formal_domain_entries": 65_536,
        "mismatches": mismatch_count,
        "max_lsb": max_lsb,
        "math_gate": "pass" if mismatch_count == 0 else "fail",
    }
    print(json.dumps(result, sort_keys=True))
    if mismatch_count != 0 or max_lsb != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
