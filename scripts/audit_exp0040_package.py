#!/usr/bin/env python3
"""Audit EXP-0040 integer references against retained W4U8 captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def stats(reference: bytes, candidate: bytes) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate byte counts differ")
    signed = [actual - expected for expected, actual in zip(reference, candidate)]
    absolute = [abs(delta) for delta in signed]
    elements = len(signed)
    if elements == 0:
        raise ValueError("empty comparison")
    return {
        "elements": elements,
        "mismatches": sum(delta != 0 for delta in signed),
        "maximum_absolute_error_u8_lsb": max(absolute),
        "mean_absolute_error_u8_lsb": sum(absolute) / elements,
        "within_1_lsb_fraction": sum(value <= 1 for value in absolute) / elements,
        "within_2_lsb_fraction": sum(value <= 2 for value in absolute) / elements,
        "mean_signed_error_u8_lsb": sum(signed) / elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()

    comparisons = {}
    for name in ("gate", "up", "middle", "down"):
        candidate_name = (
            "reference_integer_hmx_down_u8.bin"
            if name == "down"
            else f"reference_integer_hmx_{name}_u8.bin"
        )
        reference_name = (
            "reference_w4u8_down_u8.bin"
            if name == "down"
            else f"reference_w4u8_{name}_u8.bin"
        )
        comparisons[name] = stats(
            (args.package / reference_name).read_bytes(),
            (args.package / candidate_name).read_bytes(),
        )
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
