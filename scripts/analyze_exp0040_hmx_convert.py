#!/usr/bin/env python3
"""Compare one EXP-0040 HMX conversion probe with simple scale models."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("lower_word", type=lambda text: int(text, 0))
    parser.add_argument("--bias", type=int, default=0)
    args = parser.parse_args()

    output = args.output.read_bytes()
    if len(output) != 64 * 32:
        raise ValueError(f"unexpected output size: {len(output)}")
    half_scale = struct.unpack(
        "<e", struct.pack("<H", args.lower_word & 0xFFFF)
    )[0]
    scale = half_scale / 512.0
    observed: dict[int, set[int]] = {}
    for row in range(64):
        for channel in range(32):
            accumulator = (row * 4 + channel // 8) & 0xFF
            accumulator += args.bias
            observed.setdefault(accumulator, set()).add(
                output[row * 32 + channel]
            )

    print(
        f"lower_word=0x{args.lower_word:08x} half_scale={half_scale} "
        f"effective_scale={scale} bias={args.bias}"
    )
    ambiguous = [
        (accumulator, sorted(values))
        for accumulator, values in sorted(observed.items())
        if len(values) != 1
    ]
    print(f"ambiguous_mappings={len(ambiguous)} samples={ambiguous[:8]}")

    models = {
        "floor": lambda value: math.floor(value * scale),
        "ceil": lambda value: math.ceil(value * scale),
        "round_ties_even": lambda value: round(value * scale),
        "trunc": lambda value: math.trunc(value * scale),
    }
    for name, model in models.items():
        mismatches = []
        for accumulator, values in sorted(observed.items()):
            expected = max(0, min(255, model(accumulator)))
            actual = next(iter(values))
            if actual != expected:
                mismatches.append((accumulator, expected, actual))
        print(
            f"model={name} mismatches={len(mismatches)} "
            f"samples={mismatches[:16]}"
        )

    mapping = [
        (accumulator, next(iter(observed[accumulator])))
        for accumulator in sorted(observed)
        if accumulator % 8 == 0
    ]
    print(f"mapping_stride8={mapping}")


if __name__ == "__main__":
    main()
