#!/usr/bin/env python3
"""Exhaustively prove that the EXP-0117 packed-pair LUT is exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lut", type=Path)
    args = parser.parse_args()
    raw = args.lut.read_bytes()
    if len(raw) != 65536 * 2:
        raise SystemExit(f"unexpected LUT bytes: {len(raw)}")
    original = list(struct.unpack("<65536H", raw))
    if max(original) > 255:
        raise SystemExit("LUT contains a value outside U8")

    packed = original.copy()
    for gate in range(256):
        for pair in range(128):
            low = packed[gate * 256 + pair * 2]
            high = packed[gate * 256 + pair * 2 + 1]
            packed[gate * 128 + pair] = low | (high << 8)

    mismatches = 0
    for gate in range(256):
        for up in range(256):
            word = packed[gate * 128 + up // 2]
            decoded = (word >> (8 * (up & 1))) & 0xFF
            mismatches += decoded != original[gate * 256 + up]
    if mismatches:
        raise SystemExit(f"packed-pair LUT mismatches: {mismatches}")

    packed_raw = struct.pack("<32768H", *packed[:32768])
    print(json.dumps({
        "mapping_count": 65536,
        "mismatches": 0,
        "original_bytes": len(raw),
        "packed_bytes": len(packed_raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "packed_sha256": hashlib.sha256(packed_raw).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
