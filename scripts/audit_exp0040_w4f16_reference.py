#!/usr/bin/env python3
"""Report EXP-0040 integer-U8 outputs against the selected W4F16 reference."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def compare(candidate: bytes, reference_path: Path,
            scale: float, zero_point: int) -> dict[str, object]:
    reference_bytes = reference_path.read_bytes()
    if len(reference_bytes) != len(candidate) * 2:
        raise ValueError(f"wrong FP16 reference size: {reference_path}")
    count = len(candidate)
    dot = candidate_norm = reference_norm = 0.0
    absolute_sum = squared_sum = 0.0
    maximum = 0.0
    for quantized, (reference,) in zip(
            candidate, struct.iter_unpack("<e", reference_bytes)):
        actual = (quantized - zero_point) * scale
        delta = actual - reference
        absolute = abs(delta)
        absolute_sum += absolute
        squared_sum += delta * delta
        maximum = max(maximum, absolute)
        dot += actual * reference
        candidate_norm += actual * actual
        reference_norm += reference * reference
    cosine = dot / math.sqrt(candidate_norm * reference_norm)
    return {
        "elements": count,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": absolute_sum / count,
        "root_mean_square_error": math.sqrt(squared_sum / count),
        "cosine_similarity": cosine,
        "candidate_saturation_low_fraction": candidate.count(0) / count,
        "candidate_saturation_high_fraction": candidate.count(255) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("w4f16_source", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.package / "manifest.json").read_text())
    qparams = manifest["activation_contract"]
    candidate_files = {
        "gate": "reference_integer_hmx_gate_u8.bin",
        "up": "reference_integer_hmx_up_u8.bin",
        "middle": "reference_integer_hmx_middle_u8.bin",
        "down": "reference_integer_hmx_down_u8.bin",
    }
    result = {}
    for name, filename in candidate_files.items():
        result[name] = compare(
            (args.package / filename).read_bytes(),
            args.w4f16_source / f"reference_w4f16_{name}_f16.bin",
            float(qparams[name]["scale"]),
            int(qparams[name]["zero_point"]),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
