#!/usr/bin/env python3
"""Audit EXP-0163 package shapes and immutable six-seal carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_exp0155_hmx_cache import (
    BIAS_BYTES,
    HEADS,
    HEAD_DIM_TILES,
)
from prepare_exp0161_segmented_cache import (
    SEGMENT_K_BYTES,
    SEGMENT_WEIGHT_BYTES,
    TAIL_BYTES,
)


CAPACITY = 257
DECODE_STEPS = 192
MAX_SEGMENTS = 8
V_BIAS_BYTES = HEAD_DIM_TILES * BIAS_BYTES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for mode, name, suffix in (
        ("control", "control_delta_capacity257", "u8_delta"),
        ("candidate", "candidate_segmented_capacity257", "u8_segmented"),
    ):
        package = root / name
        manifest = json.loads(
            (package / "manifest.json").read_text(encoding="utf-8")
        )
        abi = manifest["cache_abi"]
        assert manifest["experiment"] == "EXP-0163"
        assert abi["mode"] == mode
        assert abi["logical_capacity"] == CAPACITY
        assert abi["decode_steps"] == DECODE_STEPS
        assert abi["reference_steps"] == list(range(DECODE_STEPS + 1))
        if mode == "candidate":
            assert abi["max_segment_slots"] == MAX_SEGMENTS
        for layer in range(28):
            for kind in ("k", "v"):
                references = sorted((package / f"layer{layer}").glob(
                    f"reference_kv_cache_{kind}_hmx_{suffix}_step*.bin"
                ))
                assert len(references) == DECODE_STEPS + 1
                expected = int(abi[f"{kind}_bytes_per_layer"])
                assert all(path.stat().st_size == expected
                           for path in references)
    print("manifest_and_reference_shape_gate=PASS")

    package = root / "candidate_segmented_capacity257" / "layer14"
    k_head_bytes = MAX_SEGMENTS * SEGMENT_K_BYTES + TAIL_BYTES
    v_head_bytes = (
        MAX_SEGMENTS * SEGMENT_WEIGHT_BYTES + V_BIAS_BYTES + TAIL_BYTES
    )
    seal_steps = (0, 0, 32, 64, 96, 128, 160, 192)
    for kind, head_bytes in (("k", k_head_bytes), ("v", v_head_bytes)):
        references = [
            (package /
             f"reference_kv_cache_{kind}_hmx_u8_segmented_step{step:02d}.bin"
             ).read_bytes()
            for step in range(DECODE_STEPS + 1)
        ]
        for segment, seal_step in enumerate(seal_steps):
            for head in range(HEADS):
                if kind == "k":
                    offset = (
                        head * head_bytes + segment * SEGMENT_K_BYTES
                    )
                    slices = ((offset, SEGMENT_K_BYTES),)
                else:
                    slices = tuple(
                        (head * head_bytes +
                         (tile * MAX_SEGMENTS + segment) * 1024, 1024)
                        for tile in range(HEAD_DIM_TILES)
                    )
                for offset, size in slices:
                    expected = references[seal_step][offset:offset + size]
                    assert any(expected)
                    for step in range(seal_step, DECODE_STEPS + 1):
                        assert references[step][offset:offset + size] == expected
    print("six_seal_immutable_segment_gate=PASS")


if __name__ == "__main__":
    main()
