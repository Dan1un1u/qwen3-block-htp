#!/usr/bin/env python3
"""Static physical-contract audit for EXP-0147 scan cells."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


PHYSICAL_M = 64
HEAD_DIM = 128
Q_HEADS_PER_GROUP = 2
KV_HEADS = 8
VTCM_BYTES = 8 * 1024 * 1024
MAX_TOTAL_KV = 4097


@dataclass(frozen=True)
class Cell:
    track: str
    logical_m: int
    past_kv: int
    total_kv: int
    padded_kv: int
    physical_chunks: int
    useful_row_utilization: float
    u8_attention_overlay_bytes: int
    fp16_attention_overlay_bytes: int
    u8_cache_bytes: int
    fp16_cache_bytes: int


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def make_cell(track: str, logical_m: int, past_kv: int) -> Cell:
    total_kv = past_kv + logical_m
    padded_kv = align_up(total_kv, 32)
    chunks = align_up(logical_m, PHYSICAL_M) // PHYSICAL_M
    physical_rows = chunks * PHYSICAL_M
    u8_overlay = 3 * padded_kv * HEAD_DIM
    fp16_overlay = 6 * padded_kv * HEAD_DIM
    return Cell(
        track=track,
        logical_m=logical_m,
        past_kv=past_kv,
        total_kv=total_kv,
        padded_kv=padded_kv,
        physical_chunks=chunks,
        useful_row_utilization=logical_m / physical_rows,
        u8_attention_overlay_bytes=u8_overlay,
        fp16_attention_overlay_bytes=fp16_overlay,
        u8_cache_bytes=2 * KV_HEADS * MAX_TOTAL_KV * HEAD_DIM,
        fp16_cache_bytes=4 * KV_HEADS * MAX_TOTAL_KV * HEAD_DIM,
    )


def main() -> None:
    cells = [make_cell("prefill", m, 0) for m in (16, 32, 64, 128)]
    cells += [make_cell("decode", 1, past) for past in (64, 256, 1024, 4096)]
    for cell in cells:
        assert cell.total_kv <= MAX_TOTAL_KV
        assert cell.u8_attention_overlay_bytes < VTCM_BYTES
        assert cell.fp16_attention_overlay_bytes < VTCM_BYTES
    print(
        json.dumps(
            {
                "experiment": "EXP-0147",
                "physical_projection_rows": PHYSICAL_M,
                "vtcm_contract_bytes": VTCM_BYTES,
                "cells": [asdict(cell) for cell in cells],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
