#!/usr/bin/env python3
"""Print compact EXP-0160 replay records from JSONL on stdin."""

from __future__ import annotations

import json
import sys


FIELDS = (
    "replay_step",
    "mode",
    "host_wall_ns",
    "w4u8_delta_reconstruction_mode",
    "attention_ticks",
    "scan_dynamic_attention_ticks",
    "scan_cache_stage_ticks",
    "u8_attention_k_pack_ticks",
    "u8_attention_v_pack_ticks",
    "scan_cache_ddr_read_bytes",
    "scan_cache_dma_descriptor_count",
    "hmx_command_count",
    "hmx_u8s8_tile_pair_count",
    "vtcm_acquired_bytes",
    "intermediate_spill_fill_count",
)


for line in sys.stdin:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("record") != "replay_profile":
        continue
    brief = {field: record.get(field) for field in FIELDS}
    brief["exact"] = all(
        int(record.get(field, -1)) == 0
        for field in (
            "output_mismatches",
            "output_max_lsb",
            "cache_mismatches",
            "cache_prefix_mismatches",
            "cache_structure_mismatches",
        )
    )
    print(json.dumps(brief, separators=(",", ":")))
