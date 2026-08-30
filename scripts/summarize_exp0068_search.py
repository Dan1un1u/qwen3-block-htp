#!/usr/bin/env python3
"""Select the EXP-0068 Attention HVX context candidate."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


MODES = ("control", "context5", "context6")
FIELDS = (
    "host_wall_ns_per_block",
    "qkv_projection_ticks",
    "attention_ticks",
    "attention_qk_norm_pool_wait_ticks",
    "u8_attention_qk_norm_rope_ticks",
    "u8_attention_softmax_ticks",
    "u8_attention_pipeline_wait_ticks",
    "gate_up_ticks",
    "down_ticks",
)


def per_block(record: dict[str, object], field: str) -> float:
    if field == "host_wall_ns_per_block":
        return float(record[field])
    return float(record[field]) / int(record["repeat_count"])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SEARCH_DIR")
    root = Path(sys.argv[1])
    modes: dict[str, object] = {}
    ranking: list[tuple[float, float, str]] = []
    for mode in MODES:
        path = root / f"search_{mode}_r10.jsonl"
        records = [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
        if len(records) != 3:
            raise SystemExit(f"wrong search sample count for {path}")
        expected_contexts = {"control": 4, "context5": 5, "context6": 6}[mode]
        for record in records:
            if record.get("experiment") != "EXP-0068":
                raise SystemExit(f"wrong experiment in {path}")
            if record.get("attention_hvx_contexts") != expected_contexts:
                raise SystemExit(f"wrong context count in {path}")
            if record.get("attention_hvx_workers_created") != expected_contexts - 1:
                raise SystemExit(f"wrong created worker count in {path}")
            if record.get("attention_hvx_workers_locked") != expected_contexts - 1:
                raise SystemExit(f"wrong locked worker count in {path}")
            if record.get("mismatches") != 0 or record.get("max_lsb") != 0:
                raise SystemExit(f"numerical mismatch in {path}")
            if record.get("intermediate_ddr_read_bytes") != 0 or \
                    record.get("intermediate_ddr_write_bytes") != 0 or \
                    record.get("intermediate_spill_fill_count") != 0:
                raise SystemExit(f"physical contract failure in {path}")
        summary: dict[str, object] = {
            "samples": len(records),
            "attention_hvx_contexts": expected_contexts,
            "attention_hvx_workers": expected_contexts - 1,
        }
        for field in FIELDS:
            summary[f"{field}_median"] = statistics.median(
                per_block(record, field) for record in records
            )
        combined = statistics.median(
            per_block(record, "qkv_projection_ticks")
            + per_block(record, "attention_ticks")
            for record in records
        )
        summary["qkv_plus_attention_ticks_median"] = combined
        modes[mode] = summary
        if mode != "control":
            ranking.append((
                combined,
                float(summary["host_wall_ns_per_block_median"]),
                mode,
            ))
    ranking.sort()
    selected_mode = ranking[0][2]
    print(json.dumps({
        "experiment": "EXP-0068",
        "selection_rule": "lowest_repeat10_qkv_plus_attention_ticks_then_host_wall",
        "selected_mode": selected_mode,
        "selected_attention_hvx_contexts": {
            "context5": 5, "context6": 6
        }[selected_mode],
        "modes": modes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
