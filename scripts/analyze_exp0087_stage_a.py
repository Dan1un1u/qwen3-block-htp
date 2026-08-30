#!/usr/bin/env python3
"""Summarize audit-only EXP-0087 Attention dependency timelines."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CONTEXTS = 6


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("{"):
            continue
        record = json.loads(line)
        if record.get("experiment") == "EXP-0087":
            records.append(record)
    if not records:
        raise SystemExit(f"no EXP-0087 records in {path}")
    return records


def median(records: list[dict], key: str) -> float:
    return float(statistics.median(record[key] for record in records))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load_records(args.jsonl)
    for record in records:
        required = {
            "variant": "W4U8",
            "numerical_audit_mode": "on",
            "w4u8_attention_timeline_enabled": 1,
            "w4u8_attention_softmax_row_slices": 2,
            "w4u8_attention_softmax_timeline_task_count": 16,
            "w4u8_attention_timeline_context_count": 6,
            "intermediate_ddr_read_bytes": 0,
            "intermediate_ddr_write_bytes": 0,
            "intermediate_spill_fill_count": 0,
            "hmx_command_count": 176,
            "hmx_u8s8_tile_pair_count": 49408,
        }
        for key, expected in required.items():
            if record.get(key) != expected:
                raise SystemExit(
                    f"{key}: expected {expected!r}, got {record.get(key)!r}"
                )
        if record.get("mismatches") != 0 or record.get("max_lsb") != 0:
            raise SystemExit("final output is not byte-exact")

    task_counts = [
        median(records, f"w4u8_attention_softmax_context{i}_tasks")
        for i in range(CONTEXTS)
    ]
    work_ticks = [
        median(records, f"w4u8_attention_softmax_context{i}_work_ticks")
        for i in range(CONTEXTS)
    ]
    summary = {
        "experiment": "EXP-0087",
        "stage": "A",
        "records": len(records),
        "output_hashes": sorted({record["output_hash"] for record in records}),
        "context_task_count_medians": task_counts,
        "context_work_tick_medians": work_ticks,
        "context_task_imbalance": max(task_counts) - min(task_counts),
        "context_work_max_to_min": max(work_ticks) / min(work_ticks),
        "qk_ready_first_ticks": median(
            records, "w4u8_attention_qk_ready_first_ticks"
        ),
        "qk_ready_last_ticks": median(
            records, "w4u8_attention_qk_ready_last_ticks"
        ),
        "softmax_start_first_ticks": median(
            records, "w4u8_attention_softmax_start_first_ticks"
        ),
        "softmax_start_last_ticks": median(
            records, "w4u8_attention_softmax_start_last_ticks"
        ),
        "softmax_end_first_ticks": median(
            records, "w4u8_attention_softmax_end_first_ticks"
        ),
        "softmax_end_last_ticks": median(
            records, "w4u8_attention_softmax_end_last_ticks"
        ),
        "softmax_task_min_ticks": median(
            records, "w4u8_attention_softmax_task_min_ticks"
        ),
        "softmax_task_max_ticks": median(
            records, "w4u8_attention_softmax_task_max_ticks"
        ),
        "softmax_task_work_ticks": median(
            records, "w4u8_attention_softmax_task_work_ticks"
        ),
        "qk_to_softmax_start_max_ticks": median(
            records, "w4u8_attention_qk_to_softmax_start_max_ticks"
        ),
        "all_slices_ready_first_ticks": median(
            records, "w4u8_attention_all_slices_ready_first_ticks"
        ),
        "all_slices_ready_last_ticks": median(
            records, "w4u8_attention_all_slices_ready_last_ticks"
        ),
        "av_start_first_ticks": median(
            records, "w4u8_attention_av_start_first_ticks"
        ),
        "av_start_last_ticks": median(
            records, "w4u8_attention_av_start_last_ticks"
        ),
        "av_end_last_ticks": median(
            records, "w4u8_attention_av_end_last_ticks"
        ),
        "all_slices_to_av_start_max_ticks": median(
            records, "w4u8_attention_all_slices_to_av_start_max_ticks"
        ),
        "pool_join_ticks": median(records, "w4u8_attention_pool_join_ticks"),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
