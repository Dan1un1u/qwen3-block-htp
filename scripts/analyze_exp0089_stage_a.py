#!/usr/bin/env python3
"""Validate the EXP-0089 audit-only Gate/Up-to-Down timeline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SAMPLES = 7
OUTPUT_HASH = "69f22eeb035e5ec5"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def require(record: dict, field: str, expected: object) -> None:
    if record.get(field) != expected:
        raise SystemExit(
            f"{field}: expected {expected!r}, got {record.get(field)!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    if (args.result_dir / "boot_id_before.txt").read_bytes() != (
            args.result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed")

    records = load(args.result_dir / "timeline.jsonl")
    main_records = [record for record in records
                    if record.get("record") is None]
    timelines = [record for record in records
                 if record.get("record") == "gate_up_down_timeline"]
    if len(main_records) != SAMPLES or len(timelines) != SAMPLES:
        raise SystemExit(
            f"expected {SAMPLES} main and timeline records, got "
            f"{len(main_records)} and {len(timelines)}"
        )
    ordinary = load(args.result_dir / "ordinary.jsonl")
    if any(record.get("record") == "gate_up_down_timeline"
           for record in ordinary):
        raise SystemExit("timeline leaked into ordinary performance mode")
    if len(ordinary) != 3:
        raise SystemExit(f"expected 3 ordinary records, got {len(ordinary)}")

    for record in main_records + ordinary:
        require(record, "experiment", "EXP-0089")
        require(record, "output_hash", OUTPUT_HASH)
        require(record, "mismatches", 0)
        require(record, "max_lsb", 0)
        require(record, "vtcm_requested_bytes", 8_388_608)
        require(record, "vtcm_acquired_bytes", 8_388_608)
        require(record, "intermediate_ddr_read_bytes", 0)
        require(record, "intermediate_ddr_write_bytes", 0)
        require(record, "intermediate_spill_fill_count", 0)
        require(record, "hmx_command_count", 176)
        require(record, "hmx_u8s8_tile_pair_count", 49_408)

    windows = []
    startup = []
    capacity = []
    for record in timelines:
        require(record, "experiment", "EXP-0089")
        require(record, "timeline_enabled", 1)
        require(record, "vtcm_requested_bytes", 8_388_608)
        require(record, "vtcm_acquired_bytes", 8_388_608)
        ordered = [
            record["gate_up_first_hmx_start_ticks"],
            record["middle_first_ready_ticks"],
            record["middle_first_half_ready_ticks"],
            record["middle_all_ready_ticks"],
            record["gate_up_join_ticks"],
            record["down_phase_start_ticks"],
            record["down_first_dma_publication_ticks"],
            record["down_first_expand_end_ticks"],
            record["down_first_hmx_start_ticks"],
            record["down_phase_end_ticks"],
        ]
        if ordered[0] < 0 or any(value <= 0 for value in ordered[1:]):
            raise SystemExit(f"missing timeline point: {ordered}")
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            raise SystemExit(f"non-monotonic timeline: {ordered}")
        if not (record["gate_up_first_hmx_start_ticks"] <=
                record["gate_up_last_hmx_end_ticks"] <=
                record["middle_all_ready_ticks"] <=
                record["gate_up_join_ticks"]):
            raise SystemExit("invalid Gate/Up HMX, activation, or join order")
        windows.append(
            record["gate_up_join_ticks"] -
            record["middle_first_half_ready_ticks"]
        )
        startup.append(
            record["down_first_hmx_start_ticks"] -
            record["down_phase_start_ticks"]
        )
        capacity.append(
            record["spare_vtcm_bytes"] >=
            record["down_first_chunk_prestage_bytes"]
        )

    window_median = float(statistics.median(windows))
    startup_median = float(statistics.median(startup))
    gate = all(capacity) and all(value > 0 for value in windows) and (
        window_median >= startup_median
    )
    summary = {
        "experiment": "EXP-0089",
        "stage": "A",
        "samples": SAMPLES,
        "timeline_is_audit_only": True,
        "output_hash": OUTPUT_HASH,
        "overlap_window_ticks": windows,
        "overlap_window_median_ticks": window_median,
        "down_startup_ticks": startup,
        "down_startup_median_ticks": startup_median,
        "spare_vtcm_bytes": timelines[0]["spare_vtcm_bytes"],
        "down_first_chunk_prestage_bytes":
            timelines[0]["down_first_chunk_prestage_bytes"],
        "capacity_gate": all(capacity),
        "stage_a_gate_pass": gate,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
