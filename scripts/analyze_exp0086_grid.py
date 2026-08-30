#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path


def load_records(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    root = Path(sys.argv[1])
    splits = (64, 80, 96, 112, 128)
    data = {
        split: {
            repeat: load_records(
                root / f"split{split}_repeat{repeat}.jsonl"
            )
            for repeat in (1, 10)
        }
        for split in splits
    }
    result = {"splits": {}, "paired_vs_split96": {}}
    for split in splits:
        result["splits"][str(split)] = {}
        for repeat in (1, 10):
            records = data[split][repeat]
            if not records:
                raise RuntimeError(f"no split={split} repeat={repeat} records")

            def median(metric, already_per_block=False):
                divisor = 1 if already_per_block else repeat
                return statistics.median(
                    float(record[metric]) / divisor for record in records
                )

            result["splits"][str(split)][f"repeat{repeat}"] = {
                "rounds": len(records),
                "host_wall_ns_per_block_median": median(
                    "host_wall_ns_per_block", True
                ),
                "down_ticks_per_block_median": median("down_ticks"),
                "down_pipeline_ticks_per_block_median": median(
                    "w4u8_mlp_down_pipeline_ticks"
                ),
                "first_hmx_start_ticks_per_block_median": median(
                    "w4u8_down_first_hmx_start_ticks"
                ),
                "continuation_wait_ticks_per_block_median": median(
                    "w4u8_down_continuation_ready_wait_ticks"
                ),
                "hmx_compute_ticks_per_block_median": median(
                    "w4u8_mlp_hmx_compute_ticks"
                ),
                "expand_work_ticks_per_block_median": median(
                    "w4u8_mlp_weight_expand_ticks"
                ),
                "output_hashes": sorted(
                    {record["output_hash"] for record in records}
                ),
                "mismatches": sorted(
                    {record["mismatches"] for record in records}
                ),
                "max_lsb": sorted(
                    {record["max_lsb"] for record in records}
                ),
            }

    for split in (64, 80, 112, 128):
        result["paired_vs_split96"][str(split)] = {}
        for repeat in (1, 10):
            control = data[96][repeat]
            candidate = data[split][repeat]
            if len(control) != len(candidate):
                raise RuntimeError("paired round count mismatch")
            summary = {}
            for metric, already_per_block in (
                ("host_wall_ns_per_block", True),
                ("down_ticks", False),
                ("w4u8_mlp_down_pipeline_ticks", False),
            ):
                changes = []
                for control_record, candidate_record in zip(
                    control, candidate
                ):
                    divisor = 1 if already_per_block else repeat
                    control_value = float(control_record[metric]) / divisor
                    candidate_value = float(candidate_record[metric]) / divisor
                    changes.append(
                        (control_value / candidate_value - 1.0) * 100.0
                    )
                summary[metric] = {
                    "paired_speed_change_percent_median": statistics.median(
                        changes
                    ),
                    "paired_speed_change_percent_rounds": changes,
                }
            result["paired_vs_split96"][str(split)][
                f"repeat{repeat}"
            ] = summary
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
