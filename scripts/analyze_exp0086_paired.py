#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    root = Path(sys.argv[1])
    result = {}
    for candidate in (64, 80, 112, 128):
        result[str(candidate)] = {}
        for repeat in (1, 10):
            control = load(
                root
                / f"candidate{candidate}_control_repeat{repeat}.jsonl"
            )
            trial = load(
                root
                / f"candidate{candidate}_candidate_repeat{repeat}.jsonl"
            )
            if len(control) != len(trial) or not control:
                raise RuntimeError("paired record count mismatch")
            comparison = {"rounds": len(control)}
            # Down timeline counters are audit-only by contract and remain
            # zero in ordinary performance records.
            for metric, already_per in (
                ("host_wall_ns_per_block", True),
                ("down_ticks", False),
                ("w4u8_mlp_down_pipeline_ticks", False),
                ("w4u8_mlp_hmx_compute_ticks", False),
                ("w4u8_mlp_weight_expand_ticks", False),
            ):
                divisor = 1 if already_per else repeat
                control_values = [
                    float(record[metric]) / divisor for record in control
                ]
                candidate_values = [
                    float(record[metric]) / divisor for record in trial
                ]
                paired = [
                    (control_value / candidate_value - 1.0) * 100.0
                    for control_value, candidate_value in zip(
                        control_values, candidate_values
                    )
                ]
                comparison[metric] = {
                    "control_median": statistics.median(control_values),
                    "candidate_median": statistics.median(candidate_values),
                    "ordinary_speed_change_percent": (
                        statistics.median(control_values)
                        / statistics.median(candidate_values)
                        - 1.0
                    )
                    * 100.0,
                    "paired_speed_change_percent_median": statistics.median(
                        paired
                    ),
                    "paired_speed_change_percent_rounds": paired,
                }
            comparison["output_hashes"] = sorted(
                {record["output_hash"] for record in trial}
            )
            comparison["mismatches"] = sorted(
                {record["mismatches"] for record in trial}
            )
            comparison["max_lsb"] = sorted(
                {record["max_lsb"] for record in trial}
            )
            result[str(candidate)][f"repeat{repeat}"] = comparison
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
