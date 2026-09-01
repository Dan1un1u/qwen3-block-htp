#!/usr/bin/env python3
"""Summarize repeated EXP-0152 full-stack replay logs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


STAGES = (
    "input_stage_ticks",
    "metadata_stage_ticks",
    "input_norm_ticks",
    "qkv_projection_ticks",
    "qk_norm_rope_ticks",
    "attention_ticks",
    "o_projection_ticks",
    "post_attention_residual_ticks",
    "post_attention_norm_ticks",
    "gate_up_ticks",
    "activation_ticks",
    "down_ticks",
    "final_residual_ticks",
    "output_stage_ticks",
    "stage_boundary_ticks",
)
QTIMER_MHZ = 19.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def median(values: list[int]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    logs = sorted(args.input.resolve().glob("device_replay_r*.log"))
    if not logs:
        raise FileNotFoundError(f"no replay logs in {args.input}")
    runs: list[list[dict[str, object]]] = []
    completion: list[dict[str, object]] = []
    for log in logs:
        profiles: list[dict[str, object]] = []
        final: dict[str, object] | None = None
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if record.get("record") == "replay_profile":
                profiles.append(record)
            if record.get("replay_sequence_complete") is True:
                final = record
        if len(profiles) != 9 or final is None:
            raise ValueError(
                f"{log}: expected 9 profiles and one completion record"
            )
        runs.append(sorted(profiles, key=lambda row: int(row["replay_step"])))
        completion.append(final)

    steps: list[dict[str, object]] = []
    for step in range(9):
        rows = [run[step] for run in runs]
        walls = [int(row["host_wall_ns"]) for row in rows]
        total_ticks = [int(row["total_ticks"]) for row in rows]
        logical_m = int(rows[0]["logical_m"])
        median_wall = median(walls)
        stage_us = {
            name: median([int(row[name]) for row in rows]) / QTIMER_MHZ
            for name in STAGES
        }
        steps.append({
            "step": step,
            "mode": rows[0]["mode"],
            "position": int(rows[0]["first_position"]),
            "logical_m": logical_m,
            "host_wall_us": {
                "first": walls[0] / 1000.0,
                "median_10": median_wall / 1000.0,
                "minimum": min(walls) / 1000.0,
                "maximum": max(walls) / 1000.0,
            },
            "useful_tokens_per_second_median_10":
                logical_m * 1.0e9 / median_wall,
            "accelerator_total_us_median_10": median(total_ticks) / QTIMER_MHZ,
            "stages_us_median_10": stage_us,
            "output_max_lsb": max(int(row["output_max_lsb"]) for row in rows),
            "output_mismatches": sum(
                int(row["output_mismatches"]) for row in rows
            ),
            "cache_mismatches": sum(
                int(row["cache_mismatches"]) for row in rows
            ),
        })

    decode_walls = [
        int(run[step]["host_wall_ns"])
        for run in runs for step in range(1, 9)
    ]
    report = {
        "experiment": "EXP-0152",
        "recipe": "W4U8",
        "measurement": "ten_independent_prefill_to_decode_replays",
        "runs": len(runs),
        "all_steps_pass": all(
            bool(record["all_steps_pass"]) for record in completion
        ),
        "steps": steps,
        "aggregate": {
            "prefill_tokens_per_second_median_10":
                steps[0]["useful_tokens_per_second_median_10"],
            "decode_tokens_per_second_all_80_median":
                1.0e9 / median(decode_walls),
            "decode_host_wall_us_all_80_median":
                median(decode_walls) / 1000.0,
            "maximum_output_lsb": max(
                int(step["output_max_lsb"]) for step in steps
            ),
            "total_output_mismatches": sum(
                int(step["output_mismatches"]) for step in steps
            ),
            "total_cache_mismatches": sum(
                int(step["cache_mismatches"]) for step in steps
            ),
        },
        "logs": [str(path) for path in logs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    if not report["all_steps_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
