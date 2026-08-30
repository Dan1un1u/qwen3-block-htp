#!/usr/bin/env python3
"""Analyze the paired EXP-0087 32-row versus 16-row Stage B gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            if record.get("experiment") == "EXP-0087":
                records.append(record)
    if not records:
        raise SystemExit(f"no EXP-0087 records in {path}")
    return records


def validate(records: list[dict], slices: int, repeat: int) -> None:
    for record in records:
        required = {
            "variant": "W4U8",
            "repeat_count": repeat,
            "numerical_audit_mode": "off",
            "w4u8_attention_timeline_requested": 0,
            "w4u8_attention_timeline_enabled": 0,
            "w4u8_attention_softmax_row_slices_requested": slices,
            "w4u8_attention_softmax_row_slices": 0,
            "attention_softmax_task_count": 8 * slices * repeat,
            "intermediate_ddr_read_bytes": 0,
            "intermediate_ddr_write_bytes": 0,
            "intermediate_spill_fill_count": 0,
            "hmx_command_count": 176 * repeat,
            "hmx_u8s8_tile_pair_count": 49408 * repeat,
            "mismatches": 0,
            "max_lsb": 0,
            "output_hash": "69f22eeb035e5ec5",
        }
        for key, expected in required.items():
            if record.get(key) != expected:
                raise SystemExit(
                    f"{key}: expected {expected!r}, got {record.get(key)!r}"
                )


def median(records: list[dict], key: str) -> float:
    return float(statistics.median(record[key] for record in records))


def improvement(control: float, candidate: float) -> float:
    return 100.0 * (control - candidate) / control


def paired_improvement(
    control: list[dict], candidate: list[dict], key: str
) -> float:
    if len(control) != len(candidate):
        raise SystemExit("paired record count mismatch")
    return float(
        statistics.median(
            improvement(c[key], x[key]) for c, x in zip(control, candidate)
        )
    )


def validate_correctness(path: Path, slices: int) -> dict:
    record = load(path)[0]
    required = {
        "variant": "W4U8",
        "numerical_audit_mode": "on",
        "w4u8_attention_softmax_row_slices_requested": slices,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": "69f22eeb035e5ec5",
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": 176,
        "hmx_u8s8_tile_pair_count": 49408,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise SystemExit(
                f"correctness {key}: expected {expected!r}, "
                f"got {record.get(key)!r}"
            )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = {
        "experiment": "EXP-0087",
        "stage": "B",
        "control_rows_per_slice": 32,
        "candidate_rows_per_slice": 16,
        "repeats": {},
    }
    gate_values = []
    for repeat in (1, 10):
        control = load(args.directory / f"control_r{repeat}.jsonl")
        candidate = load(args.directory / f"candidate_r{repeat}.jsonl")
        validate(control, 2, repeat)
        validate(candidate, 4, repeat)
        repeat_summary = {}
        for key in ("attention_ticks", "host_wall_ns_per_block"):
            control_median = median(control, key)
            candidate_median = median(candidate, key)
            ordinary = improvement(control_median, candidate_median)
            paired = paired_improvement(control, candidate, key)
            repeat_summary[key] = {
                "control_median": control_median,
                "candidate_median": candidate_median,
                "ordinary_improvement_percent": ordinary,
                "paired_improvement_percent": paired,
            }
            gate_values.extend((ordinary, paired))
        summary["repeats"][str(repeat)] = repeat_summary

    control_correctness = validate_correctness(
        args.directory / "correctness_control.jsonl", 2
    )
    candidate_correctness = validate_correctness(
        args.directory / "correctness_candidate.jsonl", 4
    )
    summary["correctness"] = {
        "control_output_hash": control_correctness["output_hash"],
        "candidate_output_hash": candidate_correctness["output_hash"],
        "candidate_probability_hash": candidate_correctness[
            "u8_attention_actual_probability_hash"
        ],
        "candidate_av_hash": candidate_correctness[
            "u8_attention_actual_av_hash"
        ],
        "byte_exact": True,
        "physical_gate": "pass",
    }
    summary["gate_values_percent"] = gate_values
    summary["gate_result"] = "pass" if all(x > 0.0 for x in gate_values) else "fail"
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
