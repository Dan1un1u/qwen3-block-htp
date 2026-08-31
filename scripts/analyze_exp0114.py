#!/usr/bin/env python3
"""Validate and report EXP-0114 W4F16 post-semaphore join barrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0110 as exp110
import analyze_exp0111 as exp111
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
EXP0109_FORMAL = exp111.EXP0109_FORMAL
TARGETS = exp111.TARGETS
W4F16_PIPELINE = exp111.W4F16_PIPELINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0110"
    exp110.validate_record(compatible, repeat, "carrier", audit=audit)
    base.require(record, "experiment", "EXP-0114")
    base.require(
        record, "w4f16_group_fence_mode",
        "join_only" if cell == "control" else "semaphore_only",
    )
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


def build_summary(result_dir: Path, exp0109_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0114")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    for cell in CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "w4f16_expand_mismatch_count":
                row["w4f16_expand_mismatch_count"],
        }
    if correctness["control"]["output_hash"] != \
            correctness["candidate"]["output_hash"]:
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    physical = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        values = exp111.metrics(sides["control"], sides["candidate"])
        comparisons[f"repeat{repeat}"] = values
        physical.append(exp111.physical_equal(
            sides["control"], sides["candidate"]
        ))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    physical_gate = all(physical)
    pc028 = {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(records[10]["control"]),
        "w4u8": exp109.module_medians(load(
            exp0109_dir / "paired_fastest_w4u8_r10.jsonl", SAMPLES)),
    }
    return {
        "experiment": "EXP-0114",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "physical_equality_gate": physical_gate,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate,
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16_w4u8": str(exp0109_dir),
            "w4f16": str(result_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["candidate"]),
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0114 — Complete profiling report", ""]
    exp107.add_pc028(lines, summary)
    exp111.add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        exp111.add_table(lines, "Primary targets", TARGETS, values)
        exp111.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp111.add_table(
            lines, "HMX/HVX/DMA and waits",
            tuple(dict.fromkeys((*exp107.OVERLAP, *W4F16_PIPELINE))),
            values,
        )
        exp111.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0114 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.exp0109_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
