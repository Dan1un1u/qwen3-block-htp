#!/usr/bin/env python3
"""Validate and report EXP-0134 actual fourth expansion context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0129 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("worker3", "worker4")
TARGETS = parent.TARGETS
W4F16_PIPELINE = parent.W4F16_PIPELINE
PHYSICAL_EQUAL_FIELDS = parent.PHYSICAL_EQUAL_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=parent.EXP0109_FORMAL)
    parser.add_argument("--exp0124-formal", type=Path,
                        default=parent.EXP0124_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0129"
    compatible["w4f16_gate_up_extra_expand_worker"] = 0
    parent.validate_record(compatible, repeat, cell, audit=audit)
    base.require(record, "experiment", "EXP-0134")
    base.require(record, "w4f16_group_fence_mode", "join_only")
    base.require(record, "w4f16_expand_claim_regions", 1)
    base.require(record, "w4f16_gate_up_extra_expand_worker",
                 0 if cell == "worker3" else 1)
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0124_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0134")
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
    if correctness["worker3"]["output_hash"] != \
            correctness["worker4"]["output_hash"]:
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    repeat_results = {}
    speed_cells = []
    physical_cells = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        values = parent.metrics(sides["worker3"], sides["worker4"])
        repeat_results[f"repeat{repeat}"] = {"metrics": values}
        records[repeat] = sides
        physical_cells.append(parent.physical_equal(
            sides["worker3"], sides["worker4"]))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_cells.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_cells)
    physical_gate = all(physical_cells)
    return {
        "experiment": "EXP-0134",
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
        "repeat_results": repeat_results,
        "pc028": parent.baseline_pc028(
            exp0109_dir, exp0124_dir, records[10]["worker3"]),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(result_dir) + "/paired_worker3_r10.jsonl",
            "w4u8": str(exp0124_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["worker4"]),
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0134 — W4F16 actual fourth-context report", ""]
    exp107.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: F16F16 uses frozen EXP-0109, W4U8 uses "
        "EXP-0124 control, and W4F16 is this experiment's three-context control.",
        "",
    ])
    parent.add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        parent.add_table(lines, "Primary targets", TARGETS, values)
        parent.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        parent.add_table(lines, "Overlapping HMX/HVX/DMA and waits",
                         tuple(dict.fromkeys((*exp107.OVERLAP,
                                              *W4F16_PIPELINE))), values)
        parent.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| EXP-0134 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal, args.exp0124_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
