#!/usr/bin/env python3
"""Validate and report EXP-0133 W4F16 balanced expansion partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0132 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("dynamic", "balanced")
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
    compatible["experiment"] = "EXP-0132"
    compatible["w4f16_expand_claim_regions"] = 1
    parent.validate_record(compatible, repeat, "claim1", audit=audit)
    base.require(record, "experiment", "EXP-0133")
    base.require(record, "w4f16_group_fence_mode", "join_only")
    base.require(record, "w4f16_expand_claim_regions",
                 1 if cell == "dynamic" else 3)
    base.require(record, "w4f16_requested_hvx_workers", 3)
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *W4F16_PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0124_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0133")
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
    if correctness["dynamic"]["output_hash"] != \
            correctness["balanced"]["output_hash"]:
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
        values = metrics(sides["dynamic"], sides["balanced"])
        repeat_results[f"repeat{repeat}"] = {"metrics": values}
        records[repeat] = sides
        physical_cells.append(physical_equal(
            sides["dynamic"], sides["balanced"]))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_cells.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_cells)
    physical_gate = all(physical_cells)
    return {
        "experiment": "EXP-0133",
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
        "pc028": parent.parent.baseline_pc028(
            exp0109_dir, exp0124_dir, records[10]["dynamic"]),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(result_dir) + "/paired_dynamic_r10.jsonl",
            "w4u8": str(exp0124_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["balanced"]),
    }


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    control = summary["pc028"]["w4f16"]
    candidate = summary["repeat10_candidate_modules"]
    total_control = control["Complete block Host wall"]
    total_candidate = candidate["Complete block Host wall"]
    lines.extend([
        "## EXP-0133 repeat-ten W4F16 module wall-time", "",
        "| Module | Dynamic claim | Balanced static | Speed |",
        "|---|---:|---:|---:|",
    ])
    for name in control:
        if name == "Complete block Host wall":
            left = f"{control[name]:.1f} us"
            right = f"{candidate[name]:.1f} us"
        else:
            left = f"{control[name]:.1f} us ({100*control[name]/total_control:.1f}%)"
            right = f"{candidate[name]:.1f} us ({100*candidate[name]/total_candidate:.1f}%)"
        speed = (control[name] / candidate[name] - 1.0) * 100.0
        lines.append(f"| {name} | {left} | {right} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0133 — W4F16 balanced-partition report", ""]
    exp107.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: F16F16 uses frozen EXP-0109, W4U8 uses "
        "EXP-0124 control, and W4F16 is this experiment's dynamic control.",
        "",
    ])
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        parent.parent.add_table(lines, "Primary targets", TARGETS, values)
        parent.parent.add_table(lines, "Additive Block Timing Ledger",
                                exp107.LEDGER, values)
        parent.parent.add_table(
            lines, "Overlapping HMX/HVX/DMA and waits",
            tuple(dict.fromkeys((*exp107.OVERLAP, *W4F16_PIPELINE))), values)
        parent.parent.add_table(lines, "Traffic, commands and residency",
                                exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| EXP-0133 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
