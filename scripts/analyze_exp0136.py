#!/usr/bin/env python3
"""Validate and report EXP-0136 four-tile Gate/Up stream subgroups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0129 as exp129
import analyze_exp0135 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("group8", "group4")
TARGETS = parent.TARGETS
W4F16_PIPELINE = parent.W4F16_PIPELINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=exp129.EXP0109_FORMAL)
    parser.add_argument("--exp0124-formal", type=Path,
                        default=exp129.EXP0124_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0135"
    parent.validate_record(compatible, repeat, "stream4", audit=audit)
    base.require(record, "experiment", "EXP-0136")
    base.require(record, "w4f16_gate_up_extra_expand_worker", 1)
    base.require(record, "w4f16_gate_up_extra_stream_worker", 1)
    base.require(record, "w4f16_gate_up_stream_group_tiles",
                 8 if cell == "group8" else 4)
    base.require(record, "w4f16_requested_hvx_workers", 4)
    base.require(record, "w4f16_hvx_workers_created", 4)
    base.require(record, "w4f16_hvx_workers_locked", 4)
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0124_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0136")
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
    if correctness["group8"]["output_hash"] != \
            correctness["group4"]["output_hash"]:
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
        values = exp129.metrics(sides["group8"], sides["group4"])
        repeat_results[f"repeat{repeat}"] = {"metrics": values}
        records[repeat] = sides
        physical_cells.append(exp129.physical_equal(
            sides["group8"], sides["group4"]))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_cells.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_cells)
    physical_gate = all(physical_cells)
    return {
        "experiment": "EXP-0136",
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
        "pc028": exp129.baseline_pc028(
            exp0109_dir, exp0124_dir, records[10]["group8"]),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(result_dir) + "/paired_group8_r10.jsonl",
            "w4u8": str(exp0124_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["group4"]),
    }


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    rows = summary["repeat10_candidate_modules"]
    lines.extend([
        "## EXP-0136 repeat-ten W4F16 module wall-time", "",
        "| Module | Group-8 control | Group-4 candidate | Speed |",
        "|---|---:|---:|---:|",
    ])
    control = summary["pc028"]["w4f16"]
    for name, candidate in rows.items():
        c = control[name]
        speed = (c / candidate - 1.0) * 100.0 if candidate else 0.0
        lines.append(
            f"| {name} | {c:.1f} us | {candidate:.1f} us | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0136 — W4F16 four-tile streaming-subgroup report", ""]
    exp107.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: F16F16 uses frozen EXP-0109, W4U8 uses "
        "EXP-0124 control, and W4F16 is this experiment's group-8 control.",
        "",
    ])
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        exp129.add_table(lines, "Primary targets", TARGETS, values)
        exp129.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp129.add_table(
            lines, "Overlapping HMX/HVX/DMA and waits",
            tuple(dict.fromkeys((*exp107.OVERLAP, *W4F16_PIPELINE))), values)
        exp129.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| EXP-0136 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
