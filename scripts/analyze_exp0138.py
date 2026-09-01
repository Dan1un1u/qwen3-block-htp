#!/usr/bin/env python3
"""Validate and report EXP-0138 initial Up-DMA overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0129 as exp129
import analyze_exp0136 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "overlap")
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0110_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0110/"
    "20260831T163813Z_d18e901339f4_formal"
)
TARGETS = parent.TARGETS
OVERLAP_FIELDS = (
    "w4f16_gate_up_initial_up_dma_wait_ticks",
    "w4f16_gate_up_first_gate_hmx_start_tick",
    "w4f16_gate_up_initial_up_dma_wait_start_tick",
    "w4f16_gate_up_initial_up_dma_overlap_count",
)
W4F16_PIPELINE = tuple(dict.fromkeys((
    *parent.W4F16_PIPELINE, *OVERLAP_FIELDS,
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0110-formal", type=Path,
                        default=EXP0110_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0136"
    parent.validate_record(compatible, repeat, "group4", audit=audit)
    base.require(record, "experiment", "EXP-0138")
    base.require(record, "w4f16_gate_up_stream_group_tiles", 4)
    base.require(
        record, "w4f16_gate_up_initial_up_dma_overlap",
        1 if cell == "overlap" else 0)
    count = int(record.get(
        "w4f16_gate_up_initial_up_dma_overlap_count", -1))
    gate_start = int(record.get(
        "w4f16_gate_up_first_gate_hmx_start_tick", -1))
    up_wait = int(record.get(
        "w4f16_gate_up_initial_up_dma_wait_start_tick", -1))
    if cell == "control":
        if count != 0 or gate_start != 0 or up_wait != 0:
            raise SystemExit("control unexpectedly used initial overlap")
    elif count != repeat or gate_start <= 0 or up_wait < gate_start:
        raise SystemExit(
            f"candidate overlap evidence failed: count={count}, "
            f"gate_start={gate_start}, up_wait={up_wait}")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *W4F16_PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def selected_pc028(exp0109_dir: Path,
                   exp0110_dir: Path) -> dict[str, dict[str, float]]:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0110_dir / "paired_carrier_r10.jsonl", SAMPLES)),
        "w4u8": exp109.module_medians(load(
            exp0109_dir / "paired_fastest_w4u8_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0138")
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
            correctness["overlap"]["output_hash"]:
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    physical_values = []
    plan_gate = True
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
                plan_gate = plan_gate and (
                    int(row["vtcm_requested_bytes"]) == VTCM_BYTES
                    and int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
                    and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES)
            sides[cell] = rows
        records[repeat] = sides
        values = metrics(sides["control"], sides["overlap"])
        comparisons[f"repeat{repeat}"] = values
        physical_values.append(exp129.physical_equal(
            sides["control"], sides["overlap"]))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    physical_gate = all(physical_values)
    return {
        "experiment": "EXP-0138",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate and plan_gate,
        "selected_cell": "overlap" if speed_gate else "control",
        "comparisons": comparisons,
        "pc028": selected_pc028(exp0109_dir, exp0110_dir),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0110_dir) + "/paired_carrier_r10.jsonl",
            "w4u8": str(exp0109_dir) + "/paired_fastest_w4u8_r10.jsonl",
        },
        "repeat10_control_modules":
            exp109.module_medians(records[10]["control"]),
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["overlap"]),
    }


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    lines.extend([
        "## PC-028 selected-baseline overview (repeat10)", "",
        "| Module | W16A16 | W4A16 selected | W4A8 selected | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        cells = []
        for key in ("f16f16", "w4f16", "w4u8"):
            value = table[key][name]
            cells.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)")
        speed = (table["w4f16"][name] / table["w4u8"][name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{speed:+.1f}% |")
    lines.append("")


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    control = summary["repeat10_control_modules"]
    candidate = summary["repeat10_candidate_modules"]
    lines.extend([
        "## EXP-0138 W4F16 module wall-time (repeat10)", "",
        "| Module | Serial startup | Overlapped startup | Speed |",
        "|---|---:|---:|---:|",
    ])
    for name in control:
        left = control[name]
        right = candidate[name]
        speed = (left / right - 1.0) * 100.0 if right else 0.0
        lines.append(
            f"| {name} | {left:.1f} us | {right:.1f} us | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0138 — Complete profiling report", ""]
    add_pc028(lines, summary)
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(
            lines, "Overlapping HMX/HVX/DMA and startup evidence",
            tuple(dict.fromkeys((*exp107.OVERLAP, *W4F16_PIPELINE))), values)
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB grant | {'PASS' if summary['fixed_8mib_vtcm_gate'] else 'FAIL'} |",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0138 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal, args.exp0110_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
