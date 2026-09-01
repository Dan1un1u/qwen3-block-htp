#!/usr/bin/env python3
"""Validate EXP-0131 post-DMA main-context Q/K preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0124 as exp124
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "main_drain")
MODE = {"control": 0, "main_drain": 1}
VTCM_BYTES = 8_388_608
EXTRA_FIELDS = (
    "w4u8_qkv_ring_main_prep_task_count",
    "w4u8_qkv_ring_main_prep_ticks",
)
TARGETS = (
    "host_wall_ns_per_block",
    "qkv_plus_qk_norm_rope_ticks",
    "qkv_projection_ticks",
    "attention_ticks",
    "total_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=exp124.EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=exp124.EXP0111_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0124"
    exp124.validate_record(compatible, repeat, "ring3", audit=audit)
    base.require(record, "experiment", "EXP-0131")
    base.require(record, "w4u8_qkv_tail_prep_mode", MODE[cell])
    if cell == "control":
        base.require(record, "w4u8_qkv_ring_main_prep_task_count", 0)
        base.require(record, "w4u8_qkv_ring_main_prep_ticks", 0)
    else:
        if int(record.get("w4u8_qkv_ring_main_prep_task_count", 0)) <= 0:
            raise SystemExit("main_drain: no Q/K pair was executed")
        if int(record.get("w4u8_qkv_ring_main_prep_ticks", 0)) <= 0:
            raise SystemExit("main_drain: missing positive main prep ticks")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp124.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *exp124.RING_FIELDS, *EXTRA_FIELDS,
        *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0131")
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
            "fused_k_operand_mismatch_count":
                row["u8_attention_fused_k_operand_mismatch_count"],
        }
    if len({value["output_hash"] for value in correctness.values()}) != 1:
        raise SystemExit("cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
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
        comparisons[f"repeat{repeat}"] = metrics(
            sides["control"], sides["main_drain"]
        )

    speed_gate = all(
        comparisons[f"repeat{repeat}"][field][key] < 0.0
        for repeat in REPEATS
        for field in (
            "qkv_plus_qk_norm_rope_ticks", "host_wall_ns_per_block"
        )
        for key in ("change_percent", "paired_change_percent_median")
    )
    physical_gate = all(
        exp124.physical_equal(records[repeat]["control"],
                              records[repeat]["main_drain"])
        for repeat in REPEATS
    )
    plan_gate = all(
        int(row["vtcm_requested_bytes"]) == VTCM_BYTES
        and int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
        and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES
        and int(row["intermediate_ddr_read_bytes"]) == 0
        and int(row["intermediate_ddr_write_bytes"]) == 0
        and int(row["intermediate_spill_fill_count"]) == 0
        for repeat in REPEATS for cell in CELLS
        for row in records[repeat][cell]
    )
    pc028 = exp124.baseline_pc028(exp0109_dir, exp0111_dir)
    pc028["w4u8"] = exp109.module_medians(
        records[10]["main_drain"]
    )
    return {
        "experiment": "EXP-0131",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate and plan_gate,
        "selected_cell": "main_drain" if speed_gate else "control",
        "comparisons": comparisons,
        "pc028": pc028,
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0131 — Complete profiling report", ""]
    exp124.add_pc028(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([
            f"## Repeat {repeat}: control versus post-DMA main drain", ""
        ])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(
            lines, "HMX/HVX/DMA, ring and waits",
            tuple(dict.fromkeys((
                *exp107.OVERLAP, *exp112.W4U8_PIPELINE,
                *exp124.RING_FIELDS, *EXTRA_FIELDS
            ))), values
        )
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| QKV module and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0131 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal, args.exp0111_formal
    )
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
