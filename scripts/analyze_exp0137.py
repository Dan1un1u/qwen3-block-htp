#!/usr/bin/env python3
"""Validate and report EXP-0137 W4U8 adaptive HVX lead credit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("cap0", "cap8", "cap16")
CAPS = {"cap0": 0, "cap8": 8, "cap16": 16}
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0110_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0110/"
    "20260831T163813Z_d18e901339f4_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "gate_up_ticks", "activation_ticks",
    "down_ticks", "total_ticks",
)
ADMISSION = (
    "w4u8_gate_up_adaptive_hvx_wait_ticks",
    "w4u8_gate_up_adaptive_hvx_wait_count",
    "w4u8_gate_up_adaptive_hvx_max_lead_regions",
    "w4u8_gate_up_streaming_region_publish_count",
    "w4u8_gate_up_streaming_region_consume_count",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
)
PIPELINE = tuple(dict.fromkeys((*exp112.W4U8_PIPELINE, *ADMISSION)))
PHYSICAL_EQUAL_FIELDS = exp112.PHYSICAL_EQUAL_FIELDS


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
    compatible["experiment"] = "EXP-0120"
    exp112_compatible = dict(compatible)
    exp112_compatible["experiment"] = "EXP-0112"
    exp112_compatible["w4u8_stream_fence_mode"] = "single_fence"
    exp112.validate_record(
        exp112_compatible, repeat, "single_fence", audit=audit)
    base.require(record, "experiment", "EXP-0137")
    base.require(record, "w4u8_stream_fence_mode", "single_fence")
    base.require(record, "w4u8_gate_up_hvx_lead_cap_regions", CAPS[cell])
    base.require(record, "w4u8_mlp_gate_up_expanded_slot_count", 16)
    base.require(record, "w4u8_mlp_gate_up_hmx_batch_n_tiles", 8)
    base.require(record, "w4u8_mlp_gate_up_hvx_workers", 3)
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")
    published = int(record.get(
        "w4u8_gate_up_streaming_region_publish_count", -1))
    consumed = int(record.get(
        "w4u8_gate_up_streaming_region_consume_count", -2))
    if published != consumed or published != 768 * repeat:
        raise SystemExit(
            f"{cell}: published/consumed region accounting failed: "
            f"{published}/{consumed}")
    if cell == "cap0" and any((
            int(record.get("w4u8_gate_up_adaptive_hvx_wait_count", -1)),
            int(record.get("w4u8_gate_up_adaptive_hvx_wait_ticks", -1)),
            int(record.get(
                "w4u8_gate_up_adaptive_hvx_max_lead_regions", -1)))):
        raise SystemExit("cap0 unexpectedly used adaptive admission")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def pc028_baselines(exp0109_dir: Path,
                    exp0110_dir: Path) -> dict[str, dict[str, float]]:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0110_dir / "paired_carrier_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0137")
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
        }
    if len({value["output_hash"] for value in correctness.values()}) != 1:
        raise SystemExit("lead-cap cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons: dict[str, dict[str, dict]] = {}
    physical_gates = {}
    admission_gates = {}
    speed_gates = {}
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = {}
        for cell in CELLS[1:]:
            values = metrics(sides["cap0"], sides[cell])
            comparisons[f"repeat{repeat}"][cell] = values
            physical_gates.setdefault(cell, []).append(
                physical_equal(sides["cap0"], sides[cell]))
            admission_gates.setdefault(cell, []).append(
                values["w4u8_gate_up_adaptive_hvx_wait_count"]
                ["candidate"] > 0.0)
            speed_gates.setdefault(cell, []).extend(
                values[field][key] < 0.0
                for field in ("gate_up_ticks", "host_wall_ns_per_block")
                for key in ("change_percent",
                            "paired_change_percent_median")
            )

    eligible = [
        cell for cell in CELLS[1:]
        if all(physical_gates[cell]) and all(admission_gates[cell])
        and all(speed_gates[cell])
    ]
    selected = min(
        eligible,
        key=lambda cell: comparisons["repeat10"][cell]
        ["host_wall_ns_per_block"]["candidate"],
        default="cap0",
    )
    plan_gate = all(
        int(row["vtcm_requested_bytes"]) == VTCM_BYTES
        and int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
        and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES
        for repeat in REPEATS for cell in CELLS
        for row in records[repeat][cell]
    )
    pc028 = pc028_baselines(exp0109_dir, exp0110_dir)
    pc028["w4u8"] = exp109.module_medians(records[10][selected])
    return {
        "experiment": "EXP-0137",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": plan_gate,
        "physical_equality_gate": {
            cell: all(physical_gates[cell]) for cell in CELLS[1:]},
        "admission_effective_gate": {
            cell: all(admission_gates[cell]) for cell in CELLS[1:]},
        "speed_gate": {
            cell: all(speed_gates[cell]) for cell in CELLS[1:]},
        "eligible_cells": eligible,
        "selected_cell": selected,
        "local_gate_pass": selected != "cap0" and plan_gate,
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0110_dir) + "/paired_carrier_r10.jsonl",
            "w4u8": str(result_dir) + f"/paired_{selected}_r10.jsonl",
        },
    }


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    lines.extend([
        "## PC-028 three-recipe overview (repeat10)", "",
        "| Module | W16A16 | W4A16 selected | W4A8 EXP-0137 | A8 vs A16 speed |",
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


def render_report(summary: dict) -> str:
    lines = ["# EXP-0137 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        for cell in CELLS[1:]:
            values = summary["comparisons"][f"repeat{repeat}"][cell]
            lines.extend([
                f"## Repeat {repeat}: cap0 versus {cell}", ""])
            exp112.add_table(lines, "Primary targets", TARGETS, values)
            exp112.add_table(lines, "Additive Block Timing Ledger",
                             exp107.LEDGER, values)
            exp112.add_table(
                lines, "HMX/HVX/DMA, admission and waits",
                tuple(dict.fromkeys((*exp107.OVERLAP, *PIPELINE))), values)
            exp112.add_table(lines, "Traffic, commands and residency",
                             exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | cap8 | cap16 |", "|---|---:|---:|",
        "| Admission effective | " + " | ".join(
            "PASS" if summary["admission_effective_gate"][cell] else "FAIL"
            for cell in CELLS[1:]) + " |",
        "| Physical equality | " + " | ".join(
            "PASS" if summary["physical_equality_gate"][cell] else "FAIL"
            for cell in CELLS[1:]) + " |",
        "| Gate/Up and Host speed | " + " | ".join(
            "PASS" if summary["speed_gate"][cell] else "FAIL"
            for cell in CELLS[1:]) + " |",
        "",
        f"Selected cell: `{summary['selected_cell']}`.",
        f"EXP-0137 local gate: `{'PASS' if summary['local_gate_pass'] else 'FAIL'}`.",
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
