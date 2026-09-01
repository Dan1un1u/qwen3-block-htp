#!/usr/bin/env python3
"""Validate and report EXP-0139 W4U8 QKV main-context prep assist."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0124 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("sequential", "ring3", "assist")
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
ASSIST_FIELDS = (
    "w4u8_qkv_main_prep_assist_count",
    "w4u8_qkv_main_prep_task_count",
    "w4u8_qkv_main_prep_ticks",
    "w4u8_qkv_dma_feed_complete_tick",
    "w4u8_qkv_main_prep_start_tick",
    "w4u8_qkv_main_prep_end_tick",
    "w4u8_qkv_main_hmx_wait_start_tick",
)
PIPELINE_FIELDS = tuple(dict.fromkeys((
    *exp112.W4U8_PIPELINE, *parent.RING_FIELDS, *ASSIST_FIELDS,
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
    compatible["experiment"] = "EXP-0124"
    parent.validate_record(
        compatible, repeat, "control" if cell == "sequential" else "ring3",
        audit=audit,
    )
    base.require(record, "experiment", "EXP-0139")
    base.require(record, "w4u8_qkv_main_prep_assist",
                 1 if cell == "assist" else 0)
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")

    assist_values = {
        field: int(record.get(field, -1)) for field in ASSIST_FIELDS
    }
    if cell != "assist":
        if any(assist_values.values()):
            raise SystemExit(
                f"{cell}: main-context assist telemetry is nonzero: "
                f"{assist_values}")
        return

    if assist_values["w4u8_qkv_main_prep_assist_count"] != repeat:
        raise SystemExit(
            f"assist: expected one exercised assist per block, got "
            f"{assist_values['w4u8_qkv_main_prep_assist_count']}")
    if assist_values["w4u8_qkv_main_prep_task_count"] < repeat:
        raise SystemExit("assist: main context did not execute a prep task/block")
    if assist_values["w4u8_qkv_main_prep_ticks"] <= 0:
        raise SystemExit("assist: missing positive main prep duration")
    feed = assist_values["w4u8_qkv_dma_feed_complete_tick"]
    start = assist_values["w4u8_qkv_main_prep_start_tick"]
    end = assist_values["w4u8_qkv_main_prep_end_tick"]
    wait = assist_values["w4u8_qkv_main_hmx_wait_start_tick"]
    if not (0 < feed <= start <= end <= wait):
        raise SystemExit(
            "assist ordering failed: "
            f"feed={feed}, start={start}, end={end}, hmx_wait={wait}")


def per_block(record: dict, field: str) -> float:
    if field == "qkv_plus_qk_norm_rope_ticks":
        return (
            float(record["qkv_projection_ticks"]) +
            float(record["qk_norm_rope_ticks"])
        ) / int(record["repeat_count"])
    return exp107.per_block(record, field)


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    control = [per_block(row, field) for row in left]
    candidate = [per_block(row, field) for row in right]
    lmed = float(statistics.median(control))
    rmed = float(statistics.median(candidate))
    paired = [
        (r / l - 1.0) * 100.0
        for l, r in zip(control, candidate) if l != 0.0
    ]
    return {
        "control": lmed,
        "candidate": rmed,
        "change_percent": ((rmed / lmed - 1.0) * 100.0
                           if lmed != 0.0 else None),
        "paired_change_percent_median": (
            float(statistics.median(paired)) if paired else None),
        "paired_change_percent_min": min(paired) if paired else None,
        "paired_change_percent_max": max(paired) if paired else None,
    }


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *PIPELINE_FIELDS, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def selected_pc028(exp0109_dir: Path,
                   exp0110_dir: Path) -> dict[str, dict[str, float]]:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0110_dir / "paired_carrier_r10.jsonl", SAMPLES)),
    }


def plan_valid(row: dict) -> bool:
    return (
        int(row["vtcm_requested_bytes"]) == VTCM_BYTES
        and int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
        and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES
        and int(row["intermediate_ddr_read_bytes"]) == 0
        and int(row["intermediate_ddr_write_bytes"]) == 0
        and int(row["intermediate_spill_fill_count"]) == 0
        and int(row["block_invocation_count"]) == int(row["repeat_count"])
    )


def faster_control(records: dict[str, list[dict]]) -> str:
    medians = {
        cell: statistics.median(
            per_block(row, "host_wall_ns_per_block")
            for row in records[cell])
        for cell in ("sequential", "ring3")
    }
    return min(medians, key=medians.get)


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0139")
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
            "stage_mismatch_count": row.get("stage_mismatch_count", 0),
        }
    if len({entry["output_hash"] for entry in correctness.values()}) != 1:
        raise SystemExit("EXP-0139 cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    plan_gate = True
    physical_gate = True
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
                plan_gate = plan_gate and plan_valid(row)
            sides[cell] = rows
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = {
            "assist_vs_ring3": metrics(sides["ring3"], sides["assist"]),
            "assist_vs_sequential": metrics(
                sides["sequential"], sides["assist"]),
            "ring3_vs_sequential": metrics(
                sides["sequential"], sides["ring3"]),
        }
        physical_gate = physical_gate and all((
            parent.physical_equal(sides["ring3"], sides["assist"]),
            parent.physical_equal(sides["sequential"], sides["assist"]),
        ))

    speed_gate = all(
        comparisons[f"repeat{repeat}"][comparison][field][key] < 0.0
        for repeat in REPEATS
        for comparison in ("assist_vs_ring3", "assist_vs_sequential")
        for field in ("qkv_plus_qk_norm_rope_ticks",
                      "host_wall_ns_per_block")
        for key in ("change_percent", "paired_change_percent_median")
    )
    selected = "assist" if speed_gate else faster_control(records[10])
    pc028 = selected_pc028(exp0109_dir, exp0110_dir)
    pc028["w4u8"] = exp109.module_medians(records[10]["assist"])
    return {
        "experiment": "EXP-0139",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "dynamic_assist_gate": True,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate and plan_gate,
        "selected_cell": selected,
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0110_dir) + "/paired_carrier_r10.jsonl",
            "w4u8": str(result_dir) + "/paired_assist_r10.jsonl",
        },
        "repeat10_modules": {
            cell: exp109.module_medians(records[10][cell])
            for cell in CELLS
        },
    }


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    lines.extend([
        "## PC-028 three-recipe overview (repeat10)", "",
        "| Module | W16A16 | W4A16 selected | W4A8 assist | A8 vs A16 speed |",
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


def add_modules(lines: list[str], summary: dict) -> None:
    table = summary["repeat10_modules"]
    totals = {cell: table[cell]["Complete block Host wall"] for cell in CELLS}
    lines.extend([
        "## EXP-0139 W4U8 module wall-time (repeat10)", "",
        "| Module | Sequential | Ring3 | Ring3 + main assist |",
        "|---|---:|---:|---:|",
    ])
    for name in table["sequential"]:
        values = []
        for cell in CELLS:
            value = table[cell][name]
            values.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[cell]:.1f}%)")
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | {values[2]} |")
    lines.append("")


def add_comparison(lines: list[str], title: str, values: dict) -> None:
    lines.extend([f"### {title}", ""])
    exp112.add_table(lines, "Primary targets", TARGETS, values)
    exp112.add_table(lines, "Additive Block Timing Ledger",
                     exp107.LEDGER, values)
    exp112.add_table(
        lines, "Overlapping HMX/HVX/DMA, QKV ring and main assist",
        tuple(dict.fromkeys((
            *exp107.OVERLAP, *exp112.W4U8_PIPELINE,
            *parent.RING_FIELDS, *ASSIST_FIELDS,
        ))), values)
    exp112.add_table(lines, "Traffic, commands and residency",
                     exp107.PHYSICAL, values)


def render_report(summary: dict) -> str:
    lines = ["# EXP-0139 — Complete profiling report", ""]
    add_pc028(lines, summary)
    add_modules(lines, summary)
    for repeat in REPEATS:
        lines.extend([f"## Repeat {repeat}", ""])
        values = summary["comparisons"][f"repeat{repeat}"]
        add_comparison(lines, "Assist versus ring3 control",
                       values["assist_vs_ring3"])
        add_comparison(lines, "Assist versus sequential control",
                       values["assist_vs_sequential"])
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Main-context assist exercised after DMA feed | {'PASS' if summary['dynamic_assist_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| QKV and Host speed versus both controls | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0139 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected experiment cell: `{summary['selected_cell']}`.",
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
