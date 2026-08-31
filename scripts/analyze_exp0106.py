#!/usr/bin/env python3
"""Validate and summarize the integrated EXP-0106 baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0099 as exp99
import validate_exp0050 as base
import validate_exp0084 as exp84


REPEATS = (1, 10)
SAMPLES = 5
W4_CELLS = (
    "fair_w4u8",
    "down4",
    "down4_softmax",
    "down4_softmax_residual",
    "integrated",
)
FAIR_CELLS = ("fair_f16f16", "fair_w4f16", "fair_w4u8")
OLD_EVIDENCE = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0084/"
    "20260830T160259Z_6dc437fe08ea_formal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_fair(record: dict, repeat: int, variant: str) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0084"
    exp84.validate_record(compatible, repeat, variant)


def validate_w4(record: dict, repeat: int, cell: str,
                audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0099"
    mode = "control" if cell == "fair_w4u8" else "batch4"
    exp99.validate_record(compatible, repeat, mode, audit=audit)
    expected = {
        "fair_w4u8": (1, 0, 0, 0),
        "down4": (4, 0, 0, 0),
        "down4_softmax": (4, 1, 0, 0),
        "down4_softmax_residual": (4, 1, 1, 0),
        "integrated": (4, 1, 1, 1),
    }[cell]
    down, softmax, residual, qk = expected
    base.require(record, "experiment", "EXP-0106")
    base.require(record, "w4u8_down_hmx_batch_outputs", down)
    base.require(record, "w4u8_qk_pair_kernel_mode", qk)
    base.require(record, "w4u8_qk_pair_kernel_mode_observed", qk)
    softmax_count = int(
        record.get("u8_attention_softmax_shuffle4_row_group_count", -1)
    )
    if (softmax_count > 0) != bool(softmax):
        raise SystemExit(f"{cell}: Softmax shuffle telemetry mismatch")
    residual_name = str(record.get("residual_mode", ""))
    if ("shuffle4" in residual_name) != bool(residual):
        raise SystemExit(f"{cell}: residual shuffle telemetry mismatch")
    quarter_count = int(record.get("w4u8_qk_quarter_pair_count", -1))
    if (quarter_count > 0) != bool(qk):
        raise SystemExit(f"{cell}: Q/K quarter telemetry mismatch")


def median(records: list[dict], field: str) -> float:
    return float(statistics.median(float(record[field]) for record in records))


def paired_change(control: list[dict], candidate: list[dict],
                  field: str) -> float:
    return float(statistics.median(
        (float(right[field]) / float(left[field]) - 1.0) * 100.0
        for left, right in zip(control, candidate)
    ))


def module_medians(records: list[dict]) -> dict[str, float]:
    modules = [dict(exp84.module_us(record)) for record in records]
    return {
        name: float(statistics.median(sample[name] for sample in modules))
        for name in modules[0]
    }


def old_records(variant: str) -> list[dict]:
    return load(
        OLD_EVIDENCE / "tri_variant" / "canonical" /
        f"{variant}_r10.jsonl", 7
    )


def build_summary(result_dir: Path) -> dict:
    if (result_dir / "boot_id_before.txt").read_bytes() != \
            (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during EXP-0106")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    for cell in FAIR_CELLS:
        record = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        variant = {
            "fair_f16f16": "F16F16",
            "fair_w4f16": "W4F16",
            "fair_w4u8": "W4U8",
        }[cell]
        if variant == "W4U8":
            validate_w4(record, 1, cell, audit=True)
        else:
            validate_fair(record, 1, variant)
    for cell in W4_CELLS[1:]:
        validate_w4(
            load(result_dir / f"correctness_{cell}.jsonl", 1)[0],
            1, cell, audit=True,
        )

    records: dict[int, dict[str, list[dict]]] = {}
    for repeat in REPEATS:
        records[repeat] = {}
        for cell in (*FAIR_CELLS[:2], *W4_CELLS):
            values = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            records[repeat][cell] = values
            if cell == "fair_f16f16":
                for record in values:
                    validate_fair(record, repeat, "F16F16")
            elif cell == "fair_w4f16":
                for record in values:
                    validate_fair(record, repeat, "W4F16")
            else:
                for record in values:
                    validate_w4(record, repeat, cell)

    component_results = {}
    for cell in W4_CELLS:
        per_repeat = {}
        for repeat in REPEATS:
            values = records[repeat][cell]
            control = records[repeat]["down4"]
            per_repeat[f"repeat{repeat}"] = {
                "host_wall_ns_per_block_median": median(
                    values, "host_wall_ns_per_block"
                ),
                "change_vs_down4_percent": (
                    median(values, "host_wall_ns_per_block") /
                    median(control, "host_wall_ns_per_block") - 1.0
                ) * 100.0,
                "paired_change_vs_down4_percent_median": paired_change(
                    control, values, "host_wall_ns_per_block"
                ),
            }
        component_results[cell] = per_repeat

    eligible = []
    for cell in W4_CELLS[2:]:
        if all(
            component_results[cell][f"repeat{repeat}"][key] < 0.0
            for repeat in REPEATS
            for key in (
                "change_vs_down4_percent",
                "paired_change_vs_down4_percent_median",
            )
        ):
            eligible.append(cell)
    selected = min(
        eligible,
        key=lambda cell: component_results[cell]["repeat10"]
            ["host_wall_ns_per_block_median"],
        default="down4",
    )

    fair = {}
    old_map = {
        "fair_f16f16": "f16f16",
        "fair_w4f16": "w4f16",
        "fair_w4u8": "w4u8",
    }
    for cell, old_variant in old_map.items():
        current = records[10][cell]
        old = old_records(old_variant)
        current_wall = median(current, "host_wall_ns_per_block")
        old_wall = median(old, "host_wall_ns_per_block")
        fair[cell] = {
            "host_wall_ns_per_block_median": current_wall,
            "change_vs_exp0084_percent":
                (current_wall / old_wall - 1.0) * 100.0,
            "modules_us": module_medians(current),
        }
    fp16_non_regression = all(
        fair[cell]["change_vs_exp0084_percent"] <= 1.0
        for cell in ("fair_f16f16", "fair_w4f16")
    )
    return {
        "experiment": "EXP-0106",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "component_results": component_results,
        "eligible_integrated_cells": eligible,
        "selected_fastest_w4u8_cell": selected,
        "strict_improvement_over_exp0099_gate": selected != "down4",
        "fair_comparison": fair,
        "fp16_no_material_regression_gate": fp16_non_regression,
        "fastest_w4u8_modules_us": module_medians(records[10][selected]),
        "local_gate_pass": selected != "down4" and fp16_non_regression,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# EXP-0106 integrated baseline profiling", "",
        f"Selected fastest W4U8 cell: `{summary['selected_fastest_w4u8_cell']}`.",
        "",
        "## Incremental W4U8 integration", "",
        "| Cell | repeat1 host µs | vs Down4 | paired | repeat10 host µs | vs Down4 | paired |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, repeats in summary["component_results"].items():
        r1 = repeats["repeat1"]
        r10 = repeats["repeat10"]
        lines.append(
            f"| {cell} | {r1['host_wall_ns_per_block_median']/1000:.3f} | "
            f"{r1['change_vs_down4_percent']:.3f}% | "
            f"{r1['paired_change_vs_down4_percent_median']:.3f}% | "
            f"{r10['host_wall_ns_per_block_median']/1000:.3f} | "
            f"{r10['change_vs_down4_percent']:.3f}% | "
            f"{r10['paired_change_vs_down4_percent_median']:.3f}% |"
        )
    lines.extend([
        "", "## Same-source fair comparison (repeat10)", "",
        "| Recipe | Host wall µs | vs EXP-0084 |",
        "|---|---:|---:|",
    ])
    for cell, values in summary["fair_comparison"].items():
        lines.append(
            f"| {cell} | {values['host_wall_ns_per_block_median']/1000:.3f} | "
            f"{values['change_vs_exp0084_percent']:.3f}% |"
        )
    lines.extend(["", "## Module wall-time table (repeat10)", ""])
    fair = summary["fair_comparison"]
    fast = summary["fastest_w4u8_modules_us"]
    lines.extend([
        "| Module | F16F16 fair µs | W4F16 fair µs | W4U8 fair µs | W4U8 fastest µs |",
        "|---|---:|---:|---:|---:|",
    ])
    names = list(fair["fair_f16f16"]["modules_us"])
    for name in names:
        lines.append(
            f"| {name} | {fair['fair_f16f16']['modules_us'][name]:.3f} | "
            f"{fair['fair_w4f16']['modules_us'][name]:.3f} | "
            f"{fair['fair_w4u8']['modules_us'][name]:.3f} | "
            f"{fast[name]:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir)
    if args.report:
        print(render_report(summary), end="")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
