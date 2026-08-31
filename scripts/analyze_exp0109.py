#!/usr/bin/env python3
"""Validate the EXP-0109 public common-layer freeze consolidation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0106 as exp106
import analyze_exp0107 as exp107
import validate_exp0050 as base
import validate_exp0084 as exp84


REPEATS = (1, 10)
SAMPLES = 5
CELLS = (
    "control_f16f16",
    "frozen_f16f16",
    "fair_w4f16",
    "fair_w4u8",
    "fastest_w4u8",
)
EXPECTED_HASH = {
    "control_f16f16": "704252c89780e695",
    "frozen_f16f16": "704252c89780e695",
    "fair_w4f16": "f18b9abbe1487231",
    "fair_w4u8": "69f22eeb035e5ec5",
    "fastest_w4u8": "69f22eeb035e5ec5",
}
DETAIL_FIELDS = tuple(dict.fromkeys((
    "host_wall_ns_per_block",
    "gate_up_ticks",
    "activation_ticks",
    "down_ticks",
    "total_ticks",
    *exp107.LEDGER,
    *exp107.OVERLAP,
    *exp107.PHYSICAL,
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    base.require(record, "experiment", "EXP-0109")
    compatible = dict(record)
    if cell in ("control_f16f16", "frozen_f16f16"):
        compatible["experiment"] = "EXP-0107"
        exp107.validate(
            compatible,
            repeat,
            "f16f16",
            cell == "frozen_f16f16",
            audit=audit,
        )
    elif cell == "fair_w4f16":
        compatible["experiment"] = "EXP-0084"
        exp84.validate_record(compatible, repeat, "W4F16")
    else:
        compatible["experiment"] = "EXP-0106"
        exp106.validate_w4(
            compatible,
            repeat,
            "fair_w4u8" if cell == "fair_w4u8" else "integrated",
            audit=audit,
        )
    if str(record.get("output_hash")) != EXPECTED_HASH[cell]:
        raise SystemExit(f"{cell}: accepted output hash changed")
    if audit and (
        int(record.get("mismatches", -1)) != 0
        or int(record.get("max_lsb", -1)) != 0
    ):
        raise SystemExit(f"{cell}: audited output mismatch")


def median_field(records: list[dict], field: str) -> float:
    return float(statistics.median(
        exp107.per_block(record, field) for record in records
    ))


def module_medians(records: list[dict]) -> dict[str, float]:
    return exp107.modules(records)


def retained_metrics(records: list[dict]) -> dict[str, float]:
    return {field: median_field(records, field) for field in DETAIL_FIELDS}


def build_summary(result_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes()
            != (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0109")
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

    records: dict[int, dict[str, list[dict]]] = {}
    retained = {}
    f16_results = {}
    fastest_results = {}
    for repeat in REPEATS:
        cells = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            cells[cell] = rows
        records[repeat] = cells
        f16_results[f"repeat{repeat}"] = exp107.metrics(
            cells["control_f16f16"], cells["frozen_f16f16"]
        )
        fastest_results[f"repeat{repeat}"] = {
            field: exp107.summarize(
                cells["fair_w4u8"], cells["fastest_w4u8"], field
            )
            for field in DETAIL_FIELDS
        }
        retained[f"repeat{repeat}"] = {
            cell: retained_metrics(cells[cell])
            for cell in ("fair_w4f16", "fair_w4u8", "fastest_w4u8")
        }

    f16_speed_gate = all(
        f16_results[f"repeat{repeat}"][field][key] < 0.0
        for repeat in REPEATS
        for field in ("gate_up_ticks", "host_wall_ns_per_block")
        for key in ("change_percent", "paired_change_percent_median")
    )
    f16_physical_gate = all(
        exp107.physical_equal(
            records[repeat]["control_f16f16"],
            records[repeat]["frozen_f16f16"],
        )
        for repeat in REPEATS
    )
    fastest_gate = all(
        fastest_results[f"repeat{repeat}"]["host_wall_ns_per_block"][key]
        < 0.0
        for repeat in REPEATS
        for key in ("change_percent", "paired_change_percent_median")
    )
    pc028 = {
        "f16f16": module_medians(records[10]["frozen_f16f16"]),
        "w4f16": module_medians(records[10]["fair_w4f16"]),
        "w4u8": module_medians(records[10]["fair_w4u8"]),
        "fastest_w4u8": module_medians(records[10]["fastest_w4u8"]),
    }
    return {
        "experiment": "EXP-0109",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness_gate": True,
        "accepted_output_hash_gate": True,
        "f16f16_speed_gate": f16_speed_gate,
        "f16f16_physical_equality_gate": f16_physical_gate,
        "retained_w4_runtime_gate": True,
        "fastest_w4u8_retention_gate": fastest_gate,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "correctness": correctness,
        "f16f16_comparison": f16_results,
        "retained_cells": retained,
        "fastest_w4u8_comparison": fastest_results,
        "pc028": pc028,
        "local_gate_pass": (
            f16_speed_gate and f16_physical_gate and fastest_gate
        ),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_compare_table(lines: list[str], title: str,
                      values: dict[str, dict], fields: tuple[str, ...]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Frozen | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        value = values[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, value['control'])} | "
            f"{base.format_value(field, value['candidate'])} | "
            f"{fmt_change(value['change_percent'])} | "
            f"{fmt_change(value['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_module_table(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {
        key: value["Complete block Host wall"] for key, value in table.items()
    }
    lines.extend([
        "## PC-028 frozen baseline module wall-time (repeat10)", "",
        "| Module | W16A16 frozen | W4A16 fair | W4A8 fair | W4A8 fastest | A8 fair vs A16 speed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        cells = []
        for key in ("f16f16", "w4f16", "w4u8", "fastest_w4u8"):
            value = table[key][name]
            if name == "Complete block Host wall":
                cells.append(f"{value:.1f} us")
            else:
                cells.append(
                    f"{value:.1f} us ({100.0 * value / totals[key]:.1f}%)"
                )
        speed = (
            table["f16f16"][name] / table["w4u8"][name] - 1.0
        ) * 100.0
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{cells[3]} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0109 — Public common-layer freeze report", ""]
    add_module_table(lines, summary)
    for repeat in REPEATS:
        lines.extend([f"## Repeat {repeat}", ""])
        values = summary["f16f16_comparison"][f"repeat{repeat}"]
        add_compare_table(
            lines,
            "F16F16 accepted interleaved component",
            values,
            ("host_wall_ns_per_block", "gate_up_ticks", "activation_ticks",
             "down_ticks", "total_ticks"),
        )
        add_compare_table(
            lines, "F16F16 additive timing ledger", values, exp107.LEDGER
        )
        add_compare_table(
            lines, "F16F16 engines and waits", values, exp107.OVERLAP
        )
        add_compare_table(
            lines, "F16F16 traffic and residency", values, exp107.PHYSICAL
        )
        fastest = summary["fastest_w4u8_comparison"][f"repeat{repeat}"]
        add_compare_table(
            lines,
            "Retained fastest W4U8 versus fair W4U8",
            fastest,
            ("host_wall_ns_per_block", "gate_up_ticks", "activation_ticks",
             "down_ticks", "total_ticks"),
        )
        lines.extend([
            "### Retained W4 cells", "",
            "| Metric | W4A16 fair | W4A8 fair | W4A8 fastest |",
            "|---|---:|---:|---:|",
        ])
        retained = summary["retained_cells"][f"repeat{repeat}"]
        for field in DETAIL_FIELDS:
            lines.append(
                f"| `{field}` | {base.format_value(field, retained['fair_w4f16'][field])} | "
                f"{base.format_value(field, retained['fair_w4u8'][field])} | "
                f"{base.format_value(field, retained['fastest_w4u8'][field])} |"
            )
        lines.append("")
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| F16F16 interleaved speed | {'PASS' if summary['f16f16_speed_gate'] else 'FAIL'} |",
        f"| F16F16 physical equality | {'PASS' if summary['f16f16_physical_equality_gate'] else 'FAIL'} |",
        f"| Retained W4 runtime plans | {'PASS' if summary['retained_w4_runtime_gate'] else 'FAIL'} |",
        f"| Fastest W4U8 retained | {'PASS' if summary['fastest_w4u8_retention_gate'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| EXP-0109 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir)
    if args.report:
        print(render_report(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
