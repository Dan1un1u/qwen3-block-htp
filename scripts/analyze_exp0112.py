#!/usr/bin/env python3
"""Validate and report EXP-0112 W4U8 Gate/Up publication fences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "single_fence", "release_only")
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0111_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0111/"
    "20260831T181039Z_eebd967a3e34_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "gate_up_ticks", "activation_ticks",
    "down_ticks", "total_ticks",
)
W4U8_PIPELINE = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "hmx_ready_wait_ticks", "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_gate_up_pipeline_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
    "w4u8_mlp_pair_publish_count", "w4u8_mlp_pair_consume_count",
    "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_gate_up_persistent_hvx_dispatch_count",
    "w4u8_gate_up_persistent_hvx_worker_count",
)
PHYSICAL_EQUAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "w4u8_mlp_pair_publish_count", "w4u8_mlp_pair_consume_count",
    "w4u8_mlp_gate_up_hmx_command_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=EXP0111_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0109"
    exp109.validate_record(
        compatible, repeat, "fastest_w4u8", audit=audit
    )
    base.require(record, "experiment", "EXP-0112")
    base.require(record, "w4u8_stream_fence_mode", cell)
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *W4U8_PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def baseline_pc028(exp0109_dir: Path, exp0111_dir: Path) -> dict:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0111_dir / "paired_candidate_r10.jsonl", SAMPLES)),
        "w4u8": exp109.module_medians(load(
            exp0109_dir / "paired_fastest_w4u8_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0112")
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
        raise SystemExit("fence cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    eligible = []
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
        comparisons[f"repeat{repeat}"] = {
            cell: metrics(sides["control"], sides[cell])
            for cell in CELLS[1:]
        }

    for cell in CELLS[1:]:
        speed = all(
            comparisons[f"repeat{repeat}"][cell][field][key] < 0.0
            for repeat in REPEATS
            for field in ("gate_up_ticks", "host_wall_ns_per_block")
            for key in ("change_percent", "paired_change_percent_median")
        )
        physical = all(
            physical_equal(records[repeat]["control"],
                           records[repeat][cell])
            for repeat in REPEATS
        )
        if speed and physical:
            eligible.append(cell)
    selected = min(
        eligible,
        key=lambda cell: comparisons["repeat10"][cell]
            ["host_wall_ns_per_block"]["candidate"],
        default="control",
    )
    pc028 = baseline_pc028(exp0109_dir, exp0111_dir)
    return {
        "experiment": "EXP-0112",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "eligible_cells": eligible,
        "selected_cell": selected,
        "speed_gate": selected != "control",
        "physical_equality_gate": selected != "control",
        "local_gate_pass": selected != "control",
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16_w4u8": str(exp0109_dir),
            "w4f16": str(exp0111_dir) + "/paired_candidate_r10.jsonl",
        },
        "repeat10_selected_modules":
            exp109.module_medians(records[10][selected]),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              metric_map: dict) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        value = metric_map[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, value['control'])} | "
            f"{base.format_value(field, value['candidate'])} | "
            f"{fmt_change(value['change_percent'])} | "
            f"{fmt_change(value['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    selected = summary["repeat10_selected_modules"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    total_selected = selected["Complete block Host wall"]
    lines.extend([
        "## PC-028 recipe wall-time table (repeat10)", "",
        "| Module | W16A16 | W4A16 specialized | W4A8 prior | W4A8 EXP-0112 | A8 candidate vs A16 speed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        values = []
        for key in ("f16f16", "w4f16", "w4u8"):
            value = table[key][name]
            values.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        value = selected[name]
        values.append(
            f"{value:.1f} us" if name == "Complete block Host wall"
            else f"{value:.1f} us ({100*value/total_selected:.1f}%)"
        )
        speed = (table["w4f16"][name] / selected[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0112 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        for cell in CELLS[1:]:
            values = summary["comparisons"][f"repeat{repeat}"][cell]
            lines.extend([f"## Repeat {repeat}: {cell}", ""])
            add_table(lines, "Primary targets", TARGETS, values)
            add_table(lines, "Additive Block Timing Ledger",
                      exp107.LEDGER, values)
            add_table(lines, "HMX/HVX/DMA, publication and waits",
                      tuple(dict.fromkeys((*exp107.OVERLAP,
                                           *W4U8_PIPELINE))), values)
            add_table(lines, "Traffic, commands and residency",
                      exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0112 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
