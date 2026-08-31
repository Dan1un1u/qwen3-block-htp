#!/usr/bin/env python3
"""Validate and report EXP-0119 Q-expansion assistance granularity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0118 as exp118


base = exp118.base
REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "assist16", "assist8", "assist4")
REGIONS = {"assist16": 16, "assist8": 8, "assist4": 4}
MODES = {
    "control": "qkvo_batch4_qk_head_pairs",
    "assist16": "qkvo_batch4_qk_head_pairs_q_preemptible_expand",
    "assist8": "qkvo_batch4_qk_head_pairs_q_assist8",
    "assist4": "qkvo_batch4_qk_head_pairs_q_assist4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=exp118.EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=exp118.EXP0111_FORMAL)
    parser.add_argument("--exp0112-formal", type=Path,
                        default=exp118.EXP0112_FORMAL)
    return parser.parse_args()


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0112"
    compatible["w4u8_qkvo_pipeline_mode"] = (
        "qkvo_batch4_qk_head_pairs"
    )
    exp118.exp112.validate_record(
        compatible, repeat, "single_fence", audit=audit
    )
    base.require(record, "experiment", "EXP-0119")
    base.require(record, "w4u8_stream_fence_mode", "single_fence")
    base.require(record, "w4u8_qkvo_pipeline_mode", MODES[cell])
    base.require(record, "attention_qk_norm_task_count", 24 * repeat)
    base.require(record, "hmx_command_count", 128 * repeat)
    base.require(record, "hmx_u8s8_tile_pair_count", 49_408 * repeat)
    base.require(record, "vtcm_requested_bytes", exp118.VTCM_BYTES)
    base.require(record, "vtcm_acquired_bytes", exp118.VTCM_BYTES)
    for field in (
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
    ):
        base.require(record, field, 0)

    assist_regions = int(record["w4u8_qkv_worker_assist_region_count"])
    assist_k = int(record["w4u8_qkv_worker_assist_k_tile_count"])
    main_regions = int(record["w4u8_qkv_main_expand_region_count"])
    main_k = int(record["w4u8_qkv_main_expand_k_tile_count"])
    max_region = int(record["w4u8_qkv_worker_max_region_k_tiles"])
    if main_k + assist_k != 8_192 * repeat:
        raise SystemExit(f"{cell}: QKV K-tile accounting does not close")
    if cell == "control":
        if assist_regions != 0 or assist_k != 0 or max_region != 0:
            raise SystemExit("control executed worker assistance")
        if main_regions != 128 * repeat:
            raise SystemExit("control output-tile accounting changed")
        return

    region = REGIONS[cell]
    expected_regions = (64 + 4_096 // region) * repeat
    if assist_regions <= 0 or assist_k != region * assist_regions:
        raise SystemExit(f"{cell}: assistance accounting is invalid")
    if main_regions + assist_regions != expected_regions:
        raise SystemExit(f"{cell}: region accounting does not close")
    if max_region != region:
        raise SystemExit(f"{cell}: wrong maximum assistance region")
    if int(record["w4u8_qkv_worker_expand_ticks"]) <= 0:
        raise SystemExit(f"{cell}: lacks worker expansion work")


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path, exp0112_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0119")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    hashes = set()
    boundaries = set()
    for cell in CELLS:
        row = base.load_jsonl(
            result_dir / f"correctness_{cell}.jsonl", 1
        )[0]
        validate_record(row, 1, cell, audit=True)
        hashes.add(row["output_hash"])
        boundary = tuple(row[field] for field in (
            "u8_input_norm_actual_hash",
            "u8_attention_actual_score_hash",
            "u8_attention_actual_probability_hash",
            "u8_attention_actual_av_hash",
        ))
        boundaries.add(boundary)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "boundary_hashes": boundary,
        }
    if len(hashes) != 1 or len(boundaries) != 1:
        raise SystemExit("a candidate changed output or an audited boundary")

    records = {}
    comparisons = {}
    eligible = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = base.load_jsonl(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = {}
        for cell in CELLS[1:]:
            values = exp118.metrics(sides["control"], sides[cell])
            speed = all(
                values[field][key] < 0.0
                for field in ("host_wall_ns_per_block",
                              "qkv_projection_ticks")
                for key in ("change_percent",
                            "paired_change_percent_median")
            )
            physical = exp118.physical_equal(
                sides["control"], sides[cell]
            )
            execution = (
                all(int(row["w4u8_qkv_worker_assist_region_count"]) > 0
                    for row in sides[cell])
                and all(int(row["w4u8_qkv_worker_max_region_k_tiles"]) ==
                        REGIONS[cell] for row in sides[cell])
            )
            comparisons[f"repeat{repeat}"][cell] = {
                "metrics": values,
                "strict_speed_gate": speed,
                "physical_equality_gate": physical,
                "granularity_execution_gate": execution,
            }

    for cell in CELLS[1:]:
        if all(
            comparisons[f"repeat{repeat}"][cell][gate]
            for repeat in REPEATS
            for gate in ("strict_speed_gate", "physical_equality_gate",
                         "granularity_execution_gate")
        ):
            eligible.append(cell)
    selected = min(
        eligible,
        key=lambda cell: comparisons["repeat10"][cell]["metrics"]
            ["host_wall_ns_per_block"]["candidate"],
        default="control",
    )
    return {
        "experiment": "EXP-0119",
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
        "comparisons": comparisons,
        "eligible_cells": eligible,
        "selected_cell": selected,
        "local_gate_pass": selected != "control",
        "local_adoption_eligible": selected != "control",
        "pc028": exp118.baseline_pc028(
            exp0109_dir, exp0111_dir, exp0112_dir),
        "repeat10_candidate_modules": exp118.exp109.module_medians(
            records[10][selected]),
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0119 — Complete profiling report", ""]
    exp118.add_pc028(lines, summary)
    for repeat in REPEATS:
        for cell in CELLS[1:]:
            result = summary["comparisons"][f"repeat{repeat}"][cell]
            values = result["metrics"]
            lines.extend([f"## Repeat {repeat}: {cell}", ""])
            exp118.add_table(lines, "Primary targets",
                             exp118.TARGETS, values)
            exp118.add_table(lines, "Additive Block Timing Ledger",
                             exp118.exp107.LEDGER, values)
            exp118.add_table(lines, "HMX/HVX/DMA, assistance and waits",
                             exp118.PIPELINE, values)
            exp118.add_table(lines, "Traffic, commands and residency",
                             exp118.exp107.PHYSICAL, values)
            lines.append(
                "Strict QKV+Host speed: **{}**; physical equality: **{}**; "
                "granularity execution: **{}**.\n".format(
                    "PASS" if result["strict_speed_gate"] else "FAIL",
                    "PASS" if result["physical_equality_gate"] else "FAIL",
                    "PASS" if result["granularity_execution_gate"] else "FAIL",
                )
            )
    lines.extend([
        "## Decision", "",
        f"Eligible cells: `{summary['eligible_cells']}`. Selected cell: "
        f"`{summary['selected_cell']}`.",
        f"EXP-0119 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        "Only the user may promote an eligible candidate to baseline.", "",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal,
        args.exp0111_formal, args.exp0112_formal,
    )
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
