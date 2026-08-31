#!/usr/bin/env python3
"""Validate and report EXP-0125 W4U8 QKV phase-aware handoff."""

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
CELLS = ("control", "handoff1", "handoff2")
EXPECTED_HANDOFF = {"control": 0, "handoff1": 1, "handoff2": 2}
VTCM_BYTES = 8_388_608
TARGETS = exp124.TARGETS
HANDOFF_FIELDS = (
    *exp124.RING_FIELDS,
    "w4u8_qkv_ring_handoff_worker_count",
    "w4u8_qkv_ring_v_expand_worker_count",
    "w4u8_qkv_ring_handoff_count",
    "w4u8_qkv_ring_final_prep_worker_count",
    "w4u8_qkv_ring_qk_expand_ticks",
    "w4u8_qkv_ring_v_expand_ticks",
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
    compatible["w4u8_qkv_ring_handoff_workers"] = 0
    exp124.validate_record(compatible, repeat, "ring3", audit=audit)
    base.require(record, "experiment", "EXP-0125")
    base.require(record, "w4u8_qkv_ring_expand_workers", 3)
    handoff = EXPECTED_HANDOFF[cell]
    expected = {
        "w4u8_qkv_ring_handoff_workers": handoff,
        "w4u8_qkv_ring_handoff_worker_count": handoff,
        "w4u8_qkv_ring_v_expand_worker_count": 3 - handoff,
        "w4u8_qkv_ring_handoff_count": handoff * repeat,
        "w4u8_qkv_ring_final_prep_worker_count": 2 + handoff,
        "w4u8_qkv_ring_expand_task_count": 128 * repeat,
        "w4u8_qkv_ring_batch_count": 32 * repeat,
        "w4u8_qkv_ring_head_publish_count": 24 * repeat,
    }
    for field, value in expected.items():
        base.require(record, field, value)
    for field in (
        "w4u8_qkv_ring_qk_expand_ticks",
        "w4u8_qkv_ring_v_expand_ticks",
    ):
        if int(record.get(field, 0)) <= 0:
            raise SystemExit(f"{cell}: missing positive {field}")


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *HANDOFF_FIELDS, *exp107.PHYSICAL,
    )))
    return {field: exp124.summarize(left, right, field)
            for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        exp124.summarize(left, right, field)["control"] ==
        exp124.summarize(left, right, field)["candidate"]
        for field in exp124.PHYSICAL_EQUAL_FIELDS
    )


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0125")
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
        raise SystemExit("handoff cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {candidate: {} for candidate in CELLS[1:]}
    for repeat in REPEATS:
        rows_by_cell = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            rows_by_cell[cell] = rows
        records[repeat] = rows_by_cell
        for candidate in CELLS[1:]:
            comparisons[candidate][f"repeat{repeat}"] = metrics(
                rows_by_cell["control"], rows_by_cell[candidate]
            )

    eligible = {}
    for candidate in CELLS[1:]:
        speed = all(
            comparisons[candidate][f"repeat{repeat}"][field][key] < 0.0
            for repeat in REPEATS
            for field in (
                "qkv_plus_qk_norm_rope_ticks", "host_wall_ns_per_block"
            )
            for key in ("change_percent", "paired_change_percent_median")
        )
        physical = all(
            physical_equal(records[repeat]["control"],
                           records[repeat][candidate])
            for repeat in REPEATS
        )
        eligible[candidate] = speed and physical
    eligible_cells = [cell for cell in CELLS[1:] if eligible[cell]]
    selected = min(
        eligible_cells,
        key=lambda cell: comparisons[cell]["repeat10"]
        ["host_wall_ns_per_block"]["candidate"],
        default="control",
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
    pc028["w4u8"] = exp109.module_medians(records[10][selected])
    return {
        "experiment": "EXP-0125",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "eligible": eligible,
        "local_gate_pass": bool(eligible_cells) and plan_gate,
        "selected_cell": selected,
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0111_dir) + "/paired_candidate_r10.jsonl",
            "w4u8": str(result_dir) + f"/paired_{selected}_r10.jsonl",
        },
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0125 — Complete profiling report", ""]
    exp124.add_pc028(lines, summary)
    for candidate in CELLS[1:]:
        for repeat in REPEATS:
            values = summary["comparisons"][candidate][f"repeat{repeat}"]
            lines.extend([
                f"## Repeat {repeat}: control versus {candidate}", ""
            ])
            exp112.add_table(lines, "Primary targets", TARGETS, values)
            exp112.add_table(lines, "Additive Block Timing Ledger",
                             exp107.LEDGER, values)
            exp112.add_table(
                lines, "HMX/HVX/DMA, handoff and waits",
                tuple(dict.fromkeys((
                    *exp107.OVERLAP, *exp112.W4U8_PIPELINE,
                    *HANDOFF_FIELDS
                ))), values
            )
            exp112.add_table(lines, "Traffic, commands and residency",
                             exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| handoff1 strict speed + physical | {'PASS' if summary['eligible']['handoff1'] else 'FAIL'} |",
        f"| handoff2 strict speed + physical | {'PASS' if summary['eligible']['handoff2'] else 'FAIL'} |",
        f"| EXP-0125 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
