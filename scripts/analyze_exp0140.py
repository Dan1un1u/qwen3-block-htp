#!/usr/bin/env python3
"""Validate and report EXP-0140 W4F16 Down join-only revalidation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0129 as exp129
import analyze_exp0136 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
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
    "host_wall_ns_per_block", "down_ticks", "gate_up_ticks",
    "activation_ticks", "total_ticks",
)
W4F16_PIPELINE = parent.W4F16_PIPELINE


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
    compatible["w4f16_group_fence_mode"] = "join_only"
    parent.validate_record(compatible, repeat, "group4", audit=audit)
    base.require(record, "experiment", "EXP-0140")
    base.require(
        record, "w4f16_group_fence_mode",
        "join_only_down" if cell == "candidate" else "join_only",
    )
    base.require(record, "w4f16_gate_up_stream_group_tiles", 4)
    base.require(record, "w4f16_gate_up_extra_expand_worker", 1)
    base.require(record, "w4f16_gate_up_extra_stream_worker", 1)
    base.require(record, "w4f16_requested_hvx_workers", 4)
    base.require(record, "w4f16_hvx_workers_created", 4)
    base.require(record, "w4f16_hvx_workers_locked", 4)
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


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
        raise SystemExit("device boot ID changed during EXP-0140")
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
            correctness["candidate"]["output_hash"]:
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    gate_up_values = []
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
        values = exp129.metrics(sides["control"], sides["candidate"])
        comparisons[f"repeat{repeat}"] = values
        physical_values.append(exp129.physical_equal(
            sides["control"], sides["candidate"]))
        for field in ("down_ticks", "host_wall_ns_per_block"):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))
        gate_up_values.extend((
            values["gate_up_ticks"]["change_percent"],
            values["gate_up_ticks"]["paired_change_percent_median"],
        ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    gate_up_preservation_gate = all(
        value is not None and value <= 0.5 for value in gate_up_values)
    physical_gate = all(physical_values)
    correctness_gate = all(
        int(value["mismatches"]) == 0
        and int(value["max_lsb"]) == 0
        and int(value["w4f16_expand_mismatch_count"]) == 0
        for value in correctness.values())
    local_gate = (
        correctness_gate and physical_gate and plan_gate and speed_gate
        and gate_up_preservation_gate)
    return {
        "experiment": "EXP-0140",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "fixed_8mib_vtcm_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gate": speed_gate,
        "gate_up_preservation_gate": gate_up_preservation_gate,
        "local_gate_pass": local_gate,
        "selected_cell": "candidate" if local_gate else "control",
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
            exp109.module_medians(records[10]["candidate"]),
    }


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    control = summary["repeat10_control_modules"]
    candidate = summary["repeat10_candidate_modules"]
    total_control = control["Complete block Host wall"]
    total_candidate = candidate["Complete block Host wall"]
    lines.extend([
        "## EXP-0140 W4F16 module wall-time (repeat10)", "",
        "| Module | Ordered Down | Join-only Down | Speed |",
        "|---|---:|---:|---:|",
    ])
    for name in control:
        if name == "Complete block Host wall":
            left = f"{control[name]:.1f} us"
            right = f"{candidate[name]:.1f} us"
        else:
            left = (f"{control[name]:.1f} us "
                    f"({100*control[name]/total_control:.1f}%)")
            right = (f"{candidate[name]:.1f} us "
                     f"({100*candidate[name]/total_candidate:.1f}%)")
        speed = (control[name] / candidate[name] - 1.0) * 100.0
        lines.append(f"| {name} | {left} | {right} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0140 — Complete profiling report", ""]
    exp107.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: F16F16 and W4U8 use frozen EXP-0109 formal "
        "evidence; selected W4F16 uses EXP-0110. The EXP-0140 control and "
        "candidate are reported separately below.", "",
    ])
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
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
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB grant | {'PASS' if summary['fixed_8mib_vtcm_gate'] else 'FAIL'} |",
        f"| Down and Host strict speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Gate/Up preservation (<=0.5% regression) | {'PASS' if summary['gate_up_preservation_gate'] else 'FAIL'} |",
        f"| EXP-0140 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
