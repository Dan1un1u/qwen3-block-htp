#!/usr/bin/env python3
"""Validate and report EXP-0123 persistent W4F16 Gate/Up expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0111 as exp111
import analyze_exp0112 as exp112
import analyze_exp0120 as exp120
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("join_only", "persistent")
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0111_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0111/"
    "20260831T181039Z_eebd967a3e34_formal"
)
EXP0120_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0120/"
    "20260831T213709Z_fa410c4c2379_formal"
)
TARGETS = exp111.TARGETS
PERSISTENT = (
    "w4f16_gate_up_persistent_dispatch_count",
    "w4f16_gate_up_persistent_job_count",
    "w4f16_gate_up_persistent_completion_count",
    "w4f16_gate_up_persistent_failure_count",
    "w4f16_gate_up_persistent_wait_ticks",
)
PIPELINE = tuple(dict.fromkeys((*exp111.W4F16_PIPELINE, *PERSISTENT)))
PHYSICAL_EQUAL_FIELDS = exp111.PHYSICAL_EQUAL_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=EXP0111_FORMAL)
    parser.add_argument("--exp0120-formal", type=Path,
                        default=EXP0120_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0111"
    compatible["w4f16_group_fence_mode"] = "join_only"
    exp111.validate_record(compatible, repeat, "candidate", audit=audit)
    base.require(record, "experiment", "EXP-0123")
    base.require(record, "w4f16_group_fence_mode", cell)
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")
    if cell == "join_only":
        for field in PERSISTENT:
            if int(record.get(field, -1)) != 0:
                raise SystemExit(f"join_only: unexpected {field}")
    else:
        expected = {
            "w4f16_gate_up_persistent_dispatch_count": 2 * repeat,
            "w4f16_gate_up_persistent_job_count": 48 * repeat,
            "w4f16_gate_up_persistent_completion_count": 96 * repeat,
            "w4f16_gate_up_persistent_failure_count": 0,
        }
        for field, value in expected.items():
            if int(record.get(field, -1)) != value:
                raise SystemExit(
                    f"persistent: {field} expected {value}, "
                    f"got {record.get(field)}"
                )


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


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path, exp0120_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0123")
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
    if len({value["output_hash"] for value in correctness.values()}) != 1:
        raise SystemExit("persistent cells produced different output hashes")

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
            sides["join_only"], sides["persistent"]
        )

    speed_gate = all(
        comparisons[f"repeat{repeat}"][field][key] < 0.0
        for repeat in REPEATS
        for field in ("gate_up_ticks", "host_wall_ns_per_block")
        for key in ("change_percent", "paired_change_percent_median")
    )
    physical_gate = all(
        physical_equal(records[repeat]["join_only"],
                       records[repeat]["persistent"])
        for repeat in REPEATS
    )
    plan_gate = all(
        int(row["vtcm_requested_bytes"]) == VTCM_BYTES and
        int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
        for repeat in REPEATS for cell in CELLS
        for row in records[repeat][cell]
    )
    selected_cell = "persistent" if speed_gate else "join_only"
    pc028 = exp120.baseline_pc028(exp0109_dir, exp0111_dir)
    pc028["w4f16"] = exp109.module_medians(
        records[10][selected_cell]
    )
    return {
        "experiment": "EXP-0123",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate and plan_gate,
        "selected_cell": selected_cell,
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(result_dir) +
                     f"/paired_{selected_cell}_r10.jsonl",
            "w4u8": str(exp0120_dir) + "/paired_ring16_r10.jsonl",
        },
    }


def render_report(summary: dict) -> str:
    lines = ["# EXP-0123 — Complete profiling report", ""]
    exp120.add_pc028(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}: join-only versus persistent", ""])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(lines, "HMX/HVX/DMA and persistent handoff",
                         tuple(dict.fromkeys((*exp107.OVERLAP,
                                              *PIPELINE))), values)
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB grant | {'PASS' if summary['fixed_8mib_vtcm_gate'] else 'FAIL'} |",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0123 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal,
        args.exp0111_formal, args.exp0120_formal
    )
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
