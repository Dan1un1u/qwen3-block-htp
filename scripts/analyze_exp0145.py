#!/usr/bin/env python3
"""Validate and report EXP-0145 W4U8 Down HMX batch-eight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0124 as exp124
import analyze_exp0143 as exp143
import analyze_exp0144 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
EXPECTED_BATCH = {"control": 4, "candidate": 8}
VTCM_BYTES = 8_388_608
OUTPUT_HASH = "69f22eeb035e5ec5"
EXP0109_FORMAL = parent.EXP0109_FORMAL
EXP0110_FORMAL = parent.EXP0110_FORMAL
TARGETS = (
    "host_wall_ns_per_block", "down_ticks", "qkv_projection_ticks",
    "attention_ticks", "gate_up_ticks", "total_ticks",
)
AUDIT_HASH_FIELDS = parent.AUDIT_HASH_FIELDS
PHYSICAL_EQUAL_FIELDS = tuple(
    field for field in exp143.PHYSICAL_EQUAL_FIELDS
    if field != "hmx_command_count"
)


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
    compatible["experiment"] = "EXP-0144"
    compatible["w4u8_down_hmx_batch_outputs"] = 4
    compatible["w4u8_mlp_down_hmx_batch_n_tiles"] = 4
    compatible["w4u8_mlp_down_hmx_command_count"] = 16 * repeat
    compatible["hmx_command_count"] = int(record["hmx_command_count"]) + (
        8 * repeat if cell == "candidate" else 0)
    parent.validate_record(compatible, repeat, "candidate", audit=audit)
    batch = EXPECTED_BATCH[cell]
    base.require(record, "experiment", "EXP-0145")
    base.require(record, "w4u8_down_hmx_batch_outputs", batch)
    base.require(record, "w4u8_mlp_down_hmx_batch_n_tiles", batch)
    base.require(record, "w4u8_mlp_down_hmx_command_count",
                 (64 // batch) * repeat)
    base.require(record, "w4u8_qk_pair_kernel_mode_observed", 3)
    base.require(record, "block_invocation_count", repeat)
    if record.get("output_hash") != OUTPUT_HASH or \
            int(record.get("mismatches", -1)) != 0 or \
            int(record.get("max_lsb", -1)) != 0:
        raise SystemExit(f"{cell}: final output is not byte-exact")


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *exp124.RING_FIELDS,
        *exp107.PHYSICAL,
        "w4u8_mlp_down_hmx_batch_n_tiles",
        "w4u8_mlp_down_hmx_command_count",
        "w4u8_mlp_down_in_command_slot_release_count",
        "w4u8_mlp_down_producer_progress_command_count",
        "w4u8_mlp_down_pipeline_ticks",
    )))
    return {field: exp124.summarize(left, right, field)
            for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        exp124.summarize(left, right, field)["control"] ==
        exp124.summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0145")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    audit_hashes = {}
    for cell in CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "down_batch": row["w4u8_mlp_down_hmx_batch_n_tiles"],
            "down_commands": row["w4u8_mlp_down_hmx_command_count"],
        }
        audit_hashes[cell] = tuple(row[field] for field in AUDIT_HASH_FIELDS)
    correctness_gate = (
        len({value["output_hash"] for value in correctness.values()}) == 1
        and audit_hashes["control"] == audit_hashes["candidate"]
    )

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    preservation_values = []
    physical_values = []
    command_gate = True
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
                    and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES
                    and int(row["intermediate_ddr_read_bytes"]) == 0
                    and int(row["intermediate_ddr_write_bytes"]) == 0
                    and int(row["intermediate_spill_fill_count"]) == 0
                )
            sides[cell] = rows
        records[repeat] = sides
        values = metrics(sides["control"], sides["candidate"])
        comparisons[f"repeat{repeat}"] = values
        physical_values.append(physical_equal(
            sides["control"], sides["candidate"]))
        for control, candidate in zip(sides["control"], sides["candidate"]):
            command_gate = command_gate and (
                int(control["w4u8_mlp_down_hmx_command_count"]) -
                    int(candidate["w4u8_mlp_down_hmx_command_count"]) ==
                    8 * repeat
                and int(control["hmx_command_count"]) -
                    int(candidate["hmx_command_count"]) == 8 * repeat
                and int(control["weight_dma_descriptor_count"]) ==
                    int(candidate["weight_dma_descriptor_count"])
                and int(control["w4u8_mlp_down_in_command_slot_release_count"]) ==
                    int(candidate["w4u8_mlp_down_in_command_slot_release_count"])
            )
        for field in ("down_ticks", "host_wall_ns_per_block"):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))
        for field in (
            "qkv_projection_ticks", "attention_ticks", "gate_up_ticks"
        ):
            preservation_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    preservation_gate = all(
        value is not None and value <= 0.5
        for value in preservation_values)
    physical_gate = all(physical_values)
    local_gate = all((correctness_gate, plan_gate, physical_gate,
                      command_gate, speed_gate, preservation_gate))
    selected = "candidate" if local_gate else "control"
    return {
        "experiment": "EXP-0145",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "command_reduction_gate": command_gate,
        "speed_gate": speed_gate,
        "upstream_preservation_gate": preservation_gate,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "local_gate_pass": local_gate,
        "selected_cell": selected,
        "comparisons": comparisons,
        "pc028": exp143.selected_pc028(
            exp0109_dir, exp0110_dir, records[10][selected]),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0110_dir) + "/paired_carrier_r10.jsonl",
            "w4u8": str(result_dir) + f"/paired_{selected}_r10.jsonl",
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
        "## EXP-0145 W4U8 module wall-time (repeat10)", "",
        "| Module | Down batch4 | Down batch8 | Speed |",
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
    lines = ["# EXP-0145 — Complete profiling report", ""]
    exp124.add_pc028(lines, summary)
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(
            lines, "Overlapping HMX/HVX/DMA, ring and waits",
            tuple(dict.fromkeys((*exp107.OVERLAP, *exp112.W4U8_PIPELINE,
                                 *exp124.RING_FIELDS))), values)
        exp112.add_table(
            lines, "Down command grouping",
            ("w4u8_mlp_down_hmx_batch_n_tiles",
             "w4u8_mlp_down_hmx_command_count",
             "w4u8_mlp_down_in_command_slot_release_count",
             "w4u8_mlp_down_producer_progress_command_count",
             "w4u8_mlp_down_pipeline_ticks"), values)
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness and audits | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality excluding command grouping | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Down command reduction | {'PASS' if summary['command_reduction_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| Down and Host strict speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| QKV, Attention and Gate/Up preservation | {'PASS' if summary['upstream_preservation_gate'] else 'FAIL'} |",
        f"| EXP-0145 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
