#!/usr/bin/env python3
"""Validate and report EXP-0143 Q/K Norm-RoPE SIMD row packing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0124 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
EXPECTED_MODE = {"control": 1, "candidate": 2}
VTCM_BYTES = 8_388_608
OUTPUT_HASH = "69f22eeb035e5ec5"
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0110_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0110/"
    "20260831T163813Z_d18e901339f4_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks",
    "qkv_projection_ticks", "u8_attention_qk_norm_rope_ticks",
    "attention_qk_norm_pool_wait_ticks", "attention_ticks",
    "gate_up_ticks", "down_ticks", "total_ticks",
)
PHYSICAL_EQUAL_FIELDS = tuple(dict.fromkeys((
    *parent.PHYSICAL_EQUAL_FIELDS,
    "weight_ddr_read_bytes", "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes", "weight_dma_descriptor_count",
    "hmx_command_count", "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count", "vtcm_peak_plan_bytes",
)))
AUDIT_HASH_FIELDS = (
    "u8_input_norm_actual_hash", "u8_attention_actual_score_hash",
    "u8_attention_actual_probability_hash",
    "u8_attention_actual_av_hash",
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
    compatible["experiment"] = "EXP-0124"
    compatible["w4u8_qk_pair_kernel_mode"] = 1
    compatible["w4u8_qk_pair_kernel_mode_observed"] = 1
    parent.validate_record(compatible, repeat, "ring3", audit=audit)
    base.require(record, "experiment", "EXP-0143")
    base.require(record, "w4u8_qk_pair_kernel_mode",
                 EXPECTED_MODE[cell])
    base.require(record, "w4u8_qk_pair_kernel_mode_observed",
                 EXPECTED_MODE[cell])
    base.require(record, "w4u8_qkv_ring_expand_workers", 3)
    base.require(record, "block_invocation_count", repeat)
    if record.get("output_hash") != OUTPUT_HASH or \
            int(record.get("mismatches", -1)) != 0 or \
            int(record.get("max_lsb", -1)) != 0:
        raise SystemExit(f"{cell}: final output is not byte-exact")


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *parent.RING_FIELDS,
        *exp107.PHYSICAL,
    )))
    return {field: parent.summarize(left, right, field)
            for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        parent.summarize(left, right, field)["control"] ==
        parent.summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def selected_pc028(exp0109_dir: Path, exp0110_dir: Path,
                   w4u8: list[dict]) -> dict[str, dict[str, float]]:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0110_dir / "paired_carrier_r10.jsonl", SAMPLES)),
        "w4u8": exp109.module_medians(w4u8),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0143")
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
            "qk_pair_mode": row["w4u8_qk_pair_kernel_mode_observed"],
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
                    and int(row["block_invocation_count"]) == repeat
                )
            sides[cell] = rows
        records[repeat] = sides
        values = metrics(sides["control"], sides["candidate"])
        comparisons[f"repeat{repeat}"] = values
        physical_values.append(physical_equal(
            sides["control"], sides["candidate"]))
        for field in (
            "qkv_plus_qk_norm_rope_ticks",
            "u8_attention_qk_norm_rope_ticks",
            "host_wall_ns_per_block",
        ):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))
        for field in ("attention_ticks", "gate_up_ticks", "down_ticks"):
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
                      speed_gate, preservation_gate))
    selected = "candidate" if local_gate else "control"
    return {
        "experiment": "EXP-0143",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "downstream_preservation_gate": preservation_gate,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "local_gate_pass": local_gate,
        "selected_cell": selected,
        "comparisons": comparisons,
        "pc028": selected_pc028(
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
        "## EXP-0143 W4U8 module wall-time (repeat10)", "",
        "| Module | memcpy row pack | SIMD row pack | Speed |",
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
    lines = ["# EXP-0143 — Complete profiling report", ""]
    parent.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: W16A16 uses frozen EXP-0109, W4A16 uses "
        "selected EXP-0110 carrier-only, and W4A8 uses the locally eligible "
        "cell from this experiment.", "",
    ])
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
                                 *parent.RING_FIELDS))), values)
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness and audits | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| QKV/QK work and Host strict speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Attention, Gate/Up and Down preservation | {'PASS' if summary['downstream_preservation_gate'] else 'FAIL'} |",
        f"| EXP-0143 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
