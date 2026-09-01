#!/usr/bin/env python3
"""Validate and report EXP-0146 W4F16 cross-projection QKV ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as common
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
OUTPUT_HASH = "f18b9abbe1487231"
VTCM_BYTES = 8_388_608
PHYSICAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "weight_ddr_read_bytes",
    "weight_dma_descriptor_count", "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "crouton_qkv_projection_count", "crouton_qkv_unpack_skipped",
    "crouton_qk_operand_count", "crouton_av_weight_count",
)
DETAIL_FIELDS = (
    "host_wall_ns_per_block", "qkv_plus_qk_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "total_ticks",
    "weight_dma_ticks", "w4f16_expand_ticks",
    "w4f16_expand_work_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_prefetch_wait_ticks", "w4f16_hmx_tail_wait_ticks",
    "projection_hmx_wait_ticks", "attention_qk_norm_pool_wait_ticks",
    "w4f16_first_expand_ticks", "w4f16_steady_expand_ticks",
    "w4f16_cross_prefetch_count", "w4f16_cross_prefetch_wait_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    rows = base.load_jsonl(path, count)
    for row in rows:
        row["qkv_plus_qk_ticks"] = (
            int(row["qkv_projection_ticks"]) +
            int(row["qk_norm_rope_ticks"])
        )
    return rows


def require_zero(record: dict, fields: tuple[str, ...], cell: str) -> None:
    for field in fields:
        if int(record.get(field, -1)) != 0:
            raise SystemExit(f"{cell}: expected {field}=0")


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    base.require(record, "experiment", "EXP-0146")
    base.require(record, "variant", "W4F16")
    base.require(record, "qkv_schedule_mode",
                 "cross_projection_ring" if cell == "candidate"
                 else "control")
    base.require(record, "w4f16_group_fence_mode", "join_only_down")
    base.require(record, "crouton_boundary_mode", "qkv_norms")
    base.require(record, "attention_pipeline_mode", "gqa_qkv_overlap")
    base.require(record, "w4f16_requested_hvx_workers", 4)
    base.require(record, "w4f16_hvx_workers_created", 4)
    base.require(record, "w4f16_hvx_workers_locked", 4)
    base.require(record, "vtcm_requested_bytes", VTCM_BYTES)
    base.require(record, "vtcm_acquired_bytes", VTCM_BYTES)
    base.require(record, "block_invocation_count", repeat)
    base.require(record, "rpc_result", 0)
    base.require(record, "output_hash", OUTPUT_HASH)
    base.require(record, "crouton_qkv_projection_count", 3)
    base.require(record, "crouton_qkv_unpack_skipped", 128)
    base.require(record, "qkv_operand_audit_tensor_count", 3)
    base.require(record, "hmx_u8s8_tile_pair_count", 0)
    if cell == "candidate":
        base.require(record, "qkv_schedule_command_count", 64)
        if int(record["qkv_schedule_trace_hash"], 16) == 0:
            raise SystemExit("candidate: empty QKV schedule trace")
    else:
        base.require(record, "qkv_schedule_command_count", 0)
    require_zero(record, (
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
        "w4f16_expand_mismatch_count", "crouton_q_operand_mismatch_count",
        "crouton_k_operand_mismatch_count", "crouton_v_operand_mismatch_count",
    ), cell)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit(f"{cell}: VTCM plan exceeds 8 MiB")
    if audit and (int(record.get("mismatches", -1)) != 0 or
                  int(record.get("max_lsb", -1)) != 0):
        raise SystemExit(f"{cell}: audited output mismatch")


def physical_equal(control: list[dict], candidate: list[dict]) -> bool:
    return all(
        common.summarize(control, candidate, field)["control"] ==
        common.summarize(control, candidate, field)["candidate"]
        for field in PHYSICAL_FIELDS
    )


def build_summary(result_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0146")
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
            "qkv_schedule_command_count": row["qkv_schedule_command_count"],
            "qkv_schedule_trace_hash": row["qkv_schedule_trace_hash"],
        }

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    physical_values = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = {
            field: common.summarize(
                sides["control"], sides["candidate"], field)
            for field in DETAIL_FIELDS
        }
        physical_values.append(physical_equal(
            sides["control"], sides["candidate"]))
        for field in ("qkv_plus_qk_ticks", "host_wall_ns_per_block"):
            values = comparisons[f"repeat{repeat}"][field]
            speed_values.extend((values["change_percent"],
                                 values["paired_change_percent_median"]))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    physical_gate = all(physical_values)
    correctness_gate = (
        correctness["control"]["output_hash"] == OUTPUT_HASH and
        correctness["candidate"]["output_hash"] == OUTPUT_HASH
    )
    return {
        "experiment": "EXP-0146",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "physical_equality_gate": physical_gate,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gate": speed_gate,
        "local_gate_pass": correctness_gate and physical_gate and speed_gate,
        "selected_cell": "candidate" if speed_gate else "control",
        "comparisons": comparisons,
        "pc028": {
            cell: common.modules(records[10][cell]) for cell in CELLS
        },
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def render_report(summary: dict) -> str:
    lines = ["# EXP-0146 — Complete profiling report", ""]
    for repeat in REPEATS:
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Metric | Control | Cross-projection ring | Delta | Paired delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for field, values in summary["comparisons"][f"repeat{repeat}"].items():
            lines.append(
                f"| `{field}` | {values['control']:.3f} | "
                f"{values['candidate']:.3f} | "
                f"{fmt(values['change_percent'])} | "
                f"{fmt(values['paired_change_percent_median'])} |"
            )
        lines.append("")
    lines.extend([
        "## PC-027 / gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical contract equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| QKV and Host strict speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0146 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
        "## PC-028 / repeat10 module wall-time", "",
        "| Module | Control | Cross-projection ring |",
        "|---|---:|---:|",
    ])
    for name in summary["pc028"]["control"]:
        lines.append(
            f"| {name} | {summary['pc028']['control'][name]:.1f} us | "
            f"{summary['pc028']['candidate'][name]:.1f} us |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
