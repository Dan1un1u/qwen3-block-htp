#!/usr/bin/env python3
"""Validate EXP-0066 audit-only probability row-sum reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0063 as report_fields
import validate_exp0065 as previous


PARENT_VALIDATE_RECORD = previous.validate_record
TARGETS = (
    "host_wall_ns_per_block",
    "u8_attention_softmax_ticks",
    "attention_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit_enabled: bool = False) -> None:
    expected_pipeline = (
        "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
        "hmx_batch_lut_templates_audit_reductions"
        if mode == "candidate"
        else "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
             "hmx_batch_lut_templates"
    )
    compatibility = dict(record)
    compatibility["experiment"] = "EXP-0065"
    compatibility["attention_pipeline_mode"] = (
        "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
        "hmx_batch_lut_templates"
    )
    PARENT_VALIDATE_RECORD(
        compatibility, repeat, "candidate", audit_enabled
    )
    base.require(record, "experiment", "EXP-0066")
    base.require(record, "attention_pipeline_mode", expected_pipeline)
    base.require(record, "hmx_command_count", 192 * repeat)
    if float(record["u8_attention_softmax_ticks"]) <= 0.0:
        raise SystemExit("non-positive Softmax ticks")


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    previous.TARGETS = TARGETS
    previous.validate_record = validate_record
    summary = previous.build_summary(result_dir, package_dir)
    invariant_fields = (
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "weight_dma_descriptor_count", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count",
        "intermediate_spill_fill_count", "w4u8_qkv_batch_count",
        "w4u8_qkvo_prefetch_count", "w4u8_qkvo_overlap_schedule_count",
        "attention_qk_norm_task_count", "vtcm_requested_bytes",
        "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    )
    overall = []
    for repeat in (1, 10):
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in invariant_fields
        )
        result["unchanged_math_traffic_commands_and_resources_gate"] = (
            invariant_gate
        )
        overall.append(result["three_target_speed_gate"] and invariant_gate)
    summary.update({
        "experiment": "EXP-0066",
        "control": "EXP-0065 unconditional probability row-sum reductions",
        "candidate": "probability row-sum reductions only in audit mode",
        "local_gate_pass": all(overall),
        "local_adoption_eligible": all(overall),
    })
    return summary


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str,
              fields: tuple[str, ...],
              metrics: dict[str, dict[str, float | None]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Unconditional control | Audit-only candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        metric = metrics[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, metric['control'])} | "
            f"{base.format_value(field, metric['candidate'])} | "
            f"{fmt_change(metric['change_percent'])} | "
            f"{fmt_change(metric['paired_change_percent_median'])} |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0066 — Complete profiling report", "",
        "The control reduces both probability halves and updates row-sum "
        "minima and maxima on every paired row even when telemetry is null. "
        "The candidate executes those diagnostic reductions only in numerical "
        "audit mode. Probability bytes, audit telemetry, HMX work and all "
        "other block arithmetic are unchanged.", "",
    ]
    for repeat in (1, 10):
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and Softmax targets",
            (
                "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
                "u8_attention_softmax_ticks", "attention_ticks",
            ), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", base.LEDGER, metrics)
        add_table(lines, "Overlapping engine work and waits", base.OVERLAP,
                  metrics)
        add_table(
            lines, "Traffic, commands, counters and residency",
            base.COUNTERS + base.RESOURCES + report_fields.EXTRA_REPORT_FIELDS,
            metrics,
        )
        lines.extend([
            f"Three-target speed gate: **{'PASS' if result['three_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/commands/resources: **{'PASS' if result['unchanged_math_traffic_commands_and_resources_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact, 0 LSB |",
        "| QK / probability / AV audit boundaries | byte-exact |",
        "| Probability row-sum audit telemetry | unchanged |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0066 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Selected Baseline is unchanged without explicit user promotion.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    if args.report:
        print(render_report(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
