#!/usr/bin/env python3
"""Validate EXP-0064 QK/AV HMX output-tile batching evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0063 as previous


OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
TARGETS = (
    "host_wall_ns_per_block",
    "u8_attention_qk_av_hmx_ticks",
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
        "u8_log2_gqa_qkv_overlap_vgather_vdeal_"
        "fused_qk_requant_hmx_batch"
        if mode == "candidate"
        else "u8_log2_gqa_qkv_overlap_vgather_vdeal_"
             "fused_qk_requant"
    )
    expected_commands = (192 if mode == "candidate" else 256) * repeat
    compatibility = dict(record)
    compatibility["experiment"] = "EXP-0063"
    compatibility["attention_pipeline_mode"] = (
        "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant"
    )
    compatibility["hmx_command_count"] = 256 * repeat
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    base.require(record, "experiment", "EXP-0064")
    base.require(record, "attention_pipeline_mode", expected_pipeline)
    base.require(record, "hmx_command_count", expected_commands)
    if float(record["u8_attention_qk_av_hmx_ticks"]) <= 0.0:
        raise SystemExit("non-positive combined QK+AV HMX ticks")


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if "u8_attention_qk_av_hmx_ticks" not in base.OVERLAP:
        base.OVERLAP += ("u8_attention_qk_av_hmx_ticks",)
    previous.TARGETS = TARGETS
    previous.validate_record = validate_record
    summary = previous.build_summary(result_dir, package_dir)
    invariant_fields = (
        "hmx_u8s8_tile_pair_count",
        "weight_dma_descriptor_count", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count",
        "intermediate_spill_fill_count", "w4u8_qkv_batch_count",
        "w4u8_qkvo_prefetch_count",
        "w4u8_qkvo_overlap_schedule_count",
        "attention_qk_norm_task_count",
        "vtcm_requested_bytes", "vtcm_acquired_bytes",
        "vtcm_peak_plan_bytes",
    )
    overall = []
    for repeat in (1, 10):
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in invariant_fields
        )
        command_gate = (
            metrics["hmx_command_count"]["control"] == 256.0 and
            metrics["hmx_command_count"]["candidate"] == 192.0
        )
        result["unchanged_math_traffic_and_resources_gate"] = (
            invariant_gate
        )
        result["attention_hmx_command_batch_gate"] = command_gate
        overall.append(
            result["three_target_speed_gate"] and
            invariant_gate and command_gate
        )
    summary.update({
        "experiment": "EXP-0064",
        "control": "EXP-0063 one HMX command per QK/AV output tile",
        "candidate": "one QK2 and one AV4 HMX command per Q head",
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
        "| Metric | Per-tile control | Per-head batch candidate | Delta | Paired delta |",
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
        "# EXP-0064 — Complete profiling report", "",
        "The control submits one integer-HMX command per QK or AV output "
        "tile. The candidate submits both QK tiles and all four AV tiles "
        "of each Q head through the worker's existing multi-output loop. "
        "Tile arithmetic, ordering, weights, biases, qparams, score and "
        "probability paths are unchanged.", "",
    ]
    for repeat in (1, 10):
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and Attention-HMX targets",
            (
                "host_wall_ns_per_block", "invocation_ticks",
                "total_ticks", "u8_attention_qk_av_hmx_ticks",
                "u8_attention_qk_hmx_ticks",
                "u8_attention_av_hmx_ticks", "attention_ticks",
            ), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", base.LEDGER, metrics)
        add_table(
            lines, "Overlapping engine work and waits",
            base.OVERLAP, metrics,
        )
        add_table(
            lines, "Traffic, commands, counters and residency",
            base.COUNTERS + base.RESOURCES + previous.EXTRA_REPORT_FIELDS,
            metrics,
        )
        lines.extend([
            f"Three-target speed gate: **{'PASS' if result['three_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/resources: **{'PASS' if result['unchanged_math_traffic_and_resources_gate'] else 'FAIL'}**; "
            f"command batching: **{'PASS' if result['attention_hmx_command_batch_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact, 0 LSB |",
        "| QK / probability / AV audit boundaries | byte-exact |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0064 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
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
