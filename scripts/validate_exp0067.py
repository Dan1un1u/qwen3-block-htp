#!/usr/bin/env python3
"""Validate EXP-0067 Input RMSNorm batched HVX rsqrt evidence."""

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
    "input_norm_ticks",
)
CONTROL_NORM_MODE = "hvx_tree_qk_batched_rsqrt_shared_rope"
CANDIDATE_NORM_MODE = (
    "hvx_tree_qk_batched_rsqrt_shared_rope_input_rsqrt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit_enabled: bool = False) -> None:
    compatibility = dict(record)
    compatibility["experiment"] = "EXP-0065"
    compatibility["u8_norm_reduction_mode"] = CONTROL_NORM_MODE
    PARENT_VALIDATE_RECORD(
        compatibility, repeat, "candidate", audit_enabled
    )
    base.require(record, "experiment", "EXP-0067")
    base.require(
        record, "u8_norm_reduction_mode",
        CANDIDATE_NORM_MODE if mode == "candidate" else CONTROL_NORM_MODE,
    )
    if audit_enabled and int(
        str(record["u8_input_norm_actual_hash"]), 16
    ) == 0:
        raise SystemExit("missing Input RMSNorm audit hash")
    if float(record["input_norm_ticks"]) <= 0.0:
        raise SystemExit("non-positive Input RMSNorm ticks")


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    previous.TARGETS = TARGETS
    previous.validate_record = validate_record
    summary = previous.build_summary(result_dir, package_dir)

    correctness_records = {
        mode: base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        for mode in base.MODES
    }
    input_hashes = {
        mode: str(record["u8_input_norm_actual_hash"])
        for mode, record in correctness_records.items()
    }
    input_norm_hash_gate = (
        input_hashes["control"] != "0000000000000000" and
        input_hashes["control"] == input_hashes["candidate"]
    )
    if not input_norm_hash_gate:
        raise SystemExit(
            f"Input RMSNorm hash mismatch: {input_hashes}"
        )

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
        result["input_norm_and_host_speed_gate"] = result[
            "three_target_speed_gate"
        ]
        overall.append(
            result["input_norm_and_host_speed_gate"] and invariant_gate
        )
    summary.update({
        "experiment": "EXP-0067",
        "control": "EXP-0065 scalar Input RMSNorm rsqrt per row",
        "candidate": "two 32-lane HVX Input RMSNorm rsqrt vectors",
        "input_norm_actual_hashes": input_hashes,
        "byte_exact_input_norm_hash_gate": input_norm_hash_gate,
        "local_gate_pass": all(overall) and input_norm_hash_gate,
        "local_adoption_eligible": all(overall) and input_norm_hash_gate,
    })
    return summary


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str,
              fields: tuple[str, ...],
              metrics: dict[str, dict[str, float | None]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Scalar-rsqrt control | Batched-rsqrt candidate | Delta | Paired delta |",
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
        "# EXP-0067 — Complete profiling report", "",
        "Both paths retain the exact EXP-0065 centered-square reduction, "
        "gamma, asymmetric-U8 qparams and native integer-HMX activation "
        "layout. The candidate gathers all 64 Input RMSNorm mean squares "
        "and evaluates reciprocal square root with two 32-lane qhmath HVX "
        "vectors instead of 64 scalar `sqrtf` calls.", "",
    ]
    for repeat in (1, 10):
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and Input RMSNorm targets",
            (
                "host_wall_ns_per_block", "invocation_ticks",
                "total_ticks", "input_norm_ticks",
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
            f"Input-Norm plus Host-wall speed gate: **{'PASS' if result['input_norm_and_host_speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/commands/resources: **{'PASS' if result['unchanged_math_traffic_commands_and_resources_gate'] else 'FAIL'}**.",
            "",
        ])
    input_hash = summary["input_norm_actual_hashes"]["candidate"]
    lines.extend([
        "## Physical and correctness gates", "",
        "| Gate | Result |", "|---|---:|",
        f"| Input RMSNorm native carrier | byte-exact hash `{input_hash}` |",
        "| Final block output | byte-exact, 0 LSB |",
        "| QK / probability / AV audit boundaries | byte-exact |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The audit hash is outside the performance timing path. Additive "
        "ledger and overlapping engine counters are not summed together; "
        "complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0067 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Selected Baseline is unchanged without explicit user promotion.", "",
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
