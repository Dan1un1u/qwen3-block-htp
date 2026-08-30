#!/usr/bin/env python3
"""Validate EXP-0083 wider W4U8 Gate/Up HMX command batches."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import validate_exp0070 as pc
import validate_exp0078 as report_base
import validate_exp0079 as previous


base = previous.base
SAMPLES = 7
SEARCH_SAMPLES = 3
REPEATS = (1, 10)
MODES = ("control", "candidate")
SEARCH_MODES = ("control", "candidate", "diagnostic32")
TARGETS = ("host_wall_ns_per_block", "gate_up_ticks")
OUTPUT_HASH = "69f22eeb035e5ec5"
BOUNDARY_HASHES = (
    "7255c2406108617c", "32aa949912e365be",
    "94f2e218f06f9627", "f853658f52032bde",
)
LEDGER = previous.LEDGER
OVERLAP = previous.OVERLAP
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *previous.EXTRA_REPORT_FIELDS,
    "w4u8_mlp_gate_up_hmx_batch_n_tiles",
    "w4u8_mlp_gate_up_expanded_slot_count",
    "w4u8_mlp_vtcm_plan_bytes",
)))

BATCH = {"control": 8, "candidate": 16, "diagnostic32": 32}
GATE_COMMANDS = {"control": 48, "candidate": 24, "diagnostic32": 12}
TOTAL_COMMANDS = {"control": 176, "candidate": 152, "diagnostic32": 140}
PLAN_BYTES = {
    # This telemetry is max(Gate/Up plan, Down plan).  With the strict
    # eight-slot control restored, Down still dominates the control maximum.
    "control": 1_708_032,
    "candidate": 1_875_968,
    "diagnostic32": 2_961_408,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit_enabled: bool = False) -> None:
    compatibility = dict(record)
    compatibility.update({
        "experiment": "EXP-0079",
        "w4u8_mlp_gate_up_hmx_batch_n_tiles": 8,
        "w4u8_mlp_gate_up_expanded_slot_count": 8,
        "w4u8_mlp_vtcm_plan_bytes": PLAN_BYTES["control"],
        "w4u8_mlp_gate_up_hmx_command_count": 48 * repeat,
        "hmx_command_count": 176 * repeat,
    })
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    fixed = {
        "experiment": "EXP-0083",
        "mlp_mode": "w4u8_streaming_persistent_mlp_hvx",
        "w4u8_mlp_gate_up_hmx_batch_n_tiles": BATCH[mode],
        "w4u8_mlp_gate_up_expanded_slot_count": BATCH[mode],
        "w4u8_mlp_vtcm_plan_bytes": PLAN_BYTES[mode],
        "w4u8_mlp_gate_up_hmx_command_count":
            GATE_COMMANDS[mode] * repeat,
        "w4u8_mlp_down_hmx_command_count": 64 * repeat,
        "w4u8_gate_up_persistent_hvx_dispatch_count": repeat,
        "w4u8_gate_up_persistent_hvx_worker_count": 3 * repeat,
        "w4u8_gate_up_transient_hvx_thread_count": 0,
        "w4u8_down_persistent_hvx_dispatch_count": repeat,
        "w4u8_down_persistent_hvx_worker_count": 5 * repeat,
        "w4u8_down_transient_hvx_thread_count": repeat,
        "hmx_command_count": TOTAL_COMMANDS[mode] * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    for field in TARGETS + ("invocation_ticks", "total_ticks"):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def summarize_search(result_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    fields = (
        "host_wall_ns_per_block", "gate_up_ticks",
        "w4u8_mlp_gate_up_pipeline_ticks",
        "w4u8_mlp_hmx_ready_wait_ticks",
        "w4u8_mlp_producer_slot_wait_ticks",
        "w4u8_mlp_expanded_slot_wait_ticks",
        "w4u8_mlp_vtcm_plan_bytes",
        "w4u8_mlp_gate_up_hmx_command_count",
    )
    for repeat in REPEATS:
        repeat_result: dict[str, object] = {}
        for mode in SEARCH_MODES:
            records = base.load_jsonl(
                result_dir / f"search_{mode}_r{repeat}.jsonl",
                SEARCH_SAMPLES,
            )
            for record in records:
                validate_record(record, repeat, mode)
            repeat_result[mode] = {
                field: statistics.median(
                    float(record[field]) for record in records
                )
                for field in fields
            }
        result[f"repeat{repeat}"] = repeat_result
    return result


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed")
    static_gate = json.loads((result_dir / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness: dict[str, object] = {}
    observed_boundaries: dict[str, tuple[str, str, str, str]] = {}
    for mode in SEARCH_MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(record, 1, mode, audit_enabled=True)
        observed_boundaries[mode] = (
            str(record["u8_input_norm_actual_hash"]),
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        if record["output_hash"] != OUTPUT_HASH:
            raise SystemExit(f"wrong final output hash for {mode}")
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "boundary_hashes": observed_boundaries[mode],
        }
    if any(value != BOUNDARY_HASHES
           for value in observed_boundaries.values()):
        raise SystemExit("a wider batch changed an audited boundary")

    search = summarize_search(result_dir)
    invariant_fields = (
        "hmx_u8s8_tile_pair_count", "weight_dma_descriptor_count",
        "weight_ddr_read_bytes", "boundary_ddr_read_bytes",
        "boundary_ddr_write_bytes", "intermediate_ddr_read_bytes",
        "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
        "intermediate_spill_fill_count", "w4u8_qkv_batch_count",
        "w4u8_qkvo_prefetch_count", "w4u8_qkvo_overlap_schedule_count",
        "attention_qk_norm_task_count", "attention_softmax_task_count",
        "u8_attention_group_count", "u8_attention_qk_execution_count",
        "u8_attention_av_execution_count", "vtcm_requested_bytes",
        "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
        "attention_hvx_workers_created", "attention_hvx_workers_locked",
        "w4u8_mlp_down_hmx_command_count",
        "w4u8_gate_up_persistent_hvx_dispatch_count",
        "w4u8_gate_up_persistent_hvx_worker_count",
        "w4u8_gate_up_transient_hvx_thread_count",
        "w4u8_down_persistent_hvx_dispatch_count",
        "w4u8_down_persistent_hvx_worker_count",
        "w4u8_down_transient_hvx_thread_count",
        "w4u8_post_residual_task_count", "w4u8_final_residual_task_count",
    )
    repeat_results: dict[str, object] = {}
    overall: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: base.load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl", SAMPLES
            ) for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        fields = tuple(dict.fromkeys((
            *TARGETS, "invocation_ticks", "total_ticks", *LEDGER,
            *OVERLAP, *base.COUNTERS, *base.RESOURCES,
            *EXTRA_REPORT_FIELDS,
        )))
        metrics = {
            field: base.summarize(
                records["control"], records["candidate"], field
            ) for field in fields
        }
        for field in (
            "w4u8_mlp_gate_up_hmx_batch_n_tiles",
            "w4u8_mlp_gate_up_expanded_slot_count",
            "w4u8_mlp_vtcm_plan_bytes",
        ):
            metrics[field]["control"] *= repeat
            metrics[field]["candidate"] *= repeat
        speed_gate = all(
            metrics[field][key] < 0.0
            for field in TARGETS
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in invariant_fields
        )
        execution_gate = (
            metrics["w4u8_mlp_gate_up_hmx_batch_n_tiles"]["control"] == 8.0
            and metrics["w4u8_mlp_gate_up_hmx_batch_n_tiles"]["candidate"] == 16.0
            and metrics["w4u8_mlp_gate_up_hmx_command_count"]["control"] == 48.0
            and metrics["w4u8_mlp_gate_up_hmx_command_count"]["candidate"] == 24.0
            and metrics["hmx_command_count"]["control"] == 176.0
            and metrics["hmx_command_count"]["candidate"] == 152.0
        )
        passed = speed_gate and invariant_gate and execution_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "wide_batch_execution_gate": execution_gate,
        }
    return {
        "experiment": "EXP-0083",
        "control": "EXP-0079 Gate/Up HMX batch eight",
        "candidate": "Gate/Up HMX batch sixteen",
        "diagnostic": "Gate/Up HMX batch thirty-two",
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "byte_exact_final_output_gate": True,
        "byte_exact_audited_boundaries_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "correctness": correctness,
        "bounded_search": search,
        "repeat_results": repeat_results,
        "local_gate_pass": all(overall),
        "local_adoption_eligible": all(overall),
    }


def add_pc028(lines: list[str], summary: dict[str, object]) -> None:
    metrics = summary["repeat_results"]["repeat10"]["metrics"]
    side = "candidate" if summary["local_gate_pass"] else "control"
    current = dict(pc.module_us(metrics, side))
    f16 = dict(zip(current, (6.6, 42.7, 397.1, 140.3, 201.5, 41.2,
                            1116.6, 459.9, 5.0, 77.5, 2488.3)))
    w4f16 = dict(zip(current, (7.4, 42.7, 439.2, 140.7, 173.9, 41.2,
                              972.7, 327.3, 5.0, 79.6, 2229.7)))
    used = "EXP-0083 batch16" if summary["local_gate_pass"] else "EXP-0079 batch8 control"
    total = current["Complete block Host wall"]
    lines.extend([
        "## PC-028 current-best three-variant module wall-time", "",
        f"{used} supplies the W4U8 column; other columns reuse valid formal evidence.", "",
        "| Module | W16A16 | W4A16 | W4A8 current best | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in current:
        if name == "Complete block Host wall":
            cells = (f"{f16[name]:.1f} us", f"{w4f16[name]:.1f} us",
                     f"{current[name]:.1f} us")
        else:
            cells = (
                f"{f16[name]:.1f} us ({100*f16[name]/f16['Complete block Host wall']:.1f}%)",
                f"{w4f16[name]:.1f} us ({100*w4f16[name]/w4f16['Complete block Host wall']:.1f}%)",
                f"{current[name]:.1f} us ({100*current[name]/total:.1f}%)",
            )
        speed = (w4f16[name] / current[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = ["# EXP-0083 — Complete profiling report", ""]
    add_pc028(lines, summary)
    lines.extend([
        "The bounded search changes only Gate/Up HMX command width and the matching phase-local expanded/pair rings. Batch sixteen and thirty-two reduce Gate/Up command handoffs while preserving every packed-W4 byte, HMX tile pair and output byte.", "",
        "## Bounded 8/16/32 search", "",
        "| Repeat | Batch | Host ns/block | Gate/Up ticks/block | Pipeline ticks/block | HMX commands/block | HMX-ready wait | Producer-slot wait | Expanded-slot wait | MLP plan bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for repeat in REPEATS:
        for mode in SEARCH_MODES:
            item = summary["bounded_search"][f"repeat{repeat}"][mode]
            lines.append(
                f"| {repeat} | {BATCH[mode]} | {item['host_wall_ns_per_block']:.1f} | {item['gate_up_ticks'] / repeat:.1f} | {item['w4u8_mlp_gate_up_pipeline_ticks'] / repeat:.1f} | {item['w4u8_mlp_gate_up_hmx_command_count'] / repeat:.1f} | {item['w4u8_mlp_hmx_ready_wait_ticks'] / repeat:.1f} | {item['w4u8_mlp_producer_slot_wait_ticks'] / repeat:.1f} | {item['w4u8_mlp_expanded_slot_wait_ticks'] / repeat:.1f} | {item['w4u8_mlp_vtcm_plan_bytes']:.0f} |"
            )
    lines.append("")
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        report_base.add_table(
            lines, "Primary wall-latency targets",
            (*TARGETS, "invocation_ticks", "total_ticks"), metrics
        )
        report_base.add_table(
            lines, "Additive Block Timing Ledger", LEDGER, metrics
        )
        report_base.add_table(
            lines, "Overlapping engine work and waits", OVERLAP, metrics
        )
        report_base.add_table(
            lines, "Traffic, commands, counters and residency",
            tuple(dict.fromkeys((
                *base.COUNTERS, *base.RESOURCES, *EXTRA_REPORT_FIELDS,
            ))), metrics
        )
        lines.extend([
            f"Two-target speed gate: **{'PASS' if result['two_target_speed_gate'] else 'FAIL'}**; unchanged physical contract: **{'PASS' if result['unchanged_physical_contract_gate'] else 'FAIL'}**; wide-batch execution: **{'PASS' if result['wide_batch_execution_gate'] else 'FAIL'}**.", "",
        ])
    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final output and audited boundaries | all three batches byte-exact to EXP-0079, 0 LSB |",
        "| Gate/Up command counts | batch8 48; batch16 24; batch32 12 |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write and spill/fill | 0 / 0 bytes; 0 |",
        "| FastRPC / HMX ownership / QNN | one / one / none |", "",
        "## Decision", "",
        f"EXP-0083 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. Selected Baseline remains unchanged without explicit user promotion.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
