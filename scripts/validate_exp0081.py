#!/usr/bin/env python3
"""Validate EXP-0081 idle Q/K-preparation worker-assisted Q expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0070 as pc
import validate_exp0078 as report_base
import validate_exp0079 as previous


base = previous.base
SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
TARGETS = ("host_wall_ns_per_block", "qkv_projection_ticks")
OUTPUT_HASH = "69f22eeb035e5ec5"
BOUNDARY_HASHES = (
    "7255c2406108617c", "32aa949912e365be",
    "94f2e218f06f9627", "f853658f52032bde",
)
LEDGER = previous.LEDGER
ASSIST_FIELDS = (
    "w4u8_qkv_worker_assist_batch_count",
    "w4u8_qkv_worker_assist_tile_count",
    "w4u8_qkv_main_expand_tile_count",
    "w4u8_qkv_worker_expand_ticks",
    "w4u8_qkv_main_expand_ticks",
    "w4u8_qkv_worker_assist_wait_ticks",
)
OVERLAP = tuple(dict.fromkeys((*previous.OVERLAP, *ASSIST_FIELDS)))
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *previous.EXTRA_REPORT_FIELDS, *ASSIST_FIELDS,
)))


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
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4_qk_head_pairs",
    })
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    fixed = {
        "experiment": "EXP-0081",
        "mlp_mode": "w4u8_streaming_persistent_mlp_hvx",
        "w4u8_qkvo_pipeline_mode": (
            "qkvo_batch4_qk_head_pairs"
            if mode == "control" else
            "qkvo_batch4_qk_head_pairs_q_worker_expand"
        ),
        "w4u8_gate_up_persistent_hvx_dispatch_count": repeat,
        "w4u8_gate_up_persistent_hvx_worker_count": 3 * repeat,
        "w4u8_gate_up_transient_hvx_thread_count": 0,
        "w4u8_down_persistent_hvx_dispatch_count": repeat,
        "w4u8_down_persistent_hvx_worker_count": 5 * repeat,
        "w4u8_down_transient_hvx_thread_count": repeat,
        "hmx_command_count": 176 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "attention_qk_norm_task_count": 24 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    expected_qkv_tiles = 128 * repeat
    assist_batches = int(record["w4u8_qkv_worker_assist_batch_count"])
    assist_tiles = int(record["w4u8_qkv_worker_assist_tile_count"])
    main_tiles = int(record["w4u8_qkv_main_expand_tile_count"])
    if mode == "control":
        if assist_batches != 0 or assist_tiles != 0:
            raise SystemExit("control executed worker-assisted expansion")
        if int(record["w4u8_qkv_worker_expand_ticks"]) != 0:
            raise SystemExit("control reported worker expansion work")
    else:
        if not (0 < assist_batches <= 16 * repeat):
            raise SystemExit("candidate assisted-batch count is invalid")
        if not (0 < assist_tiles <= 32 * repeat):
            raise SystemExit("candidate assisted-tile count is invalid")
        if int(record["w4u8_qkv_worker_expand_ticks"]) <= 0:
            raise SystemExit("candidate lacks worker expansion work")
    if main_tiles + assist_tiles != expected_qkv_tiles:
        raise SystemExit("QKV expansion tile accounting does not close")
    for field in TARGETS + ("invocation_ticks", "total_ticks"):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


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
    for mode in MODES:
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
    if (observed_boundaries["control"] != observed_boundaries["candidate"]
            or observed_boundaries["control"] != BOUNDARY_HASHES):
        raise SystemExit("candidate changed an audited boundary")

    invariant_fields = (
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "weight_dma_descriptor_count", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count",
        "intermediate_spill_fill_count", "w4u8_qkv_batch_count",
        "w4u8_qkvo_prefetch_count", "w4u8_qkvo_overlap_schedule_count",
        "attention_qk_norm_task_count", "attention_softmax_task_count",
        "u8_attention_group_count", "u8_attention_qk_execution_count",
        "u8_attention_av_execution_count", "vtcm_requested_bytes",
        "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
        "attention_hvx_workers_created", "attention_hvx_workers_locked",
        "w4u8_mlp_gate_up_hmx_command_count",
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
            metrics["w4u8_qkv_worker_assist_batch_count"]["control"] == 0.0
            and metrics["w4u8_qkv_worker_assist_batch_count"]["candidate"] > 0.0
            and metrics["w4u8_qkv_worker_assist_tile_count"]["control"] == 0.0
            and metrics["w4u8_qkv_worker_assist_tile_count"]["candidate"] > 0.0
            and metrics["attention_qk_norm_task_count"]["control"] == 24.0
            and metrics["attention_qk_norm_task_count"]["candidate"] == 24.0
        )
        passed = speed_gate and invariant_gate and execution_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "worker_assistance_execution_gate": execution_gate,
        }
    return {
        "experiment": "EXP-0081",
        "control": "EXP-0079 main-HVX QKV expansion",
        "candidate": "idle Q/K-preparation worker-assisted Q expansion",
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
    current_total = current["Complete block Host wall"]
    used = "EXP-0081 candidate" if summary["local_gate_pass"] else "EXP-0079 control"
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
                f"{current[name]:.1f} us ({100*current[name]/current_total:.1f}%)",
            )
        speed = (w4f16[name] / current[name] - 1.0) * 100.0
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = ["# EXP-0081 — Complete profiling report", ""]
    add_pc028(lines, summary)
    lines.extend([
        "The candidate atomically claims only persistent Q/K-preparation "
        "workers that are currently waiting for a future head. Claimed "
        "workers expand disjoint Q-projection W4 tiles; unavailable workers "
        "are skipped and the main HVX context immediately retains their tiles. "
        "K, V, O, HMX commands, DMA, arithmetic and all other stages are unchanged.", "",
    ])
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        report_base.add_table(
            lines, "Primary wall-latency targets",
            (*TARGETS, "invocation_ticks", "total_ticks"), metrics)
        report_base.add_table(
            lines, "Additive Block Timing Ledger", LEDGER, metrics)
        report_base.add_table(
            lines, "Overlapping engine work and waits", OVERLAP, metrics)
        report_base.add_table(
            lines, "Traffic, commands, counters and residency",
            tuple(dict.fromkeys((
                *base.COUNTERS, *base.RESOURCES,
                *EXTRA_REPORT_FIELDS,
            ))), metrics)
        lines.extend([
            f"Two-target speed gate: **{'PASS' if result['two_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged physical contract: **{'PASS' if result['unchanged_physical_contract_gate'] else 'FAIL'}**; "
            f"worker assistance execution: **{'PASS' if result['worker_assistance_execution_gate'] else 'FAIL'}**.", "",
        ])
    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final output and audited boundaries | byte-exact to EXP-0079, 0 LSB |",
        "| Q/K preparation workers and logical tasks | unchanged 5 workers; 24 logical heads/block |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write and spill/fill | 0 / 0 bytes; 0 |",
        "| FastRPC / HMX ownership / QNN | one / one / none |", "",
        "## Decision", "",
        f"EXP-0081 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Selected Baseline remains unchanged without explicit user promotion.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
