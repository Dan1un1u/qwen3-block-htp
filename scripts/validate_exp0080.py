#!/usr/bin/env python3
"""Validate EXP-0080 Q-to-K-to-V cross-projection prefetch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0079 as previous


base = previous.base
SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
TARGETS = ("host_wall_ns_per_block", "qkv_projection_ticks")
OUTPUT_HASH = "69f22eeb035e5ec5"
BOUNDARY_HASHES = previous.BOUNDARY_HASHES
LEDGER = previous.LEDGER
OVERLAP = previous.OVERLAP
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *previous.EXTRA_REPORT_FIELDS,
    "w4u8_qkv_cross_prefetch_count",
    "w4u8_qkv_cross_prefetch_adoption_count",
    "w4u8_qkv_cross_prefetch_wait_ticks",
    "w4u8_qkv_cross_prefetch_lifetime_ticks",
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
        "experiment": "EXP-0080",
        "w4u8_qkvo_pipeline_mode": (
            "qkvo_batch4_qk_head_pairs" if mode == "control" else
            "qkvo_batch4_qk_head_pairs_cross_qkv_prefetch"
        ),
        "w4u8_qkv_cross_prefetch_count": (
            0 if mode == "control" else 2 * repeat
        ),
        "w4u8_qkv_cross_prefetch_adoption_count": (
            0 if mode == "control" else 2 * repeat
        ),
        "hmx_command_count": 176 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
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
    observed: dict[str, tuple[str, str, str, str]] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(record, 1, mode, audit_enabled=True)
        observed[mode] = (
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
            "boundary_hashes": observed[mode],
        }
    if observed["control"] != observed["candidate"] or (
            observed["control"] != BOUNDARY_HASHES):
        raise SystemExit("candidate changed an audited boundary")

    invariant_fields = (
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "weight_dma_descriptor_count", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
        "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
        "w4u8_qkvo_overlap_schedule_count", "attention_qk_norm_task_count",
        "attention_softmax_task_count", "u8_attention_group_count",
        "u8_attention_qk_execution_count", "u8_attention_av_execution_count",
        "vtcm_requested_bytes", "vtcm_acquired_bytes",
        "vtcm_peak_plan_bytes", "attention_hvx_workers_created",
        "attention_hvx_workers_locked", "w4u8_mlp_gate_up_hmx_command_count",
        "w4u8_mlp_down_hmx_command_count",
        "w4u8_gate_up_persistent_hvx_dispatch_count",
        "w4u8_gate_up_persistent_hvx_worker_count",
        "w4u8_down_persistent_hvx_dispatch_count",
        "w4u8_down_persistent_hvx_worker_count",
        "w4u8_down_transient_hvx_thread_count",
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
            metrics["w4u8_qkv_cross_prefetch_count"]["control"] == 0.0
            and metrics["w4u8_qkv_cross_prefetch_count"]["candidate"] == 2.0
            and metrics["w4u8_qkv_cross_prefetch_adoption_count"]["control"] == 0.0
            and metrics["w4u8_qkv_cross_prefetch_adoption_count"]["candidate"] == 2.0
        )
        passed = speed_gate and invariant_gate and execution_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "hybrid_worker_execution_gate": execution_gate,
        }
    return {
        "experiment": "EXP-0080",
        "control": "EXP-0079 per-projection QKV first-batch DMA",
        "candidate": "Q-to-K-to-V cross-projection first-batch prefetch",
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


def render_report(summary: dict[str, object]) -> str:
    previous_targets = previous.TARGETS
    previous.TARGETS = TARGETS
    try:
        report = previous.render_report(summary)
    finally:
        previous.TARGETS = previous_targets
    report = report.replace(
        "# EXP-0079 — Complete profiling report",
        "# EXP-0080 — Complete profiling report",
    ).replace(
        "EXP-0079 candidate supplies the W4U8 column",
        "EXP-0080 candidate supplies the W4U8 column",
    ).replace(
        "EXP-0078 control supplies the W4U8 column",
        "EXP-0079 control supplies the W4U8 column",
    )
    old = (
        "The candidate preserves six Down expansion consumers but replaces "
        "five transient thread/HVX lifecycles with five existing persistent "
        "pool workers. One transient sixth worker and all queue, DMA, HMX, "
        "layout and arithmetic contracts remain unchanged."
    )
    new = (
        "The candidate starts K and V first-batch weight-and-bias DMA during "
        "the preceding Q or K tail HMX, then adopts the completed batch as the "
        "next projection's first alternating slot. Arithmetic, batch size, "
        "DMA descriptors, HMX commands and all other stages are unchanged."
    )
    report = report.replace(old, new)
    report = report.replace("hybrid worker execution", "cross-prefetch execution")
    report = report.replace(
        "| Down HVX ownership | control 0 persistent + 6 transient; candidate 5 persistent + 1 transient |",
        "| QKV cross-prefetch ownership | control 0; candidate 2 prefetches and 2 adoptions per block |",
    )
    report = report.replace("byte-exact to EXP-0078", "byte-exact to EXP-0079")
    report = report.replace("EXP-0079 local gate", "EXP-0080 local gate")
    return report


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
