#!/usr/bin/env python3
"""Validate EXP-0059 paired-row Softmax on the EXP-0058 Vdeal path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0057 as previous


OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
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
    fixed = {
        "experiment": "EXP-0059",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "residual_mode": "hvx_fused_post_norm_pool4",
        "crouton_boundary_mode": "w4u8_mlp_io_qkv_o",
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4_qk_head_pairs",
        "attention_pipeline_mode": (
            "u8_log2_gqa_qkv_overlap_vgather_vdeal_paired_softmax"
            if mode == "candidate"
            else "u8_log2_gqa_qkv_overlap_vgather_vdeal"
        ),
        "attention_hvx_contexts": 4,
        "mlp_mode": "w4u8_streaming",
        "repeat_count": repeat,
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
        "warmup_max_lsb": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "u8_attention_audit_ddr_write_bytes": (
            524_288 * repeat if audit_enabled else 0
        ),
        "u8_attention_fused_k_operand_mismatch_count": 0,
        "u8_attention_qkv_unpack_skipped": 128 * repeat,
        "hmx_command_count": 256 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": 4,
        "w4u8_qkv_batch_count": 32 * repeat,
        "w4u8_qkvo_prefetch_count": 44 * repeat,
        "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
        "attention_qk_norm_task_count": 24 * repeat,
        "w4u8_mlp_input_pack_skipped": repeat,
        "w4u8_mlp_output_unpack_skipped": repeat,
        "w4u8_mlp_input_pack_ticks": 0,
        "w4u8_mlp_output_unpack_ticks": 0,
        "w4u8_mlp_gate_up_hmx_batch_n_tiles": 8,
        "w4u8_mlp_gate_up_expanded_slot_count": 8,
        "w4u8_mlp_gate_up_hmx_command_count": 48 * repeat,
        "w4u8_mlp_down_hmx_command_count": 64 * repeat,
        "w4u8_post_residual_task_count": 16 * repeat,
        "w4u8_final_residual_task_count": 16 * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds 8 MiB")
    invocation = float(record["invocation_ticks"])
    closure = abs(invocation - float(record["ledger_named_ticks"])) / invocation
    if closure > 0.001:
        raise SystemExit(f"ledger closure exceeds 0.1%: {closure:.6%}")
    for field in TARGETS + ("invocation_ticks",):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    previous.TARGETS = TARGETS
    previous.validate_record = validate_record
    summary = previous.build_summary(result_dir, package_dir)
    summary.update({
        "experiment": "EXP-0059",
        "control": "EXP-0058 Vdeal path with independent Softmax rows",
        "candidate": "same Vdeal path with paired Softmax rows",
    })
    return summary


def fmt(value: float) -> str:
    return f"{value:,.3f}"


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0059 — Complete profiling report", "",
        "The control is the complete EXP-0058 V-pack path. The candidate "
        "changes only Softmax: corresponding rows from the two Q heads of "
        "one GQA group share one 128-byte HVX vector and one banked vlut32 "
        "operation, while normalization remains independent per head.", "",
    ]
    for repeat in previous.REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Metric (per block) | EXP-0058 control | Paired candidate | Delta | Paired delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for field in (
            "host_wall_ns_per_block", "invocation_ticks",
            "u8_attention_softmax_ticks", "attention_ticks",
            "u8_attention_v_pack_ticks", "qkv_projection_ticks",
            "gate_up_ticks", "down_ticks",
        ):
            metric = metrics[field]
            lines.append(
                f"| `{field}` | {fmt(metric['control'])} | "
                f"{fmt(metric['candidate'])} | "
                f"{fmt_change(metric['change_percent'])} | "
                f"{fmt_change(metric['paired_change_percent_median'])} |"
            )
        lines.extend([
            "",
            f"Three-target speed gate: **{'PASS' if result['three_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/resources: **{'PASS' if result['unchanged_math_traffic_and_resources_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "Final output and QK/probability/AV audit hashes are byte-exact. "
        "The exact 8 MiB VTCM request, zero intermediate DDR and spill/fill, "
        "one FastRPC execution unit and one HMX owner are preserved.", "",
        "## Decision", "",
        f"EXP-0059 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**.", "",
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
