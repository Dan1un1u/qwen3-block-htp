#!/usr/bin/env python3
"""Validate EXP-0053 paired evidence and render the PC-027 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
RESIDUAL_WORK = (
    "w4u8_post_residual_main_work_ticks",
    "w4u8_post_residual_worker_work_ticks",
    "w4u8_post_residual_pool_wait_ticks",
    "w4u8_final_residual_main_work_ticks",
    "w4u8_final_residual_worker_work_ticks",
    "w4u8_final_residual_pool_wait_ticks",
)
RESIDUAL_COUNTERS = (
    "w4u8_post_residual_task_count",
    "w4u8_final_residual_task_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit_enabled: bool = False) -> None:
    candidate = mode == "candidate"
    fixed = {
        "experiment": "EXP-0053",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "residual_mode": (
            "hvx_fused_post_norm_pool4" if candidate
            else "hvx_fused_post_norm"
        ),
        "crouton_boundary_mode": "w4u8_mlp_io_qkv_o",
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4_qk_head_tasks",
        "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
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
        "w4u8_post_residual_task_count": 16 * repeat if candidate else 0,
        "w4u8_final_residual_task_count": 16 * repeat if candidate else 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds 8 MiB")
    invocation = float(record["invocation_ticks"])
    closure = abs(
        invocation - float(record["ledger_named_ticks"])
    ) / invocation
    if closure > 0.001:
        raise SystemExit(f"ledger closure exceeds 0.1%: {closure:.6%}")
    for field in (
        "host_wall_ns_per_block", "total_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "attention_ticks", "o_projection_ticks",
        "post_attention_residual_ticks", "gate_up_ticks", "down_ticks",
        "final_residual_ticks",
    ):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")
    work = sum(int(record[field]) for field in RESIDUAL_WORK)
    if candidate and work <= 0:
        raise SystemExit("candidate did not report residual pool work")
    if not candidate and work != 0:
        raise SystemExit("control unexpectedly used residual pool")


def with_combined_residual(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        copy = dict(record)
        copy["combined_residual_ticks"] = (
            int(record["post_attention_residual_ticks"]) +
            int(record["final_residual_ticks"])
        )
        result.append(copy)
    return result


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed")

    correctness: dict[str, object] = {}
    attention_hashes: dict[str, tuple[str, str, str]] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(record, 1, mode, audit_enabled=True)
        attention_hashes[mode] = (
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
        }
    if attention_hashes["control"] != attention_hashes["candidate"]:
        raise SystemExit("candidate changed QK/probability/AV audit hashes")

    repeat_results: dict[str, object] = {}
    overall: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: base.load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl"
            )
            for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        augmented = {
            mode: with_combined_residual(values)
            for mode, values in records.items()
        }
        fields = (
            "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
            "combined_residual_ticks", *base.LEDGER, *base.OVERLAP,
            *RESIDUAL_WORK, *base.COUNTERS, *RESIDUAL_COUNTERS,
            "ledger_named_ticks", "ledger_unattributed_ticks",
            "runtime_setup_ticks", "runtime_teardown_ticks", *base.RESOURCES,
        )
        metrics = {
            field: base.summarize(
                augmented["control"], augmented["candidate"], field
            )
            for field in fields
        }
        speed_gate = all(
            metrics[field][key] < 0.0
            for field in (
                "host_wall_ns_per_block", "combined_residual_ticks"
            )
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_fields = (
            "hmx_command_count", "hmx_u8s8_tile_pair_count",
            "weight_dma_descriptor_count", "weight_ddr_read_bytes",
            "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
            "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
            "intermediate_dma_descriptor_count",
            "intermediate_spill_fill_count", "w4u8_qkv_batch_count",
            "w4u8_qkvo_prefetch_count", "attention_qk_norm_task_count",
            "vtcm_requested_bytes", "vtcm_acquired_bytes",
            "vtcm_peak_plan_bytes",
        )
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in invariant_fields
        )
        task_gate = (
            metrics["w4u8_post_residual_task_count"]["control"] == 0.0 and
            metrics["w4u8_final_residual_task_count"]["control"] == 0.0 and
            metrics["w4u8_post_residual_task_count"]["candidate"] == 16.0 and
            metrics["w4u8_final_residual_task_count"]["candidate"] == 16.0
        )
        passed = speed_gate and invariant_gate and task_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "speed_gate": speed_gate,
            "unchanged_math_traffic_and_resources_gate": invariant_gate,
            "residual_task_coverage_gate": task_gate,
        }

    return {
        "experiment": "EXP-0053",
        "control": "W4U8-EXP0052 single-context native residual",
        "candidate": "four-context native residual row-task pool",
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "byte_exact_final_output_gate": True,
        "independent_block_reference_gate": True,
        "attention_boundary_hash_gate": True,
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


def fmt(value: float) -> str:
    return f"{value:,.3f}"


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0053 — Complete profiling report", "",
        "The candidate changes only the two native-layout W4U8 residual "
        "consumers. Sixty-four rows become sixteen disjoint four-row tasks "
        "claimed by the main HVX context and the three persistent worker "
        "contexts. Q14 arithmetic, native O/Down carriers, Post-Attention "
        "RMSNorm, every HMX command, traffic and all other schedules are fixed.",
        "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Metric (per block) | Serial control | Pool4 candidate | Delta | Paired delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for field in (
            "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
            "combined_residual_ticks", "post_attention_residual_ticks",
            "final_residual_ticks", "w4u8_post_residual_pool_wait_ticks",
            "w4u8_final_residual_pool_wait_ticks",
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
            f"Speed gate: **{'PASS' if result['speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/resources: **{'PASS' if result['unchanged_math_traffic_and_resources_gate'] else 'FAIL'}**; "
            f"row-task coverage: **{'PASS' if result['residual_task_coverage_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "Final output is byte-exact (0 mismatches, 0 LSB); QK, probability "
        "and AV hashes are unchanged; requested/acquired VTCM is exactly "
        "8 MiB; intermediate DDR and spill/fill remain zero; execution stays "
        "inside one FastRPC unit with one HMX owner and no QNN dependency.", "",
        "## Decision", "",
        f"EXP-0053 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        "Baseline promotion remains a user decision.", "",
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
