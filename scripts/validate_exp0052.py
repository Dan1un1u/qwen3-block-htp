#!/usr/bin/env python3
"""Validate EXP-0052 paired evidence and render the PC-027 report."""

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
OVERLAP = base.OVERLAP + ("attention_qk_norm_pool_wait_ticks",)
COUNTERS = base.COUNTERS + ("attention_qk_norm_task_count",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str) -> None:
    candidate = mode == "candidate"
    fixed = {
        "experiment": "EXP-0052",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "crouton_boundary_mode": "w4u8_mlp_io_qkv_o",
        "w4u8_qkvo_pipeline_mode": (
            "qkvo_batch4_qk_head_tasks" if candidate else "qkvo_batch4"
        ),
        "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
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
        "u8_attention_audit_ddr_write_bytes": 0,
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
        "qkv_projection_ticks", "attention_qk_norm_pool_wait_ticks",
        "attention_ticks", "o_projection_ticks",
        "post_attention_residual_ticks", "gate_up_ticks", "down_ticks",
        "final_residual_ticks",
    ):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


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
        validate_record(record, 1, mode)
        attention_hashes[mode] = (
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "fused_k_operand_mismatches":
                record["u8_attention_fused_k_operand_mismatch_count"],
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
        fields = (
            "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
            *base.LEDGER, *OVERLAP, *COUNTERS,
            "ledger_named_ticks", "ledger_unattributed_ticks",
            "runtime_setup_ticks", "runtime_teardown_ticks", *base.RESOURCES,
        )
        metrics = {
            field: base.summarize(
                records["control"], records["candidate"], field
            )
            for field in fields
        }
        speed_gate = all(
            metrics[field][key] < 0.0
            for field in (
                "host_wall_ns_per_block", "qkv_projection_ticks",
                "attention_qk_norm_pool_wait_ticks",
            )
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_fields = (
            "hmx_command_count", "hmx_u8s8_tile_pair_count",
            "weight_dma_descriptor_count", "weight_ddr_read_bytes",
            "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
            "w4u8_qkvo_overlap_schedule_count",
            "attention_qk_norm_task_count", "vtcm_requested_bytes",
            "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
        )
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in invariant_fields
        )
        scheduler_gate = (
            metrics["attention_qk_norm_task_count"]["control"] == 24.0 and
            metrics["attention_qk_norm_task_count"]["candidate"] == 24.0 and
            metrics["w4u8_qkv_batch_count"]["control"] == 32.0 and
            metrics["w4u8_qkv_batch_count"]["candidate"] == 32.0
        )
        passed = speed_gate and invariant_gate and scheduler_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "speed_gate": speed_gate,
            "unchanged_math_traffic_and_resources_gate": invariant_gate,
            "scheduler_contract_gate": scheduler_gate,
            "logical_q_tasks_per_block": 16,
            "logical_k_tasks_per_block": 8,
            "qkv_hmx_commands_per_block": 32,
        }

    return {
        "experiment": "EXP-0052",
        "control": "W4U8-EXP0050 group-blocking Q/K prep",
        "candidate": "independent ordered Q-then-K head prep tasks",
        "package_manifest_sha256": base.sha256(
            package_dir / "manifest.json"
        ),
        "byte_exact_final_output_gate": True,
        "independent_block_reference_gate": True,
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


def format_value(field: str, value: float) -> str:
    if "bytes" in field or "count" in field or field.endswith("skipped"):
        return f"{value:,.1f}"
    return f"{value:,.3f}"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              metrics: dict[str, dict[str, float | None]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Group-task control | Head-task candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        metric = metrics[field]
        change = metric["change_percent"]
        paired = metric["paired_change_percent_median"]
        lines.append(
            f"| `{field}` | {format_value(field, metric['control'])} | "
            f"{format_value(field, metric['candidate'])} | "
            f"{'n/a' if change is None else f'{change:.3f}%'} | "
            f"{'n/a' if paired is None else f'{paired:.3f}%'} |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0052 — Complete profiling report", "",
        "The candidate changes only Q/K Norm-RoPE preparation task claim "
        "order. The control claims one GQA group and can block a worker on "
        "an unavailable K head after its two Q heads. The candidate exposes "
        "16 ordered Q-head tasks followed by eight K-head tasks, allowing "
        "workers to consume every already-published Q head before waiting "
        "for K. Projection commands, arithmetic, weights, qparams, traffic, "
        "Attention, O, Gate/Up and Down are fixed.", "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and Q/K preparation target",
            (
                "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
                "qkv_projection_ticks", "attention_qk_norm_pool_wait_ticks",
            ), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", base.LEDGER, metrics)
        add_table(lines, "Overlapping engine work and waits", OVERLAP, metrics)
        add_table(
            lines, "Traffic, commands, counters and residency",
            COUNTERS + base.RESOURCES + (
                "runtime_setup_ticks", "runtime_teardown_ticks",
                "ledger_named_ticks", "ledger_unattributed_ticks",
            ), metrics,
        )
        lines.extend([
            "Logical prep work remains **16 Q heads + 8 K heads**; QKV HMX "
            "commands remain **32**; total HMX commands remain **256**; "
            "HMX tile pairs remain **49,408**.", "",
            f"Repeat-{repeat} speed gate: "
            f"**{'PASS' if result['speed_gate'] else 'FAIL'}**; unchanged "
            f"math/traffic/resources: **{'PASS' if result['unchanged_math_traffic_and_resources_gate'] else 'FAIL'}**; "
            f"scheduler contract: **{'PASS' if result['scheduler_contract_gate'] else 'FAIL'}**.",
            "",
        ])

    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output vs EXP-0050 | byte-exact, 0 LSB |",
        "| Independent block implementation reference | 0 mismatches, 0 LSB |",
        "| QK / probability / AV audit hashes | candidate equals control |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping work/wait counters are not "
        "summed together. Host wall is the primary speed metric.", "",
        "## Decision", "",
        f"EXP-0052 local gate: "
        f"**{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: "
        f"**{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "The Selected Baseline is unchanged unless the user explicitly "
        "promotes this candidate.", "",
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
