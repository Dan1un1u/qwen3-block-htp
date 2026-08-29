#!/usr/bin/env python3
"""Validate EXP-0051 Stage-A paired evidence and render the PC-027 report."""

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
        "experiment": "EXP-0051",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "crouton_boundary_mode": "w4u8_mlp_io_qkv_o",
        "w4u8_qkvo_pipeline_mode": (
            "q_headpair" if candidate else "qkvo_batch4"
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
        "hmx_command_count": (248 if candidate else 256) * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": 4,
        "w4u8_qkv_batch_count": (24 if candidate else 32) * repeat,
        "w4u8_qkvo_prefetch_count": 44 * repeat,
        "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
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
        "qkv_projection_ticks", "attention_ticks", "o_projection_ticks",
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
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        for field, expected in {
            "experiment": "EXP-0051",
            "output_hash": OUTPUT_HASH,
            "mismatches": 0,
            "max_lsb": 0,
            "rpc_result": 0,
            "dsp_status": 3,
        }.items():
            base.require(record, field, expected)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
        }

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
            *base.LEDGER, *base.OVERLAP, *base.COUNTERS,
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
            for field in ("host_wall_ns_per_block", "qkv_projection_ticks")
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in (
                "hmx_u8s8_tile_pair_count",
                "weight_dma_descriptor_count", "weight_ddr_read_bytes",
                "vtcm_requested_bytes", "vtcm_acquired_bytes",
            )
        )
        command_gate = (
            metrics["hmx_command_count"]["control"] == 256.0 and
            metrics["hmx_command_count"]["candidate"] == 248.0 and
            metrics["w4u8_qkv_batch_count"]["control"] == 32.0 and
            metrics["w4u8_qkv_batch_count"]["candidate"] == 24.0
        )
        passed = speed_gate and invariant_gate and command_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "speed_gate": speed_gate,
            "unchanged_math_and_traffic_gate": invariant_gate,
            "command_coarsening_gate": command_gate,
            "qkv_control_commands_per_block": 32,
            "qkv_candidate_commands_per_block": 24,
            "q_commands_per_block": {"control": 16, "candidate": 8},
            "head_readiness_granularity": 1,
        }

    return {
        "experiment": "EXP-0051",
        "stage": "A-Q-only",
        "control": "W4U8-EXP0050-DOWN-COMMAND-FUSION",
        "candidate": "Q generation-safe head-pair HMX command",
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
        "stage_b_eligible": all(overall),
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
        "| Metric | Control | Q head-pair candidate | Delta | Paired delta |",
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
        "# EXP-0051 — Complete profiling report", "",
        "Stage A changes only the Q projection worker cadence. One command "
        "computes two consecutive 128-channel Q heads, but publishes the "
        "first head before waiting for the second. K/V/O, all arithmetic, "
        "weights, DMA traffic, qparams and downstream Attention are fixed.",
        "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and QKV target",
            (
                "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
                "qkv_projection_ticks",
            ), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", base.LEDGER, metrics)
        add_table(
            lines, "Overlapping engine work and waits", base.OVERLAP, metrics
        )
        add_table(
            lines, "Traffic, commands, counters and residency",
            base.COUNTERS + base.RESOURCES + (
                "runtime_setup_ticks", "runtime_teardown_ticks",
                "ledger_named_ticks", "ledger_unattributed_ticks",
            ), metrics,
        )
        lines.extend([
            "Q commands per block: **16 → 8**; total QKV commands per "
            "block: **32 → 24**; total HMX commands: **256 → 248**; HMX "
            "tile pairs remain **49,408**. Head readiness remains one head.",
            "",
            f"Repeat-{repeat} speed gate: "
            f"**{'PASS' if result['speed_gate'] else 'FAIL'}**; unchanged "
            f"math/traffic: **{'PASS' if result['unchanged_math_and_traffic_gate'] else 'FAIL'}**; "
            f"command coarsening: **{'PASS' if result['command_coarsening_gate'] else 'FAIL'}**.",
            "",
        ])

    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output vs control | byte-exact, 0 LSB |",
        "| Independent block implementation reference | 0 mismatches, 0 LSB |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping work/wait counters are not "
        "summed together. Host wall is the primary speed metric.", "",
        "## Decision", "",
        f"EXP-0051 Stage-A local gate: "
        f"**{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Stage B eligibility: **{'YES' if summary['stage_b_eligible'] else 'NO'}**. "
        "Selected Baseline is unchanged unless the user explicitly promotes it.",
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
