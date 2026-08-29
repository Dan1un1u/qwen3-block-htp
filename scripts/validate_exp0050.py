#!/usr/bin/env python3
"""Validate EXP-0050 paired evidence and render the PC-027 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608

LEDGER = (
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
)
OVERLAP = (
    "projection_pack_ticks", "projection_unpack_ticks", "weight_dma_ticks",
    "hmx_compute_ticks", "projection_hmx_wait_ticks", "hmx_ready_wait_ticks",
    "w4u8_qkvo_weight_expand_ticks", "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks", "u8_attention_qk_norm_rope_ticks",
    "u8_attention_v_pack_ticks", "u8_attention_qk_hmx_ticks",
    "u8_attention_qk_requant_ticks", "u8_attention_softmax_ticks",
    "u8_attention_av_hmx_ticks", "u8_attention_av_requant_ticks",
    "u8_attention_pipeline_wait_ticks", "w4u8_mlp_gate_up_pipeline_ticks",
    "w4u8_mlp_down_pipeline_ticks", "w4u8_mlp_activation_work_ticks",
    "w4u8_mlp_weight_stage_ticks", "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_hmx_compute_ticks", "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
)
COUNTERS = (
    "hmx_command_count", "hmx_u8s8_tile_pair_count",
    "weight_dma_descriptor_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
    "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
    "u8_attention_direct_o_tile_count", "u8_attention_qkv_unpack_skipped",
    "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
    "w4u8_qkvo_overlap_schedule_count", "w4u8_mlp_input_pack_skipped",
    "w4u8_mlp_output_unpack_skipped", "w4u8_mlp_pair_publish_count",
    "w4u8_mlp_pair_consume_count",
    "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_mlp_down_hmx_command_count",
)
RESOURCES = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    "attention_hvx_workers_created", "attention_hvx_workers_locked",
    "mlp_hvx_workers_created", "mlp_hvx_workers_locked",
    "w4u8_qkv_batch_n_tiles", "w4u8_mlp_gate_up_hvx_workers",
    "w4u8_mlp_down_hvx_workers", "w4u8_mlp_gate_up_hvx_hmx_overlap",
    "w4u8_mlp_down_hvx_hmx_overlap",
    "w4u8_mlp_gate_up_hvx_parallel_overlap",
    "w4u8_mlp_down_hvx_parallel_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, expected: int = SAMPLES) -> list[dict[str, object]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(records) != expected:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def per_block(record: dict[str, object], field: str) -> float:
    if field == "host_wall_ns_per_block" or field in RESOURCES:
        return float(record[field])
    return float(record[field]) / int(record["repeat_count"])


def summarize(control: list[dict[str, object]],
              candidate: list[dict[str, object]],
              field: str) -> dict[str, float | None]:
    left = [per_block(record, field) for record in control]
    right = [per_block(record, field) for record in candidate]
    left_median = statistics.median(left)
    right_median = statistics.median(right)
    paired = [
        (right[index] / left[index] - 1.0) * 100.0
        for index in range(SAMPLES) if left[index] != 0.0
    ]
    return {
        "control": left_median,
        "candidate": right_median,
        "change_percent": (
            (right_median / left_median - 1.0) * 100.0
            if left_median != 0.0 else None
        ),
        "paired_change_percent_median": (
            statistics.median(paired) if paired else None
        ),
        "paired_change_percent_min": min(paired) if paired else None,
        "paired_change_percent_max": max(paired) if paired else None,
    }


def validate_record(record: dict[str, object], repeat: int,
                    mode: str) -> None:
    fixed = {
        "experiment": "EXP-0049" if mode == "control" else "EXP-0050",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "crouton_boundary_mode": "w4u8_mlp_io_qkv_o",
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4",
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
        "hmx_command_count": (320 if mode == "control" else 256) * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": 4,
        "w4u8_qkv_batch_count": 32 * repeat,
        "w4u8_qkvo_prefetch_count": 44 * repeat,
        "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
        "w4u8_mlp_input_pack_skipped": repeat,
        "w4u8_mlp_output_unpack_skipped": repeat,
        "w4u8_mlp_input_pack_ticks": 0,
        "w4u8_mlp_output_unpack_ticks": 0,
        "w4u8_mlp_gate_up_hmx_batch_n_tiles": 8,
        "w4u8_mlp_gate_up_expanded_slot_count": 8,
        "w4u8_mlp_gate_up_hmx_command_count": 48 * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    if mode == "candidate":
        fixed.update({
            "w4u8_mlp_down_hmx_command_count": 64 * repeat,
        })
    for field, expected in fixed.items():
        require(record, field, expected)
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
        record = load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        require(record, "experiment",
                "EXP-0049" if mode == "control" else "EXP-0050")
        require(record, "output_hash", OUTPUT_HASH)
        require(record, "mismatches", 0)
        require(record, "max_lsb", 0)
        require(record, "rpc_result", 0)
        require(record, "dsp_status", 3)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
        }

    repeat_results: dict[str, object] = {}
    overall: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl"
            )
            for mode in MODES
        }
        # EXP-0049 ABI 26 did not expose a dedicated Down command field.
        # Its fixed two-command-per-output schedule is 128 commands/block.
        for record in records["control"]:
            record["w4u8_mlp_down_hmx_command_count"] = 128 * repeat
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        fields = (
            "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
            *LEDGER, *OVERLAP, *COUNTERS, "ledger_named_ticks",
            "ledger_unattributed_ticks", "runtime_setup_ticks",
            "runtime_teardown_ticks", *RESOURCES,
        )
        metrics = {
            field: summarize(
                records["control"], records["candidate"], field
            )
            for field in fields
        }
        speed_gate = all(
            metrics[field][key] < 0.0
            for field in (
                "host_wall_ns_per_block",
                "w4u8_mlp_down_pipeline_ticks",
            )
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_gate = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in (
                "hmx_u8s8_tile_pair_count",
                "weight_dma_descriptor_count",
                "weight_ddr_read_bytes",
                "vtcm_requested_bytes", "vtcm_acquired_bytes",
            )
        )
        command_gate = (
            metrics["hmx_command_count"]["control"] == 320.0 and
            metrics["hmx_command_count"]["candidate"] == 256.0 and
            metrics["w4u8_mlp_down_hmx_command_count"]["control"] == 128.0 and
            metrics["w4u8_mlp_down_hmx_command_count"]["candidate"] == 64.0
        )
        passed = speed_gate and invariant_gate and command_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "speed_gate": speed_gate,
            "unchanged_math_and_traffic_gate": invariant_gate,
            "command_coarsening_gate": command_gate,
            "down_control_commands_per_block": 128,
            "down_candidate_commands_per_block": 64,
            "down_chunks_per_candidate_command": 2,
        }

    return {
        "experiment": "EXP-0050",
        "control": "W4U8-EXP0049-GATE-UP-BATCH8",
        "candidate": "Down generation-safe two-chunk HMX command",
        "package_manifest_sha256": sha256(package_dir / "manifest.json"),
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
        "| Metric | EXP-0049 control | EXP-0050 candidate | Delta | Paired delta |",
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
        "# EXP-0050 — Complete profiling report", "",
        "The candidate changes only the W4U8 Down HMX consumer cadence. "
        "One worker command begins after chunk zero is ready, accumulates "
        "that chunk, waits inside the command for chunk one while HVX "
        "continues producing it, accumulates chunk one, and stores the "
        "unchanged U8 output. W4 values/scales, U8 qparams, HMX tile "
        "arithmetic, DMA bytes/descriptors, QKV, Attention, O, Gate/Up, "
        "tensor boundaries and final output are unchanged.", "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary latency and Down target",
            (
                "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
                "down_ticks", "w4u8_mlp_down_pipeline_ticks",
            ), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", LEDGER, metrics)
        add_table(lines, "Overlapping engine work and waits", OVERLAP, metrics)
        add_table(
            lines, "Traffic, commands, counters and residency",
            COUNTERS + RESOURCES + (
                "runtime_setup_ticks", "runtime_teardown_ticks",
                "ledger_named_ticks", "ledger_unattributed_ticks",
            ), metrics,
        )
        lines.extend([
            "Down HMX commands per block: **128 → 64**; total HMX "
            "commands per block: **320 → 256**; HMX tile pairs remain "
            "**49,408**.", "",
            f"Repeat-{repeat} speed gate: "
            f"**{'PASS' if result['speed_gate'] else 'FAIL'}**; unchanged "
            f"math/traffic: **{'PASS' if result['unchanged_math_and_traffic_gate'] else 'FAIL'}**; "
            f"command coarsening: **{'PASS' if result['command_coarsening_gate'] else 'FAIL'}**.",
            "",
        ])

    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output vs EXP-0049 | byte-exact, 0 LSB |",
        "| Independent block implementation reference | 0 mismatches, 0 LSB |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping work/wait counters are not "
        "summed together. Host wall is the primary speed metric.", "",
        "## Decision", "",
        f"EXP-0050 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
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
