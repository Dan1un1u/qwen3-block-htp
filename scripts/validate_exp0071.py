#!/usr/bin/env python3
"""Validate EXP-0071 parallel W4U8 Input RMSNorm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0070 as previous


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
QTIMER_TICKS_PER_US = 19.2
TARGETS = ("host_wall_ns_per_block", "input_norm_ticks")
OVERLAP = tuple(dict.fromkeys((
    *previous.OVERLAP,
    "w4u8_input_norm_main_work_ticks",
    "w4u8_input_norm_worker_work_ticks",
    "w4u8_input_norm_pool_wait_ticks",
)))
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *previous.EXTRA_REPORT_FIELDS,
    "w4u8_input_norm_task_count",
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
        "experiment": "EXP-0070",
        "u8_norm_reduction_mode": (
            "hvx_tree_qk_batched_rsqrt_shared_rope"
        ),
    })
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    expected_norm_mode = (
        "hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input"
        if mode == "candidate"
        else "hvx_tree_qk_batched_rsqrt_shared_rope"
    )
    fixed = {
        "experiment": "EXP-0071",
        "u8_norm_reduction_mode": expected_norm_mode,
        "attention_pipeline_mode": (
            "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
            "hmx_batch_lut_templates_gqa_batch"
        ),
        "attention_hvx_contexts": 6,
        "attention_hvx_workers_created": 5,
        "attention_hvx_workers_locked": 5,
        "hmx_command_count": 176 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "w4u8_input_norm_task_count": (
            16 * repeat if mode == "candidate" else 0
        ),
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    for field in TARGETS + ("invocation_ticks", "total_ticks"):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")
    if mode == "control":
        for field in (
            "w4u8_input_norm_main_work_ticks",
            "w4u8_input_norm_worker_work_ticks",
            "w4u8_input_norm_pool_wait_ticks",
        ):
            base.require(record, field, 0)
    elif float(record["w4u8_input_norm_worker_work_ticks"]) <= 0.0:
        raise SystemExit("parallel Input RMSNorm did not execute worker work")


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed")

    correctness: dict[str, object] = {}
    attention_hashes: dict[str, tuple[str, str, str]] = {}
    input_norm_hashes: dict[str, str] = {}
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
        input_norm_hashes[mode] = str(
            record["u8_input_norm_actual_hash"]
        )
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "input_norm_hash": input_norm_hashes[mode],
            "qk_hash": attention_hashes[mode][0],
            "probability_hash": attention_hashes[mode][1],
            "av_hash": attention_hashes[mode][2],
            "input_norm_task_count": record[
                "w4u8_input_norm_task_count"
            ],
        }
        if record["output_hash"] != OUTPUT_HASH:
            raise SystemExit(f"wrong final output hash for {mode}")
    if attention_hashes["control"] != attention_hashes["candidate"]:
        raise SystemExit("candidate changed QK/probability/AV audit hashes")
    if attention_hashes["control"] != (
        QK_HASH, PROBABILITY_HASH, AV_HASH
    ):
        raise SystemExit("Attention audit hashes differ from EXP-0070")
    if input_norm_hashes["control"] != input_norm_hashes["candidate"]:
        raise SystemExit("candidate changed native Input RMSNorm bytes")
    if input_norm_hashes["control"] == "0000000000000000":
        raise SystemExit("Input RMSNorm audit hash was not populated")

    repeat_results: dict[str, object] = {}
    overall: list[bool] = []
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
        "attention_hvx_workers_created", "attention_hvx_workers_locked",
        "w4u8_mlp_gate_up_hmx_command_count",
        "w4u8_mlp_down_hmx_command_count",
        "w4u8_post_residual_task_count",
        "w4u8_final_residual_task_count",
    )
    for repeat in REPEATS:
        records = {
            mode: base.load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl", SAMPLES
            )
            for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        fields = tuple(dict.fromkeys((
            *TARGETS, "invocation_ticks", "total_ticks",
            *base.LEDGER, *OVERLAP, *base.COUNTERS,
            *base.RESOURCES, *EXTRA_REPORT_FIELDS,
        )))
        metrics = {
            field: base.summarize(
                records["control"], records["candidate"], field
            )
            for field in fields
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
        task_gate = (
            metrics["w4u8_input_norm_task_count"]["control"] == 0.0
            and metrics["w4u8_input_norm_task_count"]["candidate"]
                == float(16 * repeat)
            and metrics["w4u8_input_norm_worker_work_ticks"][
                "candidate"
            ] > 0.0
        )
        passed = speed_gate and invariant_gate and task_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "parallel_task_execution_gate": task_gate,
        }

    return {
        "experiment": "EXP-0071",
        "control": "EXP-0070 serial Input RMSNorm",
        "candidate": "main plus five parallel Input RMSNorm",
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "byte_exact_final_output_gate": True,
        "byte_exact_input_norm_boundary_gate": True,
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


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              metrics: dict[str, dict[str, float | None]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | EXP-0070 control | EXP-0071 candidate | Delta | Paired delta |",
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


def add_pc028_table(lines: list[str], summary: dict[str, object]) -> None:
    metrics = summary["repeat_results"]["repeat10"]["metrics"]
    side = "candidate" if summary["local_gate_pass"] else "control"
    current = dict(previous.module_us(metrics, side))
    f16 = {
        "I/O and metadata": 6.6, "Input RMSNorm": 42.7,
        "QKV + Q/K Norm/RoPE": 397.1, "QK-Softmax-AV": 140.3,
        "O projection": 201.5, "Post-attn residual + RMSNorm": 41.2,
        "Gate/Up + SwiGLU": 1116.6, "Down projection": 459.9,
        "Final residual": 5.0, "Host/RPC and closure": 77.5,
        "Complete block Host wall": 2488.3,
    }
    w4f16 = {
        "I/O and metadata": 7.4, "Input RMSNorm": 42.7,
        "QKV + Q/K Norm/RoPE": 439.2, "QK-Softmax-AV": 140.7,
        "O projection": 173.9, "Post-attn residual + RMSNorm": 41.2,
        "Gate/Up + SwiGLU": 972.7, "Down projection": 327.3,
        "Final residual": 5.0, "Host/RPC and closure": 79.6,
        "Complete block Host wall": 2229.7,
    }
    current_total = current["Complete block Host wall"]
    used = "EXP-0071" if summary["local_gate_pass"] else "EXP-0070"
    lines.extend([
        "## PC-028 current-best three-variant module wall-time", "",
        f"{used} supplies the W4U8 column. F16F16 and the user-selected "
        "W4F16 EXP-0038 comparator reuse their retained formal evidence. "
        "Percentages are shares of complete Host wall; positive speed means "
        "W4U8 is faster than W4F16.", "",
        "| Module | W16A16 | W4A16 | W4A8 current best | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in f16:
        if name == "Complete block Host wall":
            f16_text = f"{f16[name]:.1f} us"
            w4f16_text = f"{w4f16[name]:.1f} us"
            current_text = f"{current[name]:.1f} us"
        else:
            f16_text = (
                f"{f16[name]:.1f} us "
                f"({100*f16[name]/f16['Complete block Host wall']:.1f}%)"
            )
            w4f16_text = (
                f"{w4f16[name]:.1f} us "
                f"({100*w4f16[name]/w4f16['Complete block Host wall']:.1f}%)"
            )
            current_text = (
                f"{current[name]:.1f} us "
                f"({100*current[name]/current_total:.1f}%)"
            )
        speed = (w4f16[name] / current[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {f16_text} | {w4f16_text} | "
            f"{current_text} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = ["# EXP-0071 — Complete profiling report", ""]
    add_pc028_table(lines, summary)
    lines.extend([
        "The control executes all 64 Input RMSNorm rows serially on the main "
        "HVX context. The candidate exposes sixteen four-row tasks to the "
        "existing main-plus-five context domain. Every task calls the same "
        "per-row reduction, sqrt, gamma, requant and native HMX-activation "
        "store implementation; only scheduling changes.", "",
    ])
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary wall-latency targets",
            ("host_wall_ns_per_block", "input_norm_ticks",
             "invocation_ticks", "total_ticks"), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", base.LEDGER, metrics)
        add_table(lines, "Overlapping engine work and waits", OVERLAP, metrics)
        add_table(
            lines, "Traffic, commands, counters and residency",
            tuple(dict.fromkeys((
                *base.COUNTERS, *base.RESOURCES, *EXTRA_REPORT_FIELDS,
            ))), metrics,
        )
        lines.extend([
            f"Two-target speed gate: **{'PASS' if result['two_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged physical contract: **{'PASS' if result['unchanged_physical_contract_gate'] else 'FAIL'}**; "
            f"parallel task execution: **{'PASS' if result['parallel_task_execution_gate'] else 'FAIL'}**.",
            "",
        ])
    input_hash = summary["correctness"]["candidate"]["input_norm_hash"]
    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact to EXP-0070, 0 LSB |",
        f"| Native Input RMSNorm activation | byte-exact, hash `{input_hash}` |",
        "| QK / probability / AV audit boundaries | byte-exact to EXP-0070 |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0071 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
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
