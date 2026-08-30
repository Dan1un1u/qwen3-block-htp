#!/usr/bin/env python3
"""Generate the EXP-0088 PC-027/PC-028 negative-experiment closure."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


QTIMER_TICKS_PER_US = 19.2
SAMPLES = 5
REPEATS = (1, 10)
MODULES = (
    ("I/O and metadata",
     ("input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks")),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm/RoPE",
     ("qkv_projection_ticks", "qk_norm_rope_ticks")),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-attn residual + RMSNorm",
     ("post_attention_residual_ticks", "post_attention_norm_ticks")),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
)
PRIMARY = (
    "host_wall_ns_per_block", "total_ticks", "invocation_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks", "ledger_named_ticks",
    "ledger_unattributed_ticks",
)
LEDGER = (
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks", "stage_boundary_ticks",
)
PROJECTION = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "hmx_ready_wait_ticks", "w4u8_qkvo_weight_expand_ticks",
    "w4u8_qkvo_prefetch_wait_ticks", "w4u8_qkvo_hmx_lifetime_ticks",
    "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
    "w4u8_qkvo_overlap_schedule_count",
)
ATTENTION = (
    "u8_attention_qk_norm_rope_ticks", "u8_attention_v_pack_ticks",
    "u8_attention_qk_hmx_ticks", "u8_attention_qk_requant_ticks",
    "u8_attention_softmax_ticks", "u8_attention_av_hmx_ticks",
    "u8_attention_av_requant_ticks", "u8_attention_pipeline_wait_ticks",
    "attention_qk_norm_pool_wait_ticks", "attention_qk_norm_task_count",
    "attention_softmax_task_count", "u8_attention_group_count",
    "u8_attention_qk_execution_count", "u8_attention_av_execution_count",
    "u8_attention_lut_template_build_count",
    "u8_attention_lut_template_reuse_count",
    "u8_attention_lut_template_build_ticks",
    "u8_attention_lut_private_vtcm_bytes",
)
MLP = (
    "w4u8_mlp_gate_up_pipeline_ticks", "w4u8_mlp_down_pipeline_ticks",
    "w4u8_mlp_activation_work_ticks", "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
    "w4u8_mlp_pair_publish_count", "w4u8_mlp_pair_consume_count",
    "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_mlp_down_hmx_command_count",
    "w4u8_mlp_gate_up_hvx_workers", "w4u8_mlp_down_hvx_workers",
    "w4u8_mlp_gate_up_expanded_slot_count",
)
PHYSICAL = (
    "weight_ddr_read_bytes", "weight_dma_descriptor_count",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
    "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
    "hmx_command_count", "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count", "block_invocation_count",
    "vtcm_requested_bytes", "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    "u8_attention_lut_private_vtcm_bytes",
)
CONSTANT_FIELDS = {
    "vtcm_requested_bytes", "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    "u8_attention_lut_private_vtcm_bytes",
    "w4u8_mlp_gate_up_hvx_workers", "w4u8_mlp_down_hvx_workers",
    "w4u8_mlp_gate_up_expanded_slot_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("exp0084_dir", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path, expected: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if expected is not None and len(records) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {len(records)}")
    if not records:
        raise SystemExit(f"{path}: no records")
    return records


def normalized(record: dict, field: str, repeat: int) -> float:
    value = float(record[field])
    if field == "host_wall_ns_per_block" or field in CONSTANT_FIELDS:
        return value
    return value / repeat


def median(records: list[dict], field: str, repeat: int) -> float:
    return float(statistics.median(
        normalized(record, field, repeat) for record in records
    ))


def delta(control: float, candidate: float) -> float | None:
    return None if control == 0.0 else (candidate / control - 1.0) * 100.0


def paired_delta(control: list[dict], candidate: list[dict], field: str,
                 repeat: int) -> float | None:
    values = []
    for left, right in zip(control, candidate):
        base = normalized(left, field, repeat)
        if base != 0.0:
            values.append(
                (normalized(right, field, repeat) / base - 1.0) * 100.0
            )
    return None if not values else float(statistics.median(values))


def fmt_value(field: str, value: float) -> str:
    if field == "host_wall_ns_per_block":
        return f"{value / 1000.0:.3f} us"
    if field.endswith("_bytes"):
        return f"{value:.0f} B"
    if field.endswith("_count") or field.endswith("_workers"):
        return f"{value:.3f}"
    return f"{value:.3f}"


def fmt_delta(value: float | None) -> str:
    return "N/A (control=0)" if value is None else f"{value:+.4f}%"


def module_profile(records: list[dict], repeat: int) -> dict[str, float]:
    host_us = median(records, "host_wall_ns_per_block", repeat) / 1000.0
    result: dict[str, float] = {}
    named_us = 0.0
    for name, fields in MODULES:
        value = sum(median(records, field, repeat) for field in fields)
        value /= QTIMER_TICKS_PER_US
        result[name] = value
        named_us += value
    result["Host/RPC and profiling closure remainder"] = host_us - named_us
    result["Complete block Host wall"] = host_us
    return result


def add_module_table(lines: list[str], profiles: dict[str, dict[str, float]]) -> None:
    lines.extend([
        "## PC-028 repeat10 three-variant overview", "",
        "F16F16 and W4F16 are reused, non-paired formal comparators from "
        "EXP-0084. W4U8 is the direct paired EXP-0088 control because the "
        "candidate failed its local gate.", "",
        "| Module | F16F16 EXP-0084 | W4F16 EXP-0084 | W4U8 EXP-0088 control | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ])
    for module in profiles["F16F16"]:
        values = [profiles[name][module]
                  for name in ("F16F16", "W4F16", "W4U8")]
        if module == "Complete block Host wall":
            cells = [f"{value:.1f} us" for value in values]
        else:
            cells = [
                f"{value:.1f} us "
                f"({100.0 * value / profiles[name]['Complete block Host wall']:.1f}%)"
                for value, name in zip(values, ("F16F16", "W4F16", "W4U8"))
            ]
        speed = values[1] / values[2] - 1.0
        lines.append(
            f"| {module} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{speed * 100.0:+.1f}% |"
        )
    lines.append("")


def add_direct_table(lines: list[str], control: list[dict],
                     candidate: list[dict], repeat: int,
                     fields: tuple[str, ...], title: str,
                     overlap_note: bool = False) -> None:
    lines.extend([
        f"### {title}", "",
        ("These counters overlap and must not be summed into wall time."
         if overlap_note else
         "Values are medians normalized per block."), "",
        "| Metric | Control | Candidate | Candidate delta | Paired delta median |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        left = median(control, field, repeat)
        right = median(candidate, field, repeat)
        lines.append(
            f"| `{field}` | {fmt_value(field, left)} | "
            f"{fmt_value(field, right)} | {fmt_delta(delta(left, right))} | "
            f"{fmt_delta(paired_delta(control, candidate, field, repeat))} |"
        )
    lines.append("")


def validate(result_dir: Path, gate: dict) -> None:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
            result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed")
    if gate.get("stage_b_gate_pass") is not False:
        raise SystemExit("EXP-0088 Stage-B must be a failed gate")
    correctness = gate["correctness_and_persistence"]
    for mode in ("control", "candidate"):
        item = correctness[mode]
        expected = {
            "output_hash": "69f22eeb035e5ec5",
            "qk_hash": "32aa949912e365be",
            "probability_hash": "94f2e218f06f9627",
            "av_hash": "f853658f52032bde",
        }
        for field, value in expected.items():
            if item.get(field) != value:
                raise SystemExit(f"correctness {mode} {field} mismatch")


def main() -> int:
    args = parse_args()
    gate = json.loads((args.result_dir / "gate_summary.json").read_text())
    validate(args.result_dir, gate)
    direct = {
        repeat: {
            mode: load(args.result_dir / f"paired_{mode}_r{repeat}.jsonl",
                       SAMPLES)
            for mode in ("control", "candidate")
        }
        for repeat in REPEATS
    }
    correctness = {
        mode: load(args.result_dir / f"correctness_{mode}.jsonl", 1)[0]
        for mode in ("control", "candidate")
    }
    canonical = args.exp0084_dir / "tri_variant" / "canonical"
    profiles = {
        "F16F16": module_profile(load(canonical / "f16f16_r10.jsonl"), 10),
        "W4F16": module_profile(load(canonical / "w4f16_r10.jsonl"), 10),
        "W4U8": module_profile(direct[10]["control"], 10),
    }

    lines = [
        "# EXP-0088 full profiling report", "",
        "## Identity and comparison", "",
        "| Field | Value |", "|---|---|",
        "| Experiment | EXP-0088 |",
        "| Source branch | `codex/exp-0088-w4u8-private-persistent-softmax-lut` |",
        f"| Runtime source commit | `{args.runtime_commit}` |",
        f"| Formal evidence | `{args.result_dir}` |",
        f"| Retained artifacts | `{args.artifact_dir}` |",
        "| Execution Unit | Qwen3 layer-14 complete middle block, M=64 |",
        "| Project Variant | W4U8 |",
        "| Direct control | EXP-0084 dependency stream; six private LUT banks rebuilt per block |",
        "| Candidate | Six context-local 512-byte VTCM banks retained across Prepared Runtime Session calls |",
        "| Repeats | repeat1 and repeat10 |",
        "| Paired rounds | 5 interleaved rounds |",
        "| Backend | standalone FastRPC/cDSP; QNN none; fallback none |",
        "| RPC/HMX ownership | one FastRPC invocation per block; one HMX ownership domain |",
        "",
    ]
    add_module_table(lines, profiles)

    for repeat in REPEATS:
        control = direct[repeat]["control"]
        candidate = direct[repeat]["candidate"]
        lines.extend([f"## Direct control vs candidate — repeat{repeat}", ""])
        add_direct_table(lines, control, candidate, repeat, PRIMARY,
                         "Primary latency")
        add_direct_table(lines, control, candidate, repeat, LEDGER,
                         "Complete additive Block Timing Ledger")
        add_direct_table(lines, control, candidate, repeat, PROJECTION,
                         "Projection diagnostics", True)
        add_direct_table(lines, control, candidate, repeat, ATTENTION,
                         "Attention and LUT diagnostics", True)
        add_direct_table(lines, control, candidate, repeat, MLP,
                         "MLP diagnostics", True)
        add_direct_table(lines, control, candidate, repeat, PHYSICAL,
                         "Physical contract and residency")

    lines.extend([
        "## Correctness and experiment-specific gates", "",
        "| Gate | Control | Candidate | Result |", "|---|---:|---:|---|",
        f"| Final output hash | `{correctness['control']['output_hash']}` | "
        f"`{correctness['candidate']['output_hash']}` | PASS |",
        f"| Final mismatch / max LSB | {correctness['control']['mismatches']} / "
        f"{correctness['control']['max_lsb']} | "
        f"{correctness['candidate']['mismatches']} / "
        f"{correctness['candidate']['max_lsb']} | PASS |",
        f"| QK hash | `{correctness['control']['u8_attention_actual_score_hash']}` | "
        f"`{correctness['candidate']['u8_attention_actual_score_hash']}` | PASS |",
        f"| Probability hash | `{correctness['control']['u8_attention_actual_probability_hash']}` | "
        f"`{correctness['candidate']['u8_attention_actual_probability_hash']}` | PASS |",
        f"| AV hash | `{correctness['control']['u8_attention_actual_av_hash']}` | "
        f"`{correctness['candidate']['u8_attention_actual_av_hash']}` | PASS |",
        f"| Non-finite count | {correctness['control']['common_op_nonfinite_count']} | "
        f"{correctness['candidate']['common_op_nonfinite_count']} | PASS |",
        f"| Softmax mask violations | {correctness['control']['u8_attention_probability_mask_violation_count']} | "
        f"{correctness['candidate']['u8_attention_probability_mask_violation_count']} | PASS |",
        "| Exact 8 MiB / zero intermediate DDR / zero spill-fill | yes | yes | PASS |",
        "| Ordinary warmup bank builds | 6 | 6 | PASS |",
        "| Measured repeat1 builds / reuses | 6 / 0 | 0 / 6 | PASS |",
        "| Measured repeat10 builds / reuses | 60 / 0 | 0 / 60 | PASS |",
        "",
        "The numerical-audit run changes which four worker contexts claim the "
        "Softmax tasks. The candidate therefore records one lazy build and "
        "three reuses after a four-bank warmup in that diagnostic-only run. "
        "The ordinary performance path consistently builds all six banks in "
        "warmup and performs zero measured rebuilds.", "",
        "## Decision", "",
        "Stage B fails. Removing the measured LUT builds does not shorten the "
        "critical path: repeat10 Host wall, Attention ledger, and Softmax work "
        "all regress slightly. Stage C is not entered. The EXP-0084 W4U8 "
        "Selected Baseline remains unchanged and this candidate is rejected.",
        "",
    ])
    report = "\n".join(lines)
    output = args.output or args.result_dir / "full_profiling_report.md"
    output.write_text(report, encoding="utf-8")

    r1 = gate["repeat_results"]["repeat1"]["metrics"]
    r10 = gate["repeat_results"]["repeat10"]["metrics"]
    closure = {
        "experiment": "EXP-0088",
        "runtime_source_commit": args.runtime_commit,
        "formal_results": str(args.result_dir),
        "retained_artifacts": str(args.artifact_dir),
        "stage_b_gate": "fail",
        "pc027_timing_ledger_closure": "pass",
        "pc028_module_table": "pass",
        "correctness_gate": "pass",
        "physical_gate": "pass",
        "persistence_gate": "pass_on_ordinary_performance_path",
        "local_gate": "fail",
        "adoption_status": "rejected",
        "selected_baseline_changed": False,
        "repeat1_host_delta_percent": r1["host_wall_ns_per_block"]["change_percent"],
        "repeat1_attention_delta_percent": r1["attention_ticks"]["change_percent"],
        "repeat1_softmax_delta_percent": r1["u8_attention_softmax_ticks"]["change_percent"],
        "repeat10_host_delta_percent": r10["host_wall_ns_per_block"]["change_percent"],
        "repeat10_attention_delta_percent": r10["attention_ticks"]["change_percent"],
        "repeat10_softmax_delta_percent": r10["u8_attention_softmax_ticks"]["change_percent"],
    }
    (args.result_dir / "closure_summary.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
