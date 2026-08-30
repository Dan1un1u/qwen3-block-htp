#!/usr/bin/env python3
"""Validate EXP-0077 Gate/Up cross-phase linked-DMA prefetch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0070 as pc
import validate_exp0075 as previous


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
INPUT_NORM_HASH = "7255c2406108617c"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
TARGETS = ("host_wall_ns_per_block", "post_residual_gate_up_ticks")
LEDGER = previous.LEDGER
OVERLAP = tuple(dict.fromkeys((
    *previous.OVERLAP,
    "w4u8_gate_up_cross_prefetch_wait_ticks",
    "w4u8_gate_up_cross_prefetch_lifetime_ticks",
)))
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *previous.EXTRA_REPORT_FIELDS,
    "w4u8_gate_up_cross_prefetch_count",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def add_derived(record: dict[str, object]) -> None:
    record["post_residual_gate_up_ticks"] = sum(
        int(record[field])
        for field in (
            "post_attention_residual_ticks",
            "post_attention_norm_ticks",
            "gate_up_ticks",
        )
    )


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit_enabled: bool = False) -> None:
    compatibility = dict(record)
    compatibility.update({
        "experiment": "EXP-0075",
        "mlp_mode": "w4u8_streaming",
    })
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    prefetch_count = repeat if mode == "candidate" else 0
    fixed = {
        "experiment": "EXP-0077",
        "mlp_mode": (
            "w4u8_streaming_gate_up_prefetch"
            if mode == "candidate" else "w4u8_streaming"
        ),
        "w4u8_gate_up_cross_prefetch_count": prefetch_count,
        "hmx_command_count": 176 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    if mode == "candidate":
        lifetime = int(record["w4u8_gate_up_cross_prefetch_lifetime_ticks"])
        wait = int(record["w4u8_gate_up_cross_prefetch_wait_ticks"])
        if lifetime <= 0 or wait < 0 or wait > lifetime:
            raise SystemExit("invalid cross-prefetch timing telemetry")
    else:
        base.require(record, "w4u8_gate_up_cross_prefetch_wait_ticks", 0)
        base.require(
            record, "w4u8_gate_up_cross_prefetch_lifetime_ticks", 0
        )
    add_derived(record)
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
    boundary_hashes: dict[str, tuple[str, str, str, str]] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(record, 1, mode, audit_enabled=True)
        boundary_hashes[mode] = (
            str(record["u8_input_norm_actual_hash"]),
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "input_norm_hash": boundary_hashes[mode][0],
            "qk_hash": boundary_hashes[mode][1],
            "probability_hash": boundary_hashes[mode][2],
            "av_hash": boundary_hashes[mode][3],
            "prefetch_count": record[
                "w4u8_gate_up_cross_prefetch_count"
            ],
        }
        if record["output_hash"] != OUTPUT_HASH:
            raise SystemExit(f"wrong final output hash for {mode}")
    if boundary_hashes["control"] != boundary_hashes["candidate"]:
        raise SystemExit("candidate changed an audited boundary")
    if boundary_hashes["control"] != (
        INPUT_NORM_HASH, QK_HASH, PROBABILITY_HASH, AV_HASH
    ):
        raise SystemExit("audited boundaries differ from EXP-0075")

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
        "u8_attention_group_count",
        "u8_attention_qk_execution_count",
        "u8_attention_av_execution_count",
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
            *LEDGER, *OVERLAP, *base.COUNTERS,
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
        execution_gate = (
            metrics["w4u8_gate_up_cross_prefetch_count"]["control"] == 0.0
            and metrics["w4u8_gate_up_cross_prefetch_count"][
                "candidate"
            ] == 1.0
            and metrics["w4u8_gate_up_cross_prefetch_lifetime_ticks"][
                "candidate"
            ] > 0.0
        )
        passed = speed_gate and invariant_gate and execution_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "cross_phase_prefetch_execution_gate": execution_gate,
        }

    return {
        "experiment": "EXP-0077",
        "control": "EXP-0075 Gate/Up linked DMA starts inside MLP",
        "candidate": "first four Gate/Up descriptors start before residual",
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "byte_exact_final_output_gate": True,
        "byte_exact_audited_boundaries_gate": True,
        "unchanged_weight_bytes_descriptor_hmx_gate": True,
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
        "| Metric | EXP-0075 control | EXP-0077 candidate | Delta | Paired delta |",
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
    current = dict(pc.module_us(metrics, side))
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
    used = "EXP-0077" if summary["local_gate_pass"] else "EXP-0075 control"
    lines.extend([
        "## PC-028 current-best three-variant module wall-time", "",
        f"{used} supplies the W4A8 column. Percentages are shares of "
        "complete Host wall; positive speed means W4A8 is faster than W4A16.",
        "",
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
    lines = ["# EXP-0077 — Complete profiling report", ""]
    add_pc028_table(lines, summary)
    lines.extend([
        "The control starts every linked Gate/Up DMA chain inside the MLP. "
        "The candidate starts the same first four descriptors after O "
        "projection and lets them overlap the post-attention residual/RMSNorm. "
        "It later adopts those descriptors instead of submitting them again. "
        "Weight bytes, descriptor count, HMX commands, tile pairs, arithmetic "
        "and tensor layouts remain unchanged.", "",
    ])
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary wall-latency targets",
            ("host_wall_ns_per_block", "post_residual_gate_up_ticks",
             "invocation_ticks", "total_ticks"), metrics,
        )
        add_table(lines, "Additive Block Timing Ledger", LEDGER, metrics)
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
            f"cross-phase prefetch execution: **{'PASS' if result['cross_phase_prefetch_execution_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Correctness and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact to EXP-0075, 0 LSB |",
        "| Input norm / QK / probability / AV boundaries | byte-exact to EXP-0075 |",
        "| Weight bytes / DMA descriptors / HMX commands / tile pairs | unchanged |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0077 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
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
