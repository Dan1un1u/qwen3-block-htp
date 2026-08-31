#!/usr/bin/env python3
"""Validate EXP-0101 and render its PC-027/PC-028 closure."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import validate_exp0050 as base
import validate_exp0084 as exp84


SAMPLES = 5
REPEATS = (1, 10)
MODES = ("control", "ready_only")
OUTPUT_HASH = "69f22eeb035e5ec5"
AUDIT_HASHES = (
    "7255c2406108617c", "32aa949912e365be",
    "94f2e218f06f9627", "f853658f52032bde",
)
TARGETS = ("attention_ticks", "host_wall_ns_per_block")
EXP0084_EVIDENCE = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0084/"
    "20260830T160259Z_6dc437fe08ea_formal"
)
LEDGER = exp84.LEDGER
OVERLAP = tuple(dict.fromkeys((*base.OVERLAP,)))
PHYSICAL = tuple(dict.fromkeys((
    *base.COUNTERS, *base.RESOURCES, "block_invocation_count",
    "w4u8_attention_scheduler_mode",
    "w4u8_attention_scheduler_mode_observed",
    "w4u8_attention_ready_softmax_claim_count",
    "w4u8_attention_ready_av_claim_count",
    "w4u8_attention_predependency_claim_count",
    "w4u8_attention_scheduler_idle_poll_count",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, audit: bool = False) -> None:
    compatibility = dict(record)
    compatibility["experiment"] = "EXP-0084"
    exp84.validate_record(compatibility, repeat, "W4U8")
    scheduler = 0 if mode == "control" else 1
    fixed = {
        "experiment": "EXP-0101",
        "w4u8_attention_scheduler_mode": scheduler,
        "w4u8_attention_scheduler_mode_observed": scheduler,
        "block_invocation_count": repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": 5_306_080,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "hmx_command_count": 176 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "attention_softmax_task_count": 16 * repeat,
        "u8_attention_group_count": 8 * repeat,
        "w4u8_attention_ready_softmax_claim_count": (
            0 if mode == "control" else 16 * repeat
        ),
        "w4u8_attention_ready_av_claim_count": (
            0 if mode == "control" else 8 * repeat
        ),
        "w4u8_attention_predependency_claim_count": 0,
        "u8_attention_audit_ddr_write_bytes": 524_288 if audit else 0,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    if mode == "control":
        base.require(record, "w4u8_attention_scheduler_idle_poll_count", 0)
    elif int(record.get(
            "w4u8_attention_scheduler_idle_poll_count", 0)) <= 0:
        raise SystemExit("ready_only: scheduler did not poll for readiness")
    if record.get("output_hash") != OUTPUT_HASH or \
            int(record.get("mismatches", -1)) != 0 or \
            int(record.get("max_lsb", -1)) != 0:
        raise SystemExit(f"{mode}: final output is not byte-exact")
    if audit:
        actual = (
            record.get("u8_input_norm_actual_hash"),
            record.get("u8_attention_actual_score_hash"),
            record.get("u8_attention_actual_probability_hash"),
            record.get("u8_attention_actual_av_hash"),
        )
        if actual != AUDIT_HASHES:
            raise SystemExit(f"{mode}: unexpected audited boundaries")


def per_block(record: dict[str, object], field: str) -> float:
    unscaled = {
        "host_wall_ns_per_block", "vtcm_requested_bytes",
        "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
        "w4u8_attention_scheduler_mode",
        "w4u8_attention_scheduler_mode_observed",
        "w4u8_qkv_batch_n_tiles", "w4u8_mlp_gate_up_hvx_workers",
        "w4u8_mlp_down_hvx_workers", "attention_hvx_workers_created",
        "attention_hvx_workers_locked", "mlp_hvx_workers_created",
        "mlp_hvx_workers_locked", "w4u8_mlp_gate_up_hvx_hmx_overlap",
        "w4u8_mlp_down_hvx_hmx_overlap",
        "w4u8_mlp_gate_up_hvx_parallel_overlap",
        "w4u8_mlp_down_hvx_parallel_overlap",
    }
    if field in unscaled:
        return float(record.get(field, 0.0))
    return float(record.get(field, 0.0)) / int(record["repeat_count"])


def summarize(control: list[dict[str, object]],
              candidate: list[dict[str, object]], field: str) -> dict[str, object]:
    left = [per_block(record, field) for record in control]
    right = [per_block(record, field) for record in candidate]
    left_median = float(statistics.median(left))
    right_median = float(statistics.median(right))
    paired = [
        (right_value / left_value - 1.0) * 100.0
        for left_value, right_value in zip(left, right) if left_value != 0.0
    ]
    return {
        "control": left_median,
        "candidate": right_median,
        "change_percent": (
            (right_median / left_median - 1.0) * 100.0
            if left_median != 0.0 else None
        ),
        "paired_change_percent_median": (
            float(statistics.median(paired)) if paired else None
        ),
        "paired_change_percent_min": min(paired) if paired else None,
        "paired_change_percent_max": max(paired) if paired else None,
    }


def metric_set(control: list[dict[str, object]],
               candidate: list[dict[str, object]]) -> dict[str, object]:
    fields = tuple(dict.fromkeys((
        *TARGETS, "total_ticks", *LEDGER, *OVERLAP, *PHYSICAL,
    )))
    return {field: summarize(control, candidate, field) for field in fields}


def module_medians(records: list[dict[str, object]]) -> dict[str, float]:
    samples = [dict(exp84.module_us(record)) for record in records]
    return {
        name: float(statistics.median(sample[name] for sample in samples))
        for name in samples[0]
    }


def contextual_modules(variant: str) -> dict[str, float]:
    records = exp84.load_jsonl(
        EXP0084_EVIDENCE / "tri_variant" / "canonical" /
        f"{variant}_r10.jsonl", 7
    )
    return module_medians(records)


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != \
            (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during EXP-0101")
    static_gate = json.loads((result_dir / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness: dict[str, object] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(record, 1, mode, audit=True)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "softmax_tasks": record["attention_softmax_task_count"],
            "av_groups": record["u8_attention_group_count"],
            "ready_softmax_claims": record[
                "w4u8_attention_ready_softmax_claim_count"
            ],
            "ready_av_claims": record[
                "w4u8_attention_ready_av_claim_count"
            ],
            "predependency_claims": record[
                "w4u8_attention_predependency_claim_count"
            ],
        }

    repeat_results: dict[str, object] = {}
    records_by_repeat: dict[int, dict[str, list[dict[str, object]]]] = {}
    gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: base.load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl", SAMPLES
            ) for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        metrics = metric_set(records["control"], records["ready_only"])
        speed = all(
            metrics[field][key] < 0.0
            for field in TARGETS
            for key in ("change_percent", "paired_change_percent_median")
        )
        gates.append(speed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "attention_and_host_speed_gate": speed,
        }
        records_by_repeat[repeat] = records
    local_gate = all(gates)
    selected = "ready_only" if local_gate else "control"
    return {
        "experiment": "EXP-0101",
        "control": "claim-then-wait dependency stream",
        "candidate": "ready-only nonblocking bitmask scheduler",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "static_gate": static_gate,
        "byte_exact_final_output_gate": True,
        "byte_exact_audited_boundaries_gate": True,
        "scheduler_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "correctness": correctness,
        "repeat_results": repeat_results,
        "local_gate_pass": local_gate,
        "local_adoption_eligible": local_gate,
        "pc028": {
            "f16f16": contextual_modules("f16f16"),
            "w4f16": contextual_modules("w4f16"),
            "w4u8": module_medians(records_by_repeat[10][selected]),
            "w4u8_control": module_medians(records_by_repeat[10]["control"]),
            "w4u8_candidate": module_medians(records_by_repeat[10]["ready_only"]),
            "w4u8_provenance": (
                "EXP-0101 ready-only" if local_gate else "EXP-0101 control"
            ),
            "contextual_provenance": str(EXP0084_EVIDENCE),
        },
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_metric_table(lines: list[str], title: str, fields: tuple[str, ...],
                     metrics: dict[str, dict[str, object]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
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


def add_pc028(lines: list[str], summary: dict[str, object]) -> None:
    table = summary["pc028"]
    f16, w4f16, w4u8 = table["f16f16"], table["w4f16"], table["w4u8"]
    totals = {
        "f16": f16["Complete block Host wall"],
        "w4f16": w4f16["Complete block Host wall"],
        "w4u8": w4u8["Complete block Host wall"],
    }
    lines.extend([
        "## PC-028 three-variant repeat-ten module wall-time", "",
        f"The W4U8 column uses {table['w4u8_provenance']}.", "",
        "| Module | W16A16 | W4A16 | W4A8 | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in f16:
        if name == "Complete block Host wall":
            values = (f"{f16[name]:.1f} us", f"{w4f16[name]:.1f} us",
                      f"{w4u8[name]:.1f} us")
        else:
            values = (
                f"{f16[name]:.1f} us ({100*f16[name]/totals['f16']:.1f}%)",
                f"{w4f16[name]:.1f} us ({100*w4f16[name]/totals['w4f16']:.1f}%)",
                f"{w4u8[name]:.1f} us ({100*w4u8[name]/totals['w4u8']:.1f}%)",
            )
        speed = (w4f16[name] / w4u8[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | {values[2]} | "
            f"{speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = ["# EXP-0101 — Complete profiling report", ""]
    add_pc028(lines, summary)
    lines.extend([
        "The candidate replaces claim-before-ready task reservation with "
        "ready bitmasks. QK publishes two Softmax slices and the second "
        "completed slice publishes AV. Math, task counts, HMX commands, "
        "traffic and all tensor bytes remain unchanged.", "",
    ])
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_metric_table(lines, "Primary targets", TARGETS, metrics)
        add_metric_table(lines, "Additive Block Timing Ledger", LEDGER, metrics)
        add_metric_table(lines, "Overlapping engine work and waits", OVERLAP, metrics)
        add_metric_table(lines, "Traffic, commands and residency", PHYSICAL, metrics)
        lines.extend([
            "Attention and complete Host speed gate: "
            f"**{'PASS' if result['attention_and_host_speed_gate'] else 'FAIL'}**.", "",
        ])
    lines.extend([
        "## Correctness, scheduler and physical gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block and audited boundaries | byte-exact, 0 mismatch, 0 LSB |",
        "| Ready-only task execution | 16 Softmax / 8 AV / 0 predependency claims |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR / spill-fill | 0 / 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| Weight bytes / DMA descriptors | 25,444,352 / 512 per block |",
        "| HMX commands / U8xS8 tile pairs | 176 / 49,408 per block |", "",
        "## Decision", "",
        f"EXP-0101 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Selected Baseline remains unchanged until explicit user promotion.", "",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
