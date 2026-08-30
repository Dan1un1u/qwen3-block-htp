#!/usr/bin/env python3
"""Validate EXP-0068 Attention HVX context-scaling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0063 as report_fields
import validate_exp0065 as previous


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
TARGETS = (
    "host_wall_ns_per_block",
    "qkv_plus_attention_ticks",
)
OVERLAP = tuple(dict.fromkeys((
    *base.OVERLAP,
    "attention_qk_norm_pool_wait_ticks",
    "u8_attention_qk_norm_rope_ticks",
    "u8_attention_softmax_ticks",
    "u8_attention_pipeline_wait_ticks",
)))
EXTRA_REPORT_FIELDS = tuple(dict.fromkeys((
    *report_fields.EXTRA_REPORT_FIELDS,
    "attention_qk_norm_task_count",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load_selection(result_dir: Path) -> tuple[str, int]:
    selection = json.loads((result_dir / "selection.json").read_text())
    base.require(selection, "experiment", "EXP-0068")
    mode = str(selection.get("selected_mode"))
    contexts = int(selection.get("selected_attention_hvx_contexts", 0))
    if mode not in ("context5", "context6"):
        raise SystemExit(f"invalid selected mode: {mode}")
    if contexts != {"context5": 5, "context6": 6}[mode]:
        raise SystemExit("selected mode/context mismatch")
    return mode, contexts


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, selected_contexts: int,
                    audit_enabled: bool = False) -> None:
    expected_contexts = 4 if mode == "control" else selected_contexts
    compatibility = dict(record)
    compatibility.update({
        "experiment": "EXP-0065",
        "attention_hvx_contexts": 4,
        "attention_hvx_workers_created": 3,
        "attention_hvx_workers_locked": 3,
    })
    previous.validate_record(
        compatibility, repeat, "candidate", audit_enabled
    )
    fixed = {
        "experiment": "EXP-0068",
        "attention_pipeline_mode": (
            "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
            "hmx_batch_lut_templates"
        ),
        "attention_hvx_contexts": expected_contexts,
        "attention_hvx_workers_created": expected_contexts - 1,
        "attention_hvx_workers_locked": expected_contexts - 1,
        "attention_pool_status": 0,
        "hmx_command_count": 192 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    for field in TARGETS + (
        "invocation_ticks", "qkv_projection_ticks", "attention_ticks",
        "attention_qk_norm_pool_wait_ticks",
    ):
        if field == "qkv_plus_attention_ticks":
            continue
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def add_derived(record: dict[str, object]) -> None:
    record["qkv_plus_attention_ticks"] = (
        float(record["qkv_projection_ticks"])
        + float(record["attention_ticks"])
    )


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed")
    selected_mode, selected_contexts = load_selection(result_dir)

    correctness: dict[str, object] = {}
    hashes: dict[str, tuple[str, str, str]] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(
            record, 1, mode, selected_contexts, audit_enabled=True
        )
        hashes[mode] = (
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        correctness[mode] = {
            "attention_hvx_contexts": record["attention_hvx_contexts"],
            "attention_hvx_workers_created":
                record["attention_hvx_workers_created"],
            "attention_hvx_workers_locked":
                record["attention_hvx_workers_locked"],
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "qk_hash": hashes[mode][0],
            "probability_hash": hashes[mode][1],
            "av_hash": hashes[mode][2],
        }
    if hashes["control"] != hashes["candidate"]:
        raise SystemExit("candidate changed QK/probability/AV audit hashes")
    if hashes["control"] != (QK_HASH, PROBABILITY_HASH, AV_HASH):
        raise SystemExit("Attention audit hashes differ from EXP-0065")
    for mode in MODES:
        if correctness[mode]["output_hash"] != OUTPUT_HASH:
            raise SystemExit(f"wrong final output hash for {mode}")

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
        "w4u8_mlp_gate_up_hmx_command_count",
        "w4u8_mlp_down_hmx_command_count",
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
                validate_record(record, repeat, mode, selected_contexts)
                add_derived(record)
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
        worker_gate = (
            metrics["attention_hvx_workers_created"]["control"] == 3.0
            and metrics["attention_hvx_workers_locked"]["control"] == 3.0
            and metrics["attention_hvx_workers_created"]["candidate"]
                == float(selected_contexts - 1)
            and metrics["attention_hvx_workers_locked"]["candidate"]
                == float(selected_contexts - 1)
        )
        passed = speed_gate and invariant_gate and worker_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "two_target_speed_gate": speed_gate,
            "unchanged_math_traffic_commands_and_resources_gate":
                invariant_gate,
            "selected_worker_count_gate": worker_gate,
        }

    return {
        "experiment": "EXP-0068",
        "control": "EXP-0065 main plus three Attention HVX workers",
        "candidate": (
            f"EXP-0065 main plus {selected_contexts - 1} Attention HVX "
            "workers"
        ),
        "selected_search_mode": selected_mode,
        "selected_attention_hvx_contexts": selected_contexts,
        "package_manifest_sha256": base.sha256(package_dir / "manifest.json"),
        "byte_exact_final_output_gate": True,
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


def add_table(lines: list[str], title: str,
              fields: tuple[str, ...],
              metrics: dict[str, dict[str, float | None]]) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | 4-context control | Selected candidate | Delta | Paired delta |",
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


def render_report(summary: dict[str, object]) -> str:
    contexts = int(summary["selected_attention_hvx_contexts"])
    lines = [
        "# EXP-0068 — Complete profiling report", "",
        "The control retains EXP-0065's main thread plus three persistent "
        "HVX workers. A bounded three-round search selected "
        f"{contexts} total contexts (main plus {contexts - 1} workers). "
        "Only Q/K preparation and GQA Attention task capacity changes; "
        "all arithmetic, qparams, tensor layouts, HMX commands/tile pairs, "
        "DMA traffic, MLP scheduling and physical residency are unchanged.",
        "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(
            lines, "Primary wall-latency targets",
            (
                "host_wall_ns_per_block", "invocation_ticks", "total_ticks",
                "qkv_projection_ticks", "attention_ticks",
                "qkv_plus_attention_ticks",
                "attention_qk_norm_pool_wait_ticks",
            ), metrics,
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
            f"unchanged math/traffic/commands/resources: **{'PASS' if result['unchanged_math_traffic_commands_and_resources_gate'] else 'FAIL'}**; "
            f"worker creation/lock gate: **{'PASS' if result['selected_worker_count_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact to EXP-0065, 0 LSB |",
        "| QK / probability / AV audit boundaries | byte-exact to EXP-0065 |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0068 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Selected Baseline is unchanged without explicit user promotion.",
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
