#!/usr/bin/env python3
"""Validate EXP-0069 split Attention context-domain evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_exp0050 as base
import validate_exp0068 as previous


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "candidate")
TARGETS = (
    "host_wall_ns_per_block",
    "attention_ticks",
    "qkv_plus_attention_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load_selection(result_dir: Path) -> tuple[str, int]:
    selection = json.loads((result_dir / "selection.json").read_text())
    base.require(selection, "experiment", "EXP-0069")
    base.require(selection, "qk_prep_contexts", 6)
    mode = str(selection.get("selected_mode"))
    contexts = int(selection.get("selected_attention_active_contexts", 0))
    if mode not in ("attention4", "attention5"):
        raise SystemExit(f"invalid selected mode: {mode}")
    if contexts != {"attention4": 4, "attention5": 5}[mode]:
        raise SystemExit("selected mode/context mismatch")
    return mode, contexts


def validate_record(record: dict[str, object], repeat: int,
                    mode: str, selected_active: int,
                    audit_enabled: bool = False) -> None:
    expected_active = 6 if mode == "control" else selected_active
    compatibility = dict(record)
    compatibility.update({
        "experiment": "EXP-0068",
        "attention_active_contexts": 6,
    })
    previous.validate_record(
        compatibility, repeat, "candidate", 6, audit_enabled
    )
    fixed = {
        "experiment": "EXP-0069",
        "attention_hvx_contexts": 6,
        "attention_active_contexts": expected_active,
        "attention_hvx_workers_created": 5,
        "attention_hvx_workers_locked": 5,
        "attention_pool_status": 0,
        "hmx_command_count": 192 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
    }
    for field, expected in fixed.items():
        base.require(record, field, expected)
    for field in (
        "host_wall_ns_per_block", "invocation_ticks",
        "qkv_projection_ticks", "attention_ticks",
        "attention_qk_norm_pool_wait_ticks",
    ):
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
    selected_mode, selected_active = load_selection(result_dir)

    correctness: dict[str, object] = {}
    hashes: dict[str, tuple[str, str, str]] = {}
    for mode in MODES:
        record = base.load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", 1
        )[0]
        validate_record(
            record, 1, mode, selected_active, audit_enabled=True
        )
        hashes[mode] = (
            str(record["u8_attention_actual_score_hash"]),
            str(record["u8_attention_actual_probability_hash"]),
            str(record["u8_attention_actual_av_hash"]),
        )
        correctness[mode] = {
            "attention_hvx_contexts": record["attention_hvx_contexts"],
            "attention_active_contexts": record["attention_active_contexts"],
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
    if hashes["control"] != (
        previous.QK_HASH, previous.PROBABILITY_HASH, previous.AV_HASH
    ):
        raise SystemExit("Attention audit hashes differ from EXP-0068")
    for mode in MODES:
        if correctness[mode]["output_hash"] != previous.OUTPUT_HASH:
            raise SystemExit(f"wrong final output hash for {mode}")

    invariant_fields = (
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "weight_dma_descriptor_count", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
        "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
        "w4u8_qkvo_overlap_schedule_count", "attention_qk_norm_task_count",
        "vtcm_requested_bytes", "vtcm_acquired_bytes",
        "vtcm_peak_plan_bytes", "attention_hvx_workers_created",
        "attention_hvx_workers_locked", "w4u8_mlp_gate_up_hmx_command_count",
        "w4u8_mlp_down_hmx_command_count",
    )
    repeat_results: dict[str, object] = {}
    overall: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: base.load_jsonl(
                result_dir / f"paired_{mode}_r{repeat}.jsonl", SAMPLES
            )
            for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode, selected_active)
                add_derived(record)
        fields = tuple(dict.fromkeys((
            *TARGETS, "invocation_ticks", "total_ticks", *base.LEDGER,
            *previous.OVERLAP, *base.COUNTERS, *base.RESOURCES,
            *previous.EXTRA_REPORT_FIELDS,
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
        passed = speed_gate and invariant_gate
        overall.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "three_target_speed_gate": speed_gate,
            "unchanged_math_traffic_commands_resources_and_pool_gate":
                invariant_gate,
        }

    return {
        "experiment": "EXP-0069",
        "control": "EXP-0068 prep6/Attention6",
        "candidate": f"EXP-0068 prep6/Attention{selected_active}",
        "selected_search_mode": selected_mode,
        "qk_prep_contexts": 6,
        "selected_attention_active_contexts": selected_active,
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
        "| Metric | 6/6 control | Split-domain candidate | Delta | Paired delta |",
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
    active = int(summary["selected_attention_active_contexts"])
    lines = [
        "# EXP-0069 — Complete profiling report", "",
        "Both paths create and lock the same main-plus-five persistent HVX "
        "domain and use all six contexts for Q/K preparation. The selected "
        f"candidate allows only {active} total contexts to claim the eight "
        "GQA Attention tasks. Arithmetic, qparams, layouts, HMX work, DMA "
        "traffic, VTCM and every non-Attention stage are unchanged.", "",
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
        add_table(
            lines, "Overlapping engine work and waits",
            previous.OVERLAP, metrics,
        )
        add_table(
            lines, "Traffic, commands, counters and residency",
            tuple(dict.fromkeys((
                *base.COUNTERS, *base.RESOURCES,
                *previous.EXTRA_REPORT_FIELDS,
            ))), metrics,
        )
        lines.extend([
            f"Three-target speed gate: **{'PASS' if result['three_target_speed_gate'] else 'FAIL'}**; "
            f"unchanged math/traffic/commands/resources/pool: **{'PASS' if result['unchanged_math_traffic_commands_resources_and_pool_gate'] else 'FAIL'}**.",
            "",
        ])
    lines.extend([
        "## Physical and correctness gates", "",
        "| Gate | Result |", "|---|---:|",
        "| Final block output | byte-exact to EXP-0068, 0 LSB |",
        "| QK / probability / AV audit boundaries | byte-exact to EXP-0068 |",
        "| Persistent HVX pool | six contexts created and locked in both paths |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |", "",
        "The additive ledger and overlapping engine-work counters are not "
        "summed together. Complete Host wall remains primary.", "",
        "## Decision", "",
        f"EXP-0069 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
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
