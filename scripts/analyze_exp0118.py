#!/usr/bin/env python3
"""Validate and report EXP-0118 preemptible worker-assisted Q expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112


base = exp112.base
REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0111_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0111/"
    "20260831T181039Z_eebd967a3e34_formal"
)
EXP0112_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0112/"
    "20260831T183443Z_5429c70bc6eb_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "qkv_projection_ticks",
    "attention_qk_norm_pool_wait_ticks", "total_ticks",
)
ASSIST = (
    "w4u8_qkv_worker_assist_batch_count",
    "w4u8_qkv_worker_assist_region_count",
    "w4u8_qkv_worker_assist_k_tile_count",
    "w4u8_qkv_main_expand_region_count",
    "w4u8_qkv_main_expand_k_tile_count",
    "w4u8_qkv_worker_max_region_k_tiles",
    "w4u8_qkv_worker_expand_ticks",
    "w4u8_qkv_main_expand_ticks",
    "w4u8_qkv_worker_assist_wait_ticks",
)
PIPELINE = tuple(dict.fromkeys((
    *exp107.OVERLAP, *ASSIST,
    "w4u8_qkvo_weight_expand_ticks",
    "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks",
    "u8_attention_qk_norm_rope_ticks",
)))
PHYSICAL_EQUAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "w4u8_qkv_batch_count", "w4u8_qkvo_prefetch_count",
    "w4u8_qkvo_overlap_schedule_count",
    "attention_qk_norm_task_count", "attention_softmax_task_count",
    "u8_attention_group_count", "u8_attention_qk_execution_count",
    "u8_attention_av_execution_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=EXP0111_FORMAL)
    parser.add_argument("--exp0112-formal", type=Path,
                        default=EXP0112_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0112"
    compatible["w4u8_qkvo_pipeline_mode"] = (
        "qkvo_batch4_qk_head_pairs"
    )
    exp112.validate_record(
        compatible, repeat, "single_fence", audit=audit
    )
    base.require(record, "experiment", "EXP-0118")
    base.require(record, "w4u8_stream_fence_mode", "single_fence")
    expected_mode = (
        "qkvo_batch4_qk_head_pairs" if cell == "control" else
        "qkvo_batch4_qk_head_pairs_q_preemptible_expand"
    )
    base.require(record, "w4u8_qkvo_pipeline_mode", expected_mode)
    base.require(record, "attention_qk_norm_task_count", 24 * repeat)
    base.require(record, "hmx_command_count", 128 * repeat)
    base.require(record, "hmx_u8s8_tile_pair_count", 49_408 * repeat)
    base.require(record, "vtcm_requested_bytes", VTCM_BYTES)
    base.require(record, "vtcm_acquired_bytes", VTCM_BYTES)
    for field in (
        "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
        "intermediate_dma_descriptor_count", "intermediate_spill_fill_count",
    ):
        base.require(record, field, 0)

    assisted_batches = int(record["w4u8_qkv_worker_assist_batch_count"])
    assisted_regions = int(record["w4u8_qkv_worker_assist_region_count"])
    assisted_k = int(record["w4u8_qkv_worker_assist_k_tile_count"])
    main_regions = int(record["w4u8_qkv_main_expand_region_count"])
    main_k = int(record["w4u8_qkv_main_expand_k_tile_count"])
    max_region = int(record["w4u8_qkv_worker_max_region_k_tiles"])
    if main_k + assisted_k != 8_192 * repeat:
        raise SystemExit(f"{cell}: QKV K-tile accounting does not close")
    if cell == "control":
        if any((assisted_batches, assisted_regions, assisted_k,
                max_region,
                int(record["w4u8_qkv_worker_expand_ticks"]))):
            raise SystemExit("control executed worker-assisted expansion")
        if main_regions != 128 * repeat:
            raise SystemExit("control QKV output-tile accounting changed")
    else:
        if not (0 < assisted_batches <= 16 * repeat):
            raise SystemExit("candidate assisted-batch count is invalid")
        if assisted_regions <= 0 or assisted_k != 16 * assisted_regions:
            raise SystemExit("candidate assisted-region accounting is invalid")
        if main_regions + assisted_regions != 320 * repeat:
            raise SystemExit("candidate region accounting does not close")
        if max_region != 16:
            raise SystemExit("candidate exceeded the 16-K-tile preemption bound")
        if int(record["w4u8_qkv_worker_expand_ticks"]) <= 0:
            raise SystemExit("candidate lacks worker expansion work")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def baseline_pc028(exp0109_dir: Path, exp0111_dir: Path,
                   exp0112_dir: Path) -> dict:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0111_dir / "paired_candidate_r10.jsonl", SAMPLES)),
        "w4u8_parent": exp109.module_medians(load(
            exp0112_dir / "paired_single_fence_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path, exp0112_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0118")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    hashes = set()
    boundaries = set()
    for cell in CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        hashes.add(row["output_hash"])
        boundary = tuple(row[field] for field in (
            "u8_input_norm_actual_hash",
            "u8_attention_actual_score_hash",
            "u8_attention_actual_probability_hash",
            "u8_attention_actual_av_hash",
        ))
        boundaries.add(boundary)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "boundary_hashes": boundary,
        }
    if len(hashes) != 1 or len(boundaries) != 1:
        raise SystemExit("candidate changed final output or an audited boundary")

    comparisons = {}
    records = {}
    repeat_pass = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        values = metrics(sides["control"], sides["candidate"])
        speed = all(
            values[field][key] < 0.0
            for field in ("host_wall_ns_per_block", "qkv_projection_ticks")
            for key in ("change_percent", "paired_change_percent_median")
        )
        physical = physical_equal(sides["control"], sides["candidate"])
        execution = (
            values["w4u8_qkv_worker_assist_region_count"]["control"] == 0.0
            and values["w4u8_qkv_worker_assist_region_count"]["candidate"] > 0.0
            and values["w4u8_qkv_worker_max_region_k_tiles"]["candidate"] == 16.0
        )
        comparisons[f"repeat{repeat}"] = {
            "metrics": values,
            "strict_speed_gate": speed,
            "physical_equality_gate": physical,
            "preemptible_execution_gate": execution,
        }
        repeat_pass.append(speed and physical and execution)

    return {
        "experiment": "EXP-0118",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "comparisons": comparisons,
        "local_gate_pass": all(repeat_pass),
        "local_adoption_eligible": all(repeat_pass),
        "pc028": baseline_pc028(
            exp0109_dir, exp0111_dir, exp0112_dir),
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["candidate"]),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              values: dict) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        item = values[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, item['control'])} | "
            f"{base.format_value(field, item['candidate'])} | "
            f"{fmt_change(item['change_percent'])} | "
            f"{fmt_change(item['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    candidate = summary["repeat10_candidate_modules"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    candidate_total = candidate["Complete block Host wall"]
    lines.extend([
        "## PC-028 current recipe wall-time table (repeat10)", "",
        "| Module | W16A16 | W4A16 specialized | W4A8 parent | W4A8 candidate | A8 candidate vs A16 speed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        cells = []
        for key in ("f16f16", "w4f16", "w4u8_parent"):
            value = table[key][name]
            cells.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        value = candidate[name]
        cells.append(
            f"{value:.1f} us" if name == "Complete block Host wall"
            else f"{value:.1f} us ({100*value/candidate_total:.1f}%)"
        )
        speed = (table["w4f16"][name] / candidate[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{cells[3]} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0118 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        result = summary["comparisons"][f"repeat{repeat}"]
        values = result["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(lines, "Primary targets", TARGETS, values)
        add_table(lines, "Additive Block Timing Ledger",
                  exp107.LEDGER, values)
        add_table(lines, "HMX/HVX/DMA, assistance and waits",
                  PIPELINE, values)
        add_table(lines, "Traffic, commands and residency",
                  exp107.PHYSICAL, values)
        lines.append(
            "Strict QKV+Host speed: **{}**; physical equality: **{}**; "
            "preemptible execution: **{}**.\n".format(
                "PASS" if result["strict_speed_gate"] else "FAIL",
                "PASS" if result["physical_equality_gate"] else "FAIL",
                "PASS" if result["preemptible_execution_gate"] else "FAIL",
            )
        )
    lines.extend([
        "## Decision", "",
        f"EXP-0118 local gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**. "
        f"Local adoption eligibility: **{'YES' if summary['local_adoption_eligible'] else 'NO'}**. "
        "Only the user may promote a candidate to baseline.", "",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal,
        args.exp0111_formal, args.exp0112_formal,
    )
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
