#!/usr/bin/env python3
"""Validate and report EXP-0124 W4U8 QKV producer/consumer ring."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0120 as exp120
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CORRECTNESS_CELLS = ("control", "ring1", "ring2", "ring3")
FORMAL_CELLS = ("control", "ring3")
EXPECTED_WORKERS = {"control": 0, "ring1": 1, "ring2": 2, "ring3": 3}
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0111_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0111/"
    "20260831T181039Z_eebd967a3e34_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "total_ticks",
)
RING_FIELDS = (
    "w4u8_qkvo_weight_expand_ticks", "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks", "w4u8_qkv_ring_slot_count",
    "w4u8_qkv_ring_expand_worker_count",
    "w4u8_qkv_ring_prep_worker_count",
    "w4u8_qkv_ring_dispatch_count", "w4u8_qkv_ring_batch_count",
    "w4u8_qkv_ring_expand_task_count",
    "w4u8_qkv_ring_hmx_dispatch_count",
    "w4u8_qkv_ring_head_publish_count",
    "w4u8_qkv_ring_pipeline_ticks", "w4u8_qkv_ring_dma_wait_ticks",
    "w4u8_qkv_ring_producer_slot_wait_ticks",
    "w4u8_qkv_ring_expand_ticks",
    "w4u8_qkv_ring_hmx_ready_wait_ticks",
    "w4u8_qkv_ring_hmx_compute_ticks",
    "w4u8_qkv_ring_pool_wait_ticks",
    "attention_qk_norm_worker_work_ticks",
    "attention_qk_norm_pool_wait_ticks",
)
PHYSICAL_EQUAL_FIELDS = exp112.PHYSICAL_EQUAL_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0111-formal", type=Path,
                        default=EXP0111_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0120"
    exp120.validate_record(compatible, repeat, "ring16", audit=audit)
    base.require(record, "experiment", "EXP-0124")
    base.require(record, "w4u8_qkv_ring_expand_workers",
                 EXPECTED_WORKERS[cell])
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")
    workers = EXPECTED_WORKERS[cell]
    if workers == 0:
        for field in (
            "w4u8_qkv_ring_slot_count",
            "w4u8_qkv_ring_expand_worker_count",
            "w4u8_qkv_ring_prep_worker_count",
            "w4u8_qkv_ring_dispatch_count",
            "w4u8_qkv_ring_batch_count",
            "w4u8_qkv_ring_expand_task_count",
            "w4u8_qkv_ring_hmx_dispatch_count",
            "w4u8_qkv_ring_head_publish_count",
            "w4u8_qkv_ring_pipeline_ticks",
            "w4u8_qkv_ring_dma_wait_ticks",
            "w4u8_qkv_ring_producer_slot_wait_ticks",
            "w4u8_qkv_ring_expand_ticks",
            "w4u8_qkv_ring_hmx_ready_wait_ticks",
            "w4u8_qkv_ring_hmx_compute_ticks",
            "w4u8_qkv_ring_pool_wait_ticks",
        ):
            base.require(record, field, 0)
    else:
        expected = {
            "w4u8_qkv_ring_slot_count": 4,
            "w4u8_qkv_ring_expand_worker_count": workers,
            "w4u8_qkv_ring_prep_worker_count": 5 - workers,
            "w4u8_qkv_ring_dispatch_count": repeat,
            "w4u8_qkv_ring_batch_count": 32 * repeat,
            "w4u8_qkv_ring_expand_task_count": 128 * repeat,
            "w4u8_qkv_ring_hmx_dispatch_count": repeat,
            "w4u8_qkv_ring_head_publish_count": 24 * repeat,
        }
        for field, value in expected.items():
            base.require(record, field, value)
        for field in (
            "w4u8_qkv_ring_pipeline_ticks",
            "w4u8_qkv_ring_dma_wait_ticks",
            "w4u8_qkv_ring_expand_ticks",
            "w4u8_qkv_ring_hmx_ready_wait_ticks",
            "w4u8_qkv_ring_hmx_compute_ticks",
        ):
            if int(record.get(field, 0)) <= 0:
                raise SystemExit(f"{cell}: missing positive {field}")


def per_block(record: dict, field: str) -> float:
    if field == "qkv_plus_qk_norm_rope_ticks":
        return (
            float(record["qkv_projection_ticks"]) +
            float(record["qk_norm_rope_ticks"])
        ) / int(record["repeat_count"])
    return exp107.per_block(record, field)


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    control = [per_block(row, field) for row in left]
    candidate = [per_block(row, field) for row in right]
    lmed = float(statistics.median(control))
    rmed = float(statistics.median(candidate))
    paired = [
        (r / l - 1.0) * 100.0
        for l, r in zip(control, candidate) if l != 0.0
    ]
    return {
        "control": lmed,
        "candidate": rmed,
        "change_percent": (rmed / lmed - 1.0) * 100.0
        if lmed != 0.0 else None,
        "paired_change_percent_median":
            float(statistics.median(paired)) if paired else None,
        "paired_change_percent_min": min(paired) if paired else None,
        "paired_change_percent_max": max(paired) if paired else None,
    }


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *RING_FIELDS, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def baseline_pc028(exp0109_dir: Path, exp0111_dir: Path) -> dict:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0111_dir / "paired_candidate_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0124")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    for cell in CORRECTNESS_CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
        }
    if len({value["output_hash"] for value in correctness.values()}) != 1:
        raise SystemExit("QKV ring cells produced different output hashes")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    for repeat in REPEATS:
        sides = {}
        for cell in FORMAL_CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = metrics(
            sides["control"], sides["ring3"]
        )

    speed_gate = all(
        comparisons[f"repeat{repeat}"][field][key] < 0.0
        for repeat in REPEATS
        for field in (
            "qkv_plus_qk_norm_rope_ticks", "host_wall_ns_per_block"
        )
        for key in ("change_percent", "paired_change_percent_median")
    )
    physical_gate = all(
        physical_equal(records[repeat]["control"],
                       records[repeat]["ring3"])
        for repeat in REPEATS
    )
    plan_gate = all(
        int(row["vtcm_requested_bytes"]) == VTCM_BYTES
        and int(row["vtcm_acquired_bytes"]) == VTCM_BYTES
        and 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES
        and int(row["intermediate_ddr_read_bytes"]) == 0
        and int(row["intermediate_ddr_write_bytes"]) == 0
        and int(row["intermediate_spill_fill_count"]) == 0
        for repeat in REPEATS for cell in FORMAL_CELLS
        for row in records[repeat][cell]
    )
    pc028 = baseline_pc028(exp0109_dir, exp0111_dir)
    pc028["w4u8"] = exp109.module_medians(records[10]["ring3"])
    return {
        "experiment": "EXP-0124",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "fixed_8mib_zero_ddr_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate and plan_gate,
        "selected_cell": "ring3" if speed_gate else "control",
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0111_dir) + "/paired_candidate_r10.jsonl",
            "w4u8": str(result_dir) + "/paired_ring3_r10.jsonl",
        },
    }


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    lines.extend([
        "## PC-028 three-recipe overview (repeat10)", "",
        "| Module | W16A16 | W4A16 specialized | W4A8 ring3 | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        values = []
        for key in ("f16f16", "w4f16", "w4u8"):
            value = table[key][name]
            values.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        speed = (table["w4f16"][name] / table["w4u8"][name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | {values[2]} | "
            f"{speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0124 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}: control versus ring3", ""])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(
            lines, "HMX/HVX/DMA, ring and waits",
            tuple(dict.fromkeys((
                *exp107.OVERLAP, *exp112.W4U8_PIPELINE, *RING_FIELDS
            ))), values
        )
        exp112.add_table(lines, "Traffic, commands and residency",
                         exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality except declared ring schedule | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB, zero intermediate DDR/spill | {'PASS' if summary['fixed_8mib_zero_ddr_gate'] else 'FAIL'} |",
        f"| QKV module and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0124 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal, args.exp0111_formal
    )
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
