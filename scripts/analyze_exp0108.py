#!/usr/bin/env python3
"""Validate EXP-0108 and render its PC-027/PC-028 closure."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import validate_exp0050 as base
import validate_exp0084 as exp84


REPEATS = (1, 10)
SAMPLES = 5
RECIPES = ("f16f16", "w4f16", "w4u8")
LEDGER = exp84.LEDGER
OVERLAP = tuple(dict.fromkeys((
    *base.OVERLAP, "attention_qk_norm_main_work_ticks",
    "attention_qk_norm_worker_work_ticks",
    "attention_qk_norm_pool_wait_ticks", "attention_gqa_worker_work_ticks",
    "attention_gqa_hmx_wait_ticks", "attention_gqa_queue_wait_ticks",
    "crouton_qkv_transform_ticks",
)))
PHYSICAL = tuple(dict.fromkeys((
    *base.COUNTERS, *base.RESOURCES, "block_invocation_count",
    "weight_ddr_read_bytes", "weight_dma_descriptor_count",
    "hmx_command_count", "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count", "crouton_qkv_projection_count",
    "crouton_qkv_unpack_skipped", "crouton_qk_operand_count",
    "crouton_av_weight_count", "crouton_q_operand_mismatch_count",
    "crouton_k_operand_mismatch_count", "crouton_v_operand_mismatch_count",
)))
UNSCALED = {
    "host_wall_ns_per_block", "vtcm_requested_bytes",
    "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate(record: dict, repeat: int, recipe: str,
             candidate: bool, audit: bool = False) -> None:
    variant = {
        "f16f16": "F16F16", "w4f16": "W4F16", "w4u8": "W4U8",
    }[recipe]
    compatible = dict(record)
    compatible["experiment"] = "EXP-0084"
    exp84.validate_record(compatible, repeat, variant)
    base.require(record, "experiment", "EXP-0108")
    if recipe != "w4u8":
        base.require(
            record, "crouton_boundary_mode",
            "qkv_norms" if candidate else "norms",
        )
        multiplier = repeat if candidate else 0
        expected = {
            "crouton_qkv_projection_count": 3 * multiplier,
            "crouton_qkv_unpack_skipped": 128 * multiplier,
            "crouton_qk_operand_count": 24 * multiplier,
            "crouton_av_weight_count": 8 * multiplier,
            "crouton_q_operand_mismatch_count": 0,
            "crouton_k_operand_mismatch_count": 0,
            "crouton_v_operand_mismatch_count": 0,
        }
        for field, value in expected.items():
            base.require(record, field, value)
    if audit and (int(record.get("mismatches", -1)) != 0 or
                  int(record.get("max_lsb", -1)) != 0):
        raise SystemExit(f"{recipe}: audited output mismatch")


def per_block(record: dict, field: str) -> float:
    if field == "qkv_qknorm_ticks":
        value = float(record["qkv_projection_ticks"]) + \
            float(record["qk_norm_rope_ticks"])
    else:
        value = float(record.get(field, 0.0))
    return value if field in UNSCALED else value / int(record["repeat_count"])


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
        "control": lmed, "candidate": rmed,
        "change_percent": (rmed / lmed - 1.0) * 100.0
        if lmed != 0.0 else None,
        "paired_change_percent_median":
            float(statistics.median(paired)) if paired else None,
        "paired_change_percent_min": min(paired) if paired else None,
        "paired_change_percent_max": max(paired) if paired else None,
    }


def metric_set(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        "host_wall_ns_per_block", "qkv_qknorm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "total_ticks", *LEDGER, *OVERLAP, *PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def modules(records: list[dict]) -> dict[str, float]:
    samples = [dict(exp84.module_us(record)) for record in records]
    return {
        name: float(statistics.median(row[name] for row in samples))
        for name in samples[0]
    }


def build_summary(result_dir: Path) -> dict:
    if (result_dir / "boot_id_before.txt").read_bytes() != \
            (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during EXP-0108")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    for recipe in RECIPES:
        correctness[recipe] = {}
        hashes = []
        for side in ("control", "candidate"):
            row = load(result_dir / f"correctness_{side}_{recipe}.jsonl", 1)[0]
            validate(row, 1, recipe, side == "candidate", audit=True)
            correctness[recipe][side] = {
                "output_hash": row["output_hash"],
                "mismatches": row["mismatches"], "max_lsb": row["max_lsb"],
            }
            hashes.append(row["output_hash"])
        if hashes[0] != hashes[1]:
            raise SystemExit(f"{recipe}: control/candidate hash mismatch")

    records: dict[int, dict[str, dict[str, list[dict]]]] = {}
    results = {}
    fp16_speed = []
    u8_parity = []
    for repeat in REPEATS:
        records[repeat] = {}
        results[f"repeat{repeat}"] = {}
        for recipe in RECIPES:
            sides = {}
            for side in ("control", "candidate"):
                rows = load(
                    result_dir / f"paired_{side}_{recipe}_r{repeat}.jsonl",
                    SAMPLES,
                )
                for row in rows:
                    validate(row, repeat, recipe, side == "candidate")
                sides[side] = rows
            values = metric_set(sides["control"], sides["candidate"])
            records[repeat][recipe] = sides
            results[f"repeat{repeat}"][recipe] = {"metrics": values}
            if recipe in ("f16f16", "w4f16"):
                for field in ("qkv_qknorm_ticks", "host_wall_ns_per_block"):
                    fp16_speed.extend((
                        values[field]["change_percent"],
                        values[field]["paired_change_percent_median"],
                    ))
            else:
                for field in ("qkv_qknorm_ticks", "host_wall_ns_per_block"):
                    u8_parity.extend((
                        abs(values[field]["change_percent"]),
                        abs(values[field]["paired_change_percent_median"]),
                    ))

    fp16_gate = all(value < 0.0 for value in fp16_speed)
    parity_gate = all(value <= 1.0 for value in u8_parity)
    selected = "candidate" if fp16_gate else "control"
    return {
        "experiment": "EXP-0108",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static, "correctness_gate": True,
        "fixed_8mib_vtcm_gate": True, "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True, "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True, "qnn_dependency": False,
        "correctness": correctness, "repeat_results": results,
        "fp16_qkv_and_host_speed_gate": fp16_gate,
        "w4u8_parity_gate": parity_gate,
        "local_gate_pass": fp16_gate and parity_gate,
        "selected": selected,
        "pc028": {
            recipe: modules(records[10][recipe][
                selected if recipe != "w4u8" else "control"
            ]) for recipe in RECIPES
        },
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
    f16, w4f16, w4u8 = table["f16f16"], table["w4f16"], table["w4u8"]
    totals = {"f16": f16["Complete block Host wall"],
              "w4f16": w4f16["Complete block Host wall"],
              "w4u8": w4u8["Complete block Host wall"]}
    lines.extend([
        "## PC-028 three-recipe repeat-ten module wall-time", "",
        "| Module | W16A16 | W4A16 | W4A8 | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in f16:
        if name == "Complete block Host wall":
            cells = (f"{f16[name]:.1f} us", f"{w4f16[name]:.1f} us",
                     f"{w4u8[name]:.1f} us")
        else:
            cells = (
                f"{f16[name]:.1f} us ({100*f16[name]/totals['f16']:.1f}%)",
                f"{w4f16[name]:.1f} us ({100*w4f16[name]/totals['w4f16']:.1f}%)",
                f"{w4u8[name]:.1f} us ({100*w4u8[name]/totals['w4u8']:.1f}%)",
            )
        speed = (w4f16[name] / w4u8[name] - 1.0) * 100.0
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0108 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        lines.extend([f"## Repeat {repeat}", ""])
        for recipe in RECIPES:
            values = summary["repeat_results"][f"repeat{repeat}"][recipe]["metrics"]
            lines.extend([f"### {recipe}", ""])
            add_table(lines, "Primary targets", (
                "host_wall_ns_per_block", "qkv_qknorm_ticks",
                "qkv_projection_ticks", "qk_norm_rope_ticks",
                "attention_ticks", "total_ticks",
            ), values)
            add_table(lines, "Additive Block Timing Ledger", LEDGER, values)
            add_table(lines, "Overlapping engines and waits", OVERLAP, values)
            add_table(lines, "Traffic, commands and residency", PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| F16F16 and W4F16 QKV+Host speed | {'PASS' if summary['fp16_qkv_and_host_speed_gate'] else 'FAIL'} |",
        f"| W4U8 <=1% parity | {'PASS' if summary['w4u8_parity_gate'] else 'FAIL'} |",
        "| Correctness and physical contract | PASS |",
        f"| EXP-0108 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
