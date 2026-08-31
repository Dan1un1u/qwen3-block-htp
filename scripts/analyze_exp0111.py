#!/usr/bin/env python3
"""Validate and report EXP-0111 W4F16 Gate/Up publication fences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0110 as exp110
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "gate_up_ticks", "activation_ticks",
    "down_ticks", "total_ticks",
)
W4F16_PIPELINE = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "w4f16_expand_ticks", "projection_hmx_wait_ticks",
    "projection_unpack_ticks", "hmx_ready_wait_ticks",
    "w4f16_streamed_command_count", "w4f16_expand_work_ticks",
    "w4f16_expand_region_count", "w4f16_prefetch_count",
    "w4f16_prefetch_wait_ticks", "w4f16_first_expand_ticks",
    "w4f16_steady_expand_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_hmx_tail_wait_ticks", "w4f16_cross_prefetch_count",
    "w4f16_cross_prefetch_wait_ticks",
    "w4f16_cross_prefetch_lifetime_ticks",
    "w4f16_gate_up_weight_dma_ticks", "w4f16_gate_up_expand_ticks",
    "w4f16_gate_up_expand_work_ticks",
    "w4f16_gate_up_expand_pool_wait_ticks",
    "w4f16_gate_up_prefetch_wait_ticks",
    "w4f16_gate_up_hmx_wait_ticks",
    "w4f16_gate_up_hmx_tail_wait_ticks",
    "w4f16_gate_up_unpack_ticks", "w4f16_gate_up_stream_work_ticks",
    "w4f16_gate_up_stream_ready_wait_ticks",
    "w4f16_gate_up_stream_join_wait_ticks",
    "w4f16_gate_up_hmx_command_count",
    "w4f16_gate_up_scale_init_ticks",
)
PHYSICAL_EQUAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "w4f16_gate_up_hmx_command_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0110"
    exp110.validate_record(compatible, repeat, "carrier", audit=audit)
    base.require(record, "experiment", "EXP-0111")
    base.require(
        record, "w4f16_group_fence_mode",
        "join_only" if cell == "candidate" else "control",
    )
    if int(record.get("w4f16_expand_mismatch_count", -1)) != 0:
        raise SystemExit(f"{cell}: W4F16 expansion audit failed")
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *W4F16_PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def baseline_pc028(exp0109_dir: Path,
                   current_w4f16: list[dict]) -> dict:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(current_w4f16),
        "w4u8": exp109.module_medians(load(
            exp0109_dir / "paired_fastest_w4u8_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0111")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    for cell in CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "w4f16_expand_mismatch_count":
                row["w4f16_expand_mismatch_count"],
        }
    if correctness["control"]["output_hash"] != \
            correctness["candidate"]["output_hash"]:
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    repeat_results = {}
    speed_cells = []
    physical_cells = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        values = metrics(sides["control"], sides["candidate"])
        repeat_results[f"repeat{repeat}"] = {"metrics": values}
        records[repeat] = sides
        physical_cells.append(physical_equal(
            sides["control"], sides["candidate"]))
        for field in ("gate_up_ticks", "host_wall_ns_per_block"):
            speed_cells.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_cells)
    physical_gate = all(physical_cells)
    return {
        "experiment": "EXP-0111",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "physical_equality_gate": physical_gate,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gate": speed_gate,
        "local_gate_pass": speed_gate and physical_gate,
        "repeat_results": repeat_results,
        "pc028": baseline_pc028(
            exp0109_dir, records[10]["control"]),
        "pc028_provenance": {
            "f16f16_w4u8": str(exp0109_dir),
            "w4f16": str(result_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["candidate"]),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              metric_map: dict) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        value = metric_map[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, value['control'])} | "
            f"{base.format_value(field, value['candidate'])} | "
            f"{fmt_change(value['change_percent'])} | "
            f"{fmt_change(value['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    accepted = summary["pc028"]["w4f16"]
    candidate = summary["repeat10_candidate_modules"]
    total_control = accepted["Complete block Host wall"]
    total_candidate = candidate["Complete block Host wall"]
    lines.extend([
        "## EXP-0111 repeat-ten W4F16 module wall-time", "",
        "| Module | EXP-0110 control | EXP-0111 candidate | Speed |",
        "|---|---:|---:|---:|",
    ])
    for name in accepted:
        if name == "Complete block Host wall":
            left = f"{accepted[name]:.1f} us"
            right = f"{candidate[name]:.1f} us"
        else:
            left = f"{accepted[name]:.1f} us ({100*accepted[name]/total_control:.1f}%)"
            right = f"{candidate[name]:.1f} us ({100*candidate[name]/total_candidate:.1f}%)"
        speed = (accepted[name] / candidate[name] - 1.0) * 100.0
        lines.append(f"| {name} | {left} | {right} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0111 — Complete profiling report", ""]
    exp107.add_pc028(lines, summary)
    lines.append(
        "PC-028 provenance: F16F16 and W4U8 use the frozen EXP-0109 "
        "formal evidence; W4F16 is this experiment's accepted EXP-0110 "
        "control cell."
    )
    lines.append("")
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(lines, "Primary targets", TARGETS, values)
        add_table(lines, "Additive Block Timing Ledger",
                  exp107.LEDGER, values)
        add_table(lines, "Overlapping HMX/HVX/DMA and waits",
                  tuple(dict.fromkeys((*exp107.OVERLAP,
                                       *W4F16_PIPELINE))), values)
        add_table(lines, "Traffic, commands and residency",
                  exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| EXP-0111 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.exp0109_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
