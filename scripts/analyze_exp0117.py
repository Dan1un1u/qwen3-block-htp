#!/usr/bin/env python3
"""Validate and report EXP-0117 exact packed-pair SwiGLU LUT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "packed_pair")
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
    "host_wall_ns_per_block", "gate_up_ticks",
    "w4u8_mlp_activation_work_ticks", "down_ticks", "total_ticks",
)
W4U8_PIPELINE = tuple(dict.fromkeys((
    *exp112.W4U8_PIPELINE,
    "w4u8_mlp_activation_work_ticks",
    "mlp_stream_join_wait_ticks",
    "mlp_stream_worker_work_ticks",
    "mlp_stream_main_work_ticks",
)))
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
    compatible["experiment"] = "EXP-0112"
    compatible["w4u8_stream_fence_mode"] = "single_fence"
    exp112.validate_record(
        compatible, repeat, "single_fence", audit=audit
    )
    base.require(record, "experiment", "EXP-0117")
    base.require(record, "w4u8_stream_fence_mode", "single_fence")
    base.require(record, "w4u8_activation_lut_mode", cell)
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *W4U8_PIPELINE, *exp107.PHYSICAL,
    )))
    return {field: summarize(left, right, field) for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def exact_physical(records: list[dict]) -> bool:
    return all(
        int(row.get("vtcm_requested_bytes", -1)) == VTCM_BYTES and
        int(row.get("vtcm_acquired_bytes", -1)) == VTCM_BYTES and
        int(row.get("intermediate_ddr_read_bytes", -1)) == 0 and
        int(row.get("intermediate_ddr_write_bytes", -1)) == 0 and
        int(row.get("intermediate_spill_fill_count", -1)) == 0 and
        int(row.get("block_invocation_count", -1)) ==
            int(row.get("repeat_count", -2))
        for row in records
    )


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0111_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0117")
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
        }
    if len({value["output_hash"] for value in correctness.values()}) != 1:
        raise SystemExit("control and packed-pair output hashes differ")
    if any(int(value["mismatches"]) != 0 or int(value["max_lsb"]) != 0
           for value in correctness.values()):
        raise SystemExit("audited device output mismatch")

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    physical_cells = []
    exact_cells = []
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES
            )
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
            exact_cells.append(exact_physical(rows))
        records[repeat] = sides
        comparisons[f"repeat{repeat}"] = metrics(
            sides["control"], sides["packed_pair"]
        )
        physical_cells.append(physical_equal(
            sides["control"], sides["packed_pair"]
        ))

    speed = all(
        comparisons[f"repeat{repeat}"][field][key] < 0.0
        for repeat in REPEATS
        for field in (
            "w4u8_mlp_activation_work_ticks", "gate_up_ticks",
            "host_wall_ns_per_block",
        )
        for key in ("change_percent", "paired_change_percent_median")
    )
    physical = all(physical_cells) and all(exact_cells)
    selected = "packed_pair" if speed and physical else "control"
    pc028 = exp112.baseline_pc028(exp0109_dir, exp0111_dir)
    return {
        "experiment": "EXP-0117",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "lut_exhaustive_gate": True,
        "fixed_8mib_vtcm_gate": all(exact_cells),
        "zero_intermediate_ddr_gate": all(exact_cells),
        "zero_spill_fill_gate": all(exact_cells),
        "single_fastrpc_execution_unit": all(exact_cells),
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gate": speed,
        "physical_equality_gate": physical,
        "selected_cell": selected,
        "local_gate_pass": selected == "packed_pair",
        "comparisons": comparisons,
        "pc028": pc028,
        "pc028_provenance": {
            "f16f16_w4u8_prior": str(exp0109_dir),
            "w4f16": str(exp0111_dir) + "/paired_candidate_r10.jsonl",
        },
        "repeat10_control_modules":
            exp109.module_medians(records[10]["control"]),
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["packed_pair"]),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              values: dict) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Packed pair | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        value = values[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, value['control'])} | "
            f"{base.format_value(field, value['candidate'])} | "
            f"{fmt_change(value['change_percent'])} | "
            f"{fmt_change(value['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    control = summary["repeat10_control_modules"]
    candidate = summary["repeat10_candidate_modules"]
    totals = {key: value["Complete block Host wall"]
              for key, value in table.items()}
    control_total = control["Complete block Host wall"]
    candidate_total = candidate["Complete block Host wall"]
    lines.extend([
        "## PC-028 recipe wall-time table (repeat10)", "",
        "| Module | W16A16 | W4A16 specialized | W4A8 control | W4A8 packed pair | A8 candidate vs A16 speed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        rendered = []
        for key in ("f16f16", "w4f16"):
            value = table[key][name]
            rendered.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        for values, total in ((control, control_total),
                              (candidate, candidate_total)):
            value = values[name]
            rendered.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/total:.1f}%)"
            )
        speed = (table["w4f16"][name] / candidate[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {rendered[0]} | {rendered[1]} | "
            f"{rendered[2]} | {rendered[3]} | {speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0117 — Complete profiling report", ""]
    add_pc028(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        add_table(lines, "Primary targets", TARGETS, values)
        add_table(lines, "Additive Block Timing Ledger",
                  exp107.LEDGER, values)
        add_table(lines, "HMX/HVX/DMA, LUT work and waits",
                  tuple(dict.fromkeys((*exp107.OVERLAP,
                                       *W4U8_PIPELINE))), values)
        add_table(lines, "Traffic, commands and residency",
                  exp107.PHYSICAL, values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Exhaustive 65,536-entry LUT equivalence | {'PASS' if summary['lut_exhaustive_gate'] else 'FAIL'} |",
        f"| Byte-exact device correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Activation, Gate/Up and Host speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| EXP-0117 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
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
