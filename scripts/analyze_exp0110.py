#!/usr/bin/env python3
"""Validate and report the EXP-0110 W4F16 QKV factorial."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "carrier", "prefix4", "combined")
CANDIDATES = CELLS[1:]
EXPECTED_HASH = "f18b9abbe1487231"
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
PHYSICAL_EQUAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
)
SPECIAL_FIELDS = (
    "qkv_schedule_command_count", "crouton_qkv_projection_count",
    "crouton_qkv_unpack_skipped", "crouton_qk_operand_count",
    "crouton_av_weight_count", "qkv_operand_audit_tensor_count",
    "crouton_q_operand_mismatch_count",
    "crouton_k_operand_mismatch_count",
    "crouton_v_operand_mismatch_count",
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


def fnv_words(hash_value: int, words: tuple[int, ...]) -> int:
    for word in words:
        for byte in range(4):
            hash_value ^= (word >> (8 * byte)) & 0xff
            hash_value = (hash_value * 1_099_511_628_211) & ((1 << 64) - 1)
    return hash_value


def prefix4_commands() -> list[tuple[int, int, int]]:
    commands: list[tuple[int, int, int]] = []
    batch = 2
    for group in range(4):
        for local in range(4):
            commands.append((0, group * 8 + local * batch, batch))
    for group in range(8):
        for local in range(2):
            commands.append((1, group * 4 + local * batch, batch))
    for group in range(4, 8):
        for local in range(4):
            commands.append((0, group * 8 + local * batch, batch))
    for group in range(8):
        for local in range(2):
            commands.append((2, group * 4 + local * batch, batch))
    if len(commands) != 64:
        raise AssertionError("invalid EXP-0110 command schedule")
    return commands


def expected_trace_hash(repeat: int) -> str:
    value = 1_469_598_103_934_665_603
    for _ in range(repeat):
        for command in prefix4_commands():
            value = fnv_words(value, command)
    return f"{value:016x}"


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    expected = {
        "experiment": "EXP-0110", "variant": "W4F16",
        "repeat_count": repeat, "rpc_result": 0, "dsp_status": 3,
        "numerical_status": 1, "intermediate_residency": "VTCM",
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "output_hash": EXPECTED_HASH,
        "qkv_schedule_mode": (
            "q_prefix4_k_all" if cell in ("prefix4", "combined")
            else "control"
        ),
        "crouton_boundary_mode": (
            "qkv_norms" if cell in ("carrier", "combined") else "norms"
        ),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise SystemExit(
                f"{cell} repeat{repeat} {field}: "
                f"{record.get(field)!r} != {value!r}"
            )
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC/block invocation mismatch")
    if int(record.get("w4f16_hvx_workers_created", -1)) != 3 or \
            int(record.get("w4f16_hvx_workers_locked", -1)) != 3:
        raise SystemExit(f"{cell}: W4F16 HVX pool contract changed")
    prefix = cell in ("prefix4", "combined")
    carrier = cell in ("carrier", "combined")
    expected_commands = 64 * repeat if prefix else 0
    expected_trace = expected_trace_hash(repeat) if prefix else "0" * 16
    if int(record.get("qkv_schedule_command_count", -1)) != expected_commands:
        raise SystemExit(f"{cell}: schedule command count mismatch")
    if str(record.get("qkv_schedule_trace_hash")) != expected_trace:
        raise SystemExit(
            f"{cell}: schedule trace {record.get('qkv_schedule_trace_hash')} "
            f"!= {expected_trace}"
        )
    carrier_expected = {
        "crouton_qkv_projection_count": 3 * repeat if carrier else 0,
        "crouton_qkv_unpack_skipped": 128 * repeat if carrier else 0,
        "crouton_qk_operand_count": 24 * repeat if carrier else 0,
        "crouton_av_weight_count": 8 * repeat if carrier else 0,
    }
    for field, value in carrier_expected.items():
        if int(record.get(field, -1)) != value:
            raise SystemExit(f"{cell}: {field} mismatch")
    if audit:
        for field in (
            "mismatches", "max_lsb", "crouton_q_operand_mismatch_count",
            "crouton_k_operand_mismatch_count",
            "crouton_v_operand_mismatch_count",
        ):
            if int(record.get(field, -1)) != 0:
                raise SystemExit(f"{cell}: {field} is nonzero")
        if int(record.get("qkv_operand_audit_tensor_count", -1)) != 3:
            raise SystemExit(f"{cell}: independent QKV audit did not run")


def per_block(record: dict, field: str) -> float:
    return exp107.per_block(record, field)


def summarize(left: list[dict], right: list[dict], field: str) -> dict:
    return exp107.summarize(left, right, field)


def qkv_interval(record: dict) -> float:
    return (
        float(record["qkv_projection_ticks"]) +
        float(record["qk_norm_rope_ticks"])
    ) / int(record["repeat_count"])


def summarize_qkv(left: list[dict], right: list[dict]) -> dict:
    control = [qkv_interval(row) for row in left]
    candidate = [qkv_interval(row) for row in right]
    left_median = float(statistics.median(control))
    right_median = float(statistics.median(candidate))
    paired = [
        (r / l - 1.0) * 100.0
        for l, r in zip(control, candidate) if l != 0.0
    ]
    return {
        "control": left_median, "candidate": right_median,
        "change_percent": (right_median / left_median - 1.0) * 100.0,
        "paired_change_percent_median": float(statistics.median(paired)),
        "paired_change_percent_min": min(paired),
        "paired_change_percent_max": max(paired),
    }


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        "host_wall_ns_per_block", "qkv_projection_ticks",
        "qk_norm_rope_ticks", "attention_ticks", "total_ticks",
        *exp107.LEDGER, *exp107.OVERLAP, *exp107.PHYSICAL,
        *SPECIAL_FIELDS,
    )))
    result = {field: summarize(left, right, field) for field in fields}
    result["qkv_plus_qk_norm_rope_ticks"] = summarize_qkv(left, right)
    return result


def baseline_pc028(exp0109_dir: Path) -> dict[str, dict[str, float]]:
    mapping = {
        "f16f16": "frozen_f16f16",
        "w4f16": "fair_w4f16",
        "w4u8": "fastest_w4u8",
    }
    result = {}
    for recipe, cell in mapping.items():
        rows = exp109.load(exp0109_dir / f"paired_{cell}_r10.jsonl", 5)
        result[recipe] = exp109.module_medians(rows)
    return result


def build_summary(result_dir: Path, exp0109_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0110")
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
            "qkv_operand_audit_tensor_count":
                row["qkv_operand_audit_tensor_count"],
            "q_mismatch": row["crouton_q_operand_mismatch_count"],
            "k_mismatch": row["crouton_k_operand_mismatch_count"],
            "v_mismatch": row["crouton_v_operand_mismatch_count"],
        }

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    eligible = {cell: True for cell in CANDIDATES}
    physical_gate = True
    for repeat in REPEATS:
        records[repeat] = {}
        for cell in CELLS:
            rows = load(result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            records[repeat][cell] = rows
        repeat_comparisons = {}
        for cell in CANDIDATES:
            values = metrics(records[repeat]["control"], records[repeat][cell])
            repeat_comparisons[cell] = values
            physical_gate = physical_gate and physical_equal(
                records[repeat]["control"], records[repeat][cell]
            )
            for field in (
                "host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks"
            ):
                eligible[cell] = eligible[cell] and all(
                    values[field][key] < 0.0
                    for key in (
                        "change_percent", "paired_change_percent_median"
                    )
                )
        comparisons[f"repeat{repeat}"] = repeat_comparisons

    eligible_cells = [cell for cell in CANDIDATES if eligible[cell]]
    selected = None
    if eligible_cells:
        selected = min(
            eligible_cells,
            key=lambda cell: (
                statistics.median(
                    row["host_wall_ns_per_block"]
                    for row in records[10][cell]
                ),
                statistics.median(
                    row["host_wall_ns_per_block"]
                    for row in records[1][cell]
                ),
            ),
        )
    modules = {
        cell: exp107.modules(records[10][cell]) for cell in CELLS
    }
    return {
        "experiment": "EXP-0110",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness_gate": True,
        "independent_qkv_operand_gate": True,
        "schedule_trace_gate": True,
        "physical_equality_gate": physical_gate,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "correctness": correctness,
        "comparisons": comparisons,
        "eligible": eligible,
        "selected_candidate": selected,
        "local_gate_pass": selected is not None and physical_gate,
        "factorial_modules_repeat10": modules,
        "pc028": baseline_pc028(exp0109_dir),
        "pc028_provenance": str(exp0109_dir),
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {
        key: value["Complete block Host wall"] for key, value in table.items()
    }
    lines.extend([
        "## PC-028 frozen public baseline overview (repeat10)", "",
        "| Module | W16A16 | W4A16 | W4A8 fastest | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        cells = []
        for key in ("f16f16", "w4f16", "w4u8"):
            value = table[key][name]
            cells.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        speed = (table["w4f16"][name] / table["w4u8"][name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{speed:+.1f}% |"
        )
    lines.extend(["", f"Provenance: `{summary['pc028_provenance']}`.", ""])


def add_factorial(lines: list[str], summary: dict) -> None:
    lines.extend([
        "## W4F16 factorial headline", "",
        "| Repeat | Cell | Host wall | Host delta | Host paired | QKV+prep | QKV delta | QKV paired | Eligible |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for repeat in REPEATS:
        for cell in CANDIDATES:
            values = summary["comparisons"][f"repeat{repeat}"][cell]
            host = values["host_wall_ns_per_block"]
            qkv = values["qkv_plus_qk_norm_rope_ticks"]
            lines.append(
                f"| {repeat} | {cell} | {host['candidate']/1000.0:.3f} us | "
                f"{fmt_change(host['change_percent'])} | "
                f"{fmt_change(host['paired_change_percent_median'])} | "
                f"{qkv['candidate']/19.2:.3f} us | "
                f"{fmt_change(qkv['change_percent'])} | "
                f"{fmt_change(qkv['paired_change_percent_median'])} | "
                f"{'yes' if summary['eligible'][cell] else 'no'} |"
            )
    lines.append("")


def add_metric_table(lines: list[str], title: str,
                     fields: tuple[str, ...], values: dict) -> None:
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        metric = values[field]
        lines.append(
            f"| `{field}` | {base.format_value(field, metric['control'])} | "
            f"{base.format_value(field, metric['candidate'])} | "
            f"{fmt_change(metric['change_percent'])} | "
            f"{fmt_change(metric['paired_change_percent_median'])} |"
        )
    lines.append("")


def add_factorial_modules(lines: list[str], summary: dict) -> None:
    table = summary["factorial_modules_repeat10"]
    totals = {
        cell: values["Complete block Host wall"] for cell, values in table.items()
    }
    lines.extend([
        "## EXP-0110 repeat-ten module wall-time", "",
        "| Module | Control | Carrier | Prefix4 | Combined |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in table["control"]:
        rendered = []
        for cell in CELLS:
            value = table[cell][name]
            rendered.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[cell]:.1f}%)"
            )
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0110 — W4F16 QKV specialization factorial", ""]
    add_pc028(lines, summary)
    add_factorial(lines, summary)
    add_factorial_modules(lines, summary)
    for repeat in REPEATS:
        lines.extend([f"## Repeat {repeat} complete profiling", ""])
        for cell in CANDIDATES:
            values = summary["comparisons"][f"repeat{repeat}"][cell]
            add_metric_table(
                lines, f"{cell}: headline",
                ("host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks",
                 "qkv_projection_ticks", "qk_norm_rope_ticks",
                 "attention_ticks", "total_ticks"), values,
            )
            add_metric_table(lines, f"{cell}: additive ledger",
                             exp107.LEDGER, values)
            add_metric_table(lines, f"{cell}: engines and waits",
                             exp107.OVERLAP, values)
            add_metric_table(
                lines, f"{cell}: traffic, residency, and QKV contract",
                tuple(dict.fromkeys((*exp107.PHYSICAL, *SPECIAL_FIELDS))),
                values,
            )
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Independent QKV operand audit | {'PASS' if summary['independent_qkv_operand_gate'] else 'FAIL'} |",
        f"| Exact Q-prefix trace | {'PASS' if summary['schedule_trace_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| Local speed gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        f"| Selected candidate | {summary['selected_candidate'] or 'none'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.exp0109_formal)
    if args.report:
        print(render_report(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
