#!/usr/bin/env python3
"""Validate and report EXP-0127 projection-selective QKV batch four."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0110 as exp110
import analyze_exp0111 as exp111
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "v_batch4", "kv_batch4")
CANDIDATES = CELLS[1:]
SCHEDULE_NAMES = {
    "control": "control",
    "v_batch4": "v_batch4",
    "kv_batch4": "kv_batch4",
}
EXPECTED_HMX_COMMANDS = {
    "control": 208,
    "v_batch4": 200,
    "kv_batch4": 192,
}
EXPECTED_STREAMED_COMMANDS = {
    "control": 176,
    "v_batch4": 168,
    "kv_batch4": 160,
}
VTCM_BYTES = 8_388_608
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0124_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0124/"
    "20260831T231805Z_95fadff5930d_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "gate_up_ticks", "down_ticks", "total_ticks",
)
QKV_CONTRACT = (
    "hmx_command_count", "w4f16_streamed_command_count",
    "hmx_fp16_tile_pair_count", "weight_ddr_read_bytes",
    "weight_dma_descriptor_count", "crouton_qkv_projection_count",
    "crouton_qkv_unpack_skipped", "crouton_qk_operand_count",
    "crouton_av_weight_count", "qkv_operand_audit_tensor_count",
    "crouton_q_operand_mismatch_count",
    "crouton_k_operand_mismatch_count",
    "crouton_v_operand_mismatch_count",
)
PHYSICAL_EQUAL_FIELDS = tuple(
    field for field in exp111.PHYSICAL_EQUAL_FIELDS
    if field != "hmx_command_count"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0124-formal", type=Path,
                        default=EXP0124_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    compatible = dict(record)
    compatible["experiment"] = "EXP-0110"
    compatible["qkv_schedule_mode"] = "control"
    exp110.validate_record(compatible, repeat, "carrier", audit=audit)
    base.require(record, "experiment", "EXP-0127")
    base.require(record, "w4f16_group_fence_mode", "join_only")
    base.require(
        record, "qkv_schedule_mode", SCHEDULE_NAMES[cell],
    )
    expected_hmx = EXPECTED_HMX_COMMANDS[cell] * repeat
    expected_streamed = EXPECTED_STREAMED_COMMANDS[cell] * repeat
    if int(record.get("hmx_command_count", -1)) != expected_hmx:
        raise SystemExit(f"{cell}: HMX command count mismatch")
    if int(record.get("w4f16_streamed_command_count", -1)) != \
            expected_streamed:
        raise SystemExit(f"{cell}: streamed command count mismatch")
    if int(record.get("hmx_fp16_tile_pair_count", -1)) != 98_816 * repeat:
        raise SystemExit(f"{cell}: HMX tile work changed")
    if int(record.get("block_invocation_count", -1)) != repeat:
        raise SystemExit(f"{cell}: FastRPC execution-unit count changed")


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
        (r / l - 1.0) * 100.0 for l, r in zip(control, candidate)
        if l != 0.0
    ]
    return {
        "control": left_median,
        "candidate": right_median,
        "change_percent": (right_median / left_median - 1.0) * 100.0,
        "paired_change_percent_median": float(statistics.median(paired)),
        "paired_change_percent_min": min(paired),
        "paired_change_percent_max": max(paired),
    }


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *(field for field in TARGETS
          if field != "qkv_plus_qk_norm_rope_ticks"),
        *exp107.LEDGER, *exp107.OVERLAP, *exp111.W4F16_PIPELINE,
        *exp107.PHYSICAL, *QKV_CONTRACT,
    )))
    result = {field: summarize(left, right, field) for field in fields}
    result["qkv_plus_qk_norm_rope_ticks"] = summarize_qkv(left, right)
    return result


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        summarize(left, right, field)["control"] ==
        summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def pc028(exp0109_dir: Path, exp0124_dir: Path,
          current_w4f16: list[dict]) -> dict:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(current_w4f16),
        "w4u8": exp109.module_medians(load(
            exp0124_dir / "paired_control_r10.jsonl", SAMPLES)),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0124_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0127")
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
    control_hash = correctness["control"]["output_hash"]
    if any(correctness[cell]["output_hash"] != control_hash
           for cell in CANDIDATES):
        raise SystemExit("control/candidate output hashes differ")

    records: dict[int, dict[str, list[dict]]] = {}
    repeat_results = {}
    speed_values: dict[str, list[float]] = {
        candidate: [] for candidate in CANDIDATES
    }
    physical_gates = {candidate: True for candidate in CANDIDATES}
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        repeat_results[f"repeat{repeat}"] = {}
        for candidate in CANDIDATES:
            values = metrics(sides["control"], sides[candidate])
            repeat_results[f"repeat{repeat}"][candidate] = {
                "metrics": values
            }
            physical_gates[candidate] = (
                physical_gates[candidate] and
                physical_equal(sides["control"], sides[candidate])
            )
            for field in (
                "host_wall_ns_per_block", "qkv_plus_qk_norm_rope_ticks"
            ):
                speed_values[candidate].extend((
                    values[field]["change_percent"],
                    values[field]["paired_change_percent_median"],
                ))

    speed_gates = {
        candidate: all(value is not None and value < 0.0
                       for value in speed_values[candidate])
        for candidate in CANDIDATES
    }
    eligible = [candidate for candidate in CANDIDATES
                if speed_gates[candidate] and physical_gates[candidate]]
    selected = min(
        eligible,
        key=lambda cell: statistics.median(
            float(row["host_wall_ns_per_block"])
            for row in records[10][cell]
        ),
        default="control",
    )
    module_table = pc028(exp0109_dir, exp0124_dir,
                         records[10][selected])
    return {
        "experiment": "EXP-0127",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": True,
        "independent_qkv_operand_gate": True,
        "physical_equality_gates": physical_gates,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "speed_gates": speed_gates,
        "local_gate_pass": selected != "control",
        "selected_candidate": selected,
        "repeat_results": repeat_results,
        "pc028": module_table,
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(result_dir) +
                      f"/paired_{selected}_r10.jsonl",
            "w4u8": str(exp0124_dir) + "/paired_control_r10.jsonl",
        },
        "repeat10_control_modules":
            exp109.module_medians(records[10]["control"]),
        "repeat10_modules": {
            cell: exp109.module_medians(records[10][cell])
            for cell in CELLS
        },
    }


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def add_pc028(lines: list[str], summary: dict) -> None:
    table = summary["pc028"]
    totals = {key: row["Complete block Host wall"]
              for key, row in table.items()}
    lines.extend([
        "## PC-028 three-recipe overview (repeat10)", "",
        "| Module | W16A16 | W4A16 candidate | W4A8 latest eligible | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in table["f16f16"]:
        rendered = []
        for key in ("f16f16", "w4f16", "w4u8"):
            value = table[key][name]
            rendered.append(
                f"{value:.1f} us" if name == "Complete block Host wall"
                else f"{value:.1f} us ({100*value/totals[key]:.1f}%)"
            )
        speed = (table["w4f16"][name] / table["w4u8"][name] - 1.0) * 100.0
        lines.append(f"| {name} | " + " | ".join(rendered) +
                     f" | {speed:+.1f}% |")
    lines.extend(["", f"Provenance: `{summary['pc028_provenance']}`.", ""])


def add_table(lines: list[str], title: str, fields: tuple[str, ...],
              values: dict) -> None:
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


def add_modules(lines: list[str], summary: dict) -> None:
    control = summary["repeat10_control_modules"]
    left_total = control["Complete block Host wall"]
    modules = summary["repeat10_modules"]
    lines.extend([
        "## EXP-0127 repeat-ten W4F16 module wall-time", "",
        "| Module | Batch2 control | V batch4 | Speed | K+V batch4 | Speed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in control:
        left = (f"{control[name]:.1f} us" if name == "Complete block Host wall"
                else f"{control[name]:.1f} us ({100*control[name]/left_total:.1f}%)")
        rendered = []
        for cell in CANDIDATES:
            candidate = modules[cell]
            right_total = candidate["Complete block Host wall"]
            right = (f"{candidate[name]:.1f} us"
                     if name == "Complete block Host wall" else
                     f"{candidate[name]:.1f} us ({100*candidate[name]/right_total:.1f}%)")
            speed = (control[name] / candidate[name] - 1.0) * 100.0
            rendered.extend((right, f"{speed:+.1f}%"))
        lines.append(f"| {name} | {left} | " + " | ".join(rendered) + " |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0127 — Complete profiling report", ""]
    add_pc028(lines, summary)
    add_modules(lines, summary)
    for repeat in REPEATS:
        lines.extend([f"## Repeat {repeat}", ""])
        for candidate in CANDIDATES:
            values = summary["repeat_results"][f"repeat{repeat}"][candidate]["metrics"]
            lines.extend([f"### Candidate `{candidate}`", ""])
            add_table(lines, "Primary targets", TARGETS, values)
            add_table(lines, "Additive Block Timing Ledger", exp107.LEDGER, values)
            add_table(lines, "Overlapping HMX/HVX/DMA and waits",
                      tuple(dict.fromkeys((*exp107.OVERLAP,
                                           *exp111.W4F16_PIPELINE))), values)
            add_table(lines, "Traffic, commands and residency",
                      tuple(dict.fromkeys((*exp107.PHYSICAL,
                                           *QKV_CONTRACT))), values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| V-only QKV and Host speed | {'PASS' if summary['speed_gates']['v_batch4'] else 'FAIL'} |",
        f"| K+V QKV and Host speed | {'PASS' if summary['speed_gates']['kv_batch4'] else 'FAIL'} |",
        f"| Byte-exact correctness | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Independent QKV operand audit | {'PASS' if summary['independent_qkv_operand_gate'] else 'FAIL'} |",
        f"| V-only physical contract | {'PASS' if summary['physical_equality_gates']['v_batch4'] else 'FAIL'} |",
        f"| K+V physical contract | {'PASS' if summary['physical_equality_gates']['kv_batch4'] else 'FAIL'} |",
        f"| EXP-0127 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.exp0109_formal,
                            args.exp0124_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
