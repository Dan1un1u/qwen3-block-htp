#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


SAMPLES = 7
REPEATS = (1, 10)
FP16_VARIANTS = ("f16f16", "w4f16")
STAGES = ("stage_a", "stage_b", "stage_c", "candidate")
VTCM_BYTES = 8_388_608
QTIMER_TICKS_PER_US = 19.2
EXPECTED_HASHES = {
    "F16F16": "704252c89780e695",
    "W4F16": "f18b9abbe1487231",
    "W4U8": "69f22eeb035e5ec5",
}
LEDGER = (
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks",
    "stage_boundary_ticks", "ledger_named_ticks",
    "ledger_unattributed_ticks", "invocation_ticks", "total_ticks",
)
OVERLAP = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "hmx_ready_wait_ticks", "w4f16_expand_ticks",
    "w4f16_expand_work_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_prefetch_wait_ticks", "f16f16_prefetch_wait_ticks",
    "attention_qk_norm_main_work_ticks",
    "attention_qk_norm_worker_work_ticks",
    "attention_qk_norm_pool_wait_ticks",
    "attention_softmax_main_work_ticks",
    "attention_softmax_worker_work_ticks",
    "attention_softmax_pool_wait_ticks",
    "attention_gqa_worker_work_ticks", "attention_gqa_hmx_wait_ticks",
    "attention_gqa_queue_wait_ticks", "mlp_silu_main_work_ticks",
    "mlp_silu_worker_work_ticks", "mlp_silu_pool_wait_ticks",
    "mlp_stream_worker_work_ticks", "mlp_stream_main_work_ticks",
    "mlp_stream_ready_wait_ticks", "mlp_stream_join_wait_ticks",
    "fp16_input_norm_main_work_ticks",
    "fp16_input_norm_worker_work_ticks",
    "fp16_input_norm_pool_wait_ticks",
    "fp16_post_residual_norm_main_work_ticks",
    "fp16_post_residual_norm_worker_work_ticks",
    "fp16_post_residual_norm_pool_wait_ticks",
)
PHYSICAL = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "attention_qk_norm_task_count", "fp16_qk_norm_pair_task_count",
    "fp16_input_norm_task_count", "fp16_input_norm_active_contexts",
    "fp16_post_residual_norm_task_count",
    "fp16_post_residual_norm_active_contexts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, expected: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if expected is not None and len(records) != expected:
        raise SystemExit(
            f"{path}: expected {expected} records, got {len(records)}")
    return records


def med(records: Iterable[dict], field: str) -> float:
    return float(statistics.median(float(record[field])
                                   for record in records))


def change(control: float, candidate: float) -> float | None:
    return None if control == 0.0 else (candidate / control - 1.0) * 100.0


def paired_change(control: list[dict], candidate: list[dict],
                  field: str) -> float | None:
    values = []
    for left, right in zip(control, candidate):
        base = float(left[field])
        if base != 0.0:
            values.append((float(right[field]) / base - 1.0) * 100.0)
    return None if not values else float(statistics.median(values))


def validate_record(record: dict, repeat: int,
                    expected_variant: str) -> None:
    fixed = {
        "experiment": "EXP-0084",
        "variant": expected_variant,
        "repeat_count": repeat,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "intermediate_residency": "VTCM",
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "output_hash": EXPECTED_HASHES[expected_variant],
    }
    for field, expected in fixed.items():
        if record.get(field) != expected:
            raise SystemExit(
                f"{expected_variant} {field}: {record.get(field)!r} != {expected!r}")
    if int(record.get("mismatches", -1)) != 0 or \
            int(record.get("max_lsb", -1)) != 0:
        raise SystemExit(f"{expected_variant}: output mismatch")


def stage_records(result_dir: Path, stage: str, variant: str,
                  repeat: int, side: str) -> list[dict]:
    suffix = "control" if side == "control" else stage
    records = load_jsonl(
        result_dir / "stages" / stage /
        f"{variant}_{suffix}_r{repeat}.jsonl", SAMPLES)
    expected = variant.upper()
    for record in records:
        validate_record(record, repeat, expected)
    return records


def metrics_for(control: list[dict], candidate: list[dict],
                fields: Iterable[str]) -> dict[str, dict[str, float | None]]:
    result = {}
    for field in fields:
        left = med(control, field)
        right = med(candidate, field)
        result[field] = {
            "control": left,
            "candidate": right,
            "change_percent": change(left, right),
            "paired_change_percent_median": paired_change(
                control, candidate, field),
        }
    return result


def stage_summary(result_dir: Path, stage: str) -> dict:
    fields = tuple(dict.fromkeys((
        "host_wall_ns_per_block", *LEDGER, *OVERLAP, *PHYSICAL,
    )))
    variants = {}
    gate_cells = []
    for variant in FP16_VARIANTS:
        repeat_results = {}
        for repeat in REPEATS:
            control = stage_records(
                result_dir, stage, variant, repeat, "control")
            candidate = stage_records(
                result_dir, stage, variant, repeat, "candidate")
            metrics = metrics_for(control, candidate, fields)
            repeat_results[f"repeat{repeat}"] = {"metrics": metrics}
            gate_cells.extend((
                metrics["host_wall_ns_per_block"]["change_percent"],
                metrics["host_wall_ns_per_block"]
                       ["paired_change_percent_median"],
            ))
        variants[variant] = {"repeat_results": repeat_results}
    return {
        "variants": variants,
        "host_wall_non_regression_gate": all(
            value is not None and value <= 0.0 for value in gate_cells),
    }


def module_us(record: dict) -> list[tuple[str, float]]:
    def ticks(*fields: str) -> float:
        return (
            sum(float(record[field]) for field in fields)
            / QTIMER_TICKS_PER_US
        )

    modules = [
        ("I/O and metadata",
         ticks("input_stage_ticks", "metadata_stage_ticks",
               "output_stage_ticks")),
        ("Input RMSNorm", ticks("input_norm_ticks")),
        ("QKV + Q/K Norm/RoPE",
         ticks("qkv_projection_ticks", "qk_norm_rope_ticks")),
        ("QK-Softmax-AV", ticks("attention_ticks")),
        ("O projection", ticks("o_projection_ticks")),
        ("Post-attn residual + RMSNorm",
         ticks("post_attention_residual_ticks",
               "post_attention_norm_ticks")),
        ("Gate/Up + SwiGLU",
         ticks("gate_up_ticks", "activation_ticks")),
        ("Down projection", ticks("down_ticks")),
        ("Final residual", ticks("final_residual_ticks")),
    ]
    host_us = float(record["host_wall_ns_per_block"]) / 1000.0
    modules.append(("Host/RPC and closure",
                    host_us - sum(value for _, value in modules)))
    modules.append(("Complete block Host wall", host_us))
    return modules


def median_record(records: list[dict]) -> dict:
    result = dict(records[0])
    numeric = {
        key for record in records for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for field in numeric:
        result[field] = med(records, field)
    return result


def build_summary(result_dir: Path) -> dict:
    if (result_dir / "boot_id_before.txt").read_bytes() != \
            (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during formal collection")
    static_gate = json.loads((result_dir / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")
    selection = json.loads((result_dir / "grid_selection.json").read_text())

    correctness = {}
    for variant in ("F16F16", "W4F16"):
        for mode in ("control", "stage_a", "stage_b", "stage_c", "candidate"):
            path = (
                result_dir
                / "correctness"
                / f"{variant.lower()}_{mode}.jsonl"
            )
            record = load_jsonl(path, 1)[0]
            validate_record(record, 1, variant)
            correctness[f"{variant.lower()}_{mode}"] = {
                "output_hash": record["output_hash"],
                "mismatches": record["mismatches"],
                "max_lsb": record["max_lsb"],
            }
    w4u8_correctness = load_jsonl(
        result_dir / "correctness" / "w4u8_control.jsonl", 1)[0]
    validate_record(w4u8_correctness, 1, "W4U8")
    correctness["w4u8_control"] = {
        "output_hash": w4u8_correctness["output_hash"],
        "mismatches": w4u8_correctness["mismatches"],
        "max_lsb": w4u8_correctness["max_lsb"],
    }

    stages = {stage: stage_summary(result_dir, stage) for stage in STAGES}
    w4u8 = {}
    for repeat in REPEATS:
        records = load_jsonl(
            result_dir / "w4u8" / f"w4u8_control_r{repeat}.jsonl",
            SAMPLES)
        for record in records:
            validate_record(record, repeat, "W4U8")
        w4u8[f"repeat{repeat}"] = {
            "median_record": median_record(records),
        }

    final_records = {}
    for variant in FP16_VARIANTS:
        records = stage_records(
            result_dir, "candidate", variant, 10, "candidate")
        final_records[variant] = median_record(records)
    final_records["w4u8"] = w4u8["repeat10"]["median_record"]

    local_gate = (
        stages["stage_b"]["host_wall_non_regression_gate"] and
        stages["stage_c"]["host_wall_non_regression_gate"] and
        stages["candidate"]["host_wall_non_regression_gate"]
    )
    return {
        "experiment": "EXP-0084",
        "selection": selection,
        "correctness": correctness,
        "stages": stages,
        "w4u8": w4u8,
        "final_median_records": final_records,
        "stage_a_gate": stages["stage_a"]["host_wall_non_regression_gate"],
        "stage_b_gate": stages["stage_b"]["host_wall_non_regression_gate"],
        "stage_c_gate": stages["stage_c"]["host_wall_non_regression_gate"],
        "combined_gate": stages["candidate"]["host_wall_non_regression_gate"],
        "local_gate_pass": local_gate,
        "selected_baseline_changed": False,
    }


def fmt_value(field: str, value: float) -> str:
    if field == "host_wall_ns_per_block":
        return f"{value / 1000.0:.3f} us"
    if field.endswith("_bytes"):
        return f"{value:.0f} B"
    if field.endswith("_count") or field.endswith("_contexts"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def fmt_change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}%"


def add_direct_table(lines: list[str], summary: dict,
                     stage: str, variant: str, repeat: int,
                     fields: Iterable[str], title: str) -> None:
    metrics = summary["stages"][stage]["variants"][variant][
        "repeat_results"][f"repeat{repeat}"]["metrics"]
    lines.extend([
        f"### {title}", "",
        "| Metric | Control | Candidate | Delta | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in fields:
        metric = metrics[field]
        lines.append(
            f"| `{field}` | {fmt_value(field, metric['control'])} | "
            f"{fmt_value(field, metric['candidate'])} | "
            f"{fmt_change(metric['change_percent'])} | "
            f"{fmt_change(metric['paired_change_percent_median'])} |")
    lines.append("")


def build_report(summary: dict, result_dir: Path) -> str:
    lines = [
        "# EXP-0084 full profiling report", "",
        "This report compares all three recipes from one source commit and "
        "one binary. F16F16 and W4F16 use the same bounded norm scheduling "
        "grid and selection rule; W4U8 remains byte-identical to EXP-0079.",
        "",
        "## PC-028 three-variant repeat10 module wall time", "",
        "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    records = summary["final_median_records"]
    modules = {name: dict(module_us(records[name]))
               for name in ("f16f16", "w4f16", "w4u8")}
    totals = {name: modules[name]["Complete block Host wall"]
              for name in modules}
    for module in modules["f16f16"]:
        values = {name: modules[name][module] for name in modules}
        if module == "Complete block Host wall":
            cells = [f"{values[name]:.1f} us" for name in modules]
        else:
            cells = [f"{values[name]:.1f} us "
                     f"({100.0 * values[name] / totals[name]:.1f}%)"
                     for name in modules]
        speed = values["w4f16"] / values["w4u8"] - 1.0 \
            if values["w4u8"] != 0.0 else 0.0
        lines.append(
            f"| {module} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{speed * 100.0:+.1f}% |")
    lines.extend(["", "## Stage gates", "",
                  "| Stage | Result |", "|---|---:|"])
    for stage, label in (("stage_a", "Q/K head pairs"),
                         ("stage_b", "Input RMSNorm pool"),
                         ("stage_c", "Post residual/RMSNorm pool"),
                         ("candidate", "Accepted combined candidate")):
        gate = summary["stages"][stage]["host_wall_non_regression_gate"]
        lines.append(f"| {label} | {'PASS' if gate else 'FAIL'} |")
    lines.extend([
        "", "Grid selection: "
        f"rows/task={summary['selection']['selected_rows_per_task']}, "
        f"contexts={summary['selection']['selected_contexts']}. "
        f"Rule: {summary['selection']['selection_rule']}.", "",
    ])

    stage_fields = {
        "stage_a": (
            "host_wall_ns_per_block", "qkv_projection_ticks",
            "qk_norm_rope_ticks", "attention_qk_norm_task_count",
            "fp16_qk_norm_pair_task_count",
            "attention_qk_norm_main_work_ticks",
            "attention_qk_norm_worker_work_ticks",
            "attention_qk_norm_pool_wait_ticks",
        ),
        "stage_b": (
            "host_wall_ns_per_block", "input_norm_ticks",
            "qkv_projection_ticks", "fp16_input_norm_task_count",
            "fp16_input_norm_active_contexts",
            "fp16_input_norm_main_work_ticks",
            "fp16_input_norm_worker_work_ticks",
            "fp16_input_norm_pool_wait_ticks",
        ),
        "stage_c": (
            "host_wall_ns_per_block", "post_attention_residual_ticks",
            "post_attention_norm_ticks", "gate_up_ticks",
            "fp16_post_residual_norm_task_count",
            "fp16_post_residual_norm_active_contexts",
            "fp16_post_residual_norm_main_work_ticks",
            "fp16_post_residual_norm_worker_work_ticks",
            "fp16_post_residual_norm_pool_wait_ticks",
        ),
    }
    for stage in ("stage_a", "stage_b", "stage_c"):
        for variant in FP16_VARIANTS:
            for repeat in REPEATS:
                add_direct_table(
                    lines, summary, stage, variant, repeat,
                    stage_fields[stage],
                    f"{stage.upper()} direct evidence — "
                    f"{variant.upper()} repeat{repeat}")

    direct_fields = (
        "host_wall_ns_per_block", "input_norm_ticks",
        "qkv_projection_ticks", "post_attention_residual_ticks",
        "gate_up_ticks", "down_ticks",
    )
    for variant in FP16_VARIANTS:
        for repeat in REPEATS:
            add_direct_table(
                lines, summary, "candidate", variant, repeat,
                direct_fields,
                f"Combined control vs candidate — {variant.upper()} repeat{repeat}")

    for variant in FP16_VARIANTS:
        add_direct_table(
            lines, summary, "candidate", variant, 10, LEDGER,
            f"Complete additive Block Timing Ledger — {variant.upper()} repeat10")
        add_direct_table(
            lines, summary, "candidate", variant, 10, OVERLAP,
            f"Overlapping engine work and waits — {variant.upper()} repeat10")
        add_direct_table(
            lines, summary, "candidate", variant, 10, PHYSICAL,
            f"Physical, VTCM, command and residency counters — {variant.upper()} repeat10")

    lines.extend([
        "## Correctness and physical gates", "",
        "All correctness records have `rpc_result=0`, `dsp_status=3`, "
        "`numerical_status=1`, zero mismatches, zero max LSB, the expected "
        "variant output hash, exactly 8 MiB requested/acquired VTCM, zero "
        "intermediate DDR, and zero spill/fill.", "",
        "## Evidence provenance", "",
        f"Formal result directory: `{result_dir}`", "",
        "The device boot ID is unchanged across collection. Package manifests, "
        "binary hashes, build/deploy logs, the full 3x3 grid, seven paired "
        "rounds for every stage, and seven W4U8 rounds are retained alongside "
        "this report.", "",
        f"Overall local gate: {'PASS' if summary['local_gate_pass'] else 'FAIL'}. "
        "No Selected Baseline is changed automatically.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(args.result_dir)
    if args.report:
        print(build_report(summary, args.result_dir))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
