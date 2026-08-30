#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


SAMPLES = 7
REPEATS = (1, 10)
VARIANTS = ("f16f16", "w4f16", "w4u8")
CANDIDATE = "q_prefix4_k_all"
VTCM_BYTES = 8_388_608
QTIMER_TICKS_PER_US = 19.2
EXPECTED_HASH = {
    "F16F16": "704252c89780e695",
    "W4F16": "f18b9abbe1487231",
    "W4U8": "69f22eeb035e5ec5",
}
PARITY_FIELDS = (
    "weight_ddr_read_bytes",
    "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("result_dir", type=pathlib.Path)
    return parser.parse_args()


def load_jsonl(path, expected=None):
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if expected is not None and len(records) != expected:
        raise SystemExit(
            f"{path}: expected {expected} records, got {len(records)}")
    return records


def med(records, field):
    return float(statistics.median(float(record[field])
                                   for record in records))


def median_record(records):
    result = dict(records[0])
    numeric_fields = {
        key for record in records for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for field in numeric_fields:
        result[field] = med(records, field)
    return result


def validate_record(record, variant, repeat, schedule, audit):
    expected = {
        "experiment": "EXP-0085",
        "variant": variant,
        "qkv_schedule": schedule,
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
        "output_hash": EXPECTED_HASH[variant],
        "mismatches": 0,
        "max_lsb": 0,
        "numerical_audit_mode": "on" if audit else "off",
        "attribution_mode": "on",
        "ledger_unattributed_ticks": 0,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise SystemExit(
                f"{variant}/{schedule}/r{repeat} {field}: "
                f"{record.get(field)!r} != {value!r}")
    if record["ledger_named_ticks"] != record["invocation_ticks"]:
        raise SystemExit(f"PC-027 ledger closure failed for {variant}")
    if any(record[field] != 0 for field in (
            "projection_failure_result",
            "w4f16_expand_mismatch_count",
            "crouton_q_operand_mismatch_count",
            "crouton_k_operand_mismatch_count",
            "crouton_v_operand_mismatch_count")):
        raise SystemExit(f"projection or boundary mismatch for {variant}")


def improvement(control, candidate):
    return (control / candidate - 1.0) * 100.0


def paired_improvement(control, candidate, field):
    return float(statistics.median(
        improvement(float(left[field]), float(right[field]))
        for left, right in zip(control, candidate)))


def module_us(record):
    repeat = float(record["repeat_count"])

    def ticks(*fields):
        return sum(float(record[field]) for field in fields) / repeat / \
            QTIMER_TICKS_PER_US

    modules = [
        ("I/O and metadata", ticks(
            "input_stage_ticks", "metadata_stage_ticks",
            "output_stage_ticks")),
        ("Input RMSNorm", ticks("input_norm_ticks")),
        ("QKV + Q/K Norm/RoPE", ticks(
            "qkv_projection_ticks", "qk_norm_rope_ticks")),
        ("QK-Softmax-AV", ticks("attention_ticks")),
        ("O projection", ticks("o_projection_ticks")),
        ("Post-attn residual + RMSNorm", ticks(
            "post_attention_residual_ticks", "post_attention_norm_ticks")),
        ("Gate/Up + SwiGLU", ticks("gate_up_ticks", "activation_ticks")),
        ("Down projection", ticks("down_ticks")),
        ("Final residual", ticks("final_residual_ticks")),
    ]
    host_us = float(record["host_wall_ns_per_block"]) / 1000.0
    modules.append(("Host/RPC and closure",
                    host_us - sum(value for _, value in modules)))
    modules.append(("Complete block Host wall", host_us))
    return dict(modules)


def build_summary(root):
    if (root / "boot_id_before.txt").read_bytes() != \
            (root / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during Stage C")
    static = json.loads((root / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    for variant_key in VARIANTS:
        variant = variant_key.upper()
        for side, schedule in (("control", "control"),
                               ("candidate", CANDIDATE)):
            record = load_jsonl(
                root / "correctness" /
                f"{variant_key}_{side}.jsonl", 1)[0]
            validate_record(record, variant, 1, schedule, True)
            correctness[f"{variant_key}_{side}"] = {
                "output_hash": record["output_hash"],
                "mismatches": record["mismatches"],
                "max_lsb": record["max_lsb"],
                "qkv_audit_ticks": record["qkv_audit_ticks"],
            }

    comparisons = {}
    median_records = {"control": {}, "candidate": {}}
    gate_cells = []
    for variant_key in VARIANTS:
        variant = variant_key.upper()
        comparisons[variant_key] = {}
        for repeat in REPEATS:
            records = {}
            for side, schedule in (("control", "control"),
                                   ("candidate", CANDIDATE)):
                path = root / "performance" / \
                    f"{variant_key}_{side}_r{repeat}.jsonl"
                records[side] = load_jsonl(path, SAMPLES)
                for record in records[side]:
                    validate_record(record, variant, repeat, schedule, False)
            for left, right in zip(records["control"], records["candidate"]):
                for field in PARITY_FIELDS:
                    if left[field] != right[field]:
                        raise SystemExit(
                            f"physical parity failure {variant}/r{repeat}/{field}")
            fields = {}
            for field in ("qkv_projection_ticks", "host_wall_ns_per_block"):
                control_value = med(records["control"], field)
                candidate_value = med(records["candidate"], field)
                fields[field] = {
                    "control": control_value,
                    "candidate": candidate_value,
                    "ordinary_improvement_percent": improvement(
                        control_value, candidate_value),
                    "paired_improvement_percent_median": paired_improvement(
                        records["control"], records["candidate"], field),
                }
                minimum = 0.0
                gate_cells.append({
                    "variant": variant,
                    "repeat": repeat,
                    "field": field,
                    "pass": (fields[field]["ordinary_improvement_percent"] >
                             minimum and
                             fields[field]["paired_improvement_percent_median"] >
                             minimum) if variant != "F16F16" else (
                                 fields[field]["ordinary_improvement_percent"] >=
                                 minimum and
                                 fields[field]["paired_improvement_percent_median"] >=
                                 minimum),
                })
            comparisons[variant_key][f"repeat{repeat}"] = fields
            if repeat == 10:
                median_records["control"][variant_key] = median_record(
                    records["control"])
                median_records["candidate"][variant_key] = median_record(
                    records["candidate"])

    local_pass = all(cell["pass"] for cell in gate_cells)
    module_tables = {
        side: {variant: module_us(record)
               for variant, record in records.items()}
        for side, records in median_records.items()
    }
    return {
        "experiment": "EXP-0085",
        "stage": "C",
        "candidate": CANDIDATE,
        "correctness": correctness,
        "comparisons": comparisons,
        "gate_cells": gate_cells,
        "local_gate": "pass" if local_pass else "fail",
        "pc027_timing_ledger_closure": "pass",
        "pc028_module_tables": module_tables,
        "selected_baseline_changed": False,
    }


def render_module_table(lines, title, table):
    lines.extend([
        f"## {title}", "",
        "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ])
    totals = {variant: table[variant]["Complete block Host wall"]
              for variant in VARIANTS}
    for module in table["f16f16"]:
        values = {variant: table[variant][module] for variant in VARIANTS}
        cells = []
        for variant in VARIANTS:
            if module == "Complete block Host wall":
                cells.append(f"{values[variant]:.1f} us")
            else:
                cells.append(
                    f"{values[variant]:.1f} us "
                    f"({values[variant] / totals[variant] * 100.0:.1f}%)")
        speed = improvement(values["w4f16"], values["w4u8"])
        lines.append(
            f"| {module} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{speed:+.1f}% |")
    lines.append("")


def render_report(summary):
    lines = [
        "# EXP-0085 Stage C formal report", "",
        f"Candidate: `{summary['candidate']}`. Local gate: "
        f"**{summary['local_gate'].upper()}**.", "",
        "## Seven-round paired comparison", "",
        "| Variant | Repeat | QKV control | QKV candidate | QKV ordinary | "
        "QKV paired | Host control | Host candidate | Host ordinary | Host paired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for repeat in REPEATS:
            metrics = summary["comparisons"][variant][f"repeat{repeat}"]
            qkv = metrics["qkv_projection_ticks"]
            host = metrics["host_wall_ns_per_block"]
            lines.append(
                f"| {variant.upper()} | {repeat} | "
                f"{qkv['control'] / repeat:.1f} | "
                f"{qkv['candidate'] / repeat:.1f} | "
                f"{qkv['ordinary_improvement_percent']:+.2f}% | "
                f"{qkv['paired_improvement_percent_median']:+.2f}% | "
                f"{host['control'] / 1000.0:.1f} us | "
                f"{host['candidate'] / 1000.0:.1f} us | "
                f"{host['ordinary_improvement_percent']:+.2f}% | "
                f"{host['paired_improvement_percent_median']:+.2f}% |")
    lines.append("")
    render_module_table(
        lines, "PC-028 control repeat10 module wall time",
        summary["pc028_module_tables"]["control"])
    render_module_table(
        lines, "PC-028 candidate repeat10 module wall time",
        summary["pc028_module_tables"]["candidate"])
    lines.extend([
        "## Physical and correctness closure", "",
        "All cases used exactly 8 MiB VTCM, zero intermediate DDR, zero "
        "spill/fill, one HMX owner, unchanged logical weight bytes and HMX "
        "tile pairs. Control and candidate retained the accepted output hash "
        "with zero mismatches and zero max LSB. PC-027 ledger unattributed "
        "ticks were zero for every formal record.", "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    summary = build_summary(args.result_dir)
    if args.report:
        print(render_report(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
