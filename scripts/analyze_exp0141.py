#!/usr/bin/env python3
"""Validate and report EXP-0141 W4U8 O-projection batch-eight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_exp0107 as exp107
import analyze_exp0109 as exp109
import analyze_exp0112 as exp112
import analyze_exp0124 as parent
import validate_exp0050 as base


REPEATS = (1, 10)
SAMPLES = 5
CELLS = ("control", "candidate")
VTCM_BYTES = 8_388_608
OUTPUT_HASH = "69f22eeb035e5ec5"
EXP0109_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0109/"
    "20260831T155519Z_42e2a3301292_formal"
)
EXP0110_FORMAL = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0110/"
    "20260831T163813Z_d18e901339f4_formal"
)
TARGETS = (
    "host_wall_ns_per_block", "o_projection_ticks", "gate_up_ticks",
    "down_ticks", "total_ticks",
)
COMMAND_FIELDS = (
    "weight_dma_descriptor_count", "hmx_command_count",
    "w4u8_qkvo_prefetch_count", "w4u8_qkvo_overlap_schedule_count",
    "w4u8_o_batch_count", "hmx_u8s8_tile_pair_count",
)
PHYSICAL_EQUAL_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "w4u8_mlp_pair_publish_count", "w4u8_mlp_pair_consume_count",
    "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_mlp_down_hmx_command_count",
)
AUDIT_HASH_FIELDS = (
    "u8_input_norm_actual_hash", "u8_attention_actual_score_hash",
    "u8_attention_actual_probability_hash",
    "u8_attention_actual_av_hash", "mlp_down_input_hash",
)
ZERO_AUDIT_FIELDS = (
    "u8_attention_fused_k_operand_mismatch_count",
    "u8_attention_probability_mask_violation_count",
    "crouton_q_operand_mismatch_count", "crouton_k_operand_mismatch_count",
    "crouton_v_operand_mismatch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--exp0109-formal", type=Path,
                        default=EXP0109_FORMAL)
    parser.add_argument("--exp0110-formal", type=Path,
                        default=EXP0110_FORMAL)
    return parser.parse_args()


def load(path: Path, count: int) -> list[dict]:
    return base.load_jsonl(path, count)


def validate_record(record: dict, repeat: int, cell: str,
                    audit: bool = False) -> None:
    batch = 4 if cell == "control" else 8
    fixed = {
        "experiment": "EXP-0141",
        "variant": "W4U8",
        "w4u8_stream_fence_mode": "single_fence",
        "w4u8_qkv_ring_expand_workers": 3,
        "w4u8_down_hmx_batch_outputs": 4,
        "w4u8_o_batch_n_tiles": batch,
        "w4u8_o_batch_n_tiles_observed": batch,
        "w4u8_o_batch_count": (16 if cell == "control" else 8) * repeat,
        "w4u8_qkv_ring_slot_count": 4,
        "w4u8_qkv_ring_expand_worker_count": 3,
        "w4u8_qkv_ring_prep_worker_count": 2,
        "w4u8_qkv_ring_dispatch_count": repeat,
        "w4u8_qkv_ring_batch_count": 32 * repeat,
        "w4u8_qkv_ring_expand_task_count": 128 * repeat,
        "w4u8_qkv_ring_hmx_dispatch_count": repeat,
        "w4u8_qkv_ring_head_publish_count": 24 * repeat,
        "block_invocation_count": repeat,
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
    }
    for field, value in fixed.items():
        base.require(record, field, value)
    if int(record.get("vtcm_peak_plan_bytes", 0)) <= 0 or \
            int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit(f"{cell}: invalid VTCM peak")
    if record.get("output_hash") != OUTPUT_HASH or \
            int(record.get("mismatches", -1)) != 0 or \
            int(record.get("max_lsb", -1)) != 0:
        raise SystemExit(f"{cell}: final output is not byte-exact")
    for field in ZERO_AUDIT_FIELDS:
        if int(record.get(field, -1)) != 0:
            raise SystemExit(f"{cell}: nonzero {field}")
    if audit:
        for field in AUDIT_HASH_FIELDS:
            if str(record.get(field, "0000000000000000")) == \
                    "0000000000000000":
                raise SystemExit(f"{cell}: missing audit hash {field}")


def metrics(left: list[dict], right: list[dict]) -> dict:
    fields = tuple(dict.fromkeys((
        *TARGETS, *exp107.LEDGER, *exp107.OVERLAP,
        *exp112.W4U8_PIPELINE, *parent.RING_FIELDS,
        *exp107.PHYSICAL, *COMMAND_FIELDS,
    )))
    return {field: parent.summarize(left, right, field)
            for field in fields}


def physical_equal(left: list[dict], right: list[dict]) -> bool:
    return all(
        parent.summarize(left, right, field)["control"] ==
        parent.summarize(left, right, field)["candidate"]
        for field in PHYSICAL_EQUAL_FIELDS
    )


def selected_pc028(exp0109_dir: Path, exp0110_dir: Path,
                   w4u8: list[dict]) -> dict[str, dict[str, float]]:
    return {
        "f16f16": exp109.module_medians(load(
            exp0109_dir / "paired_frozen_f16f16_r10.jsonl", SAMPLES)),
        "w4f16": exp109.module_medians(load(
            exp0110_dir / "paired_carrier_r10.jsonl", SAMPLES)),
        "w4u8": exp109.module_medians(w4u8),
    }


def build_summary(result_dir: Path, exp0109_dir: Path,
                  exp0110_dir: Path) -> dict:
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise SystemExit("device boot ID changed during EXP-0141")
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness = {}
    audit_hashes = {}
    for cell in CELLS:
        row = load(result_dir / f"correctness_{cell}.jsonl", 1)[0]
        validate_record(row, 1, cell, audit=True)
        correctness[cell] = {
            "output_hash": row["output_hash"],
            "mismatches": row["mismatches"],
            "max_lsb": row["max_lsb"],
            "o_batch_n_tiles": row["w4u8_o_batch_n_tiles_observed"],
            "o_batch_count": row["w4u8_o_batch_count"],
            "hmx_command_count": row["hmx_command_count"],
            "weight_dma_descriptor_count":
                row["weight_dma_descriptor_count"],
        }
        audit_hashes[cell] = tuple(row[field] for field in AUDIT_HASH_FIELDS)
    correctness_gate = (
        len({value["output_hash"] for value in correctness.values()}) == 1
        and audit_hashes["control"] == audit_hashes["candidate"])

    records: dict[int, dict[str, list[dict]]] = {}
    comparisons = {}
    speed_values = []
    preservation_values = []
    physical_values = []
    command_gate = True
    plan_gate = True
    for repeat in REPEATS:
        sides = {}
        for cell in CELLS:
            rows = load(
                result_dir / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for row in rows:
                validate_record(row, repeat, cell)
            sides[cell] = rows
        records[repeat] = sides
        values = metrics(sides["control"], sides["candidate"])
        comparisons[f"repeat{repeat}"] = values
        physical_values.append(physical_equal(
            sides["control"], sides["candidate"]))
        for control, candidate in zip(sides["control"], sides["candidate"]):
            command_gate = command_gate and (
                int(control["hmx_command_count"]) -
                    int(candidate["hmx_command_count"]) == 8 * repeat
                and int(control["weight_dma_descriptor_count"]) -
                    int(candidate["weight_dma_descriptor_count"]) ==
                    16 * repeat
                and int(control["w4u8_qkvo_prefetch_count"]) -
                    int(candidate["w4u8_qkvo_prefetch_count"]) == 8 * repeat
                and int(control["w4u8_qkvo_overlap_schedule_count"]) -
                    int(candidate["w4u8_qkvo_overlap_schedule_count"]) ==
                    8 * repeat)
            plan_gate = plan_gate and (
                int(control["vtcm_peak_plan_bytes"]) ==
                int(candidate["vtcm_peak_plan_bytes"]))
        for field in ("o_projection_ticks", "host_wall_ns_per_block"):
            speed_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))
        for field in ("gate_up_ticks", "down_ticks"):
            preservation_values.extend((
                values[field]["change_percent"],
                values[field]["paired_change_percent_median"],
            ))

    speed_gate = all(value is not None and value < 0.0
                     for value in speed_values)
    preservation_gate = all(
        value is not None and value <= 0.5
        for value in preservation_values)
    physical_gate = all(physical_values)
    local_gate = all((correctness_gate, physical_gate, command_gate,
                      plan_gate, speed_gate, preservation_gate))
    selected = "candidate" if local_gate else "control"
    return {
        "experiment": "EXP-0141",
        "source_commit":
            (result_dir / "source_commit.txt").read_text().strip(),
        "static_gate": static,
        "correctness": correctness,
        "correctness_gate": correctness_gate,
        "fixed_8mib_vtcm_gate": plan_gate,
        "physical_equality_gate": physical_gate,
        "command_reduction_gate": command_gate,
        "speed_gate": speed_gate,
        "downstream_preservation_gate": preservation_gate,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "local_gate_pass": local_gate,
        "selected_cell": selected,
        "comparisons": comparisons,
        "pc028": selected_pc028(
            exp0109_dir, exp0110_dir, records[10][selected]),
        "pc028_provenance": {
            "f16f16": str(exp0109_dir),
            "w4f16": str(exp0110_dir) + "/paired_carrier_r10.jsonl",
            "w4u8": str(result_dir) + f"/paired_{selected}_r10.jsonl",
        },
        "repeat10_control_modules":
            exp109.module_medians(records[10]["control"]),
        "repeat10_candidate_modules":
            exp109.module_medians(records[10]["candidate"]),
    }


def add_candidate_modules(lines: list[str], summary: dict) -> None:
    control = summary["repeat10_control_modules"]
    candidate = summary["repeat10_candidate_modules"]
    total_control = control["Complete block Host wall"]
    total_candidate = candidate["Complete block Host wall"]
    lines.extend([
        "## EXP-0141 W4U8 module wall-time (repeat10)", "",
        "| Module | O batch4 | O batch8 | Speed |",
        "|---|---:|---:|---:|",
    ])
    for name in control:
        if name == "Complete block Host wall":
            left = f"{control[name]:.1f} us"
            right = f"{candidate[name]:.1f} us"
        else:
            left = (f"{control[name]:.1f} us "
                    f"({100*control[name]/total_control:.1f}%)")
            right = (f"{candidate[name]:.1f} us "
                     f"({100*candidate[name]/total_candidate:.1f}%)")
        speed = (control[name] / candidate[name] - 1.0) * 100.0
        lines.append(f"| {name} | {left} | {right} | {speed:+.1f}% |")
    lines.append("")


def render_report(summary: dict) -> str:
    lines = ["# EXP-0141 — Complete profiling report", ""]
    exp107.add_pc028(lines, summary)
    lines.extend([
        "PC-028 provenance: F16F16 uses frozen EXP-0109, selected W4F16 "
        "uses EXP-0110, and W4U8 uses this experiment's locally eligible "
        "cell.", "",
    ])
    add_candidate_modules(lines, summary)
    for repeat in REPEATS:
        values = summary["comparisons"][f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        exp112.add_table(lines, "Primary targets", TARGETS, values)
        exp112.add_table(lines, "Additive Block Timing Ledger",
                         exp107.LEDGER, values)
        exp112.add_table(
            lines, "Overlapping HMX/HVX/DMA, ring and waits",
            tuple(dict.fromkeys((*exp107.OVERLAP, *exp112.W4U8_PIPELINE,
                                 *parent.RING_FIELDS))), values)
        exp112.add_table(lines, "Traffic, commands and residency",
                         tuple(dict.fromkeys((*exp107.PHYSICAL,
                                              *COMMAND_FIELDS))),
                         values)
    lines.extend([
        "## Gates", "", "| Gate | Result |", "|---|---:|",
        f"| Byte-exact correctness and audits | {'PASS' if summary['correctness_gate'] else 'FAIL'} |",
        f"| Physical equality | {'PASS' if summary['physical_equality_gate'] else 'FAIL'} |",
        f"| O command reduction | {'PASS' if summary['command_reduction_gate'] else 'FAIL'} |",
        f"| Exact 8 MiB grant and identical plan | {'PASS' if summary['fixed_8mib_vtcm_gate'] else 'FAIL'} |",
        f"| O and Host strict speed | {'PASS' if summary['speed_gate'] else 'FAIL'} |",
        f"| Gate/Up and Down preservation | {'PASS' if summary['downstream_preservation_gate'] else 'FAIL'} |",
        f"| EXP-0141 local gate | {'PASS' if summary['local_gate_pass'] else 'FAIL'} |",
        "", f"Selected cell: `{summary['selected_cell']}`.",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.result_dir, args.exp0109_formal, args.exp0110_formal)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
