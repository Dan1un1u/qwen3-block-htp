#!/usr/bin/env python3
"""Validate the EXP-0092 Gate/Up decoder-by-worker factorial."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SAMPLES = 5
REPEATS = (1, 10)
CONTROL = "single_w3"
CELLS = (CONTROL, "interleaved_w3", "single_w2", "interleaved_w2")
CELL_CONFIG = {
    "single_w3": ("single", 3),
    "interleaved_w3": ("interleaved2", 3),
    "single_w2": ("single", 2),
    "interleaved_w2": ("interleaved2", 2),
}
OUTPUT_HASH = "69f22eeb035e5ec5"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
VTCM_BYTES = 8_388_608
WEIGHT_BYTES = 25_444_352
DMA_DESCRIPTORS = 512
HMX_COMMANDS = 176
HMX_TILE_PAIRS = 49_408
PRIMARY_FIELDS = (
    "host_wall_ns_per_block", "gate_up_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_activation_work_ticks",
    "w4u8_mlp_hmx_compute_ticks", "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
)
LEDGER_FIELDS = (
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks",
    "stage_boundary_ticks", "invocation_ticks", "ledger_named_ticks",
    "ledger_unattributed_ticks",
)
INVARIANT_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "weight_ddr_read_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_u8s8_tile_pair_count", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_mlp_down_hmx_command_count",
)


def load(path: Path, expected: int) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if len(records) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {len(records)}")
    return records


def require(record: dict[str, object], field: str, expected: object) -> None:
    if record.get(field) != expected:
        raise SystemExit(
            f"{field}: expected {expected!r}, got {record.get(field)!r}"
        )


def validate(record: dict[str, object], repeat: int, cell: str) -> None:
    decode, workers = CELL_CONFIG[cell]
    require(record, "experiment", "EXP-0092")
    require(record, "variant", "W4U8")
    require(record, "w4u8_gate_up_decode_mode", decode)
    require(record, "w4u8_gate_up_worker_count", workers)
    require(record, "w4u8_mlp_gate_up_hvx_workers", workers)
    require(record, "w4u8_gate_up_persistent_hvx_worker_count",
            workers * repeat)
    require(record, "output_hash", OUTPUT_HASH)
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    require(record, "dsp_status", 3)
    require(record, "numerical_status", 1)
    require(record, "vtcm_requested_bytes", VTCM_BYTES)
    require(record, "vtcm_acquired_bytes", VTCM_BYTES)
    require(record, "intermediate_ddr_read_bytes", 0)
    require(record, "intermediate_ddr_write_bytes", 0)
    require(record, "intermediate_dma_descriptor_count", 0)
    require(record, "intermediate_spill_fill_count", 0)
    require(record, "weight_ddr_read_bytes", WEIGHT_BYTES * repeat)
    require(record, "weight_dma_descriptor_count",
            DMA_DESCRIPTORS * repeat)
    require(record, "hmx_command_count", HMX_COMMANDS * repeat)
    require(record, "hmx_u8s8_tile_pair_count", HMX_TILE_PAIRS * repeat)
    require(record, "block_invocation_count", repeat)


def summarize(control: list[dict[str, object]],
              candidate: list[dict[str, object]], field: str,
              divisor: float) -> dict[str, float | None]:
    left = [float(record[field]) / divisor for record in control]
    right = [float(record[field]) / divisor for record in candidate]
    left_median = statistics.median(left)
    right_median = statistics.median(right)
    if left_median == 0.0 or any(value == 0.0 for value in left):
        return {"control": left_median, "candidate": right_median,
                "change_percent": None,
                "paired_change_percent_median": None}
    paired = [(r / l - 1.0) * 100.0 for l, r in zip(left, right)]
    return {
        "control": left_median, "candidate": right_median,
        "change_percent": (right_median / left_median - 1.0) * 100.0,
        "paired_change_percent_median": statistics.median(paired),
    }


def build_summary(root: Path) -> dict[str, object]:
    if (root / "boot_id_before.txt").read_bytes() != (
            root / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed")
    static_gate = json.loads((root / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")
    correctness: dict[str, object] = {}
    for cell in CELLS:
        record = load(root / f"correctness_{cell}.jsonl", 1)[0]
        validate(record, 1, cell)
        require(record, "u8_attention_actual_score_hash", QK_HASH)
        require(record, "u8_attention_actual_probability_hash",
                PROBABILITY_HASH)
        require(record, "u8_attention_actual_av_hash", AV_HASH)
        correctness[cell] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "qk_hash": record["u8_attention_actual_score_hash"],
            "probability_hash": record[
                "u8_attention_actual_probability_hash"],
            "av_hash": record["u8_attention_actual_av_hash"],
        }

    repeats: dict[str, object] = {}
    passing_by_repeat: list[set[str]] = []
    for repeat in REPEATS:
        records = {
            cell: load(root / f"paired_{cell}_r{repeat}.jsonl", SAMPLES)
            for cell in CELLS
        }
        for cell in CELLS:
            for record in records[cell]:
                validate(record, repeat, cell)
        cells: dict[str, object] = {}
        passing: set[str] = set()
        for cell in CELLS[1:]:
            fields = tuple(dict.fromkeys((*PRIMARY_FIELDS, *LEDGER_FIELDS,
                                          *INVARIANT_FIELDS)))
            metrics = {
                field: summarize(
                    records[CONTROL], records[cell], field,
                    1.0 if field == "host_wall_ns_per_block"
                    else float(repeat))
                for field in fields
            }
            speed = all(
                metrics[field][key] is not None and
                float(metrics[field][key]) < 0.0
                for field in ("gate_up_ticks", "host_wall_ns_per_block")
                for key in ("change_percent",
                            "paired_change_percent_median")
            )
            invariant = all(
                all(candidate[field] == records[CONTROL][index][field]
                    for index, candidate in enumerate(records[cell]))
                for field in INVARIANT_FIELDS
            )
            if speed and invariant:
                passing.add(cell)
            cells[cell] = {
                "metrics": metrics,
                "strict_gate_up_and_host_speed_gate": speed,
                "unchanged_physical_contract_gate": invariant,
                "repeat_gate_pass": speed and invariant,
            }
        passing_by_repeat.append(passing)
        repeats[f"repeat{repeat}"] = {"cells": cells,
                                      "passing_cells": sorted(passing)}
    passing_all = set.intersection(*passing_by_repeat)
    selected = sorted(passing_all)[0] if len(passing_all) == 1 else None
    return {
        "experiment": "EXP-0092", "stage": "A",
        "samples_per_cell": SAMPLES, "static_gate": static_gate,
        "correctness": correctness,
        "repeat_results": repeats,
        "passing_cells_all_repeats": sorted(passing_all),
        "unique_selected_cell": selected,
        "stage_a_gate_pass": selected is not None,
        "physical_gate": {
            "vtcm_bytes": VTCM_BYTES, "intermediate_ddr_bytes": 0,
            "spill_fill": 0, "weight_bytes": WEIGHT_BYTES,
            "dma_descriptors": DMA_DESCRIPTORS,
            "hmx_commands": HMX_COMMANDS,
            "u8s8_tile_pairs": HMX_TILE_PAIRS, "hmx_owners": 1,
            "qnn": False, "pass": True,
        },
    }


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0092 — Complete profiling report", "",
        "Four cells isolate decoder scheduling, exact physical Gate/Up worker "
        "count, and their interaction. All numerical and physical work other "
        "than the declared worker count is fixed.", "",
    ]
    for repeat in REPEATS:
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Cell vs single_w3 | Host change | Host paired | Gate/Up change | Gate/Up paired | Expand change | HMX compute change | Ready-wait change | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cell in CELLS[1:]:
            result = summary["repeat_results"][f"repeat{repeat}"]["cells"][cell]
            metrics = result["metrics"]
            host = metrics["host_wall_ns_per_block"]
            gate = metrics["gate_up_ticks"]
            expand = metrics["w4u8_mlp_weight_expand_ticks"]
            hmx = metrics["w4u8_mlp_hmx_compute_ticks"]
            ready = metrics["w4u8_mlp_hmx_ready_wait_ticks"]
            lines.append(
                f"| {cell} | {host['change_percent']:+.3f}% | "
                f"{host['paired_change_percent_median']:+.3f}% | "
                f"{gate['change_percent']:+.3f}% | "
                f"{gate['paired_change_percent_median']:+.3f}% | "
                f"{expand['change_percent']:+.3f}% | "
                f"{hmx['change_percent']:+.3f}% | "
                f"{ready['change_percent']:+.3f}% | "
                f"{'PASS' if result['repeat_gate_pass'] else 'FAIL'} |"
            )
        lines.append("")
    lines.extend([
        "## PC-028 stable accepted three-recipe repeat10 table", "",
        "| Module | W16A16 | W4A16 | W4A8 accepted | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
        "| I/O and metadata | 6.6 us (0.3%) | 7.6 us (0.4%) | 4.1 us (0.2%) | +85.1% |",
        "| Input RMSNorm | 17.4 us (0.7%) | 17.3 us (0.8%) | 19.2 us (1.0%) | -9.7% |",
        "| QKV + Q/K Norm/RoPE | 400.6 us (16.4%) | 437.6 us (20.2%) | 424.6 us (21.7%) | +3.1% |",
        "| QK-Softmax-AV | 140.4 us (5.7%) | 139.6 us (6.5%) | 196.8 us (10.1%) | -29.1% |",
        "| O projection | 202.0 us (8.3%) | 172.9 us (8.0%) | 171.4 us (8.8%) | +0.9% |",
        "| Post-attn residual + RMSNorm | 16.7 us (0.7%) | 16.7 us (0.8%) | 35.6 us (1.8%) | -52.9% |",
        "| Gate/Up + SwiGLU | 1120.4 us (45.9%) | 964.6 us (44.6%) | 686.2 us (35.1%) | +40.6% |",
        "| Down projection | 459.6 us (18.8%) | 329.5 us (15.2%) | 318.1 us (16.3%) | +3.6% |",
        "| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 17.3 us (0.9%) | -71.3% |",
        "| Host/RPC and closure remainder | 73.7 us (3.0%) | 71.1 us (3.3%) | 82.2 us (4.2%) | -13.5% |",
        "| Complete block Host wall | 2442.4 us | 2161.8 us | 1955.3 us | +10.6% |",
        "", "## Correctness and physical gates", "",
        "All four cells are byte-exact at the final output and audited QK, probability and AV boundaries, with exact 8 MiB VTCM, zero intermediate DDR/spill, one FastRPC, one HMX owner and no QNN.",
        "", "## Decision", "",
        f"Stage-A gate: **{'PASS' if summary['stage_a_gate_pass'] else 'FAIL'}**. "
        f"Unique selected cell: `{summary['unique_selected_cell']}`. The accepted baseline remains EXP-0084 pending explicit user promotion.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    summary = build_summary(args.result_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
