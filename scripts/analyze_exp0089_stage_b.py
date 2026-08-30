#!/usr/bin/env python3
"""Validate EXP-0089 Stage-B Gate/Up-to-Down producer overlap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SAMPLES = 5
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
MIDDLE_HASH = "a1458e36b7fc9ad3"
DOWN_HASH = "f64264224127030f"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
WEIGHT_BYTES = 25_444_352
DESCRIPTORS = 512
HMX_COMMANDS = 176
HMX_TILE_PAIRS = 49_408
VTCM_BYTES = 8_388_608
VTCM_PEAK = 5_306_080

LEDGER_FIELDS = (
    "input_stage_ticks",
    "metadata_stage_ticks",
    "input_norm_ticks",
    "qkv_projection_ticks",
    "qk_norm_rope_ticks",
    "attention_ticks",
    "o_projection_ticks",
    "post_attention_residual_ticks",
    "post_attention_norm_ticks",
    "gate_up_ticks",
    "activation_ticks",
    "down_ticks",
    "final_residual_ticks",
    "output_stage_ticks",
    "runtime_setup_ticks",
    "runtime_teardown_ticks",
    "stage_boundary_ticks",
    "invocation_ticks",
    "ledger_named_ticks",
    "ledger_unattributed_ticks",
)

OVERLAP_FIELDS = (
    "w4u8_mlp_gate_up_pipeline_ticks",
    "w4u8_mlp_down_pipeline_ticks",
    "w4u8_mlp_activation_work_ticks",
    "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
    "weight_dma_ticks",
    "hmx_compute_ticks",
    "hmx_ready_wait_ticks",
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


def validate_primary(record: dict[str, object], repeat: int) -> None:
    require(record, "experiment", "EXP-0089")
    require(record, "variant", "W4U8")
    require(record, "output_hash", OUTPUT_HASH)
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    require(record, "dsp_status", 3)
    require(record, "numerical_status", 1)
    require(record, "vtcm_requested_bytes", VTCM_BYTES)
    require(record, "vtcm_acquired_bytes", VTCM_BYTES)
    require(record, "vtcm_peak_plan_bytes", VTCM_PEAK)
    require(record, "intermediate_ddr_read_bytes", 0)
    require(record, "intermediate_ddr_write_bytes", 0)
    require(record, "intermediate_dma_descriptor_count", 0)
    require(record, "intermediate_spill_fill_count", 0)
    require(record, "weight_ddr_read_bytes", WEIGHT_BYTES * repeat)
    require(record, "weight_dma_descriptor_count", DESCRIPTORS * repeat)
    require(record, "hmx_command_count", HMX_COMMANDS * repeat)
    require(record, "hmx_u8s8_tile_pair_count", HMX_TILE_PAIRS * repeat)
    require(record, "block_invocation_count", repeat)
    require(record, "w4u8_mlp_gate_up_hmx_command_count", 48 * repeat)
    require(record, "w4u8_mlp_down_hmx_command_count", 64 * repeat)


def validate_boundary(record: dict[str, object]) -> None:
    require(record, "experiment", "EXP-0089")
    require(record, "record", "gate_up_down_audit")
    require(record, "middle_activation_hash", MIDDLE_HASH)
    require(record, "down_output_hash", DOWN_HASH)


def validate_prestage(record: dict[str, object]) -> None:
    require(record, "experiment", "EXP-0089")
    require(record, "record", "gate_up_down_prestage")
    require(record, "enabled", 1)
    require(record, "output_count", 4)
    require(record, "dma_descriptor_count", 2)
    require(record, "expand_count", 4)
    require(record, "consumed", 1)
    require(record, "ring_bytes", 1_183_744)
    require(record, "ring_offset", 2_013_184)
    require(record, "gate_up_dma_descriptor_count", 386)
    require(record, "down_dma_descriptor_count", 30)
    require(record, "trigger_output_base", 288)
    if not (
        int(record["start_ticks"])
        < int(record["first_dma_publication_ticks"])
        <= int(record["dma_end_ticks"])
        <= int(record["expand_end_ticks"])
    ):
        raise SystemExit(f"invalid pre-stage ordering: {record}")
    if int(record["first_expand_end_ticks"]) < int(
            record["first_dma_publication_ticks"]):
        raise SystemExit(f"expand completed before DMA publication: {record}")


def median(records: list[dict[str, object]], field: str) -> float:
    return float(statistics.median(float(record[field]) for record in records))


def summarize(control: list[dict[str, object]],
              candidate: list[dict[str, object]], field: str) -> dict[str, float]:
    left = [float(record[field]) for record in control]
    right = [float(record[field]) for record in candidate]
    control_median = statistics.median(left)
    candidate_median = statistics.median(right)
    paired = [(r / l - 1.0) * 100.0 for l, r in zip(left, right)]
    return {
        "control": control_median,
        "candidate": candidate_median,
        "change_percent": (candidate_median / control_median - 1.0) * 100.0,
        "paired_change_percent_median": statistics.median(paired),
    }


def summarize_combined(control: list[dict[str, object]],
                       candidate: list[dict[str, object]]) -> dict[str, float]:
    left = [float(r["gate_up_ticks"]) + float(r["down_ticks"])
            for r in control]
    right = [float(r["gate_up_ticks"]) + float(r["down_ticks"])
             for r in candidate]
    control_median = statistics.median(left)
    candidate_median = statistics.median(right)
    paired = [(r / l - 1.0) * 100.0 for l, r in zip(left, right)]
    return {
        "control": control_median,
        "candidate": candidate_median,
        "change_percent": (candidate_median / control_median - 1.0) * 100.0,
        "paired_change_percent_median": statistics.median(paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    if (root / "boot_id_before.txt").read_bytes() != (
            root / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed")

    correctness: dict[str, object] = {}
    for mode in MODES:
        primary = load(root / f"correctness_{mode}.jsonl", 1)[0]
        boundary = load(root / f"correctness_{mode}_boundary.jsonl", 1)[0]
        validate_primary(primary, 1)
        validate_boundary(boundary)
        require(primary, "u8_attention_actual_score_hash", QK_HASH)
        require(primary, "u8_attention_actual_probability_hash",
                PROBABILITY_HASH)
        require(primary, "u8_attention_actual_av_hash", AV_HASH)
        correctness[mode] = {
            "output_hash": primary["output_hash"],
            "middle_activation_hash": boundary["middle_activation_hash"],
            "down_output_hash": boundary["down_output_hash"],
            "qk_hash": primary["u8_attention_actual_score_hash"],
            "probability_hash": primary[
                "u8_attention_actual_probability_hash"],
            "av_hash": primary["u8_attention_actual_av_hash"],
            "mismatches": primary["mismatches"],
            "max_lsb": primary["max_lsb"],
        }
    validate_prestage(load(
        root / "correctness_candidate_prestage.jsonl", 1)[0])

    repeat_results: dict[str, object] = {}
    stage_gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load(root / f"paired_{mode}_r{repeat}.jsonl", SAMPLES)
            for mode in MODES
        }
        for mode in MODES:
            for record in records[mode]:
                validate_primary(record, repeat)
        prestage_records = load(
            root / f"paired_candidate_r{repeat}_prestage.jsonl", SAMPLES)
        for record in prestage_records:
            validate_prestage(record)

        gate_up = summarize(
            records["control"], records["candidate"], "gate_up_ticks")
        down = summarize(
            records["control"], records["candidate"], "down_ticks")
        combined = summarize_combined(
            records["control"], records["candidate"])
        host = summarize(
            records["control"], records["candidate"],
            "host_wall_ns_per_block")
        gate_up_non_regress = all(
            gate_up[key] <= 0.0
            for key in ("change_percent", "paired_change_percent_median")
        )
        strict_speed = all(
            metric[key] < 0.0
            for metric in (combined, host)
            for key in ("change_percent", "paired_change_percent_median")
        )
        passed = gate_up_non_regress and strict_speed
        stage_gates.append(passed)

        ledger = {
            field: summarize(records["control"], records["candidate"], field)
            for field in LEDGER_FIELDS
        }
        overlap = {
            field: summarize(records["control"], records["candidate"], field)
            for field in OVERLAP_FIELDS
        }
        prestage = {
            field: median(prestage_records, field)
            for field in (
                "start_ticks", "first_dma_publication_ticks",
                "dma_end_ticks", "first_expand_end_ticks",
                "expand_end_ticks", "dma_work_ticks", "expand_work_ticks",
            )
        }
        repeat_results[f"repeat{repeat}"] = {
            "gate_up": gate_up,
            "down": down,
            "gate_up_plus_down": combined,
            "complete_host": host,
            "gate_up_non_regress": gate_up_non_regress,
            "strict_combined_and_host_speed_gate": strict_speed,
            "repeat_gate_pass": passed,
            "timing_ledger": ledger,
            "overlap_counters": overlap,
            "prestage_timeline_median": prestage,
        }

    print(json.dumps({
        "experiment": "EXP-0089",
        "stage": "B",
        "samples_per_cell": SAMPLES,
        "correctness": correctness,
        "physical_gate": {
            "vtcm_requested_bytes": VTCM_BYTES,
            "vtcm_peak_plan_bytes": VTCM_PEAK,
            "intermediate_ddr_bytes": 0,
            "intermediate_spill_fill_count": 0,
            "weight_ddr_bytes_per_block": WEIGHT_BYTES,
            "weight_dma_descriptors_per_block": DESCRIPTORS,
            "hmx_commands_per_block": HMX_COMMANDS,
            "u8s8_tile_pairs_per_block": HMX_TILE_PAIRS,
            "hmx_owners": 1,
            "pass": True,
        },
        "repeat_results": repeat_results,
        "stage_b_gate_pass": all(stage_gates),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
