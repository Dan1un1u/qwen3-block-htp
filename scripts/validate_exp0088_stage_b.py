#!/usr/bin/env python3
"""Validate the EXP-0088 private persistent Softmax-LUT speed gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SAMPLES = 5
REPEATS = (1, 10)
MODES = ("control", "candidate")
TARGETS = (
    "host_wall_ns_per_block",
    "attention_ticks",
    "u8_attention_softmax_ticks",
)
OUTPUT_HASH = "69f22eeb035e5ec5"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
CONTROL_PIPELINE = (
    "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_"
    "hmx_batch_lut_templates_gqa_batch_dependency_stream"
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


def validate(record: dict[str, object], repeat: int, mode: str,
             audit: bool = False) -> None:
    require(record, "experiment", "EXP-0088")
    require(record, "attention_pipeline_mode", CONTROL_PIPELINE +
            ("_private_persistent_lut" if mode == "candidate" else ""))
    require(record, "output_hash", OUTPUT_HASH)
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    require(record, "vtcm_requested_bytes", 8_388_608)
    require(record, "vtcm_acquired_bytes", 8_388_608)
    require(record, "intermediate_ddr_read_bytes", 0)
    require(record, "intermediate_ddr_write_bytes", 0)
    require(record, "intermediate_dma_descriptor_count", 0)
    require(record, "intermediate_spill_fill_count", 0)
    require(record, "hmx_command_count", 176 * repeat)
    require(record, "hmx_u8s8_tile_pair_count", 49_408 * repeat)
    require(record, "block_invocation_count", repeat)
    if audit:
        require(record, "u8_attention_actual_score_hash", QK_HASH)
        require(record, "u8_attention_actual_probability_hash",
                PROBABILITY_HASH)
        require(record, "u8_attention_actual_av_hash", AV_HASH)
    if mode == "control":
        require(record, "warmup_u8_attention_lut_template_reuse_count", 0)
        require(record, "u8_attention_lut_template_reuse_count", 0)
        require(record, "u8_attention_lut_private_vtcm_bytes", 0)
        if not audit:
            require(record, "warmup_u8_attention_lut_template_build_count", 6)
            require(record, "u8_attention_lut_template_build_count", 6 * repeat)
    else:
        require(record, "u8_attention_lut_private_vtcm_bytes", 3_072)
        if audit:
            warmup_builds = int(
                record["warmup_u8_attention_lut_template_build_count"]
            )
            measured_builds = int(
                record["u8_attention_lut_template_build_count"]
            )
            measured_reuses = int(
                record["u8_attention_lut_template_reuse_count"]
            )
            if not 1 <= warmup_builds <= 6:
                raise SystemExit(
                    "candidate audit warmup must activate between one and "
                    f"six private contexts, got {warmup_builds}"
                )
            if measured_reuses <= 0:
                raise SystemExit(
                    "candidate audit measured invocation did not reuse any "
                    "warmup-built private bank"
                )
            if measured_builds + measured_reuses != warmup_builds:
                raise SystemExit(
                    "candidate audit active-context count changed between "
                    "warmup and measured invocation: "
                    f"{measured_builds} + {measured_reuses} != {warmup_builds}"
                )
        else:
            require(record, "u8_attention_lut_template_build_count", 0)
            require(record, "u8_attention_lut_template_reuse_count", 6 * repeat)
            require(record, "u8_attention_lut_template_build_ticks", 0)
            require(record, "warmup_u8_attention_lut_template_build_count", 6)
            require(record, "warmup_u8_attention_lut_template_reuse_count", 0)


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
        record = load(root / f"correctness_{mode}.jsonl", 1)[0]
        validate(record, 1, mode, audit=True)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "qk_hash": record["u8_attention_actual_score_hash"],
            "probability_hash": record["u8_attention_actual_probability_hash"],
            "av_hash": record["u8_attention_actual_av_hash"],
            "warmup_builds": record["warmup_u8_attention_lut_template_build_count"],
            "measured_builds": record["u8_attention_lut_template_build_count"],
            "measured_reuses": record["u8_attention_lut_template_reuse_count"],
        }

    repeat_results: dict[str, object] = {}
    gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load(root / f"paired_{mode}_r{repeat}.jsonl", SAMPLES)
            for mode in MODES
        }
        for mode in MODES:
            for record in records[mode]:
                validate(record, repeat, mode)
        metrics = {
            field: summarize(records["control"], records["candidate"], field)
            for field in TARGETS
        }
        passed = all(
            metrics[field][key] < 0.0
            for field in TARGETS
            for key in ("change_percent", "paired_change_percent_median")
        )
        gates.append(passed)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "three_target_speed_gate": passed,
            "control_build_count": 6 * repeat,
            "candidate_build_count": 0,
            "candidate_reuse_count": 6 * repeat,
        }

    print(json.dumps({
        "experiment": "EXP-0088",
        "stage": "B",
        "correctness_and_persistence": correctness,
        "repeat_results": repeat_results,
        "stage_b_gate_pass": all(gates),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
