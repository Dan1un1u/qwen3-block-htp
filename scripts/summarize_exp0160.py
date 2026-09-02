#!/usr/bin/env python3
"""Validate and summarize EXP-0160 reconstruction scheduling A/B."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0159 as common


MODES = ("serial", "pipeline")
LAYERS = 28
CANDIDATE_BASE_READ = 4_014_080
CANDIDATE_DECODE_WRITE = 57_344


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def read_run(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sequence_pass = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("record") == "replay_profile":
            records.append(value)
        if value.get("replay_sequence_complete") is True:
            sequence_pass = value.get("all_steps_pass") is True
    if len(records) != 9 or [int(item["replay_step"]) for item in records] != list(range(9)):
        raise ValueError(f"incomplete or unordered replay in {path}")
    if not sequence_pass:
        raise ValueError(f"failed replay sequence in {path}")
    return records


def load_runs(result_dir: Path, mode: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"round_*_{mode}.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected 10 {mode} runs, got {len(paths)}")
    return [read_run(path) for path in paths]


def validate(runs: dict[str, list[list[dict[str, object]]]]) -> tuple[bool, bool, bool, bool]:
    correctness = True
    physical = True
    structural = True
    invariant = True
    expected_modes = {"serial": 0, "pipeline": 2}
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            for record in run:
                step = int(record["replay_step"])
                correctness &= (
                    int(record["experiment"]) == 160
                    and record["variant"] == "W4U8"
                    and int(record["w4u8_delta_reconstruction_mode"])
                    == expected_modes[mode]
                    and int(record["dsp_status"]) == 3
                    and int(record["numerical_status"]) == 1
                    and int(record["output_mismatches"]) == 0
                    and int(record["output_max_lsb"]) == 0
                    and int(record["cache_mismatches"]) == 0
                    and int(record["cache_prefix_mismatches"]) == 0
                    and int(record["cache_structure_mismatches"]) == 0
                )
                physical &= (
                    int(record["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(record["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(record["intermediate_ddr_read_bytes"]) == 0
                    and int(record["intermediate_ddr_write_bytes"]) == 0
                    and int(record["intermediate_spill_fill_count"]) == 0
                    and record["intermediate_residency"] == "VTCM"
                    and record["qnn"] == "none"
                    and int(record["block_invocation_count"]) == LAYERS
                    and int(record["ledger_unattributed_ticks"]) == 0
                )
                if step == 0:
                    structural &= (
                        int(record["scan_cache_ddr_read_bytes"]) == 0
                        and int(record["scan_cache_ddr_write_bytes"])
                        == CANDIDATE_BASE_READ
                    )
                else:
                    structural &= (
                        int(record["scan_cache_ddr_read_bytes"])
                        == CANDIDATE_BASE_READ + CANDIDATE_DECODE_WRITE * step
                        and int(record["scan_cache_ddr_write_bytes"])
                        == CANDIDATE_DECODE_WRITE
                        and int(record["u8_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_dma_descriptor_count"])
                        == 64 * LAYERS
                    )
                    if mode == "pipeline":
                        structural &= (
                            int(record["scan_attention_overlay_required_bytes"])
                            > 0
                            and int(record["scan_attention_overlay_required_bytes"])
                            <= int(record["scan_attention_overlay_capacity_bytes"])
                        )
    for serial_run, pipeline_run in zip(runs["serial"], runs["pipeline"]):
        for serial, pipeline in zip(serial_run, pipeline_run):
            invariant &= (
                int(serial["hmx_command_count"]) == int(pipeline["hmx_command_count"])
                and int(serial["hmx_u8s8_tile_pair_count"])
                == int(pipeline["hmx_u8s8_tile_pair_count"])
                and int(serial["weight_ddr_read_bytes"])
                == int(pipeline["weight_ddr_read_bytes"])
            )
    return correctness, physical, structural, invariant


def median_decode_field(runs: list[list[dict[str, object]]], field: str) -> float:
    return float(statistics.median([
        statistics.mean(float(record.get(field, 0)) for record in run[1:])
        for run in runs
    ]))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_runs(result_dir, mode) for mode in MODES}
    correctness, physical, structural, invariant = validate(runs)
    summaries = {mode: common.summarize_runs(value) for mode, value in runs.items()}
    for mode, mode_runs in runs.items():
        summaries[mode]["decode_attention_us"] = (
            median_decode_field(mode_runs, "attention_ticks") / common.TICKS_PER_US
        )
        summaries[mode]["decode_pipeline_wait_us"] = (
            median_decode_field(mode_runs, "u8_attention_pipeline_wait_ticks")
            / common.TICKS_PER_US
        )
        summaries[mode]["overlay_required_bytes"] = median_decode_field(
            mode_runs, "scan_attention_overlay_required_bytes"
        )

    serial = summaries["serial"]
    pipeline = summaries["pipeline"]
    serial_prefill = float(serial["prefill_host_wall_us"])
    pipeline_prefill = float(pipeline["prefill_host_wall_us"])
    serial_decode = float(serial["decode_host_wall_us_per_token"])
    pipeline_decode = float(pipeline["decode_host_wall_us_per_token"])
    speed = {
        "prefill_speed_improvement_percent": common.pct_speed(serial_prefill, pipeline_prefill),
        "prefill_gate_pass": pipeline_prefill <= 1.01 * serial_prefill,
        "decode_speed_improvement_percent": common.pct_speed(serial_decode, pipeline_decode),
        "decode_gate_pass": pipeline_decode < serial_decode,
        "attention_speed_improvement_percent": common.pct_speed(
            float(serial["decode_attention_us"]),
            float(pipeline["decode_attention_us"]),
        ),
    }
    all_gates = (
        correctness and physical and structural and invariant
        and bool(speed["prefill_gate_pass"])
        and bool(speed["decode_gate_pass"])
    )
    summary = {
        "experiment": "EXP-0160",
        "execution_unit": "real Qwen3 layers0-27 M64 prefill then decode positions64-71",
        "measurement_contract": "same_binary_10_rotated_serial_pipeline_replay_runs",
        "correctness_pass": correctness,
        "physical_pass": physical,
        "structural_pass": structural,
        "unchanged_hmx_and_projection_contract_pass": invariant,
        "speed_gates": speed,
        "all_gates_pass": all_gates,
        "modes": summaries,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0160 full profiling report", "",
        "One binary, ten rotated complete replays, real layers 0-27.", "",
        "| Reconstruction | Prefill M64 | Prefill tok/s | Decode/token | Decode tok/s | Attention/decode |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        data = summaries[mode]
        lines.append(
            f"| {mode} | {float(data['prefill_host_wall_us']):.3f} us | "
            f"{float(data['prefill_tokens_per_second']):.3f} | "
            f"{float(data['decode_host_wall_us_per_token']):.3f} us | "
            f"{float(data['decode_tokens_per_second']):.3f} | "
            f"{float(data['decode_attention_us']):.3f} us |"
        )
    lines += [
        "",
        f"Pipeline prefill speed change: {float(speed['prefill_speed_improvement_percent']):+.3f}%.",
        f"Pipeline decode speed change: {float(speed['decode_speed_improvement_percent']):+.3f}%.",
        f"Pipeline Attention speed change: {float(speed['attention_speed_improvement_percent']):+.3f}%.",
        "", "## Validation", "",
        f"- Exact output/cache correctness: {'PASS' if correctness else 'FAIL'}",
        f"- Physical residency contract: {'PASS' if physical else 'FAIL'}",
        f"- Direct/two-slot reconstruction structure: {'PASS' if structural else 'FAIL'}",
        f"- Unchanged HMX/projection work: {'PASS' if invariant else 'FAIL'}",
        f"- Prefill <= 1% regression: {'PASS' if speed['prefill_gate_pass'] else 'FAIL'}",
        f"- Decode strictly faster: {'PASS' if speed['decode_gate_pass'] else 'FAIL'}",
        f"- Overall gate: {'PASS' if all_gates else 'FAIL'}", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not all_gates:
        raise SystemExit("EXP-0160 validation gate failed")


if __name__ == "__main__":
    main()
