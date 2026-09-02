#!/usr/bin/env python3
"""Validate and summarize EXP-0157 prefill HMX-carrier reuse."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
MODES = ("duplicate", "reuse")
LAYERS = 28
EXPECTED_REUSED_BYTES = 4_014_080
MODULES = {
    "I/O and metadata": (
        "input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks"
    ),
    "Input RMSNorm": ("input_norm_ticks",),
    "QKV + Q/K Norm-RoPE preparation": (
        "qkv_projection_ticks", "qk_norm_rope_ticks"
    ),
    "QK-Softmax-AV": ("attention_ticks",),
    "O projection": ("o_projection_ticks",),
    "Post-attention residual + RMSNorm": (
        "post_attention_residual_ticks", "post_attention_norm_ticks"
    ),
    "Gate/Up + SwiGLU": ("gate_up_ticks", "activation_ticks"),
    "Down projection": ("down_ticks",),
    "Final residual": ("final_residual_ticks",),
    "KV-cache carrier conversion/update": ("scan_cache_pack_ticks",),
    "KV-cache append DMA": ("scan_cache_append_ticks",),
    "DSP orchestration/bookkeeping": (
        "block_orchestration_ticks", "layer_bookkeeping_ticks",
        "runtime_setup_ticks", "runtime_teardown_ticks",
        "stage_boundary_ticks",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--exp0154-result", type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0154/"
            "20260902_profile_rerun_v1/final"
        ),
    )
    parser.add_argument(
        "--exp0156-summary", type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0156/"
            "20260902T054011Z_5cec25408a33_formal/summary.json"
        ),
    )
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("record") == "replay_profile":
            records.append(value)
    if len(records) != 9:
        raise ValueError(f"incomplete replay in {path}: {len(records)}")
    if [int(item["replay_step"]) for item in records] != list(range(9)):
        raise ValueError(f"unordered replay in {path}")
    return records


def load_runs(result_dir: Path, mode: str) -> list[list[dict[str, object]]]:
    runs = [
        read_records(path)
        for path in sorted((result_dir / "raw").glob(f"round_*_{mode}.jsonl"))
    ]
    if len(runs) != 10:
        raise ValueError(f"expected 10 {mode} runs, got {len(runs)}")
    return runs


def load_old_runs(result_dir: Path, recipe: str) -> list[list[dict[str, object]]]:
    runs = [
        read_records(path)
        for path in sorted(
            (result_dir / "rotated_logs").glob(f"r??_{recipe}.log")
        )
    ]
    if len(runs) != 10:
        raise ValueError(f"expected 10 EXP-0154 {recipe} runs, got {len(runs)}")
    return runs


def stage_modules(record: dict[str, object]) -> dict[str, float]:
    values = {
        name: sum(float(record.get(field, 0)) for field in fields)
        / TICKS_PER_US
        for name, fields in MODULES.items()
    }
    values["Runtime/orchestration remainder"] = max(
        0.0, float(record["host_wall_ns"]) / 1000.0 - sum(values.values())
    )
    return values


def summarize_runs(runs: list[list[dict[str, object]]]) -> dict[str, object]:
    prefill_wall = median([
        float(run[0]["host_wall_ns"]) / 1000.0 for run in runs
    ])
    decode_wall = median([
        statistics.mean(
            float(record["host_wall_ns"]) / 1000.0 for record in run[1:]
        )
        for run in runs
    ])
    prefill_modules = {
        name: median([stage_modules(run[0])[name] for run in runs])
        for name in list(MODULES) + ["Runtime/orchestration remainder"]
    }
    decode_modules = {
        name: median([
            statistics.mean(stage_modules(record)[name] for record in run[1:])
            for run in runs
        ])
        for name in list(MODULES) + ["Runtime/orchestration remainder"]
    }
    return {
        "prefill_host_wall_us": prefill_wall,
        "prefill_tokens_per_second": 64_000_000.0 / prefill_wall,
        "decode_host_wall_us_per_token": decode_wall,
        "decode_tokens_per_second": 1_000_000.0 / decode_wall,
        "prefill_modules_us": prefill_modules,
        "decode_modules_us": decode_modules,
    }


def median_prefill_field(
    runs: list[list[dict[str, object]]], field: str
) -> float:
    return median([float(run[0].get(field, 0)) for run in runs])


def median_decode_field(
    runs: list[list[dict[str, object]]], field: str
) -> float:
    return median([
        statistics.mean(float(record.get(field, 0)) for record in run[1:])
        for run in runs
    ])


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_runs(result_dir, mode) for mode in MODES}

    correctness_pass = True
    physical_pass = True
    structural_pass = True
    for mode, mode_runs in runs.items():
        expected_mode = 0 if mode == "duplicate" else 1
        for run in mode_runs:
            for record in run:
                correctness_pass &= (
                    int(record["experiment"]) == 157
                    and int(record["output_mismatches"]) == 0
                    and int(record["output_max_lsb"]) == 0
                    and int(record["cache_mismatches"]) == 0
                    and int(record["cache_prefix_mismatches"]) == 0
                    and int(record["cache_structure_mismatches"]) == 0
                )
                physical_pass &= (
                    int(record["w4u8_prefill_cache_mode"]) == expected_mode
                    and int(record["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(record["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(record["intermediate_ddr_read_bytes"]) == 0
                    and int(record["intermediate_ddr_write_bytes"]) == 0
                    and int(record["intermediate_spill_fill_count"]) == 0
                    and record["intermediate_residency"] == "VTCM"
                    and record["qnn"] == "none"
                    and int(record["block_invocation_count"]) == LAYERS
                    and int(record["ledger_unattributed_ticks"]) == 0
                )
                if record["mode"] == "decode":
                    structural_pass &= (
                        int(record["u8_cache_full_prefix_pack_count"]) == 0
                        and int(record["u8_cache_native_prefill_build_count"]) == 0
                        and int(record["u8_cache_native_prefill_reuse_count"]) == 0
                        and int(record["u8_cache_native_incremental_append_count"])
                        == LAYERS
                    )
            prefill = run[0]
            if mode == "duplicate":
                structural_pass &= (
                    int(prefill["u8_cache_native_prefill_build_count"]) == LAYERS
                    and int(prefill["u8_cache_native_prefill_reuse_count"]) == 0
                    and int(prefill[
                        "u8_cache_native_prefill_reused_carrier_bytes"
                    ]) == 0
                )
            else:
                structural_pass &= (
                    int(prefill["u8_cache_native_prefill_build_count"]) == 0
                    and int(prefill["u8_cache_native_prefill_reuse_count"])
                    == LAYERS
                    and int(prefill[
                        "u8_cache_native_prefill_reused_carrier_bytes"
                    ]) == EXPECTED_REUSED_BYTES
                )

    mode_summary = {
        mode: summarize_runs(mode_runs) for mode, mode_runs in runs.items()
    }
    counter_fields = (
        "scan_cache_pack_ticks", "scan_cache_append_ticks",
        "scan_cache_dma_descriptor_count", "scan_cache_ddr_read_bytes",
        "scan_cache_ddr_write_bytes", "u8_attention_k_pack_ticks",
        "u8_attention_v_pack_ticks", "u8_cache_native_append_update_ticks",
        "u8_cache_native_prefill_build_count",
        "u8_cache_native_prefill_reuse_count",
        "u8_cache_native_prefill_reused_carrier_bytes",
        "u8_cache_native_incremental_append_count",
        "u8_cache_full_prefix_pack_count", "hmx_compute_ticks",
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "vtcm_peak_plan_bytes", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    )
    for mode, mode_runs in runs.items():
        mode_summary[mode]["prefill_counters"] = {
            field: median_prefill_field(mode_runs, field)
            for field in counter_fields
        }
        mode_summary[mode]["decode_counters"] = {
            field: median_decode_field(mode_runs, field)
            for field in counter_fields
        }

    control = mode_summary["duplicate"]
    candidate = mode_summary["reuse"]
    prefill_speed_gate = (
        float(candidate["prefill_host_wall_us"])
        < float(control["prefill_host_wall_us"])
    )
    decode_preservation_gate = (
        float(candidate["decode_host_wall_us_per_token"])
        <= 1.02 * float(control["decode_host_wall_us_per_token"])
    )

    old = {
        recipe: summarize_runs(load_old_runs(args.exp0154_result, recipe))
        for recipe in ("f16f16", "w4f16")
    }
    exp0156 = json.loads(args.exp0156_summary.read_text(encoding="utf-8"))
    row_major_prefill = float(
        exp0156["layouts"]["row_major"]["prefill_host_wall_us"]
    )
    summary = {
        "experiment": "EXP-0157",
        "execution_unit": (
            "real Qwen3 layers0-27 M64 prefill then decode positions64-71"
        ),
        "measurement_contract": (
            "repeat_count_1_stateful_replay_with_10_rotated_formal_pairs"
        ),
        "correctness_pass": correctness_pass,
        "physical_pass": physical_pass,
        "structural_pass": structural_pass,
        "prefill_speed_gate_pass": prefill_speed_gate,
        "decode_preservation_gate_pass": decode_preservation_gate,
        "modes": mode_summary,
        "comparison": {
            "prefill_latency_change_percent": (
                float(candidate["prefill_host_wall_us"])
                / float(control["prefill_host_wall_us"]) - 1.0
            ) * 100.0,
            "prefill_speedup_x": (
                float(control["prefill_host_wall_us"])
                / float(candidate["prefill_host_wall_us"])
            ),
            "decode_latency_change_percent": (
                float(candidate["decode_host_wall_us_per_token"])
                / float(control["decode_host_wall_us_per_token"]) - 1.0
            ) * 100.0,
            "candidate_prefill_change_vs_EXP0156_row_major_percent": (
                float(candidate["prefill_host_wall_us"])
                / row_major_prefill - 1.0
            ) * 100.0,
        },
        "three_recipe_overview": {
            "F16F16_EXP0154": old["f16f16"],
            "W4F16_EXP0154": old["w4f16"],
            "W4U8_EXP0157_reuse": candidate,
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0157 full profiling report", "",
        "Stateful replay uses kernel repeat_count=1; repeat-ten evidence is the "
        "median of ten rotated complete prefill/decode A/B pairs.", "",
        "## Three-recipe decode overview", "",
        "F16F16 and W4F16 reuse unchanged EXP-0154 evidence; W4U8 is the "
        "EXP-0157 reuse candidate.", "",
        "| Module | F16F16 | W4F16 | W4U8 reuse | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    module_order = list(MODULES) + ["Runtime/orchestration remainder"]
    f16 = old["f16f16"]
    w4f16 = old["w4f16"]
    w4u8 = candidate
    for module in module_order:
        f16_us = float(f16["decode_modules_us"][module])
        w4f16_us = float(w4f16["decode_modules_us"][module])
        w4u8_us = float(w4u8["decode_modules_us"][module])
        lines.append(
            f"| {module} | {f16_us:.3f} us "
            f"({100*f16_us/float(f16['decode_host_wall_us_per_token']):.1f}%) | "
            f"{w4f16_us:.3f} us "
            f"({100*w4f16_us/float(w4f16['decode_host_wall_us_per_token']):.1f}%) | "
            f"{w4u8_us:.3f} us "
            f"({100*w4u8_us/float(w4u8['decode_host_wall_us_per_token']):.1f}%) | "
            f"{(w4f16_us/w4u8_us-1)*100:+.1f}% |"
        )
    lines.append(
        f"| Complete Host wall | {float(f16['decode_host_wall_us_per_token']):.3f} us | "
        f"{float(w4f16['decode_host_wall_us_per_token']):.3f} us | "
        f"{float(w4u8['decode_host_wall_us_per_token']):.3f} us | "
        f"{(float(w4f16['decode_host_wall_us_per_token'])/float(w4u8['decode_host_wall_us_per_token'])-1)*100:+.1f}% |"
    )
    lines += [
        "", "## Direct HMX-native prefill-initialization A/B", "",
        "| Mode | Prefill M64 | Prefill tok/s | Decode/token | Decode tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        data = mode_summary[mode]
        lines.append(
            f"| {mode} | {float(data['prefill_host_wall_us']):.3f} us | "
            f"{float(data['prefill_tokens_per_second']):.3f} | "
            f"{float(data['decode_host_wall_us_per_token']):.3f} us | "
            f"{float(data['decode_tokens_per_second']):.3f} |"
        )
    lines += [
        "", "## Prefill module ledger", "",
        "| Module | Duplicate control | Reuse candidate | Candidate change |",
        "|---|---:|---:|---:|",
    ]
    for module in module_order:
        control_us = float(control["prefill_modules_us"][module])
        candidate_us = float(candidate["prefill_modules_us"][module])
        delta = ((candidate_us / control_us - 1.0) * 100.0
                 if control_us != 0.0 else 0.0)
        lines.append(
            f"| {module} | {control_us:.3f} us | {candidate_us:.3f} us | "
            f"{delta:+.2f}% |"
        )
    lines += [
        "", "## Physical and cache counters", "",
        "| Counter | Duplicate prefill | Reuse prefill | Duplicate decode | Reuse decode |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in counter_fields:
        lines.append(
            f"| {field} | {control['prefill_counters'][field]:.3f} | "
            f"{candidate['prefill_counters'][field]:.3f} | "
            f"{control['decode_counters'][field]:.3f} | "
            f"{candidate['decode_counters'][field]:.3f} |"
        )
    lines += [
        "",
        f"Correctness gate: **{'PASS' if correctness_pass else 'FAIL'}**.  ",
        f"Structural gate: **{'PASS' if structural_pass else 'FAIL'}**.  ",
        f"Physical gate: **{'PASS' if physical_pass else 'FAIL'}**.  ",
        f"Prefill speed gate: **{'PASS' if prefill_speed_gate else 'FAIL'}**.  ",
        f"Decode preservation gate: **{'PASS' if decode_preservation_gate else 'FAIL'}**.",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all((correctness_pass, structural_pass, physical_pass,
                prefill_speed_gate, decode_preservation_gate)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
