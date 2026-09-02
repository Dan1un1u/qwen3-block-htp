#!/usr/bin/env python3
"""Validate and summarize EXP-0158 A16 HMX-native KV-cache parity."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
LAYERS = 28
PREFILL_TOKENS = 64
EXPECTED_REUSED_BYTES = 7_340_032
EXPECTED_NATIVE_PREFILL_WRITE_BYTES = 8_257_536
EXPECTED_DECODE_WRITE_BYTES = 114_688
VARIANTS = {
    "f16f16_row_major": ("F16F16", "row_major"),
    "f16f16_hmx_native_f16": ("F16F16", "hmx_native_f16"),
    "w4f16_row_major": ("W4F16", "row_major"),
    "w4f16_hmx_native_f16": ("W4F16", "hmx_native_f16"),
}
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
        "--exp0157-summary", type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0157/"
            "20260902T061809Z_0a3557d075dc_formal/summary.json"
        ),
    )
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


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
    if len(records) != 9:
        raise ValueError(f"incomplete replay in {path}: {len(records)} profiles")
    if [int(item["replay_step"]) for item in records] != list(range(9)):
        raise ValueError(f"unordered replay in {path}")
    if not sequence_pass:
        raise ValueError(f"failed replay sequence in {path}")
    return records


def load_runs(result_dir: Path, key: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"round_*_{key}.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected 10 {key} runs, got {len(paths)}")
    return [read_run(path) for path in paths]


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
    order = list(MODULES) + ["Runtime/orchestration remainder"]
    prefill_modules = {
        name: median([stage_modules(run[0])[name] for run in runs])
        for name in order
    }
    decode_modules = {
        name: median([
            statistics.mean(stage_modules(record)[name] for record in run[1:])
            for run in runs
        ])
        for name in order
    }
    return {
        "prefill_host_wall_us": prefill_wall,
        "prefill_tokens_per_second": PREFILL_TOKENS * 1_000_000.0 / prefill_wall,
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


def pct_speed(control_us: float, candidate_us: float) -> float:
    return (control_us / candidate_us - 1.0) * 100.0


def fmt_cell(value: float, total: float) -> str:
    return f"{value:.3f} us ({100.0 * value / total:.1f}%)"


def validate_runs(
    all_runs: dict[str, list[list[dict[str, object]]]]
) -> tuple[bool, bool, bool]:
    correctness = True
    physical = True
    structural = True
    for key, runs in all_runs.items():
        variant, layout = VARIANTS[key]
        native = layout == "hmx_native_f16"
        for run in runs:
            for record in run:
                correctness &= (
                    int(record["experiment"]) == 158
                    and record["variant"] == variant
                    and int(record["dsp_status"]) == 3
                    and int(record["numerical_status"]) == 1
                    and int(record["output_nonfinite_count"]) == 0
                    and math.isfinite(float(record["output_cosine"]))
                    and float(record["output_cosine"]) >= 0.99999
                    and math.isfinite(float(record["output_nrmse"]))
                    and float(record["output_nrmse"]) <= 0.003
                    and int(record["cache_prefix_mismatches"]) == 0
                    and int(record["cache_structure_mismatches"]) == 0
                    and int(record["cache_nonfinite_count"]) == 0
                    and int(record["cache_tensor_count"]) == 2 * LAYERS
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
                if native and record["mode"] == "prefill":
                    structural &= (
                        int(record["f16_cache_native_prefill_reuse_count"])
                        == LAYERS
                        and int(record[
                            "f16_cache_native_prefill_reused_carrier_bytes"
                        ]) == EXPECTED_REUSED_BYTES
                        and int(record[
                            "f16_cache_native_incremental_append_count"
                        ]) == 0
                        and int(record["f16_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_ddr_write_bytes"])
                        == EXPECTED_NATIVE_PREFILL_WRITE_BYTES
                    )
                elif native:
                    structural &= (
                        int(record["f16_cache_native_prefill_reuse_count"]) == 0
                        and int(record[
                            "f16_cache_native_incremental_append_count"
                        ]) == LAYERS
                        and int(record["f16_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_ddr_write_bytes"])
                        == EXPECTED_DECODE_WRITE_BYTES
                    )
                else:
                    structural &= (
                        int(record["f16_cache_native_prefill_reuse_count"]) == 0
                        and int(record[
                            "f16_cache_native_incremental_append_count"
                        ]) == 0
                        and int(record[
                            "f16_cache_native_prefill_reused_carrier_bytes"
                        ]) == 0
                        and int(record[
                            "f16_cache_native_append_update_ticks"
                        ]) == 0
                        and int(record["f16_cache_full_prefix_pack_count"])
                        == (0 if record["mode"] == "prefill" else 16 * LAYERS)
                    )
    return correctness, physical, structural


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {key: load_runs(result_dir, key) for key in VARIANTS}
    correctness, physical, structural = validate_runs(runs)
    summaries = {key: summarize_runs(value) for key, value in runs.items()}

    counter_fields = (
        "scan_cache_pack_ticks", "scan_cache_append_ticks",
        "scan_cache_stage_ticks", "scan_cache_dma_descriptor_count",
        "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
        "f16_cache_native_append_update_ticks",
        "f16_cache_native_prefill_reuse_count",
        "f16_cache_native_prefill_reused_carrier_bytes",
        "f16_cache_native_incremental_append_count",
        "f16_cache_full_prefix_pack_count", "attention_qk_pack_ticks",
        "attention_av_pack_ticks", "attention_qk_hmx_ticks",
        "attention_av_hmx_ticks", "hmx_compute_ticks", "hmx_command_count",
        "hmx_fp16_tile_pair_count", "vtcm_peak_plan_bytes",
        "weight_ddr_read_bytes", "boundary_ddr_read_bytes",
        "boundary_ddr_write_bytes",
    )
    for key, key_runs in runs.items():
        summaries[key]["prefill_counters"] = {
            field: median_prefill_field(key_runs, field)
            for field in counter_fields
        }
        summaries[key]["decode_counters"] = {
            field: median_decode_field(key_runs, field)
            for field in counter_fields
        }

    recipe_gates: dict[str, object] = {}
    for recipe in ("f16f16", "w4f16"):
        control = summaries[f"{recipe}_row_major"]
        candidate = summaries[f"{recipe}_hmx_native_f16"]
        prefill_control = float(control["prefill_host_wall_us"])
        prefill_candidate = float(candidate["prefill_host_wall_us"])
        decode_control = float(control["decode_host_wall_us_per_token"])
        decode_candidate = float(candidate["decode_host_wall_us_per_token"])
        recipe_gates[recipe] = {
            "prefill_speed_gate_pass": prefill_candidate < prefill_control,
            "decode_speed_gate_pass": decode_candidate < decode_control,
            "prefill_latency_change_percent": (
                prefill_candidate / prefill_control - 1.0
            ) * 100.0,
            "prefill_speed_improvement_percent": pct_speed(
                prefill_control, prefill_candidate
            ),
            "decode_latency_change_percent": (
                decode_candidate / decode_control - 1.0
            ) * 100.0,
            "decode_speed_improvement_percent": pct_speed(
                decode_control, decode_candidate
            ),
        }

    exp0157 = json.loads(args.exp0157_summary.read_text(encoding="utf-8"))
    w4u8 = exp0157["modes"]["reuse"]
    three_recipe = {
        "F16F16_EXP0158_hmx_native_f16": summaries["f16f16_hmx_native_f16"],
        "W4F16_EXP0158_hmx_native_f16": summaries["w4f16_hmx_native_f16"],
        "W4U8_EXP0157_reuse": w4u8,
    }
    summary = {
        "experiment": "EXP-0158",
        "execution_unit": (
            "real Qwen3 layers0-27 M64 prefill then decode positions64-71"
        ),
        "measurement_contract": (
            "repeat_count_1_stateful_replay_with_10_rotated_four_variant_runs"
        ),
        "cache_contract": (
            "exact_M64_HMX_carrier_plus_contiguous_8_token_delta_journal"
        ),
        "correctness_pass": correctness,
        "physical_pass": physical,
        "structural_pass": structural,
        "recipe_gates": recipe_gates,
        "all_speed_gates_pass": all(
            bool(gate["prefill_speed_gate_pass"])
            and bool(gate["decode_speed_gate_pass"])
            for gate in recipe_gates.values()
        ),
        "variants": summaries,
        "three_recipe_overview": three_recipe,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0158 full profiling report", "",
        "All timings use one prepared-session invocation per full 28-layer "
        "prefill/decode step. Each cell is the median across ten rotated "
        "complete four-variant runs; decode is the per-run mean of positions "
        "64-71 before taking the median.", "",
        "## Direct cache-layout A/B", "",
        "| Recipe / layout | Prefill M64 | Prefill tok/s | Decode/token | Decode tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in VARIANTS:
        data = summaries[key]
        lines.append(
            f"| {key} | {float(data['prefill_host_wall_us']):.3f} us | "
            f"{float(data['prefill_tokens_per_second']):.3f} | "
            f"{float(data['decode_host_wall_us_per_token']):.3f} us | "
            f"{float(data['decode_tokens_per_second']):.3f} |"
        )
    lines += ["", "## A/B speed gates", "",
              "| Recipe | Prefill speed | Decode speed | Gate |",
              "|---|---:|---:|---:|"]
    for recipe, gate in recipe_gates.items():
        passed = gate["prefill_speed_gate_pass"] and gate["decode_speed_gate_pass"]
        lines.append(
            f"| {recipe} | {float(gate['prefill_speed_improvement_percent']):+.3f}% | "
            f"{float(gate['decode_speed_improvement_percent']):+.3f}% | "
            f"{'PASS' if passed else 'FAIL'} |"
        )

    module_order = list(MODULES) + ["Runtime/orchestration remainder"]
    for stage, title in (("prefill", "Prefill"), ("decode", "Decode")):
        lines += ["", f"## Three-recipe {title.lower()} module ledger", "",
                  "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
                  "|---|---:|---:|---:|---:|"]
        f16 = three_recipe["F16F16_EXP0158_hmx_native_f16"]
        w4 = three_recipe["W4F16_EXP0158_hmx_native_f16"]
        u8 = three_recipe["W4U8_EXP0157_reuse"]
        wall_key = "prefill_host_wall_us" if stage == "prefill" else "decode_host_wall_us_per_token"
        modules_key = f"{stage}_modules_us"
        for module in module_order:
            f16_us = float(f16[modules_key][module])
            w4_us = float(w4[modules_key][module])
            u8_us = float(u8[modules_key][module])
            lines.append(
                f"| {module} | {fmt_cell(f16_us, float(f16[wall_key]))} | "
                f"{fmt_cell(w4_us, float(w4[wall_key]))} | "
                f"{fmt_cell(u8_us, float(u8[wall_key]))} | "
                f"{pct_speed(w4_us, u8_us):+.1f}% |"
            )
        lines.append(
            f"| Complete Host wall | {float(f16[wall_key]):.3f} us | "
            f"{float(w4[wall_key]):.3f} us | {float(u8[wall_key]):.3f} us | "
            f"{pct_speed(float(w4[wall_key]), float(u8[wall_key])):+.1f}% |"
        )

    lines += ["", "## Validation", "",
              f"- Numerical/correctness: {'PASS' if correctness else 'FAIL'}",
              f"- Physical contract: {'PASS' if physical else 'FAIL'}",
              f"- Structural cache contract: {'PASS' if structural else 'FAIL'}",
              f"- All strict speed gates: {'PASS' if summary['all_speed_gates_pass'] else 'FAIL'}",
              ""]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not (correctness and physical and structural):
        raise SystemExit("EXP-0158 validation gate failed")


if __name__ == "__main__":
    main()
