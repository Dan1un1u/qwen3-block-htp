#!/usr/bin/env python3
"""Summarize EXP-0156 full-stack HMX-native cache scaling."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
LAYOUTS = ("row_major", "hmx_native_u8")
LAYERS = 28
SCALING_POINTS = (1, 3, 7, 14, 28)
EXP0155_SINGLE_LAYER_US = 2358.307375
SCALING_LIMIT_US = 1.25 * LAYERS * EXP0155_SINGLE_LAYER_US
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--exp0154-result",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0154/"
            "20260902_profile_rerun_v1/final"
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


def load_new_runs(result_dir: Path, layout: str) -> list[list[dict[str, object]]]:
    runs = [
        read_records(path)
        for path in sorted((result_dir / "raw").glob(f"round_*_{layout}.jsonl"))
    ]
    if len(runs) != 10:
        raise ValueError(f"expected 10 {layout} runs, got {len(runs)}")
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


def wall_and_modules(
    runs: list[list[dict[str, object]]],
) -> dict[str, object]:
    prefill = median([
        float(run[0]["host_wall_ns"]) / 1000.0 for run in runs
    ])
    decode_per_run = [
        statistics.mean(
            float(record["host_wall_ns"]) / 1000.0 for record in run[1:]
        )
        for run in runs
    ]
    decode = median(decode_per_run)
    modules: dict[str, float] = {}
    for name, fields in MODULES.items():
        modules[name] = median([
            statistics.mean(
                sum(float(record.get(field, 0)) for field in fields)
                / TICKS_PER_US
                for record in run[1:]
            )
            for run in runs
        ])
    modules["Runtime/orchestration remainder"] = max(
        0.0, decode - sum(modules.values())
    )
    return {
        "prefill_host_wall_us": prefill,
        "decode_host_wall_us_per_token": decode,
        "decode_tokens_per_second": 1_000_000.0 / decode,
        "decode_modules_us": modules,
    }


def median_decode_field(
    runs: list[list[dict[str, object]]], field: str
) -> float:
    return median([
        statistics.mean(float(record.get(field, 0)) for record in run[1:])
        for run in runs
    ])


def per_layer_decode_us(
    runs: list[list[dict[str, object]]],
) -> list[float]:
    values: list[float] = []
    for layer in range(LAYERS):
        key = f"slice_layer_{layer}"
        values.append(median([
            statistics.mean(
                float(record[key]["layer_ticks"]) / TICKS_PER_US
                for record in run[1:]
            )
            for run in runs
        ]))
    return values


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {
        layout: load_new_runs(result_dir, layout) for layout in LAYOUTS
    }

    correctness_pass = True
    physical_pass = True
    for layout, layout_runs in runs.items():
        for run in layout_runs:
            for record in run:
                correctness_pass &= (
                    int(record["output_mismatches"]) == 0
                    and int(record["cache_mismatches"]) == 0
                    and int(record["cache_prefix_mismatches"]) == 0
                    and int(record["cache_structure_mismatches"]) == 0
                )
                physical_pass &= (
                    int(record["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(record["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(record["intermediate_ddr_read_bytes"]) == 0
                    and int(record["intermediate_ddr_write_bytes"]) == 0
                    and int(record["intermediate_spill_fill_count"]) == 0
                )
                if layout == "hmx_native_u8" and record["mode"] == "decode":
                    physical_pass &= (
                        int(record["u8_cache_full_prefix_pack_count"]) == 0
                    )

    layout_summary = {
        layout: wall_and_modules(layout_runs)
        for layout, layout_runs in runs.items()
    }
    for layout, layout_runs in runs.items():
        layout_summary[layout]["cache_metrics"] = {
            field: median_decode_field(layout_runs, field)
            for field in (
                "u8_attention_k_pack_ticks",
                "u8_attention_v_pack_ticks",
                "u8_cache_native_append_update_ticks",
                "u8_cache_full_prefix_pack_count",
                "scan_cache_ddr_read_bytes",
                "scan_cache_ddr_write_bytes",
            )
        }

    row = layout_summary["row_major"]
    native = layout_summary["hmx_native_u8"]
    speed_gate = (
        float(native["decode_host_wall_us_per_token"])
        < float(row["decode_host_wall_us_per_token"])
    )
    scaling_gate = (
        float(native["decode_host_wall_us_per_token"]) <= SCALING_LIMIT_US
    )

    layer_values = {
        layout: per_layer_decode_us(layout_runs)
        for layout, layout_runs in runs.items()
    }
    scaling: dict[str, object] = {}
    for point in SCALING_POINTS:
        row_sum = sum(layer_values["row_major"][:point])
        native_sum = sum(layer_values["hmx_native_u8"][:point])
        scaling[str(point)] = {
            "row_major_cumulative_layer_us": row_sum,
            "hmx_native_cumulative_layer_us": native_sum,
            "speedup_x": row_sum / native_sum,
            "hmx_native_us_per_layer": native_sum / point,
        }

    old = {
        recipe: wall_and_modules(load_old_runs(args.exp0154_result, recipe))
        for recipe in ("f16f16", "w4f16")
    }
    overview = {
        "F16F16_EXP0154": old["f16f16"],
        "W4F16_EXP0154": old["w4f16"],
        "W4U8_EXP0156_HMX_native": native,
    }
    summary = {
        "experiment": "EXP-0156",
        "execution_unit": (
            "real Qwen3 layers0-27 M64 prefill then decode positions64-71"
        ),
        "correctness_pass": correctness_pass,
        "physical_pass": physical_pass,
        "speed_gate_pass": speed_gate,
        "scaling_gate_pass": scaling_gate,
        "scaling_limit_us": SCALING_LIMIT_US,
        "layouts": layout_summary,
        "comparison": {
            "decode_speedup_x": (
                float(row["decode_host_wall_us_per_token"])
                / float(native["decode_host_wall_us_per_token"])
            ),
            "decode_latency_reduction_percent": (
                1.0
                - float(native["decode_host_wall_us_per_token"])
                / float(row["decode_host_wall_us_per_token"])
            ) * 100.0,
            "prefill_latency_change_percent": (
                float(native["prefill_host_wall_us"])
                / float(row["prefill_host_wall_us"])
                - 1.0
            ) * 100.0,
        },
        "layer_decode_us": layer_values,
        "cumulative_scaling": scaling,
        "three_recipe_overview": overview,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0156 full profiling report",
        "",
        "## Three-recipe decode overview",
        "",
        "F16F16 and W4F16 reuse the unchanged EXP-0154 formal evidence; "
        "W4U8 is the EXP-0156 HMX-native candidate.",
        "",
        "| Module | F16F16 | W4F16 | W4U8 HMX-native | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    module_order = list(MODULES) + ["Runtime/orchestration remainder"]
    f16 = old["f16f16"]
    w4f16 = old["w4f16"]
    w4u8 = native
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
        "",
        "## Direct cache-layout A/B",
        "",
        "| Layout | Prefill M64 | Decode/token | Decode tok/s |",
        "|---|---:|---:|---:|",
    ]
    for layout in LAYOUTS:
        data = layout_summary[layout]
        lines.append(
            f"| {layout} | {float(data['prefill_host_wall_us']):.3f} us | "
            f"{float(data['decode_host_wall_us_per_token']):.3f} us | "
            f"{float(data['decode_tokens_per_second']):.3f} |"
        )
    lines += [
        "",
        f"Correctness gate: **{'PASS' if correctness_pass else 'FAIL'}**.  ",
        f"Physical gate: **{'PASS' if physical_pass else 'FAIL'}**.  ",
        f"Speed gate: **{'PASS' if speed_gate else 'FAIL'}**.  ",
        f"Scaling gate: **{'PASS' if scaling_gate else 'FAIL'}** "
        f"(limit {SCALING_LIMIT_US:.3f} us/token).",
        "",
        "## Cumulative per-layer decode ledger",
        "",
        "| Layers | row-major | HMX-native | Speedup | HMX-native/layer |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in SCALING_POINTS:
        data = scaling[str(point)]
        lines.append(
            f"| {point} | {data['row_major_cumulative_layer_us']:.3f} us | "
            f"{data['hmx_native_cumulative_layer_us']:.3f} us | "
            f"{data['speedup_x']:.3f}x | "
            f"{data['hmx_native_us_per_layer']:.3f} us |"
        )
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not (correctness_pass and physical_pass and speed_gate and scaling_gate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
