#!/usr/bin/env python3
"""Summarize the rotated EXP-0155 row-major versus HMX-cache replay."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
LAYOUTS = ("row_major", "hmx_native_u8")

MODULES = {
    "I/O and metadata": ("input_stage_ticks", "metadata_stage_ticks",
                         "output_stage_ticks"),
    "Input RMSNorm": ("input_norm_ticks",),
    "QKV + Q/K Norm-RoPE preparation": ("qkv_projection_ticks",
                                           "qk_norm_rope_ticks"),
    "QK-Softmax-AV": ("attention_ticks",),
    "O projection": ("o_projection_ticks",),
    "Post-attention residual + RMSNorm": (
        "post_attention_residual_ticks", "post_attention_norm_ticks"),
    "Gate/Up + SwiGLU": ("gate_up_ticks", "activation_ticks"),
    "Down projection": ("down_ticks",),
    "Final residual": ("final_residual_ticks",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("record") == "replay_profile":
            records.append(value)
    if len(records) != 9 or [int(item["replay_step"]) for item in records] != list(range(9)):
        raise ValueError(f"incomplete replay in {path}: {len(records)} records")
    return records


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs: dict[str, list[list[dict[str, object]]]] = {layout: [] for layout in LAYOUTS}
    for layout in LAYOUTS:
        for path in sorted((result_dir / "raw").glob(f"round_*_{layout}.jsonl")):
            runs[layout].append(read_records(path))
        if len(runs[layout]) != 10:
            raise ValueError(f"expected 10 {layout} runs, got {len(runs[layout])}")

    physical_pass = True
    correctness_pass = True
    for layout in LAYOUTS:
        for run in runs[layout]:
            for record in run:
                correctness_pass &= (
                    int(record["output_mismatches"]) == 0 and
                    int(record["cache_mismatches"]) == 0 and
                    int(record["cache_prefix_mismatches"]) == 0)
                physical_pass &= (
                    int(record["vtcm_requested_bytes"]) == 8 * 1024 * 1024 and
                    int(record["vtcm_acquired_bytes"]) == 8 * 1024 * 1024 and
                    int(record["intermediate_ddr_read_bytes"]) == 0 and
                    int(record["intermediate_ddr_write_bytes"]) == 0 and
                    int(record["intermediate_spill_fill_count"]) == 0)
                if layout == "hmx_native_u8":
                    physical_pass &= int(record["u8_cache_full_prefix_pack_count"]) == 0

    summary: dict[str, object] = {
        "experiment": "EXP-0155",
        "execution_unit": "real layer14 M64 prefill then decode positions 64-71",
        "rounds": 10,
        "correctness_pass": correctness_pass,
        "physical_pass": physical_pass,
        "layouts": {},
    }
    for layout in LAYOUTS:
        prefill_us = [float(run[0]["host_wall_ns"]) / 1000.0 for run in runs[layout]]
        decode_mean_us = [
            statistics.mean(float(item["host_wall_ns"]) / 1000.0 for item in run[1:])
            for run in runs[layout]
        ]
        decode_sequence_us = [
            sum(float(item["host_wall_ns"]) / 1000.0 for item in run[1:])
            for run in runs[layout]
        ]
        complete_sequence_us = [
            float(run[0]["host_wall_ns"]) / 1000.0 + decode_sequence_us[index]
            for index, run in enumerate(runs[layout])
        ]
        module_us: dict[str, float] = {}
        for name, fields in MODULES.items():
            per_run = [
                statistics.mean(
                    sum(float(item.get(field, 0)) for field in fields) /
                    TICKS_PER_US
                    for item in run[1:]
                )
                for run in runs[layout]
            ]
            module_us[name] = median(per_run)
        decode_wall = median(decode_mean_us)
        accounted = sum(module_us.values())
        module_us["Runtime/orchestration remainder"] = max(0.0, decode_wall - accounted)
        cache_metrics = {}
        for field in (
            "u8_attention_k_pack_ticks", "u8_attention_v_pack_ticks",
            "u8_cache_native_append_update_ticks", "scan_cache_pack_ticks",
            "scan_cache_stage_ticks", "scan_cache_append_ticks",
            "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
            "u8_cache_full_prefix_pack_count",
        ):
            per_run = [
                statistics.mean(float(item.get(field, 0)) for item in run[1:])
                for run in runs[layout]
            ]
            cache_metrics[field] = median(per_run)
        summary["layouts"][layout] = {
            "prefill_host_wall_us_median": median(prefill_us),
            "decode_host_wall_us_per_token_median": decode_wall,
            "decode_eight_token_sequence_us_median": median(decode_sequence_us),
            "prefill_plus_decode_sequence_us_median": median(complete_sequence_us),
            "decode_modules_us": module_us,
            "decode_cache_metrics": cache_metrics,
        }

    row = summary["layouts"]["row_major"]
    native = summary["layouts"]["hmx_native_u8"]
    comparison = {
        "prefill_latency_change_percent": (
            float(native["prefill_host_wall_us_median"]) /
            float(row["prefill_host_wall_us_median"]) - 1.0) * 100.0,
        "decode_speedup_x": (
            float(row["decode_host_wall_us_per_token_median"]) /
            float(native["decode_host_wall_us_per_token_median"])),
        "decode_latency_reduction_percent": (
            1.0 - float(native["decode_host_wall_us_per_token_median"]) /
            float(row["decode_host_wall_us_per_token_median"])) * 100.0,
        "complete_sequence_speedup_x": (
            float(row["prefill_plus_decode_sequence_us_median"]) /
            float(native["prefill_plus_decode_sequence_us_median"])),
    }
    summary["comparison"] = comparison
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# EXP-0155 formal profiling report", "",
        f"Correctness gate: **{'PASS' if correctness_pass else 'FAIL'}**.  ",
        f"Physical gate: **{'PASS' if physical_pass else 'FAIL'}**.", "",
        "| Layout | Prefill M64 | Decode/token L64-L71 | Eight decode tokens | Prefill + eight decode |",
        "|---|---:|---:|---:|---:|",
    ]
    for layout in LAYOUTS:
        data = summary["layouts"][layout]
        lines.append(
            f"| {layout} | {data['prefill_host_wall_us_median']:.3f} us | "
            f"{data['decode_host_wall_us_per_token_median']:.3f} us | "
            f"{data['decode_eight_token_sequence_us_median']:.3f} us | "
            f"{data['prefill_plus_decode_sequence_us_median']:.3f} us |")
    lines += [
        "",
        f"HMX-native decode speedup: **{comparison['decode_speedup_x']:.3f}x** "
        f"({comparison['decode_latency_reduction_percent']:.2f}% latency reduction).  ",
        f"Prefill latency change: **{comparison['prefill_latency_change_percent']:+.2f}%**.  ",
        f"Complete replay speedup: **{comparison['complete_sequence_speedup_x']:.3f}x**.",
        "", "## Decode wall attribution", "",
        "| Module | row_major | hmx_native_u8 |",
        "|---|---:|---:|",
    ]
    for name in list(MODULES) + ["Runtime/orchestration remainder"]:
        lines.append(
            f"| {name} | {row['decode_modules_us'][name]:.3f} us | "
            f"{native['decode_modules_us'][name]:.3f} us |")
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_pass or not physical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
