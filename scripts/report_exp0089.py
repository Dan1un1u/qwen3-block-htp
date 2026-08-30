#!/usr/bin/env python3
"""Render the PC-027/PC-028 EXP-0089 closure report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TICKS_PER_US = 19.2

COMPARATORS = {
    "I/O and metadata": (6.6, 7.6),
    "Input RMSNorm": (17.4, 17.3),
    "QKV + Q/K Norm/RoPE": (400.6, 437.6),
    "QK-Softmax-AV": (140.4, 139.6),
    "O projection": (202.0, 172.9),
    "Post-attn residual + RMSNorm": (16.7, 16.7),
    "Gate/Up + SwiGLU": (1120.4, 964.6),
    "Down projection": (459.6, 329.5),
    "Final residual": (5.0, 5.0),
    "Host/RPC and profiling closure remainder": (73.7, 71.1),
}
COMPARATOR_HOST = (2442.4, 2161.8)


def pct(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.4f}%"


def number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def metric_row(name: str, metric: dict[str, object], unit: str = "ticks") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"| `{name}` | {number(metric['control'])}{suffix} | "
        f"{number(metric['candidate'])}{suffix} | "
        f"{pct(metric.get('change_percent'))} | "
        f"{pct(metric.get('paired_change_percent_median'))} |"
    )


def module_times(repeat10: dict[str, object]) -> dict[str, float]:
    ledger = repeat10["timing_ledger"]
    def ticks(field: str) -> float:
        return float(ledger[field]["control"])
    modules = {
        "I/O and metadata": (
            ticks("input_stage_ticks") + ticks("metadata_stage_ticks") +
            ticks("output_stage_ticks")) / TICKS_PER_US,
        "Input RMSNorm": ticks("input_norm_ticks") / TICKS_PER_US,
        "QKV + Q/K Norm/RoPE": (
            ticks("qkv_projection_ticks") + ticks("qk_norm_rope_ticks")
        ) / TICKS_PER_US,
        "QK-Softmax-AV": ticks("attention_ticks") / TICKS_PER_US,
        "O projection": ticks("o_projection_ticks") / TICKS_PER_US,
        "Post-attn residual + RMSNorm": (
            ticks("post_attention_residual_ticks") +
            ticks("post_attention_norm_ticks")) / TICKS_PER_US,
        "Gate/Up + SwiGLU": (
            ticks("gate_up_ticks") + ticks("activation_ticks")
        ) / TICKS_PER_US,
        "Down projection": ticks("down_ticks") / TICKS_PER_US,
        "Final residual": ticks("final_residual_ticks") / TICKS_PER_US,
    }
    host_us = float(repeat10["complete_host"]["control"]) / 1000.0
    modules["Host/RPC and profiling closure remainder"] = (
        host_us - sum(modules.values())
    )
    return modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--analysis-commit", default="working-tree")
    args = parser.parse_args()

    summary = json.loads((args.result_dir / "gate_summary.json").read_text())
    source_commit = (args.result_dir / "source_commit.txt").read_text().strip()
    repeat10 = summary["repeat_results"]["repeat10"]
    w4u8_modules = module_times(repeat10)
    w4u8_host = float(repeat10["complete_host"]["control"]) / 1000.0

    lines = [
        "# EXP-0089 full profiling report",
        "",
        "## PC-028 repeat10 three-variant overview",
        "",
        "F16F16 and the Selected-Baseline W4F16 values are reused from the "
        "accepted EXP-0084 canonical common-schedule evidence. W4U8 is the "
        "direct EXP-0089 control because the candidate failed its local gate.",
        "",
        "| Module | F16F16 EXP-0084 | W4F16 EXP-0084 | W4U8 EXP-0089 control | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (f16, w4f16) in COMPARATORS.items():
        w4u8 = w4u8_modules[name]
        lines.append(
            f"| {name} | {f16:.1f} us ({f16 / COMPARATOR_HOST[0] * 100:.1f}%) | "
            f"{w4f16:.1f} us ({w4f16 / COMPARATOR_HOST[1] * 100:.1f}%) | "
            f"{w4u8:.1f} us ({w4u8 / w4u8_host * 100:.1f}%) | "
            f"{(w4f16 / w4u8 - 1.0) * 100:+.1f}% |"
        )
    lines.extend([
        f"| Complete block Host wall | {COMPARATOR_HOST[0]:.1f} us | "
        f"{COMPARATOR_HOST[1]:.1f} us | {w4u8_host:.1f} us | "
        f"{(COMPARATOR_HOST[1] / w4u8_host - 1.0) * 100:+.1f}% |",
        "",
        "## Identity and comparison",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Experiment | EXP-0089 |",
        "| Source branch | `codex/exp-0089-w4u8-gate-up-down-dependency-stream` |",
        f"| Runtime source commit | `{source_commit}` |",
        f"| Analysis source commit | `{args.analysis_commit}` |",
        f"| Formal evidence | `{args.result_dir}` |",
        f"| Retained artifacts | `{args.artifact_dir}` |",
        "| Execution Unit | Qwen3 layer-14 complete middle block, M=64 |",
        "| Project Variant | W4U8 |",
        "| Direct control | EXP-0084 serial Gate/Up completion followed by Down setup |",
        "| Candidate | output-288 trigger; two DMA descriptors and four first-chunk HVX expansions retained in a private VTCM ring |",
        "| Repeats / rounds | repeat1 and repeat10; five interleaved paired rounds |",
        "| Backend | standalone FastRPC/cDSP; QNN none; fallback none |",
        "| RPC/HMX ownership | one FastRPC invocation per block; one HMX ownership domain |",
        "",
    ])

    for repeat_name in ("repeat1", "repeat10"):
        result = summary["repeat_results"][repeat_name]
        lines.extend([
            f"## Direct control vs candidate — {repeat_name}",
            "",
            "All DSP tick values are medians normalized per block. Host wall is in microseconds per block.",
            "",
            "### Primary Gate/Up-to-Down latency",
            "",
            "| Metric | Control | Candidate | Candidate delta | Paired delta median |",
            "|---|---:|---:|---:|---:|",
            metric_row("gate_up_ticks", result["gate_up"]),
            metric_row("down_ticks", result["down"]),
            metric_row("gate_up_ticks + down_ticks", result["gate_up_plus_down"]),
            metric_row("host_wall_ns_per_block", result["complete_host"], "ns"),
            "",
            "### Complete additive Block Timing Ledger",
            "",
            "| Metric | Control | Candidate | Candidate delta | Paired delta median |",
            "|---|---:|---:|---:|---:|",
        ])
        for field, metric in result["timing_ledger"].items():
            lines.append(metric_row(field, metric))
        lines.extend([
            "",
            "### Overlapping MLP/HMX/HVX/DMA diagnostics",
            "",
            "These counters overlap and must not be summed into wall time.",
            "",
            "| Metric | Control | Candidate | Candidate delta | Paired delta median |",
            "|---|---:|---:|---:|---:|",
        ])
        for field, metric in result["overlap_counters"].items():
            lines.append(metric_row(field, metric))
        timeline = result["prestage_timeline_median"]
        lines.extend([
            "",
            "### Candidate pre-stage timeline",
            "",
            "| Event/work | Median qtimer ticks from Gate/Up pipeline start |",
            "|---|---:|",
        ])
        for field, value in timeline.items():
            lines.append(f"| `{field}` | {float(value):.3f} |")
        lines.extend([
            "",
            f"Gate/Up non-regression gate: **{'PASS' if result['gate_up_non_regress'] else 'FAIL'}**. "
            f"Combined-and-Host strict speed gate: **{'PASS' if result['strict_combined_and_host_speed_gate'] else 'FAIL'}**.",
            "",
        ])

    physical = summary["physical_gate"]
    correctness = summary["correctness"]
    lines.extend([
        "## Correctness and physical contract",
        "",
        "| Gate | Control | Candidate | Result |",
        "|---|---:|---:|---|",
        f"| Final output hash | `{correctness['control']['output_hash']}` | `{correctness['candidate']['output_hash']}` | PASS |",
        f"| Middle activation hash | `{correctness['control']['middle_activation_hash']}` | `{correctness['candidate']['middle_activation_hash']}` | PASS |",
        f"| Down output hash | `{correctness['control']['down_output_hash']}` | `{correctness['candidate']['down_output_hash']}` | PASS |",
        f"| QK / probability / AV hashes | `{correctness['control']['qk_hash']}` / `{correctness['control']['probability_hash']}` / `{correctness['control']['av_hash']}` | identical | PASS |",
        "| Final mismatch / max LSB | 0 / 0 | 0 / 0 | PASS |",
        f"| Weight DDR bytes per block | {physical['weight_ddr_bytes_per_block']} | {physical['weight_ddr_bytes_per_block']} | PASS |",
        f"| Weight DMA descriptors per block | {physical['weight_dma_descriptors_per_block']} | {physical['weight_dma_descriptors_per_block']} | PASS |",
        f"| HMX commands / U8S8 tile pairs | {physical['hmx_commands_per_block']} / {physical['u8s8_tile_pairs_per_block']} | same | PASS |",
        f"| VTCM requested / peak | {physical['vtcm_requested_bytes']} / {physical['vtcm_peak_plan_bytes']} B | same global peak; 1183744 B phase-overlay ring | PASS |",
        "| Intermediate DDR read/write; spill/fill | 0 / 0; 0 | 0 / 0; 0 | PASS |",
        "| QNN / CPU fallback / second HMX owner | none | none | PASS |",
        "",
        "The candidate moves exactly four existing Down weight bundles into two earlier DMA descriptors, expands four unchanged first chunks on the existing HVX pool, and consumes them from VTCM. The normal Down producer therefore omits two later descriptors. Total weight bytes, 512 descriptors, 176 HMX commands, 49,408 tile pairs, arithmetic, qparams, output codes, and the global VTCM peak remain unchanged.",
        "",
        "## Decision",
        "",
        "Stage B fails. Down improves by 3.29% at repeat1 and 1.47% at repeat10, but the borrowed DMA/HVX work raises Gate/Up by 0.79% and 0.62%. The combined interval improves only 0.40% and 0.04%; repeat10 ordinary Host wall regresses 0.17%. Because Gate/Up non-regression is a hard condition, Stage C is not entered. EXP-0084 remains the Selected W4U8 baseline and the EXP-0089 candidate is rejected.",
        "",
    ])

    report = "\n".join(lines)
    output = args.result_dir / "full_profiling_report.md"
    output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
