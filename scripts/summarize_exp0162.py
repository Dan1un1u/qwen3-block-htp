#!/usr/bin/env python3
"""Validate and summarize EXP-0162 dynamic segmented-cache full-stack A/B."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
LAYERS = 28
PREFILL_TOKENS = 64
STEPS = tuple(range(41))
MODES = ("control", "candidate")
WINDOWS = {
    "prefill": (0,),
    "early_decode_64_71": tuple(range(1, 9)),
    "preseal_decode_72_94": tuple(range(9, 32)),
    "seal_position_95": (32,),
    "postseal_decode_96_103": tuple(range(33, 41)),
    "all_decode_64_103": tuple(range(1, 41)),
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
        "runtime_setup_ticks", "runtime_teardown_ticks", "stage_boundary_ticks",
    ),
}
OVERLAP_FIELDS = (
    "weight_dma_ticks", "hmx_compute_ticks", "hmx_ready_wait_ticks",
    "w4u8_qkvo_weight_expand_ticks", "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks", "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks", "u8_attention_k_pack_ticks",
    "u8_attention_v_pack_ticks", "u8_attention_qk_hmx_ticks",
    "u8_attention_softmax_ticks", "u8_attention_av_hmx_ticks",
    "u8_attention_pipeline_wait_ticks", "scan_cache_stage_ticks",
    "scan_cache_pack_ticks", "scan_cache_append_ticks",
)
COUNT_FIELDS = (
    "hmx_command_count", "hmx_u8s8_tile_pair_count",
    "weight_dma_descriptor_count", "boundary_dma_descriptor_count",
    "scan_cache_dma_descriptor_count", "weight_ddr_read_bytes",
    "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
    "intermediate_spill_fill_count", "vtcm_requested_bytes",
    "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
)
SEALED_BYTES = 1_892_352


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--exp0158-summary", type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0158/"
            "20260902T071708Z_264c911a65a3_formal/summary.json"
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
    if len(records) != len(STEPS):
        raise ValueError(f"incomplete replay in {path}: {len(records)} records")
    if tuple(int(item["replay_step"]) for item in records) != STEPS:
        raise ValueError(f"unordered replay in {path}")
    if not sequence_pass:
        raise ValueError(f"failed replay sequence in {path}")
    return records


def load_runs(result_dir: Path, mode: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"round_*_{mode}.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected 10 {mode} runs, got {len(paths)}")
    return [read_run(path) for path in paths]


def module_values(record: dict[str, object]) -> dict[str, float]:
    values = {
        name: sum(float(record.get(field, 0)) for field in fields) / TICKS_PER_US
        for name, fields in MODULES.items()
    }
    values["Runtime/orchestration remainder"] = max(
        0.0, float(record["host_wall_ns"]) / 1000.0 - sum(values.values())
    )
    return values


def per_run_mean(
    run: list[dict[str, object]], steps: tuple[int, ...], field: str
) -> float:
    return float(statistics.mean(float(run[step].get(field, 0)) for step in steps))


def summarize_window(
    runs: list[list[dict[str, object]]], steps: tuple[int, ...]
) -> dict[str, object]:
    wall_us = median([
        per_run_mean(run, steps, "host_wall_ns") / 1000.0 for run in runs
    ])
    module_order = list(MODULES) + ["Runtime/orchestration remainder"]
    modules = {
        name: median([
            statistics.mean(module_values(run[step])[name] for step in steps)
            for run in runs
        ])
        for name in module_order
    }
    overlap_us = {
        field: median([
            per_run_mean(run, steps, field) / TICKS_PER_US for run in runs
        ])
        for field in OVERLAP_FIELDS
    }
    counters = {
        field: median([per_run_mean(run, steps, field) for run in runs])
        for field in COUNT_FIELDS
    }
    result: dict[str, object] = {
        "host_wall_us_per_step": wall_us,
        "modules_us": modules,
        "overlap_us_per_step": overlap_us,
        "counters_per_step": counters,
    }
    if steps == (0,):
        result["tokens_per_second"] = PREFILL_TOKENS * 1_000_000.0 / wall_us
    else:
        result["tokens_per_second"] = 1_000_000.0 / wall_us
    return result


def validate(
    runs: dict[str, list[list[dict[str, object]]]]
) -> dict[str, bool]:
    correctness = True
    physical = True
    invocation = True
    lifecycle = True
    invariant = True
    expected_reconstruction = {"control": 2, "candidate": 0}
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            for step, record in enumerate(run):
                correctness &= (
                    int(record["experiment"]) == 162
                    and record["variant"] == "W4U8"
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
                    and int(record["ledger_unattributed_ticks"]) == 0
                    and record["intermediate_residency"] == "VTCM"
                    and record["qnn"] == "none"
                )
                invocation &= (
                    int(record["prepared_session_run_index"]) == step + 1
                    and int(record["block_invocation_count"]) == LAYERS
                )
                invariant &= (
                    int(record["weight_ddr_read_bytes"]) == 712_441_856
                    and int(record["w4u8_delta_reconstruction_mode"])
                    == expected_reconstruction[mode]
                )
                if mode == "candidate":
                    lifecycle &= int(record["u8_cache_full_prefix_pack_count"]) == 0
                    if step == 0:
                        lifecycle &= (
                            int(record["u8_cache_native_prefill_reuse_count"]) == LAYERS
                            and int(record["u8_cache_segment_tail_append_count"]) == 0
                            and int(record["u8_cache_segment_seal_count"]) == 0
                        )
                    else:
                        lifecycle &= (
                            int(record["u8_cache_segment_tail_append_count"]) == LAYERS
                            and int(record["u8_cache_native_incremental_append_count"]) == LAYERS
                        )
                        if step == 32:
                            lifecycle &= (
                                int(record["u8_cache_segment_seal_count"]) == LAYERS
                                and int(record["u8_cache_segment_sealed_bytes"])
                                == SEALED_BYTES
                                and int(record["scan_cache_ddr_write_bytes"])
                                == SEALED_BYTES + 57_344
                            )
                        else:
                            lifecycle &= (
                                int(record["u8_cache_segment_seal_count"]) == 0
                                and int(record["u8_cache_segment_sealed_bytes"]) == 0
                            )
                    lifecycle &= (
                        int(record["scan_attention_overlay_required_bytes"])
                        <= int(record["scan_attention_overlay_capacity_bytes"])
                    )
    for control_run, candidate_run in zip(runs["control"], runs["candidate"]):
        for control, candidate in zip(control_run, candidate_run):
            invariant &= (
                control["output_hash"] == candidate["output_hash"]
                and int(control["weight_ddr_read_bytes"])
                == int(candidate["weight_ddr_read_bytes"])
            )
    return {
        "correctness_pass": correctness,
        "physical_pass": physical,
        "one_rpc_per_step_pass": invocation,
        "cache_lifecycle_pass": lifecycle,
        "unchanged_math_and_projection_pass": invariant,
    }


def paired_speed(
    control_runs: list[list[dict[str, object]]],
    candidate_runs: list[list[dict[str, object]]],
    steps: tuple[int, ...],
) -> dict[str, float]:
    control = [per_run_mean(run, steps, "host_wall_ns") / 1000.0 for run in control_runs]
    candidate = [per_run_mean(run, steps, "host_wall_ns") / 1000.0 for run in candidate_runs]
    improvements = [(left / right - 1.0) * 100.0 for left, right in zip(control, candidate)]
    deltas = [right - left for left, right in zip(control, candidate)]
    return {
        "control_median_us": median(control),
        "candidate_median_us": median(candidate),
        "paired_median_delta_us": median(deltas),
        "paired_median_speed_improvement_percent": median(improvements),
        "paired_min_speed_improvement_percent": min(improvements),
        "paired_max_speed_improvement_percent": max(improvements),
    }


def fmt_cell(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_runs(result_dir, mode) for mode in MODES}
    gates = validate(runs)
    windows = {
        mode: {name: summarize_window(mode_runs, steps) for name, steps in WINDOWS.items()}
        for mode, mode_runs in runs.items()
    }
    speed = {
        name: paired_speed(runs["control"], runs["candidate"], steps)
        for name, steps in WINDOWS.items()
    }
    speed_gates = {
        "prefill_at_most_one_percent_regression":
            speed["prefill"]["candidate_median_us"]
            <= 1.01 * speed["prefill"]["control_median_us"],
        "early_decode_non_regression":
            speed["early_decode_64_71"]["paired_median_delta_us"] <= 0.0,
        "postseal_decode_strict_improvement":
            speed["postseal_decode_96_103"]["paired_median_delta_us"] < 0.0,
        "complete_decode_strict_improvement":
            speed["all_decode_64_103"]["paired_median_delta_us"] < 0.0,
    }
    all_gates = all(gates.values()) and all(speed_gates.values())

    old = json.loads(args.exp0158_summary.read_text(encoding="utf-8"))[
        "three_recipe_overview"
    ]
    f16 = old["F16F16_EXP0158_hmx_native_f16"]
    w4f16 = old["W4F16_EXP0158_hmx_native_f16"]
    current = windows["candidate"]
    three_recipe = {
        "F16F16_EXP0158_hmx_native_f16": f16,
        "W4F16_EXP0158_hmx_native_f16": w4f16,
        "W4U8_EXP0162_dynamic_segmented": {
            "prefill_host_wall_us": current["prefill"]["host_wall_us_per_step"],
            "prefill_modules_us": current["prefill"]["modules_us"],
            "decode_host_wall_us_per_token": current["early_decode_64_71"]["host_wall_us_per_step"],
            "decode_modules_us": current["early_decode_64_71"]["modules_us"],
        },
    }
    summary = {
        "experiment": "EXP-0162",
        "execution_unit": "real Qwen3 layers0-27 M64 prefill then decode positions64-103",
        "measurement_contract": "same_binary_10_rotated_control_candidate_sessions",
        "gates": gates,
        "speed_gates": speed_gates,
        "all_gates_pass": all_gates,
        "speed": speed,
        "windows": windows,
        "three_recipe_overview": three_recipe,
        "provenance": {
            "f16f16_and_w4f16": str(args.exp0158_summary),
            "w4u8": str(result_dir),
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# EXP-0162 full profiling report", "",
        "One binary, ten rotated complete sessions. Each session is one real 28-layer M64 prefill followed by forty stateful decode tokens at positions 64-103.", "",
        "## Direct dynamic-cache A/B", "",
        "| Window | Control | Dynamic segmented V4 | Paired speed change |",
        "|---|---:|---:|---:|",
    ]
    for name in WINDOWS:
        item = speed[name]
        lines.append(
            f"| {name} | {item['control_median_us']:.3f} us | "
            f"{item['candidate_median_us']:.3f} us | "
            f"{item['paired_median_speed_improvement_percent']:+.3f}% |"
        )

    for window_name, title in (
        ("prefill", "Prefill M64 module ledger"),
        ("early_decode_64_71", "Early decode L64-L71 module ledger"),
        ("seal_position_95", "Segment-seal position L95 module ledger"),
        ("postseal_decode_96_103", "Post-seal decode L96-L103 module ledger"),
        ("all_decode_64_103", "Complete decode L64-L103 module ledger"),
    ):
        control = windows["control"][window_name]
        candidate = windows["candidate"][window_name]
        control_wall = float(control["host_wall_us_per_step"])
        candidate_wall = float(candidate["host_wall_us_per_step"])
        lines += ["", f"## {title}", "", "| Module | Control | Candidate | Candidate speed |", "|---|---:|---:|---:|"]
        for module in list(MODULES) + ["Runtime/orchestration remainder"]:
            left = float(control["modules_us"][module])
            right = float(candidate["modules_us"][module])
            change = (left / right - 1.0) * 100.0 if right else 0.0
            lines.append(
                f"| {module} | {fmt_cell(left, control_wall)} | "
                f"{fmt_cell(right, candidate_wall)} | {change:+.1f}% |"
            )
        lines.append(
            f"| Complete Host wall | {control_wall:.3f} us | "
            f"{candidate_wall:.3f} us | "
            f"{(control_wall / candidate_wall - 1.0) * 100.0:+.3f}% |"
        )

    lines += ["", "## Decode overlap and physical counters", "", "| Counter (per decode token) | Control | Candidate |", "|---|---:|---:|"]
    cdec = windows["control"]["all_decode_64_103"]
    ndec = windows["candidate"]["all_decode_64_103"]
    for field in OVERLAP_FIELDS:
        lines.append(
            f"| {field} | {float(cdec['overlap_us_per_step'][field]):.3f} us | "
            f"{float(ndec['overlap_us_per_step'][field]):.3f} us |"
        )
    for field in COUNT_FIELDS:
        lines.append(
            f"| {field} | {float(cdec['counters_per_step'][field]):.3f} | "
            f"{float(ndec['counters_per_step'][field]):.3f} |"
        )

    module_order = list(MODULES) + ["Runtime/orchestration remainder"]
    lines += ["", "## Three-recipe prefill overview", "", "F16F16 and W4F16 reuse EXP-0158; W4U8 is this experiment's candidate.", "", "| Module | F16F16 | W4F16 | W4U8 EXP-0162 | W4U8 speed vs W4F16 |", "|---|---:|---:|---:|---:|"]
    for module in module_order:
        left = float(f16["prefill_modules_us"][module])
        middle = float(w4f16["prefill_modules_us"][module])
        right = float(current["prefill"]["modules_us"][module])
        lines.append(
            f"| {module} | {fmt_cell(left, float(f16['prefill_host_wall_us']))} | "
            f"{fmt_cell(middle, float(w4f16['prefill_host_wall_us']))} | "
            f"{fmt_cell(right, float(current['prefill']['host_wall_us_per_step']))} | "
            f"{(middle / right - 1.0) * 100.0:+.1f}% |"
        )
    lines.append(
        f"| Complete Host wall | {float(f16['prefill_host_wall_us']):.3f} us | "
        f"{float(w4f16['prefill_host_wall_us']):.3f} us | "
        f"{float(current['prefill']['host_wall_us_per_step']):.3f} us | "
        f"{(float(w4f16['prefill_host_wall_us']) / float(current['prefill']['host_wall_us_per_step']) - 1.0) * 100.0:+.1f}% |"
    )

    lines += ["", "## Three-recipe early-decode overview", "", "| Module | F16F16 | W4F16 | W4U8 EXP-0162 | W4U8 speed vs W4F16 |", "|---|---:|---:|---:|---:|"]
    for module in module_order:
        left = float(f16["decode_modules_us"][module])
        middle = float(w4f16["decode_modules_us"][module])
        right = float(current["early_decode_64_71"]["modules_us"][module])
        lines.append(
            f"| {module} | {fmt_cell(left, float(f16['decode_host_wall_us_per_token']))} | "
            f"{fmt_cell(middle, float(w4f16['decode_host_wall_us_per_token']))} | "
            f"{fmt_cell(right, float(current['early_decode_64_71']['host_wall_us_per_step']))} | "
            f"{(middle / right - 1.0) * 100.0:+.1f}% |"
        )
    lines.append(
        f"| Complete Host wall | {float(f16['decode_host_wall_us_per_token']):.3f} us | "
        f"{float(w4f16['decode_host_wall_us_per_token']):.3f} us | "
        f"{float(current['early_decode_64_71']['host_wall_us_per_step']):.3f} us | "
        f"{(float(w4f16['decode_host_wall_us_per_token']) / float(current['early_decode_64_71']['host_wall_us_per_step']) - 1.0) * 100.0:+.1f}% |"
    )

    lines += ["", "## Gates", ""]
    for name, passed in gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    for name, passed in speed_gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines += [
        f"- Overall gate: {'PASS' if all_gates else 'FAIL'}", "",
        "The one-RPC-per-step contract makes an internal repeat10 run inapplicable. Its formal replacement is ten complete, order-rotated stateful sessions; all additive and overlapping counters are retained per step.",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result_dir": str(result_dir),
        "all_gates_pass": all_gates,
        "speed": speed,
        **gates,
        **speed_gates,
    }, indent=2, sort_keys=True))
    if not all_gates:
        raise SystemExit("EXP-0162 gate failed")


if __name__ == "__main__":
    main()
