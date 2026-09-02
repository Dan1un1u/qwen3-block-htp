#!/usr/bin/env python3
"""Validate and summarize EXP-0159 W4U8 delta-journal cache."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TICKS_PER_US = 19.2
LAYERS = 28
PREFILL_TOKENS = 64
MODES = ("control", "candidate")
CONTROL_PREFILL_WRITE = 5_906_432
CANDIDATE_PREFILL_WRITE = 4_014_080
CONTROL_DECODE_READ = 7_798_784
CONTROL_DECODE_WRITE = 1_892_352
CANDIDATE_DECODE_WRITE = 57_344
CANDIDATE_BASE_READ = 4_014_080
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
    if len(records) != 9:
        raise ValueError(f"incomplete replay in {path}: {len(records)}")
    if [int(item["replay_step"]) for item in records] != list(range(9)):
        raise ValueError(f"unordered replay in {path}")
    if not sequence_pass:
        raise ValueError(f"failed replay sequence in {path}")
    return records


def load_runs(result_dir: Path, mode: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"round_*_{mode}.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected 10 {mode} runs, got {len(paths)}")
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
    return {
        "prefill_host_wall_us": prefill_wall,
        "prefill_tokens_per_second": PREFILL_TOKENS * 1_000_000.0 / prefill_wall,
        "decode_host_wall_us_per_token": decode_wall,
        "decode_tokens_per_second": 1_000_000.0 / decode_wall,
        "prefill_modules_us": {
            name: median([stage_modules(run[0])[name] for run in runs])
            for name in order
        },
        "decode_modules_us": {
            name: median([
                statistics.mean(
                    stage_modules(record)[name] for record in run[1:]
                )
                for run in runs
            ])
            for name in order
        },
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
    if candidate_us == 0.0:
        return 0.0 if control_us == 0.0 else float("inf")
    return (control_us / candidate_us - 1.0) * 100.0


def fmt_cell(value: float, total: float) -> str:
    return f"{value:.3f} us ({100.0 * value / total:.1f}%)"


def validate(
    runs: dict[str, list[list[dict[str, object]]]]
) -> tuple[bool, bool, bool, bool]:
    correctness = True
    physical = True
    structural = True
    invariant = True
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            for record in run:
                step = int(record["replay_step"])
                correctness &= (
                    int(record["experiment"]) == 159
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
                    and record["intermediate_residency"] == "VTCM"
                    and record["qnn"] == "none"
                    and int(record["block_invocation_count"]) == LAYERS
                    and int(record["ledger_unattributed_ticks"]) == 0
                )
                if record["mode"] == "prefill":
                    expected_write = (
                        CONTROL_PREFILL_WRITE if mode == "control"
                        else CANDIDATE_PREFILL_WRITE
                    )
                    structural &= (
                        int(record["u8_cache_native_prefill_reuse_count"])
                        == LAYERS
                        and int(record[
                            "u8_cache_native_prefill_reused_carrier_bytes"
                        ]) == CANDIDATE_PREFILL_WRITE
                        and int(record[
                            "u8_cache_native_incremental_append_count"
                        ]) == 0
                        and int(record["u8_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_ddr_read_bytes"]) == 0
                        and int(record["scan_cache_ddr_write_bytes"])
                        == expected_write
                        and int(record["scan_cache_dma_descriptor_count"])
                        == 32 * LAYERS
                    )
                elif mode == "control":
                    structural &= (
                        int(record[
                            "u8_cache_native_incremental_append_count"
                        ]) == LAYERS
                        and int(record["u8_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_ddr_read_bytes"])
                        == CONTROL_DECODE_READ
                        and int(record["scan_cache_ddr_write_bytes"])
                        == CONTROL_DECODE_WRITE
                        and int(record["scan_cache_dma_descriptor_count"])
                        == 128 * LAYERS
                    )
                else:
                    structural &= (
                        int(record[
                            "u8_cache_native_incremental_append_count"
                        ]) == LAYERS
                        and int(record["u8_cache_full_prefix_pack_count"]) == 0
                        and int(record["scan_cache_ddr_read_bytes"])
                        == CANDIDATE_BASE_READ + CANDIDATE_DECODE_WRITE * step
                        and int(record["scan_cache_ddr_write_bytes"])
                        == CANDIDATE_DECODE_WRITE
                        and int(record["scan_cache_dma_descriptor_count"])
                        == 64 * LAYERS
                    )
    # Cross-mode invariants are checked by aligned round and replay step.
    for control_run, candidate_run in zip(runs["control"], runs["candidate"]):
        for control_record, candidate_record in zip(control_run, candidate_run):
            invariant &= (
                int(control_record["hmx_command_count"])
                == int(candidate_record["hmx_command_count"])
                and int(control_record["hmx_u8s8_tile_pair_count"])
                == int(candidate_record["hmx_u8s8_tile_pair_count"])
                and int(control_record["weight_ddr_read_bytes"])
                == int(candidate_record["weight_ddr_read_bytes"])
            )
    return correctness, physical, structural, invariant


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_runs(result_dir, mode) for mode in MODES}
    correctness, physical, structural, invariant = validate(runs)
    summaries = {mode: summarize_runs(value) for mode, value in runs.items()}

    counter_fields = (
        "scan_cache_pack_ticks", "scan_cache_append_ticks",
        "scan_cache_stage_ticks", "scan_cache_dma_descriptor_count",
        "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
        "u8_attention_k_pack_ticks", "u8_attention_v_pack_ticks",
        "u8_cache_native_append_update_ticks",
        "u8_cache_native_prefill_reuse_count",
        "u8_cache_native_prefill_reused_carrier_bytes",
        "u8_cache_native_incremental_append_count",
        "u8_cache_full_prefix_pack_count", "hmx_compute_ticks",
        "hmx_command_count", "hmx_u8s8_tile_pair_count",
        "vtcm_peak_plan_bytes", "weight_ddr_read_bytes",
        "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    )
    for mode, mode_runs in runs.items():
        summaries[mode]["prefill_counters"] = {
            field: median_prefill_field(mode_runs, field)
            for field in counter_fields
        }
        summaries[mode]["decode_counters"] = {
            field: median_decode_field(mode_runs, field)
            for field in counter_fields
        }

    control = summaries["control"]
    candidate = summaries["candidate"]
    prefill_control = float(control["prefill_host_wall_us"])
    prefill_candidate = float(candidate["prefill_host_wall_us"])
    decode_control = float(control["decode_host_wall_us_per_token"])
    decode_candidate = float(candidate["decode_host_wall_us_per_token"])
    speed = {
        "prefill_latency_change_percent":
            (prefill_candidate / prefill_control - 1.0) * 100.0,
        "prefill_speed_improvement_percent":
            pct_speed(prefill_control, prefill_candidate),
        "prefill_gate_pass": prefill_candidate <= 1.01 * prefill_control,
        "decode_latency_change_percent":
            (decode_candidate / decode_control - 1.0) * 100.0,
        "decode_speed_improvement_percent":
            pct_speed(decode_control, decode_candidate),
        "decode_gate_pass": decode_candidate < decode_control,
    }

    exp0158 = json.loads(args.exp0158_summary.read_text(encoding="utf-8"))
    latest = {
        "F16F16_EXP0158_hmx_native_f16":
            exp0158["variants"]["f16f16_hmx_native_f16"],
        "W4F16_EXP0158_hmx_native_f16":
            exp0158["variants"]["w4f16_hmx_native_f16"],
        "W4U8_EXP0159_delta_journal": candidate,
    }
    all_gates = (
        correctness and physical and structural and invariant
        and bool(speed["prefill_gate_pass"])
        and bool(speed["decode_gate_pass"])
    )
    summary = {
        "experiment": "EXP-0159",
        "execution_unit": (
            "real Qwen3 layers0-27 M64 prefill then decode positions64-71"
        ),
        "measurement_contract": (
            "same_binary_10_rotated_control_candidate_replay_runs"
        ),
        "control_cache": "capacity96 persistent HMX tile DDR RMW",
        "candidate_cache": (
            "compact immutable M64 HMX carrier plus contiguous U8 delta journal"
        ),
        "correctness_pass": correctness,
        "physical_pass": physical,
        "structural_pass": structural,
        "unchanged_hmx_and_projection_contract_pass": invariant,
        "speed_gates": speed,
        "all_gates_pass": all_gates,
        "modes": summaries,
        "latest_three_recipe_overview": latest,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0159 full profiling report", "",
        "The direct A/B uses one binary and ten rotated complete replays. "
        "Each replay contains one real 28-layer M64 prefill followed by "
        "eight stateful decode tokens at positions 64-71.", "",
        "## Direct cache A/B", "",
        "| Cache contract | Prefill M64 | Prefill tok/s | Decode/token | Decode tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        data = summaries[mode]
        lines.append(
            f"| {mode} | {float(data['prefill_host_wall_us']):.3f} us | "
            f"{float(data['prefill_tokens_per_second']):.3f} | "
            f"{float(data['decode_host_wall_us_per_token']):.3f} us | "
            f"{float(data['decode_tokens_per_second']):.3f} |"
        )
    lines += [
        "",
        f"Candidate prefill speed change: "
        f"{float(speed['prefill_speed_improvement_percent']):+.3f}%.",
        f"Candidate decode speed change: "
        f"{float(speed['decode_speed_improvement_percent']):+.3f}%.",
    ]

    order = list(MODULES) + ["Runtime/orchestration remainder"]
    for stage, title in (("prefill", "Prefill"), ("decode", "Decode")):
        wall_key = (
            "prefill_host_wall_us" if stage == "prefill"
            else "decode_host_wall_us_per_token"
        )
        modules_key = f"{stage}_modules_us"
        lines += [
            "", f"## {title} module ledger", "",
            "| Module | Control | Candidate | Candidate speed |",
            "|---|---:|---:|---:|",
        ]
        for module in order:
            control_us = float(control[modules_key][module])
            candidate_us = float(candidate[modules_key][module])
            lines.append(
                f"| {module} | {fmt_cell(control_us, float(control[wall_key]))} | "
                f"{fmt_cell(candidate_us, float(candidate[wall_key]))} | "
                f"{pct_speed(control_us, candidate_us):+.1f}% |"
            )
        lines.append(
            f"| Complete Host wall | {float(control[wall_key]):.3f} us | "
            f"{float(candidate[wall_key]):.3f} us | "
            f"{pct_speed(float(control[wall_key]), float(candidate[wall_key])):+.1f}% |"
        )

    lines += [
        "", "## Validation", "",
        f"- Exact output/cache correctness: {'PASS' if correctness else 'FAIL'}",
        f"- Physical residency contract: {'PASS' if physical else 'FAIL'}",
        f"- Zero-RMW journal contract: {'PASS' if structural else 'FAIL'}",
        f"- Unchanged HMX/projection work: {'PASS' if invariant else 'FAIL'}",
        f"- Prefill <= 1% regression: {'PASS' if speed['prefill_gate_pass'] else 'FAIL'}",
        f"- Decode strictly faster: {'PASS' if speed['decode_gate_pass'] else 'FAIL'}",
        f"- Overall gate: {'PASS' if all_gates else 'FAIL'}", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not all_gates:
        raise SystemExit("EXP-0159 validation gate failed")


if __name__ == "__main__":
    main()
