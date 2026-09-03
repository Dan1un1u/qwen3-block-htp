#!/usr/bin/env python3
"""Validate and summarize EXP-0163 six-seal full-stack W4U8 A/B."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import summarize_exp0162 as base


EXPERIMENT = 163
LAYERS = 28
PREFILL = 64
DECODE_STEPS = 192
STEPS = tuple(range(DECODE_STEPS + 1))
SEAL_STEPS = (32, 64, 96, 128, 160, 192)
SEAL_POSITIONS = tuple(PREFILL + step - 1 for step in SEAL_STEPS)
SEALED_BYTES = 1_892_352
APPEND_BYTES = 57_344

WINDOWS = {
    "prefill": (0,),
    "early_decode_64_71": tuple(range(1, 9)),
    "segment_64_95": tuple(range(1, 33)),
    "segment_96_127": tuple(range(33, 65)),
    "segment_128_159": tuple(range(65, 97)),
    "segment_160_191": tuple(range(97, 129)),
    "segment_192_223": tuple(range(129, 161)),
    "tail_224_255": tuple(range(161, 193)),
    "all_decode_64_255": tuple(range(1, 193)),
    **{
        f"seal_position_{position}": (step,)
        for step, position in zip(SEAL_STEPS, SEAL_POSITIONS)
    },
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
                    int(record["experiment"]) == EXPERIMENT
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
                if mode != "candidate":
                    continue
                lifecycle &= int(record["u8_cache_full_prefix_pack_count"]) == 0
                expected_valid = PREFILL + step
                expected_padded = int(math.ceil(expected_valid / 32.0) * 32)
                lifecycle &= (
                    int(record["scan_total_kv_length"]) == expected_valid
                    and int(record["scan_padded_kv_length"]) == expected_padded
                )
                if step == 0:
                    lifecycle &= (
                        int(record["u8_cache_native_prefill_reuse_count"])
                        == LAYERS
                        and int(record["u8_cache_segment_tail_append_count"])
                        == 0
                        and int(record["u8_cache_segment_seal_count"]) == 0
                    )
                else:
                    lifecycle &= (
                        int(record["u8_cache_segment_tail_append_count"])
                        == LAYERS
                        and int(record[
                            "u8_cache_native_incremental_append_count"
                        ]) == LAYERS
                    )
                    if step in SEAL_STEPS:
                        lifecycle &= (
                            int(record["u8_cache_segment_seal_count"])
                            == LAYERS
                            and int(record["u8_cache_segment_sealed_bytes"])
                            == SEALED_BYTES
                            and int(record["scan_cache_ddr_write_bytes"])
                            == SEALED_BYTES + APPEND_BYTES
                        )
                    else:
                        lifecycle &= (
                            int(record["u8_cache_segment_seal_count"]) == 0
                            and int(record["u8_cache_segment_sealed_bytes"]) == 0
                            and int(record["scan_cache_ddr_write_bytes"])
                            == APPEND_BYTES
                        )
                    lifecycle &= (
                        int(record["scan_attention_overlay_required_bytes"])
                        <= int(record[
                            "scan_attention_overlay_capacity_bytes"
                        ])
                    )
    for control_run, candidate_run in zip(
        runs["control"], runs["candidate"]
    ):
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
        "six_seal_lifecycle_pass": lifecycle,
        "unchanged_math_and_projection_pass": invariant,
    }


def fmt_cell(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    base.STEPS = STEPS
    runs = {
        mode: base.load_runs(result_dir, mode) for mode in base.MODES
    }
    gates = validate(runs)
    windows = {
        mode: {
            name: base.summarize_window(mode_runs, steps)
            for name, steps in WINDOWS.items()
        }
        for mode, mode_runs in runs.items()
    }
    speed = {
        name: base.paired_speed(
            runs["control"], runs["candidate"], steps
        )
        for name, steps in WINDOWS.items()
    }
    speed_gates = {
        "prefill_at_most_one_percent_regression":
            speed["prefill"]["candidate_median_us"]
            <= 1.01 * speed["prefill"]["control_median_us"],
        "early_decode_non_regression":
            speed["early_decode_64_71"]["paired_median_delta_us"] <= 0.0,
        "tail_224_255_strict_improvement":
            speed["tail_224_255"]["paired_median_delta_us"] < 0.0,
        "complete_192_token_decode_strict_improvement":
            speed["all_decode_64_255"]["paired_median_delta_us"] < 0.0,
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
        "W4U8_EXP0163_multiseal": {
            "prefill_host_wall_us": current["prefill"][
                "host_wall_us_per_step"
            ],
            "prefill_modules_us": current["prefill"]["modules_us"],
            "early_decode_host_wall_us_per_token": current[
                "early_decode_64_71"
            ]["host_wall_us_per_step"],
            "early_decode_modules_us": current[
                "early_decode_64_71"
            ]["modules_us"],
            "complete_decode_host_wall_us_per_token": current[
                "all_decode_64_255"
            ]["host_wall_us_per_step"],
            "complete_decode_modules_us": current[
                "all_decode_64_255"
            ]["modules_us"],
        },
    }
    summary = {
        "experiment": "EXP-0163",
        "execution_unit": (
            "real Qwen3 layers0-27 M64 prefill then decode positions64-255"
        ),
        "measurement_contract": (
            "same_binary_10_rotated_control_candidate_sessions"
        ),
        "seal_positions": list(SEAL_POSITIONS),
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0163 full profiling report", "",
        "Ten order-rotated complete sessions. Each session is one real "
        "28-layer M64 prefill followed by 192 stateful decode tokens at "
        "positions 64-255, crossing six dynamic segment seals.", "",
        "## Direct dynamic-cache A/B", "",
        "| Window | Monolithic delta control | Dynamic segmented | Paired speed change |",
        "|---|---:|---:|---:|",
    ]
    ordered = (
        "prefill", "early_decode_64_71", "segment_64_95",
        "segment_96_127", "segment_128_159", "segment_160_191",
        "segment_192_223", "tail_224_255", "all_decode_64_255",
        *(f"seal_position_{position}" for position in SEAL_POSITIONS),
    )
    for name in ordered:
        item = speed[name]
        lines.append(
            f"| {name} | {item['control_median_us']:.3f} us | "
            f"{item['candidate_median_us']:.3f} us | "
            f"{item['paired_median_speed_improvement_percent']:+.3f}% |"
        )

    for window_name, title in (
        ("prefill", "Prefill M64 module ledger"),
        ("early_decode_64_71", "Early decode L64-L71 module ledger"),
        ("tail_224_255", "Long-tail decode L224-L255 module ledger"),
        ("all_decode_64_255", "Complete decode L64-L255 module ledger"),
    ):
        control = windows["control"][window_name]
        candidate = windows["candidate"][window_name]
        control_wall = float(control["host_wall_us_per_step"])
        candidate_wall = float(candidate["host_wall_us_per_step"])
        lines += [
            "", f"## {title}", "",
            "| Module | Control | Candidate | Candidate speed |",
            "|---|---:|---:|---:|",
        ]
        for module in list(base.MODULES) + [
            "Runtime/orchestration remainder"
        ]:
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

    lines += ["", "## Gates", ""]
    for name, passed in gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    for name, passed in speed_gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines += [
        f"- Overall gate: {'PASS' if all_gates else 'FAIL'}", "",
        "Seal latency is reported rather than required to win individually; "
        "the complete 192-token and final 32-token windows are the speed gates.",
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
        raise SystemExit("EXP-0163 gate failed")


if __name__ == "__main__":
    main()
