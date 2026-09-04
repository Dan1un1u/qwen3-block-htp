#!/usr/bin/env python3
"""Validate the EXP-0174 three-cell continuous-decode short gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import summarize_exp0164 as base  # noqa: E402


STEPS = 17
LAYERS = 28
CELLS = ("post_batch8", "inline_batch8", "inline_batch16")
MODES = {"post_batch8": 0, "inline_batch8": 1, "inline_batch16": 2}
BATCHES = {"post_batch8": 8, "inline_batch8": 8, "inline_batch16": 16}
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
DIAGNOSTICS = (
    "w4u8_mlp_gate_up_pipeline_ticks",
    "w4u8_mlp_activation_work_ticks",
    "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0174-w4u8-decode-gate-up-inline-publish",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cell(result_dir: Path, cell: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{cell}.log"))
    if len(paths) != 5:
        raise ValueError(f"expected five {cell} logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        records = base.read_json_lines(path)
        steps = [item for item in records
                 if "generation_step" in item and "record" not in item]
        profiles = [item for item in records
                    if item.get("record") == "generation_profile"]
        finals = [item for item in records
                  if item.get("generation_sequence_complete") is True]
        if len(steps) != STEPS or len(profiles) != STEPS or len(finals) != 1:
            raise ValueError(f"incomplete {cell} run: {path}")
        if not (finals[0].get("all_steps_pass") is True
                and int(finals[0].get("requested_steps", -1)) == STEPS
                and int(finals[0].get("completed_steps", -1)) == STEPS):
            raise ValueError(f"failed final record: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if not (int(step.get("experiment", -1)) == 174
                    and int(profile.get("experiment", -1)) == 174
                    and int(step.get("generation_step", -1)) == index
                    and int(profile.get("generation_step", -1)) == index
                    and int(step.get("generation_mode", -1)) == 9):
                raise ValueError(f"identity mismatch: {path}:{index}")
            profile["_step_record"] = step
            profile["_source_log"] = str(path)
        runs.append(profiles)
    return runs


def run_mean(run: list[dict[str, object]], indices: tuple[int, ...],
             getter: Callable[[dict[str, object]], float]) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_metric(runs: list[list[dict[str, object]]],
                  indices: tuple[int, ...],
                  getter: Callable[[dict[str, object]], float]) -> float:
    return float(statistics.median(
        run_mean(run, indices, getter) for run in runs))


def tick_us(record: dict[str, object], field: str) -> float:
    return float(record[field]) / base.TICKS_PER_US


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    return f"{(control / candidate - 1.0) * 100.0:+.2f}%"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {cell: load_cell(result_dir, cell) for cell in CELLS}
    prefill_indices = (0,)
    decode_indices = tuple(range(1, STEPS))

    rows = {
        cell: {
            "prefill": {
                name: median_metric(
                    runs[cell], prefill_indices,
                    lambda record, name=name: base.generation_row_us(
                        record, name),
                ) for name in ROWS
            },
            "decode": {
                name: median_metric(
                    runs[cell], decode_indices,
                    lambda record, name=name: base.generation_row_us(
                        record, name),
                ) for name in ROWS
            },
        } for cell in CELLS
    }
    diagnostics = {
        cell: {
            field.replace("_ticks", "_us"): median_metric(
                runs[cell], decode_indices,
                lambda record, field=field: tick_us(record, field),
            ) for field in DIAGNOSTICS
        } for cell in CELLS
    }

    physical = True
    schedule = True
    cache = True
    sequences: list[tuple[int, ...]] = []
    hashes: list[tuple[str, ...]] = []
    for cell, cell_runs in runs.items():
        for run in cell_runs:
            sequences.append(tuple(
                int(profile["_step_record"]["selected_token_id"])
                for profile in run))
            hashes.append(tuple(str(profile["output_hash"]) for profile in run))
            for index, profile in enumerate(run):
                step = profile["_step_record"]
                expected_before = 0 if index == 0 else 63 + index
                expected_after = 64 if index == 0 else 64 + index
                expected_inline = index > 0 and cell != "post_batch8"
                expected_release = 10752 if expected_inline else 0
                expected_publish = 5376 if expected_inline else 0
                expected_batch = 8 if index == 0 else BATCHES[cell]
                schedule &= bool(
                    int(profile["w4u8_decode_gate_up_mode"]) == MODES[cell]
                    and int(profile["w4u8_mlp_gate_up_hmx_batch_n_tiles"])
                        == expected_batch
                    and int(profile[
                        "w4u8_mlp_gate_up_inline_slot_release_count"])
                        == expected_release
                    and int(profile[
                        "w4u8_mlp_gate_up_inline_pair_publish_count"])
                        == expected_publish
                )
                physical &= bool(
                    int(step["rpc_result"]) == 0 and bool(step["pass"])
                    and profile["backend"] == "standalone_fastrpc_dsp"
                    and profile["qnn"] == "none"
                    and int(profile["block_invocation_count"]) == LAYERS
                    and int(profile["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(profile["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(profile["intermediate_ddr_read_bytes"]) == 0
                    and int(profile["intermediate_ddr_write_bytes"]) == 0
                    and int(profile["intermediate_spill_fill_count"]) == 0
                    and int(profile["boundary_ddr_write_bytes"]) == 0
                    and int(profile["ledger_unattributed_ticks"]) == 0
                )
                cache &= bool(
                    int(profile["first_position"]) == expected_before
                    and int(profile["valid_length"]) == expected_after
                    and int(profile["cache_mismatches"]) == 0
                    and int(profile["cache_structure_mismatches"]) == 0
                )

    control_wall = rows["post_batch8"]["decode"]["Complete Host wall"]
    control_gate = rows["post_batch8"]["decode"]["Gate/Up projection"]
    eligible: dict[str, bool] = {}
    direct: dict[str, dict[str, float]] = {}
    for cell in CELLS:
        prefill_wall = rows[cell]["prefill"]["Complete Host wall"]
        decode_wall = rows[cell]["decode"]["Complete Host wall"]
        gate_wall = rows[cell]["decode"]["Gate/Up projection"]
        direct[cell] = {
            "prefill_wall_us": prefill_wall,
            "prefill_tok_s": 64e6 / prefill_wall,
            "decode_wall_us_per_token": decode_wall,
            "decode_tok_s": 1e6 / decode_wall,
            "decode_speed_percent_vs_control":
                (control_wall / decode_wall - 1.0) * 100.0,
            "gate_up_speed_percent_vs_control":
                (control_gate / gate_wall - 1.0) * 100.0,
        }
        eligible[cell] = bool(
            cell != "post_batch8"
            and decode_wall < control_wall and gate_wall < control_gate)

    gates = {
        "all_15_sessions_physical_contract": physical,
        "requested_and_observed_inline_geometry": schedule,
        "all_cache_lengths_synchronous": cache,
        "all_15_sessions_byte_exact_output_hashes": len(set(hashes)) == 1,
        "all_15_sessions_identical_token_sequence": len(set(sequences)) == 1,
        "any_inline_cell_beats_control_gate_up_and_complete_decode_wall":
            any(eligible.values()),
    }
    selected = min(
        CELLS, key=lambda cell: direct[cell]["decode_wall_us_per_token"])
    conclusion = (
        "advance_to_formal" if any(eligible.values())
        else "reject_inline_publication_without_193_step_formal_run"
    )
    summary = {
        "experiment": "EXP-0174",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "short_evidence": str(result_dir),
        "sessions_per_cell": 5,
        "continuous_decode_tokens_per_session": STEPS - 1,
        "gates": gates,
        "eligible": eligible,
        "best_observed_cell": selected,
        "conclusion": conclusion,
        "direct": direct,
        "module_rows_us": rows,
        "diagnostics_us": diagnostics,
        "quality_gate": "disabled_by_contract",
        "formal_193_step_sections": "not_run_by_approved_short_gate_stop_condition",
        "provenance": {
            "static_gate_sha256": sha256_file(
                result_dir / "static_gate.json"),
            "logs": {path.name: sha256_file(path) for path in sorted(
                (result_dir / "raw").glob("*.log"))},
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")

    lines = [
        "# EXP-0174 decode Gate/Up inline pair publication short gate", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "The approved stop condition skips the 193-step formal run when no "
        "inline cell improves both Gate/Up and complete decode wall.", "",
        "## Direct full-stack short-gate result", "",
        "| Cell | Prefill tok/s | Decode wall/token | Decode tok/s | "
        "Decode speed vs control | Gate/Up speed vs control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
            f"{item['decode_speed_percent_vs_control']:+.3f}% | "
            f"{item['gate_up_speed_percent_vs_control']:+.3f}% |")
    lines += ["", "## Complete decode module table", "",
              "| Module | post_batch8 | inline_batch8 | inline_batch16 |",
              "|---|---:|---:|---:|"]
    walls = {cell: rows[cell]["decode"]["Complete Host wall"] for cell in CELLS}
    for name in ROWS:
        lines.append(
            f"| {name} | {fmt(rows['post_batch8']['decode'][name], walls['post_batch8'])} | "
            f"{fmt(rows['inline_batch8']['decode'][name], walls['inline_batch8'])} | "
            f"{fmt(rows['inline_batch16']['decode'][name], walls['inline_batch16'])} |")
    lines += ["", "## Gate/Up overlap diagnostics", "",
              "| Metric | post_batch8 | inline_batch8 | inline_batch16 |",
              "|---|---:|---:|---:|"]
    for field in DIAGNOSTICS:
        name = field.replace("_ticks", "_us")
        lines.append(
            f"| {name} | {diagnostics['post_batch8'][name]:.3f} | "
            f"{diagnostics['inline_batch8'][name]:.3f} | "
            f"{diagnostics['inline_batch16'][name]:.3f} |")
    lines += ["", "## Gates", ""]
    for name, value in gates.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines += ["", "## Conclusion", "", conclusion, ""]
    (result_dir / "short_gate_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "experiment": "EXP-0174",
        "best_observed_cell": selected,
        "conclusion": conclusion,
        "direct": direct,
        "gates": gates,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
