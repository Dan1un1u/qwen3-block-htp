#!/usr/bin/env python3
"""Validate the EXP-0178 64-row versus direct row-four common-op gate."""

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
CELLS = ("full", "row4")
ROWS_PER_CELL = {"full": 64, "row4": 4, "row4_poison": 4}
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
TARGET_ROWS = (
    "Input RMSNorm",
    "Post-attention residual + RMSNorm",
    "Final residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0178-w4u8-decode-row4-common-ops",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path, expected_steps: int = STEPS) -> list[dict[str, object]]:
    records = base.read_json_lines(path)
    steps = [item for item in records
             if "generation_step" in item and "record" not in item]
    profiles = [item for item in records
                if item.get("record") == "generation_profile"]
    finals = [item for item in records
              if item.get("generation_sequence_complete") is True]
    if (len(steps) != expected_steps or len(profiles) != expected_steps
            or len(finals) != 1):
        raise ValueError(f"incomplete run: {path}")
    if not (finals[0].get("all_steps_pass") is True
            and int(finals[0].get("requested_steps", -1)) == expected_steps
            and int(finals[0].get("completed_steps", -1)) == expected_steps):
        raise ValueError(f"failed final record: {path}")
    for index, (step, profile) in enumerate(zip(steps, profiles)):
        if not (int(step.get("experiment", -1)) == 178
                and int(profile.get("experiment", -1)) == 178
                and int(step.get("generation_step", -1)) == index
                and int(profile.get("generation_step", -1)) == index
                and int(step.get("generation_mode", -1)) == 9):
            raise ValueError(f"identity mismatch: {path}:{index}")
        profile["_step_record"] = step
        profile["_source_log"] = str(path)
    return profiles


def load_cell(result_dir: Path, cell: str,
              rounds: int = 5) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{cell}.log"))
    if len(paths) != rounds:
        raise ValueError(f"expected {rounds} {cell} logs, got {len(paths)}")
    return [load_run(path) for path in paths]


def run_mean(run: list[dict[str, object]], indices: tuple[int, ...],
             getter: Callable[[dict[str, object]], float]) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_metric(runs: list[list[dict[str, object]]],
                  indices: tuple[int, ...],
                  getter: Callable[[dict[str, object]], float]) -> float:
    return float(statistics.median(
        run_mean(run, indices, getter) for run in runs))


def sequence(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_token_id"])
                 for item in run)


def logit_codes(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_logit_half_bits"])
                 for item in run)


def output_hashes(run: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(item["output_hash"]) for item in run)


def validate_physical(run: list[dict[str, object]]) -> bool:
    for index, profile in enumerate(run):
        step = profile["_step_record"]
        expected_before = 0 if index == 0 else 63 + index
        expected_after = 64 if index == 0 else 64 + index
        if not (
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
            and int(profile["first_position"]) == expected_before
            and int(profile["valid_length"]) == expected_after
            and int(profile["cache_mismatches"]) == 0
            and int(profile["cache_structure_mismatches"]) == 0
        ):
            return False
    return True


def validate_dispatch(run: list[dict[str, object]], cell: str) -> bool:
    requested = ROWS_PER_CELL[cell]
    poison = 1 if cell == "row4_poison" else 0
    for index, profile in enumerate(run):
        if int(profile["w4u8_decode_common_op_rows"]) != requested:
            return False
        if int(profile["w4u8_decode_common_padding_poison"]) != poison:
            return False
        if index == 0:
            if not (
                int(profile["w4u8_common_op_rows_observed"]) == 64
                and int(profile["w4u8_input_norm_direct_row4_call_count"]) == 0
                and int(profile["w4u8_post_residual_direct_row4_call_count"]) == 0
                and int(profile["w4u8_final_residual_direct_row4_call_count"]) == 0
                and int(profile["w4u8_common_padding_poison_count"]) == 0
            ):
                return False
        elif requested == 64:
            if not (
                int(profile["w4u8_common_op_rows_observed"]) == 64
                and int(profile["w4u8_input_norm_direct_row4_call_count"]) == 0
                and int(profile["w4u8_post_residual_direct_row4_call_count"]) == 0
                and int(profile["w4u8_final_residual_direct_row4_call_count"]) == 0
                and int(profile["w4u8_common_padding_poison_count"]) == 0
            ):
                return False
        elif not (
            int(profile["w4u8_common_op_rows_observed"]) == 4
            and int(profile["w4u8_input_norm_direct_row4_call_count"]) == LAYERS
            and int(profile["w4u8_post_residual_direct_row4_call_count"]) == LAYERS
            and int(profile["w4u8_final_residual_direct_row4_call_count"]) == LAYERS
            and int(profile["w4u8_common_padding_poison_count"])
                == poison * LAYERS
            and int(profile["w4u8_input_norm_task_count"]) == 0
            and int(profile["w4u8_post_residual_task_count"]) == 0
            and int(profile["w4u8_final_residual_task_count"]) == 0
        ):
            return False
    return True


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    if candidate == 0.0:
        return "+0.00%" if control == 0.0 else "removed/fused"
    return f"{(control / candidate - 1.0) * 100.0:+.2f}%"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {cell: load_cell(result_dir, cell) for cell in CELLS}
    poison = load_run(result_dir / "raw" / "padding_poison.log")
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
    reference = runs["full"][0]
    all_runs = [run for cell in CELLS for run in runs[cell]] + [poison]
    physical = all(validate_physical(run) for run in all_runs)
    dispatch = all(validate_dispatch(run, cell)
                   for cell in CELLS for run in runs[cell])
    poison_dispatch = validate_dispatch(poison, "row4_poison")
    exact_sequences = all(sequence(run) == sequence(reference)
                          for run in all_runs)
    exact_logits = all(logit_codes(run) == logit_codes(reference)
                       for run in all_runs)
    exact_hashes = all(output_hashes(run) == output_hashes(reference)
                       for run in all_runs)
    invariant_hmx = all(
        tuple(int(item["hmx_u8s8_tile_pair_count"]) for item in run)
        == tuple(int(item["hmx_u8s8_tile_pair_count"])
                 for item in reference)
        for run in all_runs)

    walls = {cell: rows[cell]["decode"]["Complete Host wall"]
             for cell in CELLS}
    target = {
        cell: sum(rows[cell]["decode"][name] for name in TARGET_ROWS)
        for cell in CELLS
    }
    direct = {}
    for cell in CELLS:
        prefill_wall = rows[cell]["prefill"]["Complete Host wall"]
        decode_wall = walls[cell]
        direct[cell] = {
            "prefill_wall_us": prefill_wall,
            "prefill_tok_s": 64e6 / prefill_wall,
            "decode_wall_us_per_token": decode_wall,
            "decode_tok_s": 1e6 / decode_wall,
            "target_stage_us_per_token": target[cell],
            "decode_speed_percent_vs_control":
                (walls["full"] / decode_wall - 1.0) * 100.0,
        }

    paired = []
    for control, candidate in zip(runs["full"], runs["row4"]):
        control_wall = run_mean(
            control, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall"))
        candidate_wall = run_mean(
            candidate, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall"))
        control_target = sum(run_mean(
            control, decode_indices,
            lambda record, name=name: base.generation_row_us(record, name))
            for name in TARGET_ROWS)
        candidate_target = sum(run_mean(
            candidate, decode_indices,
            lambda record, name=name: base.generation_row_us(record, name))
            for name in TARGET_ROWS)
        paired.append({
            "decode_wall_saved_us": control_wall - candidate_wall,
            "decode_wall_speed_percent":
                (control_wall / candidate_wall - 1.0) * 100.0,
            "target_stage_speed_percent":
                (control_target / candidate_target - 1.0) * 100.0,
        })
    pair_wins = sum(item["decode_wall_saved_us"] > 0.0 for item in paired)

    gates = {
        "all_11_sessions_physical_contract": physical,
        "requested_and_observed_common_op_row_extent": dispatch,
        "padding_poison_dispatch": poison_dispatch,
        "unchanged_HMX_tile_work": invariant_hmx,
        "byte_exact_valid_hidden_hashes": exact_hashes,
        "byte_exact_selected_logit_codes": exact_logits,
        "identical_token_sequences": exact_sequences,
        "padding_poison_invariance": (
            output_hashes(poison) == output_hashes(reference)
            and logit_codes(poison) == logit_codes(reference)
            and sequence(poison) == sequence(reference)),
        "row4_lower_combined_target_stage_wall":
            target["row4"] < target["full"],
        "row4_lower_complete_decode_wall": walls["row4"] < walls["full"],
        "row4_wins_all_five_rotated_pairs": pair_wins == 5,
    }
    conclusion = (
        "advance_to_formal" if all(gates.values())
        else "reject_row4_without_193_step_formal_run"
    )
    summary = {
        "experiment": "EXP-0178",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "short_evidence": str(result_dir),
        "sessions_per_performance_cell": 5,
        "padding_poison_sessions": 1,
        "continuous_decode_tokens_per_session": STEPS - 1,
        "gates": gates,
        "conclusion": conclusion,
        "direct": direct,
        "module_rows_us": rows,
        "paired_rounds": paired,
        "quality_gate": "disabled_by_contract",
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
        "# EXP-0178 row-four common-op short gate", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack short-gate result", "",
        "| Cell | Prefill tok/s | Decode wall/token | Decode tok/s | "
        "Target three stages | Decode speed vs control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
            f"{item['target_stage_us_per_token']:.3f} us | "
            f"{item['decode_speed_percent_vs_control']:+.3f}% |")
    lines += ["", "## Complete decode module table", "",
              "| Module | Full-64 control | Row-4 candidate | "
              "Candidate speed |", "|---|---:|---:|---:|"]
    for name in ROWS:
        control = rows["full"]["decode"][name]
        candidate = rows["row4"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, walls['full'])} | "
            f"{fmt(candidate, walls['row4'])} | "
            f"{fmt_speed(control, candidate)} |")
    lines += ["", "## Gates", ""]
    for name, value in gates.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines += ["", "## Conclusion", "", conclusion, ""]
    (result_dir / "short_gate_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "experiment": "EXP-0178", "conclusion": conclusion,
        "direct": direct, "paired_rounds": paired, "gates": gates,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    if conclusion != "advance_to_formal":
        raise SystemExit("EXP-0178 short gate failed")


if __name__ == "__main__":
    main()
