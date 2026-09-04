#!/usr/bin/env python3
"""Validate EXP-0179 decode-only row-four Q/K Norm-RoPE evidence."""

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
import summarize_exp0167 as exp167  # noqa: E402
import summarize_exp0173 as exp173  # noqa: E402
import summarize_exp0178_short as exp178  # noqa: E402


LAYERS = 28
CELLS = ("full", "row4")
ROWS = exp178.ROWS
TARGET_ROW = "QKV + Q/K Norm-RoPE preparation"
Q_PAIR_CALLS = LAYERS * 8
K_PAIR_CALLS = LAYERS * 4
PAIR_CALLS = Q_PAIR_CALLS + K_PAIR_CALLS
FULL_ROWS_PROCESSED = LAYERS * (16 + 8) * 64
ROW4_ROWS_PROCESSED = LAYERS * (16 + 8) * 4
DIAGNOSTICS = (
    "qkv_projection_ticks",
    "u8_attention_qk_norm_rope_ticks",
    "w4u8_qkvo_weight_expand_ticks",
    "weight_dma_ticks",
    "projection_hmx_wait_ticks",
)
COUNTERS = (
    "w4u8_decode_q_pair_row4_call_count",
    "w4u8_decode_k_pair_row4_call_count",
    "w4u8_decode_qk_rows_processed",
    "w4u8_decode_k_temp_carrier_skipped_count",
    "w4u8_qk_padding_poison_pair_count",
    "hmx_u8s8_tile_pair_count",
    "hmx_command_count",
    "weight_ddr_read_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rounds", type=int, choices=(5, 10), required=True)
    parser.add_argument("--steps", type=int, choices=(17, 193), required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0179-w4u8-decode-row4-qk-norm-rope",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path, expected_steps: int) -> list[dict[str, object]]:
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
    final = finals[0]
    if not (final.get("all_steps_pass") is True
            and int(final.get("requested_steps", -1)) == expected_steps
            and int(final.get("completed_steps", -1)) == expected_steps):
        raise ValueError(f"failed final record: {path}")
    for index, (step, profile) in enumerate(zip(steps, profiles)):
        if not (int(step.get("experiment", -1)) == 179
                and int(profile.get("experiment", -1)) == 179
                and int(step.get("generation_step", -1)) == index
                and int(profile.get("generation_step", -1)) == index
                and int(step.get("generation_mode", -1)) == 9):
            raise ValueError(f"identity mismatch: {path}:{index}")
        profile["_step_record"] = step
        profile["_source_log"] = str(path)
    return profiles


def load_cell(result_dir: Path, cell: str, rounds: int,
              steps: int) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{cell}.log"))
    if len(paths) != rounds:
        raise ValueError(
            f"expected {rounds} {cell} logs, got {len(paths)}")
    return [load_run(path, steps) for path in paths]


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


def validate_dispatch(run: list[dict[str, object]], cell: str,
                      poison: bool = False) -> bool:
    requested = 64 if cell == "full" else 4
    poison_value = 1 if poison else 0
    for index, profile in enumerate(run):
        if not (
            int(profile["w4u8_decode_qk_norm_rope_rows"]) == requested
            and int(profile["w4u8_decode_qk_padding_poison"])
                == poison_value
        ):
            return False
        if index == 0:
            if not (
                int(profile["w4u8_qk_norm_rope_rows_observed"]) == 0
                and int(profile["w4u8_decode_q_pair_row4_call_count"]) == 0
                and int(profile["w4u8_decode_k_pair_row4_call_count"]) == 0
                and int(profile["w4u8_decode_qk_rows_processed"]) == 0
                and int(profile[
                    "w4u8_decode_k_temp_carrier_skipped_count"]) == 0
                and int(profile[
                    "w4u8_qk_padding_poison_pair_count"]) == 0
            ):
                return False
        elif cell == "full":
            if not (
                int(profile["w4u8_qk_norm_rope_rows_observed"]) == 64
                and int(profile["w4u8_decode_q_pair_row4_call_count"]) == 0
                and int(profile["w4u8_decode_k_pair_row4_call_count"]) == 0
                and int(profile["w4u8_decode_qk_rows_processed"])
                    == FULL_ROWS_PROCESSED
                and int(profile[
                    "w4u8_decode_k_temp_carrier_skipped_count"]) == 0
                and int(profile[
                    "w4u8_qk_padding_poison_pair_count"]) == 0
            ):
                return False
        elif not (
            int(profile["w4u8_qk_norm_rope_rows_observed"]) == 4
            and int(profile["w4u8_decode_q_pair_row4_call_count"])
                == Q_PAIR_CALLS
            and int(profile["w4u8_decode_k_pair_row4_call_count"])
                == K_PAIR_CALLS
            and int(profile["w4u8_decode_qk_rows_processed"])
                == ROW4_ROWS_PROCESSED
            and int(profile["w4u8_decode_k_temp_carrier_skipped_count"])
                == LAYERS * 8
            and int(profile["w4u8_qk_padding_poison_pair_count"])
                == (PAIR_CALLS if poison else 0)
        ):
            return False
    return True


def validate_qk_boundary(result_dir: Path) -> bool:
    control = load_run(result_dir / "qk_boundary_control.log", 2)
    candidate = load_run(result_dir / "qk_boundary_row4.log", 2)
    c_decode = control[1]
    r_decode = candidate[1]
    return bool(
        validate_dispatch(control, "full")
        and validate_dispatch(candidate, "row4")
        and int(c_decode["w4u8_decode_q_valid_row_hash"]) != 0
        and int(c_decode["w4u8_decode_k_valid_row_hash"]) != 0
        and int(c_decode["w4u8_decode_q_valid_row_hash"])
            == int(r_decode["w4u8_decode_q_valid_row_hash"])
        and int(c_decode["w4u8_decode_k_valid_row_hash"])
            == int(r_decode["w4u8_decode_k_valid_row_hash"])
        and output_hashes(control) == output_hashes(candidate)
        and sequence(control) == sequence(candidate)
        and logit_codes(control) == logit_codes(candidate)
    )


def invariant_field(reference: list[dict[str, object]],
                    run: list[dict[str, object]], field: str) -> bool:
    return tuple(int(item[field]) for item in run) == tuple(
        int(item[field]) for item in reference)


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    if candidate > 0.0:
        return f"{(control / candidate - 1.0) * 100.0:+.2f}%"
    return "+0.00%" if control == 0.0 else "removed/fused"


def main() -> None:
    args = parse_args()
    if args.formal != (args.rounds == 10 and args.steps == 193):
        raise ValueError("formal requires ten 193-step rotated pairs")
    result_dir = args.result_dir.resolve()
    runs = {
        cell: load_cell(result_dir, cell, args.rounds, args.steps)
        for cell in CELLS
    }
    poison = load_run(result_dir / "padding_poison.log", args.steps)
    all_runs = [run for cell in CELLS for run in runs[cell]]
    reference = runs["full"][0]
    prefill_indices = (0,)
    decode_indices = tuple(range(1, args.steps))
    module_rows = {
        cell: {
            "prefill": {
                name: median_metric(
                    runs[cell], prefill_indices,
                    lambda record, name=name:
                    base.generation_row_us(record, name),
                ) for name in ROWS
            },
            "decode": {
                name: median_metric(
                    runs[cell], decode_indices,
                    lambda record, name=name:
                    base.generation_row_us(record, name),
                ) for name in ROWS
            },
        } for cell in CELLS
    }
    diagnostics = {
        cell: {
            field.replace("_ticks", "_us"): median_metric(
                runs[cell], decode_indices,
                lambda record, field=field:
                float(record[field]) / base.TICKS_PER_US,
            ) for field in DIAGNOSTICS
        } for cell in CELLS
    }
    counters = {
        cell: {
            field: median_metric(
                runs[cell], decode_indices,
                lambda record, field=field: float(record[field]),
            ) for field in COUNTERS
        } for cell in CELLS
    }
    control_us = module_rows["full"]["decode"]["Complete Host wall"]
    candidate_us = module_rows["row4"]["decode"]["Complete Host wall"]
    control_target = module_rows["full"]["decode"][TARGET_ROW]
    candidate_target = module_rows["row4"]["decode"][TARGET_ROW]

    session_decode = {
        cell: [run_mean(
            run, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall")) for run in runs[cell]]
        for cell in CELLS
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_decode["full"], session_decode["row4"])]
    pair_wins = sum(value > 0.0 for value in paired_saved_us)

    exact_sequences = all(sequence(run) == sequence(reference)
                          for run in all_runs + [poison])
    exact_logits = all(logit_codes(run) == logit_codes(reference)
                       for run in all_runs + [poison])
    exact_hashes = all(output_hashes(run) == output_hashes(reference)
                       for run in all_runs + [poison])
    invariant_fields = all(
        invariant_field(reference, run, field)
        for run in all_runs + [poison]
        for field in (
            "hmx_u8s8_tile_pair_count", "hmx_command_count",
            "weight_ddr_read_bytes",
        )
    )

    gates: dict[str, bool] = {
        "all_performance_sessions_physical_contract": all(
            exp178.validate_physical(run) for run in all_runs),
        "requested_and_observed_qk_row_extent": all(
            validate_dispatch(run, cell)
            for cell in CELLS for run in runs[cell]),
        "qk_valid_row_boundary_hashes_byte_exact":
            validate_qk_boundary(result_dir),
        "all_sessions_byte_exact_valid_hidden_hashes": exact_hashes,
        "all_sessions_identical_selected_logit_codes": exact_logits,
        "all_sessions_identical_token_sequences": exact_sequences,
        "unchanged_qkv_hmx_and_weight_work": invariant_fields,
        "padding_poison_physical_contract":
            exp178.validate_physical(poison),
        "padding_poison_dispatch":
            validate_dispatch(poison, "row4", poison=True),
        "padding_poison_invariance": bool(
            output_hashes(poison) == output_hashes(reference)
            and logit_codes(poison) == logit_codes(reference)
            and sequence(poison) == sequence(reference)),
        "prefill_uses_unchanged_full_row_path": all(
            int(run[0]["w4u8_qk_norm_rope_rows_observed"]) == 0
            and int(run[0]["w4u8_decode_q_pair_row4_call_count"]) == 0
            and int(run[0]["w4u8_decode_k_pair_row4_call_count"]) == 0
            for run in all_runs),
        "candidate_strictly_faster_qkv_plus_qk_preparation":
            candidate_target < control_target,
        "candidate_strictly_faster_median_decode_wall":
            candidate_us < control_us,
        "candidate_wins_required_rotated_pairs":
            pair_wins >= (6 if args.formal else 5),
    }
    if args.formal:
        audit = json.loads(
            (result_dir / "audit/independent_reference.json").read_text())
        gates.update({
            "first16_EXP0168_tokens_and_codes_exact": bool(
                list(sequence(reference)[:16]) == exp173.EXPECTED_TOKENS
                and list(logit_codes(reference)[:16])
                    == exp173.EXPECTED_CODES),
            "independent_integer_head_reference_193_of_193": bool(
                audit.get("summary", {}).get("implementation_gate")
                    == "pass"
                and int(audit.get("summary", {}).get(
                    "verified_steps", -1)) == args.steps
                and int(audit.get("summary", {}).get(
                    "token_and_code_matches", -1)) == args.steps),
            "EXP0163_transformer_and_cache_regression":
                exp167.validate_exp0163_regression(
                    result_dir / "w4u8_exp0163_regression.log"),
        })
    gates["all_required"] = all(gates.values())

    direct = {
        cell: {
            "prefill_wall_us":
                module_rows[cell]["prefill"]["Complete Host wall"],
            "prefill_tok_s": 64e6 /
                module_rows[cell]["prefill"]["Complete Host wall"],
            "decode_wall_us_per_token":
                module_rows[cell]["decode"]["Complete Host wall"],
            "decode_tok_s": 1e6 /
                module_rows[cell]["decode"]["Complete Host wall"],
            "qkv_qk_prep_us_per_token":
                module_rows[cell]["decode"][TARGET_ROW],
        } for cell in CELLS
    }
    direct["decode_speed_percent"] = (
        control_us / candidate_us - 1.0) * 100.0
    direct["decode_wall_reduction_percent"] = (
        1.0 - candidate_us / control_us) * 100.0
    direct["qkv_qk_prep_speed_percent"] = (
        control_target / candidate_target - 1.0) * 100.0
    direct["pair_wins"] = pair_wins
    direct["paired_saved_us"] = paired_saved_us

    provenance = {
        "static_gate_sha256": sha256_file(result_dir / "static_gate.json"),
        "qk_boundary_control_sha256": sha256_file(
            result_dir / "qk_boundary_control.log"),
        "qk_boundary_row4_sha256": sha256_file(
            result_dir / "qk_boundary_row4.log"),
        "padding_poison_sha256": sha256_file(
            result_dir / "padding_poison.log"),
        "logs": {path.name: sha256_file(path) for path in sorted(
            (result_dir / "raw").glob("*.log"))},
    }
    if args.formal:
        provenance.update({
            "audit_reference_sha256": sha256_file(
                result_dir / "audit/independent_reference.json"),
            "audit_device_sha256": sha256_file(
                result_dir / "audit/device.jsonl"),
            "regression_sha256": sha256_file(
                result_dir / "w4u8_exp0163_regression.log"),
        })
    conclusion = (
        "locally_eligible_pending_user_promotion"
        if args.formal and gates["all_required"]
        else "advance_to_formal"
        if not args.formal and gates["all_required"]
        else "reject_candidate"
    )
    summary = {
        "experiment": "EXP-0179",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "evidence": str(result_dir),
        "formal": args.formal,
        "sessions_per_cell": args.rounds,
        "continuous_decode_tokens_per_session": args.steps - 1,
        "gates": gates,
        "conclusion": conclusion,
        "direct": direct,
        "module_rows_us": module_rows,
        "diagnostics_us": diagnostics,
        "counters": counters,
        "quality_gate": "disabled_by_contract",
        "provenance": provenance,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")

    lines = [
        "# EXP-0179 decode W4U8 row-four Q/K Norm-RoPE", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Cell | Prefill wall | Prefill tok/s | Decode wall/token | "
        "Decode tok/s | QKV + Q/K preparation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_wall_us']:.3f} us | "
            f"{item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
            f"{item['qkv_qk_prep_us_per_token']:.3f} us |")
    lines += [
        f"| Change | diagnostic only | diagnostic only | "
        f"{-direct['decode_wall_reduction_percent']:.3f}% | "
        f"{direct['decode_speed_percent']:+.3f}% | "
        f"{direct['qkv_qk_prep_speed_percent']:+.3f}% |", "",
        "## Complete decode module table", "",
        "| Module | Full-64 control | Row-4 candidate | Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        control = module_rows["full"]["decode"][name]
        candidate = module_rows["row4"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, control_us)} | "
            f"{fmt(candidate, candidate_us)} | "
            f"{fmt_speed(control, candidate)} |")
    lines += ["", "## Q/K preparation diagnostics", "",
              "| Diagnostic | Full-64 | Row-4 | Change |",
              "|---|---:|---:|---:|"]
    for field in DIAGNOSTICS:
        name = field.replace("_ticks", "_us")
        control = diagnostics["full"][name]
        candidate = diagnostics["row4"][name]
        lines.append(
            f"| {name} | {control:.3f} us | {candidate:.3f} us | "
            f"{candidate - control:+.3f} us |")
    for field in COUNTERS:
        control = counters["full"][field]
        candidate = counters["row4"][field]
        lines.append(
            f"| {field} | {control:.0f} | {candidate:.0f} | "
            f"{candidate - control:+.0f} |")
    lines += ["", "## Rotated-pair stability", "",
              f"Candidate wins: {pair_wins}/{args.rounds}; paired wall "
              "saved (us/token): "
              + ", ".join(f"{value:+.3f}"
                          for value in paired_saved_us),
              "", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "## Conclusion", "", conclusion, "",
              "The candidate changes only continuous logical-M1 W4U8 "
              "Q/K normalization and RoPE from 64 physical rows to the "
              "first four SIMD-aligned rows and removes the decode-unused "
              "temporary K carrier. M64 prefill and all downstream math "
              "remain unchanged. Semantic quality remains disabled by "
              "contract.", ""]
    report_name = (
        "full_profiling_report.md" if args.formal
        else "short_gate_report.md")
    (result_dir / report_name).write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "experiment": "EXP-0179",
        "conclusion": conclusion,
        "control_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": direct["decode_speed_percent"],
        "qkv_qk_prep_speed_percent":
            direct["qkv_qk_prep_speed_percent"],
        "pair_wins": pair_wins,
        "gates": gates,
        "result_dir": str(result_dir),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    if not gates["all_required"]:
        raise SystemExit("EXP-0179 required gate failed")


if __name__ == "__main__":
    main()
