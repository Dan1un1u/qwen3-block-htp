#!/usr/bin/env python3
"""Validate and summarize EXP-0173 W4U8 LM-head batch eight vs sixteen."""

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


STEPS = 193
LAYERS = 28
MODES = ("batch8", "batch16")
MODE_GROUPS = {"batch8": 8, "batch16": 16}
EXPECTED_COMMANDS = {"batch8": 594, "batch16": 297}
EXPECTED_PREFETCHES = {"batch8": 593, "batch16": 296}
EXPECTED_TOKENS = [
    124491, 51272, 51272, 9092, 51272, 128014, 23186, 85301,
    23186, 23186, 23186, 23186, 105260, 37440, 23186, 5205,
]
EXPECTED_CODES = [
    159, 162, 160, 161, 161, 160, 162, 164,
    163, 165, 165, 160, 156, 156, 165, 158,
]
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
HEAD_FIELDS = (
    "generation_lm_head_weight_dma_ticks",
    "generation_lm_head_scale_dma_ticks",
    "generation_lm_head_expand_ticks",
    "generation_lm_head_hmx_ticks",
    "generation_lm_head_argmax_ticks",
    "generation_lm_head_weight_dma_wait_ticks",
    "generation_lm_head_hmx_tail_wait_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0173-w4u8-decode-lm-head-batch16",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mode(result_dir: Path, mode: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{mode}.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten {mode} logs, got {len(paths)}")
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
            raise ValueError(f"incomplete {mode} run: {path}")
        if not (finals[0].get("all_steps_pass") is True
                and int(finals[0].get("requested_steps", -1)) == STEPS
                and int(finals[0].get("completed_steps", -1)) == STEPS):
            raise ValueError(f"failed final record: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if not (int(step.get("experiment", -1)) == 173
                    and int(profile.get("experiment", -1)) == 173
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


def summarize_rows(runs: list[list[dict[str, object]]],
                   indices: tuple[int, ...]) -> dict[str, float]:
    return {
        name: median_metric(
            runs, indices,
            lambda record, name=name: base.generation_row_us(record, name),
        ) for name in ROWS
    }


def tick_us(record: dict[str, object], field: str) -> float:
    return float(record[field]) / base.TICKS_PER_US


def validate_audit(path: Path) -> bool:
    profiles = [item for item in base.read_json_lines(path)
                if item.get("record") == "generation_profile"]
    return bool(
        len(profiles) == STEPS
        and int(profiles[0].get("generation_lm_head_batch_n_tiles", -1)) == 8
        and all(
            int(item.get("w4u8_decode_lm_head_group_tiles", -1)) == 16
            and int(item.get("generation_lm_head_batch_n_tiles", -1)) == 16
            and int(item.get("generation_lm_head_command_count", -1)) == 297
            and int(item.get("generation_lm_head_prefetch_count", -1)) == 296
            for item in profiles[1:]
        )
    )


def validate_runs(runs: dict[str, list[list[dict[str, object]]]]) -> dict[str, bool]:
    physical = True
    schedule = True
    cache = True
    sequences: list[tuple[int, ...]] = []
    hashes: list[tuple[str, ...]] = []
    codes: list[tuple[int, ...]] = []
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            tokens_run: list[int] = []
            hashes_run: list[str] = []
            codes_run: list[int] = []
            for index, profile in enumerate(run):
                step = profile["_step_record"]
                tokens_run.append(int(step["selected_token_id"]))
                codes_run.append(int(step["selected_logit_half_bits"]))
                hashes_run.append(str(profile["output_hash"]))
                expected_before = 0 if index == 0 else 63 + index
                expected_after = 64 if index == 0 else 64 + index
                expected_group = 8 if index == 0 else MODE_GROUPS[mode]
                expected_commands = 594 if index == 0 else EXPECTED_COMMANDS[mode]
                expected_prefetches = 593 if index == 0 else EXPECTED_PREFETCHES[mode]
                schedule &= bool(
                    int(profile["w4u8_decode_lm_head_group_tiles"])
                        == MODE_GROUPS[mode]
                    and int(profile["generation_lm_head_batch_n_tiles"])
                        == expected_group
                    and int(profile["generation_lm_head_command_count"])
                        == expected_commands
                    and int(profile["generation_lm_head_prefetch_count"])
                        == expected_prefetches
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
                for layer_index in range(LAYERS):
                    layer = profile[f"slice_layer_{layer_index}"]
                    cache &= bool(
                        int(layer["cache_valid_before"]) == expected_before
                        and int(layer["cache_valid_after"]) == expected_after)
                    physical &= bool(
                        int(layer["hidden_ddr_read_bytes"]) == 0
                        and int(layer["hidden_ddr_write_bytes"]) == 0
                        and int(layer["layer_unattributed_ticks"]) == 0)
            sequences.append(tuple(tokens_run))
            codes.append(tuple(codes_run))
            hashes.append(tuple(hashes_run))
    return {
        "all_timed_physical_contracts": physical,
        "requested_and_observed_batch_geometry": schedule,
        "all_layer_cache_lengths_synchronous": cache,
        "all_20_sessions_byte_exact_output_hashes": len(set(hashes)) == 1,
        "all_20_sessions_identical_token_sequence": len(set(sequences)) == 1,
        "first16_EXP0168_tokens_and_codes_exact": bool(
            list(sequences[0][:16]) == EXPECTED_TOKENS
            and list(codes[0][:16]) == EXPECTED_CODES),
    }


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    if candidate > 0.0:
        return f"{(control / candidate - 1.0) * 100.0:+.2f}%"
    return "+0.00%" if control == 0.0 else "removed/fused"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_mode(result_dir, mode) for mode in MODES}
    prefill_indices = (0,)
    decode_indices = tuple(range(1, STEPS))
    module_rows = {
        mode: {
            "prefill": summarize_rows(runs[mode], prefill_indices),
            "decode": summarize_rows(runs[mode], decode_indices),
        } for mode in MODES
    }
    head = {
        mode: {
            field.replace("_ticks", "_us"): median_metric(
                runs[mode], decode_indices,
                lambda record, field=field: tick_us(record, field))
            for field in HEAD_FIELDS
        } for mode in MODES
    }
    head["batch8"]["commands"] = median_metric(
        runs["batch8"], decode_indices,
        lambda record: float(record["generation_lm_head_command_count"]))
    head["batch16"]["commands"] = median_metric(
        runs["batch16"], decode_indices,
        lambda record: float(record["generation_lm_head_command_count"]))
    head["batch8"]["prefetches"] = median_metric(
        runs["batch8"], decode_indices,
        lambda record: float(record["generation_lm_head_prefetch_count"]))
    head["batch16"]["prefetches"] = median_metric(
        runs["batch16"], decode_indices,
        lambda record: float(record["generation_lm_head_prefetch_count"]))

    gates = validate_runs(runs)
    audit = json.loads(
        (result_dir / "audit/independent_reference.json").read_text())
    gates["audit_batch16_geometry_193_steps"] = validate_audit(
        result_dir / "audit/device.jsonl")
    gates["independent_integer_head_reference_193_of_193"] = bool(
        audit.get("summary", {}).get("implementation_gate") == "pass"
        and int(audit.get("summary", {}).get("verified_steps", -1)) == STEPS
        and int(audit.get("summary", {}).get("token_and_code_matches", -1))
            == STEPS)
    gates["EXP0163_transformer_and_cache_regression"] = (
        exp167.validate_exp0163_regression(
            result_dir / "w4u8_exp0163_regression.log"))

    session_decode = {
        mode: [run_mean(run, decode_indices, base.host_wall_us)
               for run in runs[mode]]
        for mode in MODES
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_decode["batch8"], session_decode["batch16"])]
    control_us = module_rows["batch8"]["decode"]["Complete Host wall"]
    candidate_us = module_rows["batch16"]["decode"]["Complete Host wall"]
    gates["candidate_strictly_faster_median_decode_wall"] = (
        candidate_us < control_us)
    gates["candidate_wins_all_ten_rotated_pairs"] = all(
        saved > 0.0 for saved in paired_saved_us)
    gates["prefill_remains_batch8"] = all(
        int(run[0]["generation_lm_head_batch_n_tiles"]) == 8
        for mode_runs in runs.values() for run in mode_runs)
    gates["all_required"] = all(gates.values())

    direct = {
        mode: {
            "prefill_wall_us": module_rows[mode]["prefill"]["Complete Host wall"],
            "prefill_tok_s": 64e6 / module_rows[mode]["prefill"]["Complete Host wall"],
            "decode_wall_us_per_token": module_rows[mode]["decode"]["Complete Host wall"],
            "decode_tok_s": 1e6 / module_rows[mode]["decode"]["Complete Host wall"],
        } for mode in MODES
    }
    direct["decode_speed_percent"] = (control_us / candidate_us - 1.0) * 100.0
    direct["decode_wall_reduction_percent"] = (1.0 - candidate_us / control_us) * 100.0
    direct["pair_wins"] = sum(saved > 0.0 for saved in paired_saved_us)
    direct["paired_saved_us"] = paired_saved_us

    summary = {
        "experiment": "EXP-0173",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "sessions_per_mode": 10,
        "prefill_tokens": 64,
        "continuous_decode_tokens_per_session": 192,
        "gates": gates,
        "direct": direct,
        "module_rows_us": module_rows,
        "lm_head_us": head,
        "quality_gate": "disabled_by_contract",
        "provenance": {
            "audit_reference_sha256": sha256_file(
                result_dir / "audit/independent_reference.json"),
            "audit_device_sha256": sha256_file(
                result_dir / "audit/device.jsonl"),
            "regression_sha256": sha256_file(
                result_dir / "w4u8_exp0163_regression.log"),
            "logs": {path.name: sha256_file(path) for path in sorted(
                (result_dir / "raw").glob("*.log"))},
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")

    control_wall = control_us
    candidate_wall = candidate_us
    lines = [
        "# EXP-0173 decode W4U8 LM-head batch sixteen", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Mode | Prefill tok/s | Decode wall/token | Decode tok/s |",
        "|---|---:|---:|---:|",
        f"| Batch8 control | {direct['batch8']['prefill_tok_s']:.3f} | {control_us:.3f} us | {1e6/control_us:.3f} |",
        f"| Batch16 candidate | {direct['batch16']['prefill_tok_s']:.3f} | {candidate_us:.3f} us | {1e6/candidate_us:.3f} |",
        f"| Change | preservation diagnostic | {-direct['decode_wall_reduction_percent']:.3f}% wall | +{direct['decode_speed_percent']:.3f}% |",
        "", "## Complete decode module table", "",
        "| Module | Batch8 control | Batch16 candidate | Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        control = module_rows["batch8"]["decode"][name]
        candidate = module_rows["batch16"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, control_wall)} | "
            f"{fmt(candidate, candidate_wall)} | "
            f"{fmt_speed(control, candidate)} |")
    lines += [
        "", "## LM-head diagnostics", "",
        "| Diagnostic | Batch8 | Batch16 | Change |",
        "|---|---:|---:|---:|",
    ]
    for field in head["batch8"]:
        control = head["batch8"][field]
        candidate = head["batch16"][field]
        suffix = "" if field in ("commands", "prefetches") else " us"
        lines.append(
            f"| {field} | {control:.3f}{suffix} | {candidate:.3f}{suffix} | "
            f"{candidate - control:+.3f}{suffix} |")
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The candidate changes only the M=1 W4U8 LM-head coarse group "
        "geometry and phase-overlaid VTCM carriers. Prefill remains batch "
        "eight; Transformer, cache and token mathematics remain unchanged. "
        "Semantic quality remains disabled by contract.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "control_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": direct["decode_speed_percent"],
        "pair_wins": direct["pair_wins"],
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0173 required gate failed")


if __name__ == "__main__":
    main()
