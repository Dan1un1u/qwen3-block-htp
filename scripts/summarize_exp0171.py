#!/usr/bin/env python3
"""Validate and summarize EXP-0171 full-tile versus row4 decode SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import summarize_exp0164 as base  # noqa: E402
import summarize_exp0167 as exp167  # noqa: E402
import summarize_exp0170 as exp170  # noqa: E402

STEPS = 193
LAYERS = 28
PAIR_CALLS_PER_LAYER = 192
CALLS_PER_DECODE_STEP = LAYERS * PAIR_CALLS_PER_LAYER
MODES = ("full_tile", "decode_row4")
MODE_IDS = {"full_tile": 0, "decode_row4": 1}
ROWS = exp170.ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0171-w4u8-decode-row4-swiglu",
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
            raise ValueError(f"incomplete run: {path}")
        if not (finals[0].get("all_steps_pass") is True
                and int(finals[0].get("requested_steps", -1)) == STEPS):
            raise ValueError(f"failed final record: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if not (int(step.get("experiment", -1)) == 171
                    and int(profile.get("experiment", -1)) == 171
                    and int(step.get("generation_step", -1)) == index
                    and int(profile.get("generation_step", -1)) == index):
                raise ValueError(f"identity mismatch: {path}:{index}")
            profile["_step_record"] = step
        runs.append(profiles)
    return runs


def run_mean(run: list[dict[str, object]], indices: tuple[int, ...],
             getter) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_metric(runs: list[list[dict[str, object]]],
                  indices: tuple[int, ...], getter) -> float:
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


def validate_audit(path: Path) -> bool:
    profiles = [item for item in base.read_json_lines(path)
                if item.get("record") == "generation_profile"]
    return bool(
        len(profiles) == STEPS
        and int(profiles[0].get("w4u8_decode_swiglu_row4_call_count", -1)) == 0
        and all(
            int(item.get("w4u8_decode_swiglu_mode", -1)) == 1
            and int(item.get("w4u8_decode_swiglu_row4_call_count", -1))
                == CALLS_PER_DECODE_STEP
            and int(item.get("w4u8_decode_swiglu_row4_mismatch_count", -1)) == 0
            for item in profiles[1:]
        )
    )


def validate_runs(runs: dict[str, list[list[dict[str, object]]]]) -> dict[str, bool]:
    physical = True
    mode_contract = True
    cache = True
    sequences: list[tuple[int, ...]] = []
    hashes: list[tuple[str, ...]] = []
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            token_run: list[int] = []
            hash_run: list[str] = []
            for index, profile in enumerate(run):
                step = profile["_step_record"]
                token_run.append(int(step["selected_token_id"]))
                hash_run.append(str(profile["output_hash"]))
                expected_calls = (
                    CALLS_PER_DECODE_STEP
                    if mode == "decode_row4" and index > 0 else 0)
                mode_contract &= bool(
                    int(profile["w4u8_decode_swiglu_mode"]) == MODE_IDS[mode]
                    and int(profile["w4u8_decode_swiglu_row4_call_count"])
                        == expected_calls
                    and int(profile["w4u8_decode_swiglu_row4_mismatch_count"])
                        == 0)
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
                    and int(profile["ledger_unattributed_ticks"]) == 0)
                expected_before = 0 if index == 0 else 63 + index
                expected_after = 64 if index == 0 else 64 + index
                cache &= bool(
                    int(profile["first_position"]) == expected_before
                    and int(profile["valid_length"]) == expected_after
                    and int(profile["cache_mismatches"]) == 0
                    and int(profile["cache_structure_mismatches"]) == 0)
            sequences.append(tuple(token_run))
            hashes.append(tuple(hash_run))
    return {
        "all_timed_physical_contracts": physical,
        "requested_and_observed_row4_contract": mode_contract,
        "all_layer_cache_lengths_synchronous": cache,
        "all_20_sessions_byte_exact_output_hashes": len(set(hashes)) == 1,
        "all_20_sessions_identical_token_sequence": len(set(sequences)) == 1,
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
    decode_indices = tuple(range(1, STEPS))
    prefill_indices = (0,)
    rows = {
        mode: {
            "prefill": summarize_rows(runs[mode], prefill_indices),
            "decode": summarize_rows(runs[mode], decode_indices),
        } for mode in MODES
    }
    gates = validate_runs(runs)
    gates["on_device_row4_exact_1032192_calls"] = validate_audit(
        result_dir / "audit/device.jsonl")
    independent = json.loads(
        (result_dir / "audit/independent_reference.json").read_text())
    gates["independent_reference_193_of_193"] = bool(
        independent.get("summary", {}).get("implementation_gate") == "pass"
        and int(independent.get("summary", {}).get("verified_steps", -1))
            == STEPS)
    gates["EXP0163_transformer_and_cache_regression"] = (
        exp167.validate_exp0163_regression(
            result_dir / "w4u8_exp0163_regression.log"))

    session_decode = {
        mode: [run_mean(run, decode_indices, base.host_wall_us)
               for run in runs[mode]]
        for mode in MODES
    }
    paired_saved = [control - candidate for control, candidate in zip(
        session_decode["full_tile"], session_decode["decode_row4"])]
    control_us = rows["full_tile"]["decode"]["Complete Host wall"]
    candidate_us = rows["decode_row4"]["decode"]["Complete Host wall"]
    gates["candidate_strictly_faster_median_decode_wall"] = (
        candidate_us < control_us)
    gates["candidate_wins_all_ten_rotated_pairs"] = all(
        saved > 0.0 for saved in paired_saved)
    gates["all_required"] = all(gates.values())

    summary = {
        "experiment": "EXP-0171",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "sessions_per_mode": 10,
        "prefill_tokens": 64,
        "continuous_decode_tokens_per_session": 192,
        "gates": gates,
        "direct": {
            "full_tile": {
                "prefill_tok_s": 64e6 / rows["full_tile"]["prefill"]["Complete Host wall"],
                "decode_wall_us_per_token": control_us,
                "decode_tok_s": 1e6 / control_us,
            },
            "decode_row4": {
                "prefill_tok_s": 64e6 / rows["decode_row4"]["prefill"]["Complete Host wall"],
                "decode_wall_us_per_token": candidate_us,
                "decode_tok_s": 1e6 / candidate_us,
            },
            "decode_speed_percent": (control_us / candidate_us - 1.0) * 100.0,
            "decode_wall_reduction_percent": (1.0 - candidate_us / control_us) * 100.0,
            "pair_wins": sum(saved > 0.0 for saved in paired_saved),
            "paired_saved_us": paired_saved,
        },
        "module_rows_us": rows,
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

    lines = [
        "# EXP-0171 row-selective W4U8 decode SwiGLU", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Mode | Prefill tok/s | Decode wall/token | Decode tok/s |",
        "|---|---:|---:|---:|",
        f"| Full tile control | {summary['direct']['full_tile']['prefill_tok_s']:.3f} | {control_us:.3f} us | {1e6/control_us:.3f} |",
        f"| Decode row4 | {summary['direct']['decode_row4']['prefill_tok_s']:.3f} | {candidate_us:.3f} us | {1e6/candidate_us:.3f} |",
        f"| Change | diagnostic only | {summary['direct']['decode_wall_reduction_percent']:+.3f}% wall | {summary['direct']['decode_speed_percent']:+.3f}% |",
        "", "## Complete decode module table", "",
        "| Module | Full tile | Decode row4 | Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        control = rows["full_tile"]["decode"][name]
        candidate = rows["decode_row4"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, control_us)} | "
            f"{fmt(candidate, candidate_us)} | "
            f"{fmt_speed(control, candidate)} |")
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The candidate changes only M=1 W4U8 Gate/Up SwiGLU "
        "publication. Prefill, projections, cache ABI, Attention, Down and "
        "LM head remain unchanged.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "control_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": summary["direct"]["decode_speed_percent"],
        "pair_wins": summary["direct"]["pair_wins"],
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0171 required gate failed")


if __name__ == "__main__":
    main()
