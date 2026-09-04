#!/usr/bin/env python3
"""Validate and summarize the formal EXP-0176 O-projection batch gate."""

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


STEPS = 193
LAYERS = 28
CELLS = ("o_batch4", "o_batch8")
BATCHES = {"o_batch4": 4, "o_batch8": 8}
O_COMMANDS = {"o_batch4": 448, "o_batch8": 224}
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
DIAGNOSTICS = (
    "weight_dma_ticks",
    "w4u8_qkvo_weight_expand_ticks",
    "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks",
    "projection_hmx_wait_ticks",
)
COUNTERS = (
    "w4u8_o_batch_count",
    "hmx_u8s8_tile_pair_count",
    "hmx_command_count",
    "w4u8_qkvo_prefetch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0176-w4u8-decode-o-batch8",
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
    if len(paths) != 10:
        raise ValueError(f"expected ten {cell} logs, got {len(paths)}")
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
            if not (int(step.get("experiment", -1)) == 176
                    and int(profile.get("experiment", -1)) == 176
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
    if len(profiles) != STEPS:
        return False
    return bool(
        int(profiles[0].get("experiment", -1)) == 176
        and int(profiles[0].get("w4u8_decode_o_batch_n_tiles", -1)) == 8
        and int(profiles[0].get("w4u8_o_batch_n_tiles_observed", -1)) == 4
        and int(profiles[0].get("w4u8_o_batch_count", -1)) == 448
        and all(
            int(item.get("experiment", -1)) == 176
            and int(item.get("w4u8_decode_o_batch_n_tiles", -1)) == 8
            and int(item.get("w4u8_o_batch_n_tiles_observed", -1)) == 8
            and int(item.get("w4u8_o_batch_count", -1)) == 224
            for item in profiles[1:]
        )
    )


def validate_runs(
    runs: dict[str, list[list[dict[str, object]]]],
) -> tuple[dict[str, bool], list[tuple[int, ...]], list[tuple[int, ...]]]:
    physical = True
    schedule = True
    cache = True
    sequences: list[tuple[int, ...]] = []
    hashes: list[tuple[str, ...]] = []
    codes: list[tuple[int, ...]] = []
    hmx_work: dict[str, list[int]] = {cell: [] for cell in CELLS}
    for cell, cell_runs in runs.items():
        for run in cell_runs:
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
                expected_batch = 4 if index == 0 else BATCHES[cell]
                expected_commands = 448 if index == 0 else O_COMMANDS[cell]
                schedule &= bool(
                    int(profile["w4u8_decode_o_batch_n_tiles"])
                        == BATCHES[cell]
                    and int(profile["w4u8_o_batch_n_tiles_observed"])
                        == expected_batch
                    and int(profile["w4u8_o_batch_count"])
                        == expected_commands
                )
                physical &= bool(
                    int(step["rpc_result"]) == 0 and bool(step["pass"])
                    and profile["backend"] == "standalone_fastrpc_dsp"
                    and profile["qnn"] == "none"
                    and int(profile["block_invocation_count"]) == LAYERS
                    and int(profile["vtcm_requested_bytes"])
                        == 8 * 1024 * 1024
                    and int(profile["vtcm_acquired_bytes"])
                        == 8 * 1024 * 1024
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
                if index > 0:
                    hmx_work[cell].append(
                        int(profile["hmx_u8s8_tile_pair_count"]))
            sequences.append(tuple(tokens_run))
            codes.append(tuple(codes_run))
            hashes.append(tuple(hashes_run))
    gates = {
        "all_20_sessions_physical_contract": physical,
        "requested_and_observed_O_batch_geometry": schedule,
        "all_layer_cache_lengths_synchronous": cache,
        "all_20_sessions_byte_exact_output_hashes": len(set(hashes)) == 1,
        "all_20_sessions_identical_token_sequence": len(set(sequences)) == 1,
        "first16_EXP0168_tokens_and_codes_exact": bool(
            list(sequences[0][:16]) == exp173.EXPECTED_TOKENS
            and list(codes[0][:16]) == exp173.EXPECTED_CODES),
        "unchanged_HMX_tile_pair_work": bool(
            hmx_work["o_batch4"] and hmx_work["o_batch8"]
            and set(hmx_work["o_batch4"]) == set(hmx_work["o_batch8"])),
    }
    return gates, sequences, codes


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    if candidate > 0.0:
        return f"{(control / candidate - 1.0) * 100.0:+.2f}%"
    return "+0.00%" if control == 0.0 else "removed/fused"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {cell: load_cell(result_dir, cell) for cell in CELLS}
    prefill_indices = (0,)
    decode_indices = tuple(range(1, STEPS))
    module_rows = {
        cell: {
            "prefill": summarize_rows(runs[cell], prefill_indices),
            "decode": summarize_rows(runs[cell], decode_indices),
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
    counters = {
        cell: {
            field: median_metric(
                runs[cell], decode_indices,
                lambda record, field=field: float(record.get(field, 0)),
            ) for field in COUNTERS
        } for cell in CELLS
    }

    gates, _, _ = validate_runs(runs)
    audit = json.loads(
        (result_dir / "audit/independent_reference.json").read_text())
    gates["audit_O_batch8_geometry_193_steps"] = validate_audit(
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
        cell: [run_mean(run, decode_indices, base.host_wall_us)
               for run in runs[cell]]
        for cell in CELLS
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_decode["o_batch4"], session_decode["o_batch8"])]
    pair_wins = sum(saved > 0.0 for saved in paired_saved_us)
    control_us = module_rows["o_batch4"]["decode"]["Complete Host wall"]
    candidate_us = module_rows["o_batch8"]["decode"]["Complete Host wall"]
    gates["candidate_strictly_faster_median_decode_wall"] = (
        candidate_us < control_us)
    gates["candidate_wins_majority_of_ten_rotated_pairs"] = pair_wins >= 6
    gates["prefill_remains_O_batch4"] = all(
        int(run[0]["w4u8_o_batch_n_tiles_observed"]) == 4
        for cell_runs in runs.values() for run in cell_runs)
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
        } for cell in CELLS
    }
    direct["decode_speed_percent"] = (
        control_us / candidate_us - 1.0) * 100.0
    direct["decode_wall_reduction_percent"] = (
        1.0 - candidate_us / control_us) * 100.0
    direct["pair_wins"] = pair_wins
    direct["paired_saved_us"] = paired_saved_us

    summary = {
        "experiment": "EXP-0176",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "sessions_per_cell": 10,
        "prefill_tokens": 64,
        "continuous_decode_tokens_per_session": 192,
        "gates": gates,
        "direct": direct,
        "module_rows_us": module_rows,
        "diagnostics_us": diagnostics,
        "counters": counters,
        "quality_gate": "disabled_by_contract",
        "provenance": {
            "static_gate_sha256": sha256_file(
                result_dir / "static_gate.json"),
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
        "# EXP-0176 decode W4U8 O-projection batch-eight", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Cell | Prefill Host wall | Prefill tok/s | Decode wall/token | "
        "Decode tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_wall_us']:.3f} us | "
            f"{item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} |")
    lines += [
        f"| Change | diagnostic only | diagnostic only | "
        f"{-direct['decode_wall_reduction_percent']:.3f}% | "
        f"{direct['decode_speed_percent']:+.3f}% |",
        "", "## Complete decode module table", "",
        "| Module | O batch4 control | O batch8 candidate | Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        control = module_rows["o_batch4"]["decode"][name]
        candidate = module_rows["o_batch8"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, control_us)} | "
            f"{fmt(candidate, candidate_us)} | "
            f"{fmt_speed(control, candidate)} |")
    lines += [
        "", "## O-pipeline diagnostics", "",
        "| Diagnostic | O batch4 | O batch8 | Change |",
        "|---|---:|---:|---:|",
    ]
    for field in DIAGNOSTICS:
        name = field.replace("_ticks", "_us")
        control = diagnostics["o_batch4"][name]
        candidate = diagnostics["o_batch8"][name]
        lines.append(
            f"| {name} | {control:.3f} us | {candidate:.3f} us | "
            f"{candidate - control:+.3f} us |")
    for field in COUNTERS:
        control = counters["o_batch4"][field]
        candidate = counters["o_batch8"][field]
        lines.append(
            f"| {field} | {control:.0f} | {candidate:.0f} | "
            f"{candidate - control:+.0f} |")
    lines += [
        "", "## Rotated-pair stability", "",
        f"Candidate wins: {pair_wins}/10; paired wall saved (us/token): "
        + ", ".join(f"{value:+.3f}" for value in paired_saved_us),
        "", "## Gates", "", "| Gate | Result |", "|---|---:|",
    ]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The candidate changes only decode-M1 W4U8 O-projection HMX "
        "batch width. Prefill remains batch four; HMX tile work, Transformer "
        "mathematics, cache and output tokens remain unchanged. Semantic "
        "quality remains disabled by contract.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "control_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": direct["decode_speed_percent"],
        "pair_wins": pair_wins,
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0176 required gate failed")


if __name__ == "__main__":
    main()
