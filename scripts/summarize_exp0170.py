#!/usr/bin/env python3
"""Validate and summarize EXP-0170 scalar versus HVX decode Softmax."""

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
DECODE_CALLS_PER_STEP = LAYERS * 8
MODES = ("scalar", "hvx_tile4")
MODE_IDS = {"scalar": 0, "hvx_tile4": 1}
EXPECTED_TOKENS = [
    124491, 51272, 51272, 9092, 51272, 128014, 23186, 85301,
    23186, 23186, 23186, 23186, 105260, 37440, 23186, 5205,
]
EXPECTED_CODES = [
    159, 162, 160, 161, 161, 160, 162, 164,
    163, 165, 165, 160, 156, 156, 165, 158,
]
BUCKETS = tuple(
    (f"L{first}-{first + 31}", tuple(range(1 + first - 64,
                                          1 + first - 64 + 32)))
    for first in range(64, 256, 32)
)
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
ATTENTION_FIELDS = (
    "u8_attention_qk_norm_rope_ticks",
    "u8_attention_k_pack_ticks", "u8_attention_v_pack_ticks",
    "u8_attention_qk_hmx_ticks", "u8_attention_softmax_ticks",
    "u8_attention_av_hmx_ticks", "u8_attention_av_requant_ticks",
    "u8_attention_pipeline_wait_ticks", "attention_unattributed_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0170-w4u8-decode-hvx-softmax",
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
            if not (int(step.get("experiment", -1)) == 170
                    and int(profile.get("experiment", -1)) == 170
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
    return float(statistics.median(run_mean(run, indices, getter)
                                   for run in runs))


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


def slope(runs: list[list[dict[str, object]]],
          getter: Callable[[dict[str, object]], float]) -> float:
    values = [float(statistics.median(getter(run[index]) for run in runs))
              for index in range(1, STEPS)]
    xs = list(range(64, 256))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    return sum((x - x_mean) * (y - y_mean)
               for x, y in zip(xs, values)) / sum(
                   (x - x_mean) ** 2 for x in xs)


def validate_audit(path: Path) -> bool:
    records = base.read_json_lines(path)
    profiles = [item for item in records
                if item.get("record") == "generation_profile"]
    return bool(
        len(profiles) == STEPS
        and int(profiles[0].get("w4u8_decode_softmax_hvx_tile4_call_count", -1)) == 0
        and all(
            int(item.get("w4u8_decode_softmax_mode", -1)) == 1
            and int(item.get("w4u8_decode_softmax_hvx_tile4_call_count", -1))
                == DECODE_CALLS_PER_STEP
            and int(item.get("w4u8_decode_softmax_hvx_tile4_mismatch_count", -1)) == 0
            for item in profiles[1:]
        )
    )


def validate_runs(runs: dict[str, list[list[dict[str, object]]]]) -> dict[str, bool]:
    physical = True
    requested_and_observed = True
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
                requested_and_observed &= bool(
                    int(profile["w4u8_decode_softmax_mode"]) == MODE_IDS[mode]
                    and int(profile["w4u8_decode_softmax_hvx_tile4_call_count"])
                        == (DECODE_CALLS_PER_STEP
                            if mode == "hvx_tile4" and index > 0 else 0)
                    and int(profile["w4u8_decode_softmax_hvx_tile4_mismatch_count"]) == 0
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
        "requested_and_observed_kernel_mode": requested_and_observed,
        "all_layer_cache_lengths_synchronous": cache,
        "all_20_sessions_byte_exact_output_hashes": len(set(hashes)) == 1,
        "all_20_sessions_identical_token_sequence": len(set(sequences)) == 1,
        "first16_EXP0168_tokens_and_codes_exact": bool(
            list(sequences[0][:16]) == EXPECTED_TOKENS
            and list(codes[0][:16]) == EXPECTED_CODES),
    }


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = {mode: load_mode(result_dir, mode) for mode in MODES}
    decode_indices = tuple(range(1, STEPS))
    prefill_indices = (0,)
    gates = validate_runs(runs)
    audit = json.loads(
        (result_dir / "audit/independent_reference.json").read_text())
    gates["on_device_probability_byte_exact_43008_calls"] = validate_audit(
        result_dir / "audit/device.jsonl")
    gates["independent_integer_head_reference_193_of_193"] = bool(
        audit.get("summary", {}).get("implementation_gate") == "pass"
        and int(audit.get("summary", {}).get("verified_steps", -1)) == STEPS
        and int(audit.get("summary", {}).get("token_and_code_matches", -1))
            == STEPS)
    gates["EXP0163_transformer_and_cache_regression"] = (
        exp167.validate_exp0163_regression(
            result_dir / "w4u8_exp0163_regression.log"))

    module_rows = {
        mode: {
            "prefill": summarize_rows(runs[mode], prefill_indices),
            "decode": summarize_rows(runs[mode], decode_indices),
        } for mode in MODES
    }
    session_decode = {
        mode: [run_mean(run, decode_indices, base.host_wall_us)
               for run in runs[mode]]
        for mode in MODES
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_decode["scalar"], session_decode["hvx_tile4"])]
    control_us = module_rows["scalar"]["decode"]["Complete Host wall"]
    candidate_us = module_rows["hvx_tile4"]["decode"]["Complete Host wall"]
    gates["candidate_strictly_faster_median_decode_wall"] = (
        candidate_us < control_us)
    gates["candidate_wins_all_ten_rotated_pairs"] = all(
        saved > 0.0 for saved in paired_saved_us)

    bucket_summary: dict[str, dict[str, dict[str, float]]] = {}
    for label, indices in BUCKETS:
        bucket_summary[label] = {
            mode: {
                "wall_us": median_metric(runs[mode], indices,
                                         base.host_wall_us),
                "softmax_us": median_metric(
                    runs[mode], indices,
                    lambda record: tick_us(
                        record, "u8_attention_softmax_ticks")),
                "attention_us": median_metric(
                    runs[mode], indices,
                    lambda record: base.generation_row_us(
                        record, "QK-Softmax-AV")),
            } for mode in MODES
        }

    attention = {
        mode: {
            field.replace("_ticks", "_us"): median_metric(
                runs[mode], decode_indices,
                lambda record, field=field: tick_us(record, field))
            for field in ATTENTION_FIELDS
        } for mode in MODES
    }
    slopes = {
        mode: {
            "host_wall_us_per_cache_token": slope(
                runs[mode], base.host_wall_us),
            "softmax_us_per_cache_token": slope(
                runs[mode], lambda record: tick_us(
                    record, "u8_attention_softmax_ticks")),
            "attention_us_per_cache_token": slope(
                runs[mode], lambda record: base.generation_row_us(
                    record, "QK-Softmax-AV")),
        } for mode in MODES
    }
    gates["all_required"] = all(gates.values())

    summary = {
        "experiment": "EXP-0170",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "sessions_per_mode": 10,
        "prefill_tokens": 64,
        "continuous_decode_tokens_per_session": 192,
        "gates": gates,
        "direct": {
            "scalar": {
                "prefill_wall_us": module_rows["scalar"]["prefill"]["Complete Host wall"],
                "prefill_tok_s": 64e6 / module_rows["scalar"]["prefill"]["Complete Host wall"],
                "decode_wall_us_per_token": control_us,
                "decode_tok_s": 1e6 / control_us,
            },
            "hvx_tile4": {
                "prefill_wall_us": module_rows["hvx_tile4"]["prefill"]["Complete Host wall"],
                "prefill_tok_s": 64e6 / module_rows["hvx_tile4"]["prefill"]["Complete Host wall"],
                "decode_wall_us_per_token": candidate_us,
                "decode_tok_s": 1e6 / candidate_us,
            },
            "decode_speed_percent": (control_us / candidate_us - 1.0) * 100.0,
            "decode_wall_reduction_percent": (1.0 - candidate_us / control_us) * 100.0,
            "pair_wins": sum(saved > 0.0 for saved in paired_saved_us),
            "paired_saved_us": paired_saved_us,
        },
        "module_rows_us": module_rows,
        "attention_us": attention,
        "buckets": bucket_summary,
        "linear_slopes": slopes,
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

    scalar_wall = control_us
    candidate_wall = candidate_us
    lines = [
        "# EXP-0170 HVX-tiled W4U8 decode Softmax", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Mode | Prefill tok/s | Decode wall/token | Decode tok/s |",
        "|---|---:|---:|---:|",
        f"| Scalar control | {summary['direct']['scalar']['prefill_tok_s']:.3f} | {control_us:.3f} us | {1e6/control_us:.3f} |",
        f"| HVX tile4 | {summary['direct']['hvx_tile4']['prefill_tok_s']:.3f} | {candidate_us:.3f} us | {1e6/candidate_us:.3f} |",
        f"| Change | diagnostic only | {-summary['direct']['decode_wall_reduction_percent']:.3f}% wall | +{summary['direct']['decode_speed_percent']:.3f}% |",
        "", "## Complete decode module table", "",
        "| Module | Scalar control | HVX tile4 | Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        scalar = module_rows["scalar"]["decode"][name]
        candidate = module_rows["hvx_tile4"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(scalar, scalar_wall)} | "
            f"{fmt(candidate, candidate_wall)} | "
            f"{(scalar / candidate - 1.0) * 100.0:+.2f}% |")
    lines += [
        "", "## Cache-length buckets", "",
        "| L before decode | Scalar wall | HVX wall | Scalar Softmax | HVX Softmax | Speed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, _ in BUCKETS:
        scalar = bucket_summary[label]["scalar"]
        candidate = bucket_summary[label]["hvx_tile4"]
        lines.append(
            f"| {label} | {scalar['wall_us']:.3f} us | "
            f"{candidate['wall_us']:.3f} us | {scalar['softmax_us']:.3f} us | "
            f"{candidate['softmax_us']:.3f} us | "
            f"{(scalar['wall_us']/candidate['wall_us']-1.0)*100.0:+.2f}% |")
    lines += ["", "## Attention substages", "",
              "| Substage | Scalar | HVX tile4 | Change |",
              "|---|---:|---:|---:|"]
    for field in attention["scalar"]:
        scalar = attention["scalar"][field]
        candidate = attention["hvx_tile4"][field]
        lines.append(
            f"| {field} | {scalar:.3f} us | {candidate:.3f} us | "
            f"{candidate - scalar:+.3f} us |")
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The candidate changes only M=1 dynamic W4U8 Softmax. "
        "Prefill, cache ABI and DMA, QK/AV HMX, projections, MLP and LM head "
        "remain unchanged. Semantic quality remains disabled by contract.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "scalar_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": summary["direct"]["decode_speed_percent"],
        "pair_wins": summary["direct"]["pair_wins"],
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0170 required gate failed")


if __name__ == "__main__":
    main()
