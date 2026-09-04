#!/usr/bin/env python3
"""Validate and summarize formal EXP-0178 row-four common operators."""

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
import summarize_exp0178_short as short  # noqa: E402


STEPS = 193
LAYERS = 28
CELLS = ("full", "row4")
ROWS = short.ROWS
TARGET_ROWS = short.TARGET_ROWS
DIAGNOSTICS = (
    "w4u8_input_norm_main_work_ticks",
    "w4u8_input_norm_worker_work_ticks",
    "w4u8_input_norm_pool_wait_ticks",
    "w4u8_post_residual_main_work_ticks",
    "w4u8_post_residual_worker_work_ticks",
    "w4u8_post_residual_pool_wait_ticks",
    "w4u8_final_residual_main_work_ticks",
    "w4u8_final_residual_worker_work_ticks",
    "w4u8_final_residual_pool_wait_ticks",
    "weight_dma_ticks",
    "w4u8_qkvo_weight_expand_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks",
    "u8_attention_qk_hmx_ticks",
    "u8_attention_softmax_ticks",
    "u8_attention_av_hmx_ticks",
    "u8_attention_av_requant_ticks",
    "projection_hmx_wait_ticks",
)
COUNTERS = (
    "w4u8_input_norm_task_count",
    "w4u8_post_residual_task_count",
    "w4u8_final_residual_task_count",
    "w4u8_input_norm_direct_row4_call_count",
    "w4u8_post_residual_direct_row4_call_count",
    "w4u8_final_residual_direct_row4_call_count",
    "w4u8_common_padding_poison_count",
    "hmx_u8s8_tile_pair_count",
    "hmx_command_count",
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


def load_cell(result_dir: Path, cell: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{cell}.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten {cell} logs, got {len(paths)}")
    return [short.load_run(path, STEPS) for path in paths]


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


def sequence(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_token_id"])
                 for item in run)


def logit_codes(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_logit_half_bits"])
                 for item in run)


def output_hashes(run: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(item["output_hash"]) for item in run)


def validate_audit(path: Path) -> bool:
    records = base.read_json_lines(path)
    profiles = [item for item in records
                if item.get("record") == "generation_profile"]
    return bool(
        len(profiles) == STEPS
        and int(profiles[0].get("experiment", -1)) == 178
        and int(profiles[0].get("w4u8_common_op_rows_observed", -1)) == 64
        and int(profiles[0].get(
            "w4u8_input_norm_direct_row4_call_count", -1)) == 0
        and all(
            int(item.get("experiment", -1)) == 178
            and int(item.get("w4u8_decode_common_op_rows", -1)) == 4
            and int(item.get("w4u8_common_op_rows_observed", -1)) == 4
            and int(item.get(
                "w4u8_input_norm_direct_row4_call_count", -1)) == LAYERS
            and int(item.get(
                "w4u8_post_residual_direct_row4_call_count", -1)) == LAYERS
            and int(item.get(
                "w4u8_final_residual_direct_row4_call_count", -1)) == LAYERS
            for item in profiles[1:]
        )
    )


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
    poison = short.load_run(result_dir / "padding_poison.log", STEPS)
    all_runs = [run for cell in CELLS for run in runs[cell]]
    reference = runs["full"][0]
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
                lambda record, field=field: float(record[field]),
            ) for field in COUNTERS
        } for cell in CELLS
    }

    gates = {
        "all_20_performance_sessions_physical_contract": all(
            short.validate_physical(run) for run in all_runs),
        "requested_and_observed_common_op_row_extent": all(
            short.validate_dispatch(run, cell)
            for cell in CELLS for run in runs[cell]),
        "all_20_sessions_byte_exact_valid_hidden_hashes": len({
            output_hashes(run) for run in all_runs}) == 1,
        "all_20_sessions_identical_selected_logit_codes": len({
            logit_codes(run) for run in all_runs}) == 1,
        "all_20_sessions_identical_token_sequences": len({
            sequence(run) for run in all_runs}) == 1,
        "first16_EXP0168_tokens_and_codes_exact": bool(
            list(sequence(reference)[:16]) == exp173.EXPECTED_TOKENS
            and list(logit_codes(reference)[:16]) == exp173.EXPECTED_CODES),
        "unchanged_HMX_tile_work": all(
            tuple(int(item["hmx_u8s8_tile_pair_count"]) for item in run)
            == tuple(int(item["hmx_u8s8_tile_pair_count"])
                     for item in reference)
            for run in all_runs),
        "padding_poison_physical_contract":
            short.validate_physical(poison),
        "padding_poison_dispatch":
            short.validate_dispatch(poison, "row4_poison"),
        "padding_poison_invariance": bool(
            output_hashes(poison) == output_hashes(reference)
            and logit_codes(poison) == logit_codes(reference)
            and sequence(poison) == sequence(reference)),
        "audit_row4_geometry_193_steps": validate_audit(
            result_dir / "audit/device.jsonl"),
    }
    audit = json.loads(
        (result_dir / "audit/independent_reference.json").read_text())
    gates["independent_integer_head_reference_193_of_193"] = bool(
        audit.get("summary", {}).get("implementation_gate") == "pass"
        and int(audit.get("summary", {}).get("verified_steps", -1)) == STEPS
        and int(audit.get("summary", {}).get("token_and_code_matches", -1))
            == STEPS)
    gates["EXP0163_transformer_and_cache_regression"] = (
        exp167.validate_exp0163_regression(
            result_dir / "w4u8_exp0163_regression.log"))

    session_decode = {
        cell: [run_mean(
            run, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall")) for run in runs[cell]]
        for cell in CELLS
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_decode["full"], session_decode["row4"])]
    pair_wins = sum(saved > 0.0 for saved in paired_saved_us)
    control_us = module_rows["full"]["decode"]["Complete Host wall"]
    candidate_us = module_rows["row4"]["decode"]["Complete Host wall"]
    target = {
        cell: sum(module_rows[cell]["decode"][name]
                  for name in TARGET_ROWS) for cell in CELLS
    }
    gates["candidate_strictly_faster_combined_target_stage_wall"] = (
        target["row4"] < target["full"])
    gates["candidate_strictly_faster_median_decode_wall"] = (
        candidate_us < control_us)
    gates["candidate_wins_majority_of_ten_rotated_pairs"] = pair_wins >= 6
    gates["prefill_uses_unchanged_full_row_path"] = all(
        int(run[0]["w4u8_common_op_rows_observed"]) == 64
        and int(run[0]["w4u8_input_norm_direct_row4_call_count"]) == 0
        and int(run[0]["w4u8_post_residual_direct_row4_call_count"]) == 0
        and int(run[0]["w4u8_final_residual_direct_row4_call_count"]) == 0
        for run in all_runs)
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
            "target_stage_us_per_token": target[cell],
        } for cell in CELLS
    }
    direct["decode_speed_percent"] = (
        control_us / candidate_us - 1.0) * 100.0
    direct["decode_wall_reduction_percent"] = (
        1.0 - candidate_us / control_us) * 100.0
    direct["target_stage_speed_percent"] = (
        target["full"] / target["row4"] - 1.0) * 100.0
    direct["pair_wins"] = pair_wins
    direct["paired_saved_us"] = paired_saved_us

    summary = {
        "experiment": "EXP-0178",
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
            "padding_poison_sha256": sha256_file(
                result_dir / "padding_poison.log"),
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
        "# EXP-0178 decode W4U8 row-four common nonlinear operators", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Cell | Prefill Host wall | Prefill tok/s | Decode wall/token | "
        "Decode tok/s | Target three stages |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_wall_us']:.3f} us | "
            f"{item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
            f"{item['target_stage_us_per_token']:.3f} us |")
    lines += [
        f"| Change | diagnostic only | diagnostic only | "
        f"{-direct['decode_wall_reduction_percent']:.3f}% | "
        f"{direct['decode_speed_percent']:+.3f}% | "
        f"{direct['target_stage_speed_percent']:+.3f}% |", "",
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
    lines += ["", "## Common-op and pipeline diagnostics", "",
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
              f"Candidate wins: {pair_wins}/10; paired wall saved "
              "(us/token): "
              + ", ".join(f"{value:+.3f}" for value in paired_saved_us),
              "", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The candidate changes only the logical-M1 physical row extent "
        "of three W4U8 common nonlinear stages. It preserves exact valid-row "
        "math, all projection and Attention HMX work, cache state and every "
        "token. Semantic quality remains disabled by contract.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "control_decode_tok_s": 1e6 / control_us,
        "candidate_decode_tok_s": 1e6 / candidate_us,
        "decode_speed_percent": direct["decode_speed_percent"],
        "target_stage_speed_percent": direct["target_stage_speed_percent"],
        "pair_wins": pair_wins,
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0178 required gate failed")


if __name__ == "__main__":
    main()
