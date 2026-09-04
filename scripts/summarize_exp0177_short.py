#!/usr/bin/env python3
"""Validate the EXP-0177 full-carrier versus row-four AV short gate."""

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
VECTORS_PER_GROUP = {"full": 128, "row4": 8, "row4_poison": 8}
GROUPS_PER_STACK = LAYERS * 8
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0177-w4u8-decode-av-row4-requant",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> list[dict[str, object]]:
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
            and int(finals[0].get("requested_steps", -1)) == STEPS
            and int(finals[0].get("completed_steps", -1)) == STEPS):
        raise ValueError(f"failed final record: {path}")
    for index, (step, profile) in enumerate(zip(steps, profiles)):
        if not (int(step.get("experiment", -1)) == 177
                and int(profile.get("experiment", -1)) == 177
                and int(step.get("generation_step", -1)) == index
                and int(profile.get("generation_step", -1)) == index
                and int(step.get("generation_mode", -1)) == 9):
            raise ValueError(f"identity mismatch: {path}:{index}")
        profile["_step_record"] = step
        profile["_source_log"] = str(path)
    return profiles


def load_cell(result_dir: Path, cell: str) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"pair_??_{cell}.log"))
    if len(paths) != 5:
        raise ValueError(f"expected five {cell} logs, got {len(paths)}")
    return [load_run(path) for path in paths]


def run_mean(run: list[dict[str, object]], indices: tuple[int, ...],
             getter: Callable[[dict[str, object]], float]) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_metric(runs: list[list[dict[str, object]]],
                  indices: tuple[int, ...],
                  getter: Callable[[dict[str, object]], float]) -> float:
    return float(statistics.median(
        run_mean(run, indices, getter) for run in runs))


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_speed(control: float, candidate: float) -> str:
    if candidate == 0.0:
        return "+0.00%" if control == 0.0 else "removed/fused"
    return f"{(control / candidate - 1.0) * 100.0:+.2f}%"


def sequence(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_token_id"])
                 for item in run)


def logit_codes(run: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(item["_step_record"]["selected_logit_half_bits"])
                 for item in run)


def output_hashes(run: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(item["output_hash"]) for item in run)


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

    physical = True
    dispatch = True
    cache = True
    all_runs = [run for cell in CELLS for run in runs[cell]] + [poison]
    for run in all_runs:
        source_name = Path(str(run[0]["_source_log"])).name
        cell = ("row4_poison" if source_name == "padding_poison.log"
                else ("row4" if "row4" in source_name else "full"))
        for index, profile in enumerate(run):
            step = profile["_step_record"]
            expected_before = 0 if index == 0 else 63 + index
            expected_after = 64 if index == 0 else 64 + index
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
            requested_rows = ROWS_PER_CELL[cell]
            requested_poison = 1 if cell == "row4_poison" else 0
            if index == 0:
                dispatch &= bool(
                    int(profile["w4u8_decode_av_requant_rows"])
                        == requested_rows
                    and int(profile["w4u8_decode_av_padding_poison"])
                        == requested_poison
                    and int(profile["w4u8_av_requant_call_count"]) == 0
                    and int(profile["w4u8_av_requant_vector_count"]) == 0
                    and int(profile["w4u8_av_padding_poison_count"]) == 0
                )
            else:
                dispatch &= bool(
                    int(profile["w4u8_decode_av_requant_rows"])
                        == requested_rows
                    and int(profile["w4u8_decode_av_padding_poison"])
                        == requested_poison
                    and int(profile["w4u8_av_requant_rows_observed"])
                        == requested_rows
                    and int(profile["w4u8_av_requant_call_count"])
                        == GROUPS_PER_STACK
                    and int(profile["w4u8_av_requant_vector_count"])
                        == GROUPS_PER_STACK * VECTORS_PER_GROUP[cell]
                    and int(profile["w4u8_av_padding_poison_count"])
                        == requested_poison * GROUPS_PER_STACK
                )

    reference_run = runs["full"][0]
    exact_sequences = all(sequence(run) == sequence(reference_run)
                          for run in all_runs)
    exact_logits = all(logit_codes(run) == logit_codes(reference_run)
                       for run in all_runs)
    exact_hashes = all(output_hashes(run) == output_hashes(reference_run)
                       for run in all_runs)
    invariant_hmx = all(
        tuple(int(item["hmx_u8s8_tile_pair_count"]) for item in run)
        == tuple(int(item["hmx_u8s8_tile_pair_count"])
                 for item in reference_run)
        for run in all_runs)

    control_wall = rows["full"]["decode"]["Complete Host wall"]
    candidate_wall = rows["row4"]["decode"]["Complete Host wall"]
    control_attention = rows["full"]["decode"]["QK-Softmax-AV"]
    candidate_attention = rows["row4"]["decode"]["QK-Softmax-AV"]
    direct = {}
    for cell in CELLS:
        prefill_wall = rows[cell]["prefill"]["Complete Host wall"]
        decode_wall = rows[cell]["decode"]["Complete Host wall"]
        direct[cell] = {
            "prefill_wall_us": prefill_wall,
            "prefill_tok_s": 64e6 / prefill_wall,
            "decode_wall_us_per_token": decode_wall,
            "decode_tok_s": 1e6 / decode_wall,
            "decode_speed_percent_vs_control":
                (control_wall / decode_wall - 1.0) * 100.0,
        }

    paired = []
    for control, candidate in zip(runs["full"], runs["row4"]):
        control_wall_round = run_mean(
            control, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall"))
        candidate_wall_round = run_mean(
            candidate, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall"))
        control_attention_round = run_mean(
            control, decode_indices,
            lambda record: base.generation_row_us(
                record, "QK-Softmax-AV"))
        candidate_attention_round = run_mean(
            candidate, decode_indices,
            lambda record: base.generation_row_us(
                record, "QK-Softmax-AV"))
        paired.append({
            "decode_wall_speed_percent":
                (control_wall_round / candidate_wall_round - 1.0) * 100.0,
            "attention_speed_percent":
                (control_attention_round / candidate_attention_round - 1.0)
                * 100.0,
        })

    gates = {
        "all_11_sessions_physical_contract": physical,
        "cache_lengths_and_structure": cache,
        "requested_and_observed_AV_row_extent": dispatch,
        "unchanged_QK_AV_HMX_tile_work": invariant_hmx,
        "byte_exact_valid_hidden_hashes": exact_hashes,
        "byte_exact_selected_logit_codes": exact_logits,
        "identical_token_sequences": exact_sequences,
        "padding_poison_invariance": (
            output_hashes(poison) == output_hashes(reference_run)
            and logit_codes(poison) == logit_codes(reference_run)
            and sequence(poison) == sequence(reference_run)),
        "row4_lower_Attention_wall": candidate_attention < control_attention,
        "row4_lower_complete_decode_wall": candidate_wall < control_wall,
    }
    conclusion = (
        "advance_to_formal" if all(gates.values())
        else "reject_row4_without_193_step_formal_run"
    )
    summary = {
        "experiment": "EXP-0177",
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
        "formal_193_step_sections": (
            "pending" if conclusion == "advance_to_formal"
            else "not_run_by_approved_short_gate_stop_condition"
        ),
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

    walls = {cell: rows[cell]["decode"]["Complete Host wall"]
             for cell in CELLS}
    lines = [
        "# EXP-0177 row-selective AV requantization short gate", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack short-gate result", "",
        "| Cell | Prefill tok/s | Decode wall/token | Decode tok/s | "
        "Decode speed vs control |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
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
        "experiment": "EXP-0177", "conclusion": conclusion,
        "direct": direct, "paired_rounds": paired, "gates": gates,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
