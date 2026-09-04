#!/usr/bin/env python3
"""Validate EXP-0181 Attention-side V-cache publication evidence."""

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
KV_HEADS = 8
HEAD_TILES = 4
EXPERIMENT_NUMBER = 181
EXPERIMENT_NAME = "EXP-0181"
DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0181-w4u8-decode-attention-publish-v-tail"
)
CANDIDATE_FORMAT = 11
CANDIDATE_LABEL = "Attention-publish-v6"
REPORT_TITLE = "Attention-side quartet V-cache publication"
VTCM_TAIL_MODE = False
VTCM_TAIL_ATLAS_BYTES = LAYERS * KV_HEADS * 32 * 128
CELLS = ("control", "quartet")
ROWS = exp178.ROWS
TARGET_ROW = "QK-Softmax-AV"
DIAGNOSTICS = (
    "u8_attention_v_pack_ticks",
    "scan_cache_stage_ticks",
    "scan_cache_append_ticks",
    "u8_cache_native_append_update_ticks",
    "u8_attention_pipeline_wait_ticks",
)
COUNTERS = (
    "scan_cache_dma_descriptor_count",
    "scan_cache_ddr_read_bytes",
    "scan_cache_ddr_write_bytes",
    "u8_cache_v_quartet_append_count",
    "u8_cache_v_quartet_publish_count",
    "u8_cache_v_quartet_attention_publish_count",
    "u8_cache_v_quartet_partial_pack_rows",
    "u8_cache_v_quartet_full_tile_rmw_count",
    "u8_cache_v_quartet_native_load_bytes",
    "u8_cache_v_vtcm_tail_init_count",
    "u8_cache_v_vtcm_tail_row_update_count",
    "u8_cache_v_vtcm_tail_publish_count",
    "u8_cache_v_vtcm_tail_seal_count",
    "u8_cache_v_vtcm_tail_partial_pack_rows",
    "u8_cache_v_vtcm_tail_init_bytes",
    "u8_cache_v_vtcm_tail_native_load_bytes",
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
        default=DEFAULT_SOURCE_BRANCH,
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
        if not (int(step.get("experiment", -1)) == EXPERIMENT_NUMBER
                and int(profile.get("experiment", -1)) == EXPERIMENT_NUMBER
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
        raise ValueError(f"expected {rounds} {cell} logs, got {len(paths)}")
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


def validate_layout(run: list[dict[str, object]], cell: str) -> bool:
    candidate = cell == "quartet"
    for index, profile in enumerate(run):
        if not (
            int(profile["kv_cache_k_format"]) == 8
            and int(profile["kv_cache_v_format"]) == (
                CANDIDATE_FORMAT if candidate else 9)
            and int(profile.get(
                "u8_cache_v_quartet_full_tile_rmw_count", 0)) == 0
        ):
            return False
        quartet_fields = (
            "u8_cache_v_quartet_append_count",
            "u8_cache_v_quartet_publish_count",
            "u8_cache_v_quartet_attention_publish_count",
            "u8_cache_v_quartet_partial_pack_rows",
            "u8_cache_v_quartet_native_load_bytes",
        )
        vtcm_fields = (
            "u8_cache_v_vtcm_tail_init_count",
            "u8_cache_v_vtcm_tail_row_update_count",
            "u8_cache_v_vtcm_tail_publish_count",
            "u8_cache_v_vtcm_tail_seal_count",
            "u8_cache_v_vtcm_tail_partial_pack_rows",
            "u8_cache_v_vtcm_tail_init_bytes",
            "u8_cache_v_vtcm_tail_native_load_bytes",
        )
        if not candidate:
            if any(int(profile.get(field, 0)) != 0
                   for field in quartet_fields + vtcm_fields):
                return False
            continue
        if index == 0:
            if VTCM_TAIL_MODE:
                if (any(int(profile.get(field, 0)) != 0
                        for field in quartet_fields)
                        or int(profile.get(
                            "u8_cache_v_vtcm_tail_init_count", 0)) != 1
                        or int(profile.get(
                            "u8_cache_v_vtcm_tail_init_bytes", 0)) !=
                            VTCM_TAIL_ATLAS_BYTES
                        or any(int(profile.get(field, 0)) != 0
                               for field in vtcm_fields
                               if field not in (
                                   "u8_cache_v_vtcm_tail_init_count",
                                   "u8_cache_v_vtcm_tail_init_bytes"))):
                    return False
            elif any(int(profile.get(field, 0)) != 0
                     for field in quartet_fields):
                return False
            continue
        valid = int(profile["valid_length"])
        tail_rows = valid % 32
        groups = (tail_rows + 3) // 4
        expected_append = LAYERS * KV_HEADS
        expected_publish = expected_append if valid % 4 == 0 else 0
        expected_attention_publish = (expected_append
            if valid % 4 == 0 and tail_rows != 0 else 0)
        expected_partial = expected_append * (tail_rows % 4)
        expected_native_bytes = (
            LAYERS * KV_HEADS * HEAD_TILES * groups * 128
        )
        if VTCM_TAIL_MODE:
            if (any(int(profile.get(field, 0)) != 0
                    for field in quartet_fields)
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_init_count", 0)) != 0
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_init_bytes", 0)) != 0
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_row_update_count", 0)) !=
                        expected_append
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_publish_count", 0)) !=
                        expected_publish
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_seal_count", 0)) !=
                        (expected_append if tail_rows == 0 else 0)
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_partial_pack_rows", 0)) !=
                        expected_partial
                    or int(profile.get(
                        "u8_cache_v_vtcm_tail_native_load_bytes", 0)) !=
                        expected_native_bytes):
                return False
            continue
        if not (
            int(profile["u8_cache_v_quartet_append_count"])
                == expected_append
            and int(profile["u8_cache_v_quartet_publish_count"])
                == expected_publish
            and int(profile["u8_cache_v_quartet_attention_publish_count"])
                == expected_attention_publish
            and int(profile["u8_cache_v_quartet_partial_pack_rows"])
                == expected_partial
            and int(profile["u8_cache_v_quartet_native_load_bytes"])
                == expected_native_bytes
        ):
            return False
    return True


def validate_vtcm_read_contract(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> bool:
    if not VTCM_TAIL_MODE or len(control) != len(candidate):
        return not VTCM_TAIL_MODE
    for index, (control_profile, candidate_profile) in enumerate(
            zip(control, candidate)):
        control_read = int(control_profile["scan_cache_ddr_read_bytes"])
        candidate_read = int(candidate_profile["scan_cache_ddr_read_bytes"])
        if index == 0:
            expected_saved = 0
        else:
            tail_rows = int(candidate_profile["valid_length"]) % 32
            expected_saved = LAYERS * KV_HEADS * (
                tail_rows * 128 + (4096 if tail_rows == 0 else 0))
        if control_read - candidate_read != expected_saved:
            return False
        if (int(candidate_profile["scan_cache_ddr_write_bytes"])
                != int(control_profile["scan_cache_ddr_write_bytes"])):
            return False
        if (int(candidate_profile["vtcm_peak_plan_bytes"])
                - int(control_profile["vtcm_peak_plan_bytes"])
                != VTCM_TAIL_ATLAS_BYTES):
            return False
    return True


def validate_boundary(result_dir: Path) -> bool:
    control = load_run(result_dir / "boundary_control.log", 34)
    candidate = load_run(result_dir / "boundary_quartet.log", 34)
    return bool(
        exp178.validate_physical(control)
        and exp178.validate_physical(candidate)
        and validate_layout(control, "control")
        and validate_layout(candidate, "quartet")
        and validate_vtcm_read_contract(control, candidate)
        and sequence(control) == sequence(candidate)
        and logit_codes(control) == logit_codes(candidate)
        and output_hashes(control) == output_hashes(candidate)
    )


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
    reference = runs["control"][0]
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
                lambda record, field=field:
                float(record.get(field, 0)),
            ) for field in COUNTERS
        } for cell in CELLS
    }
    control_wall = module_rows["control"]["decode"]["Complete Host wall"]
    candidate_wall = module_rows["quartet"]["decode"]["Complete Host wall"]
    control_attention = module_rows["control"]["decode"][TARGET_ROW]
    candidate_attention = module_rows["quartet"]["decode"][TARGET_ROW]
    session_walls = {
        cell: [run_mean(
            run, decode_indices,
            lambda record: base.generation_row_us(
                record, "Complete Host wall")) for run in runs[cell]]
        for cell in CELLS
    }
    paired_saved_us = [control - candidate for control, candidate in zip(
        session_walls["control"], session_walls["quartet"])]
    pair_wins = sum(value > 0.0 for value in paired_saved_us)
    exact_sequences = all(sequence(run) == sequence(reference)
                          for run in all_runs + [poison])
    exact_logits = all(logit_codes(run) == logit_codes(reference)
                       for run in all_runs + [poison])
    exact_hashes = all(output_hashes(run) == output_hashes(reference)
                       for run in all_runs + [poison])
    invariant_work = all(
        tuple(int(item[field]) for item in run) ==
        tuple(int(item[field]) for item in reference)
        for run in all_runs + [poison]
        for field in (
            "hmx_u8s8_tile_pair_count", "hmx_command_count",
            "weight_ddr_read_bytes",
        )
    )
    gates: dict[str, bool] = {
        "all_performance_sessions_physical_contract": all(
            exp178.validate_physical(run) for run in all_runs),
        "control_and_candidate_layout_dispatch": all(
            validate_layout(run, cell)
            for cell in CELLS for run in runs[cell]),
        "vtcm_tail_exact_DDR_read_elimination_and_journal_writes": all(
            validate_vtcm_read_contract(control, candidate)
            for control, candidate in zip(
                runs["control"], runs["quartet"])),
        "quartet_and_segment_boundary_34_step_exact":
            validate_boundary(result_dir),
        "all_sessions_byte_exact_valid_hidden_hashes": exact_hashes,
        "all_sessions_identical_selected_logit_codes": exact_logits,
        "all_sessions_identical_token_sequences": exact_sequences,
        "unchanged_hmx_and_weight_work": invariant_work,
        "zero_full_tile_read_modify_write": all(
            int(item.get(
                "u8_cache_v_quartet_full_tile_rmw_count", 0)) == 0
            for run in runs["quartet"] for item in run),
        "padding_poison_physical_contract":
            exp178.validate_physical(poison),
        "padding_poison_quartet_dispatch":
            validate_layout(poison, "quartet"),
        "padding_poison_invariance": bool(
            output_hashes(poison) == output_hashes(reference)
            and logit_codes(poison) == logit_codes(reference)
            and sequence(poison) == sequence(reference)),
        "candidate_strictly_faster_attention":
            candidate_attention < control_attention,
        "candidate_strictly_faster_median_decode_wall":
            candidate_wall < control_wall,
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
                audit.get("summary", {}).get("implementation_gate") == "pass"
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
            "attention_us_per_token":
                module_rows[cell]["decode"][TARGET_ROW],
        } for cell in CELLS
    }
    direct["decode_speed_percent"] = (
        control_wall / candidate_wall - 1.0) * 100.0
    direct["attention_speed_percent"] = (
        control_attention / candidate_attention - 1.0) * 100.0
    direct["pair_wins"] = pair_wins
    direct["paired_saved_us"] = paired_saved_us

    provenance = {
        "static_gate_sha256": sha256_file(result_dir / "static_gate.json"),
        "boundary_control_sha256": sha256_file(
            result_dir / "boundary_control.log"),
        "boundary_quartet_sha256": sha256_file(
            result_dir / "boundary_quartet.log"),
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
        "experiment": EXPERIMENT_NAME,
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
        f"# {EXPERIMENT_NAME} {REPORT_TITLE}", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Cell | Prefill wall | Prefill tok/s | Decode wall/token | "
        "Decode tok/s | QK-Softmax-AV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        item = direct[cell]
        lines.append(
            f"| {cell} | {item['prefill_wall_us']:.3f} us | "
            f"{item['prefill_tok_s']:.3f} | "
            f"{item['decode_wall_us_per_token']:.3f} us | "
            f"{item['decode_tok_s']:.3f} | "
            f"{item['attention_us_per_token']:.3f} us |")
    lines += [
        f"| Change | diagnostic only | diagnostic only | "
        f"{candidate_wall - control_wall:+.3f} us | "
        f"{direct['decode_speed_percent']:+.3f}% | "
        f"{direct['attention_speed_percent']:+.3f}% |", "",
        "## Complete decode module table", "",
        f"| Module | Segmented-v4 control | {CANDIDATE_LABEL} candidate | "
        "Candidate speed |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        control = module_rows["control"]["decode"][name]
        candidate = module_rows["quartet"]["decode"][name]
        lines.append(
            f"| {name} | {fmt(control, control_wall)} | "
            f"{fmt(candidate, candidate_wall)} | "
            f"{fmt_speed(control, candidate)} |")
    lines += ["", "## Cache and Attention diagnostics", "",
              f"| Field | Segmented-v4 | {CANDIDATE_LABEL} | Delta |",
              "|---|---:|---:|---:|"]
    for field in DIAGNOSTICS:
        name = field.replace("_ticks", "_us")
        control = diagnostics["control"][name]
        candidate = diagnostics["quartet"][name]
        lines.append(
            f"| {name} | {control:.3f} us | {candidate:.3f} us | "
            f"{candidate - control:+.3f} us |")
    for field in COUNTERS:
        control = counters["control"][field]
        candidate = counters["quartet"][field]
        lines.append(
            f"| {field} | {control:.0f} | {candidate:.0f} | "
            f"{candidate - control:+.0f} |")
    lines += ["", "## Rotated-pair stability", "",
              f"Candidate wins: {pair_wins}/{args.rounds}; paired wall "
              "saved (us/token): " + ", ".join(
                  f"{value:+.3f}" for value in paired_saved_us),
              "", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "## Conclusion", "", conclusion, "",
              ("The candidate keeps the mutable V-tail carrier in the "
               "prepared-session VTCM allocation while retaining the "
               "segmented-v4 DDR journal and immutable segments. Attention "
               "loads and publishes quartet carriers entirely in VTCM."
               if VTCM_TAIL_MODE else
               "The candidate changes only the W4U8 mutable V-cache "
               "publication schedule. Attention converts the already-loaded "
               "fourth-row group for immediate AV consumption and persistent "
               "reuse; the append path performs no redundant group read.") +
              " K cache, Attention math, qparams, HMX work, projections, MLP "
              "and M64 prefill math remain unchanged.", ""]
    report_name = (
        "full_profiling_report.md" if args.formal
        else "short_gate_report.md")
    (result_dir / report_name).write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "experiment": EXPERIMENT_NAME,
        "conclusion": conclusion,
        "control_decode_tok_s": 1e6 / control_wall,
        "candidate_decode_tok_s": 1e6 / candidate_wall,
        "decode_speed_percent": direct["decode_speed_percent"],
        "attention_speed_percent": direct["attention_speed_percent"],
        "pair_wins": pair_wins,
        "gates": gates,
        "result_dir": str(result_dir),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    if not gates["all_required"]:
        raise SystemExit(f"{EXPERIMENT_NAME} required gate failed")


if __name__ == "__main__":
    main()
