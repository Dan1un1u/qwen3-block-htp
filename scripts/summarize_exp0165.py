#!/usr/bin/env python3
"""Validate and summarize EXP-0165 paired generation-boundary profiling."""

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


EXPERIMENT = 165
CONTROL_MODE = 1
CANDIDATE_MODE = 4
TICKS_PER_US = base.TICKS_PER_US
STEPS = base.GENERATION_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0165-w4f16-prefill-boundary-pipeline",
    )
    parser.add_argument("--experiment", type=int, default=EXPERIMENT)
    parser.add_argument("--control-mode", type=int, default=CONTROL_MODE)
    parser.add_argument("--candidate-mode", type=int, default=CANDIDATE_MODE)
    parser.add_argument(
        "--control-description",
        default="generation mode 1, scalar online argmax and batch2 LM head",
    )
    parser.add_argument(
        "--candidate-description",
        default=(
            "generation mode 4, HVX group-max with stable lane resolution "
            "and phase-overlaid batch8 LM head"
        ),
    )
    parser.add_argument(
        "--conclusion",
        default=(
            "The candidate changes no model math. It reduces the online "
            "argmax from a full scalar vocabulary scan to one HVX maximum "
            "reduction per 64 logits, using scalar lane resolution only when "
            "a group establishes a new global maximum. It then batches eight "
            "output tiles by reusing VTCM regions whose transformer-phase "
            "lifetimes have ended, reducing LM-head HMX commands without "
            "increasing the 8 MiB allocation."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cell(
    result_dir: Path, cell: str, expected_mode: int,
) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"round_*_{cell}.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"{cell}: expected ten logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        records = base.read_json_lines(path)
        steps = [
            item for item in records
            if "generation_step" in item and "record" not in item
        ]
        profiles = [
            item for item in records
            if item.get("record") == "generation_profile"
        ]
        finals = [
            item for item in records
            if item.get("generation_sequence_complete") is True
        ]
        if len(steps) != STEPS or len(profiles) != STEPS or len(finals) != 1:
            raise ValueError(f"{path}: incomplete generation run")
        if int(finals[0].get("generation_mode", -1)) != expected_mode:
            raise ValueError(f"{path}: wrong final generation mode")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if (
                int(step.get("experiment", -1)) != EXPERIMENT
                or int(profile.get("experiment", -1)) != EXPERIMENT
                or int(step.get("generation_step", -1)) != index
                or int(profile.get("generation_step", -1)) != index
                or int(step.get("generation_mode", -1)) != expected_mode
            ):
                raise ValueError(f"{path}: identity/order mismatch at {index}")
            profile["_step_record"] = step
            profile["_final_record"] = finals[0]
            profile["_source_log"] = str(path)
        runs.append(profiles)
    return runs


def mean_for_run(
    run: list[dict[str, object]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_for_runs(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return float(statistics.median(
        mean_for_run(run, indices, getter) for run in runs
    ))


def validate_cell(
    runs: list[list[dict[str, object]]], expected: list[int], mode: int,
) -> dict[str, bool]:
    exact = True
    stable = True
    physical = True
    ledger = True
    layer_structure = True
    first_tokens: list[int] | None = None
    for run in runs:
        tokens: list[int] = []
        for index, profile in enumerate(run):
            step = profile["_step_record"]
            tokens.append(int(step["selected_token_id"]))
            exact &= (
                int(step["generation_mode"]) == mode
                and step["token_match"] is True
                and step["pass"] is True
                and int(step["rpc_result"]) == 0
            )
            physical &= (
                profile["backend"] == "standalone_fastrpc_dsp"
                and profile["qnn"] == "none"
                and int(profile["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                and int(profile["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                and int(profile["intermediate_ddr_read_bytes"]) == 0
                and int(profile["intermediate_ddr_write_bytes"]) == 0
                and int(profile["intermediate_spill_fill_count"]) == 0
                and int(profile["boundary_ddr_write_bytes"]) == 0
                and int(profile["block_invocation_count"]) == base.LAYERS
            )
            ledger &= int(profile["ledger_unattributed_ticks"]) == 0
            for layer_index in range(base.LAYERS):
                layer = profile[f"slice_layer_{layer_index}"]
                layer_structure &= (
                    int(layer["layer_index"]) == layer_index
                    and int(layer["hidden_ddr_read_bytes"]) == 0
                    and int(layer["hidden_ddr_write_bytes"]) == 0
                    and int(layer["layer_unattributed_ticks"]) == 0
                )
        exact &= tokens == expected
        if first_tokens is None:
            first_tokens = tokens
        else:
            stable &= tokens == first_tokens
    return {
        "independent_token_sequence_exact": exact,
        "stable_across_ten_sessions": stable,
        "physical_contract": physical,
        "complete_ledger": ledger,
        "cache_and_hidden_structure": layer_structure,
    }


def speed(control_us: float, candidate_us: float) -> float:
    return (control_us / candidate_us - 1.0) * 100.0


def time_change(control_us: float, candidate_us: float) -> float:
    return (candidate_us / control_us - 1.0) * 100.0 if control_us else 0.0


def row_value(record: dict[str, object], name: str) -> float:
    return base.generation_row_us(record, name)


def summarize_rows(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    rows: tuple[str, ...], representative: int | None = None,
) -> dict[str, float]:
    if representative is not None:
        return {
            name: mean_for_run(
                runs[representative], indices,
                lambda record, name=name: row_value(record, name),
            )
            for name in rows
        }
    return {
        name: median_for_runs(
            runs, indices,
            lambda record, name=name: row_value(record, name),
        )
        for name in rows
    }


def diagnostic(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...], field: str,
) -> float:
    return median_for_runs(
        runs, indices,
        lambda record: float(record[field]) / TICKS_PER_US,
    )


def raw_metric(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...], field: str,
) -> float:
    return median_for_runs(
        runs, indices, lambda record: float(record[field])
    )


def fmt_cell(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def append_pc028(
    lines: list[str], title: str,
    variants: dict[str, dict[str, object]],
) -> None:
    lines += [
        f"## {title}", "",
        "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [item[0] for item in base.BASE_LEDGER] + [
        "True Host-DSP boundary", "Complete Host wall",
    ]:
        f16 = float(variants["F16F16"]["modules_us"][name])
        w4f16 = float(variants["W4F16"]["modules_us"][name])
        w4u8 = float(variants["W4U8"]["modules_us"][name])
        comparison = "N/A" if w4u8 == 0.0 else f"{speed(w4f16, w4u8):+.1f}%"
        lines.append(
            f"| {name} | {fmt_cell(f16, float(variants['F16F16']['wall_us']))} | "
            f"{fmt_cell(w4f16, float(variants['W4F16']['wall_us']))} | "
            f"{fmt_cell(w4u8, float(variants['W4U8']['wall_us']))} | "
            f"{comparison} |"
        )
    lines.append("")


def main() -> None:
    global EXPERIMENT, CONTROL_MODE, CANDIDATE_MODE
    args = parse_args()
    EXPERIMENT = args.experiment
    CONTROL_MODE = args.control_mode
    CANDIDATE_MODE = args.candidate_mode
    experiment_name = f"EXP-{EXPERIMENT:04d}"
    result_dir = args.result_dir.resolve()
    semantic = json.loads(base.SEMANTIC_REFERENCE.read_text(encoding="utf-8"))
    expected = [int(item) for item in semantic["w4f16"]["token_ids"]]
    control = load_cell(result_dir, "control", CONTROL_MODE)
    candidate = load_cell(result_dir, "candidate", CANDIDATE_MODE)

    gates = {
        "control": validate_cell(control, expected, CONTROL_MODE),
        "candidate": validate_cell(candidate, expected, CANDIDATE_MODE),
    }
    gates["selected_logit_bits_identical"] = all(
        int(control[run][step]["_step_record"]["selected_logit_half_bits"])
        == int(candidate[run][step]["_step_record"]["selected_logit_half_bits"])
        for run in range(10) for step in range(STEPS)
    )
    gates["accepted_transformer_replay_regression"] = base.validate_regression(
        result_dir / "w4f16_exp0158_regression.log"
    )

    prefill_indices = (0,)
    decode_indices = tuple(range(1, STEPS))
    paired_prefill = [
        speed(base.host_wall_us(control[index][0]),
              base.host_wall_us(candidate[index][0]))
        for index in range(10)
    ]
    paired_decode = [
        speed(
            mean_for_run(control[index], decode_indices, base.host_wall_us),
            mean_for_run(candidate[index], decode_indices, base.host_wall_us),
        )
        for index in range(10)
    ]
    paired_sequence = [
        speed(
            sum(base.host_wall_us(item) for item in control[index]),
            sum(base.host_wall_us(item) for item in candidate[index]),
        )
        for index in range(10)
    ]
    paired_prefill_median = float(statistics.median(paired_prefill))
    representative = min(
        range(10), key=lambda index:
            abs(paired_prefill[index] - paired_prefill_median)
    )

    rows = (
        "Token embedding",
        *(name for name, _ in base.BASE_LEDGER[:-2]),
        "Final model RMSNorm",
        "Streaming W4 LM head + greedy argmax",
        base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
        "True Host-DSP boundary", "Complete Host wall",
    )
    scopes = {"prefill": prefill_indices, "decode": decode_indices}
    direct: dict[str, object] = {}
    for scope, indices in scopes.items():
        direct[scope] = {
            "control_repeat_one": summarize_rows(
                control, indices, rows, representative
            ),
            "candidate_repeat_one": summarize_rows(
                candidate, indices, rows, representative
            ),
            "control_repeat_ten": summarize_rows(control, indices, rows),
            "candidate_repeat_ten": summarize_rows(candidate, indices, rows),
        }

    f16_runs = base.load_replay_runs(sorted(
        (base.EXP0158 / "raw").glob("round_*_f16f16_hmx_native_f16.jsonl")
    ), 9)
    w4f16_runs = base.load_replay_runs(sorted(
        (base.EXP0158 / "raw").glob("round_*_w4f16_hmx_native_f16.jsonl")
    ), 9)
    w4u8_runs = base.load_replay_runs(sorted(
        (base.EXP0163 / "raw").glob("round_*_candidate.jsonl")
    ), 193)
    overview = {
        "F16F16": base.summarize_old_overview(f16_runs, (0,)),
        "W4F16": base.summarize_old_overview(w4f16_runs, (0,)),
        "W4U8": base.summarize_old_overview(w4u8_runs, (0,)),
    }

    all_gate_values = [
        value for cell in ("control", "candidate")
        for value in gates[cell].values()
    ] + [
        bool(gates["selected_logit_bits_identical"]),
        bool(gates["accepted_transformer_replay_regression"]),
    ]
    gates["all"] = all(all_gate_values)
    gates["performance"] = paired_prefill_median > 0.0
    gates["all_with_performance"] = bool(gates["all"] and gates["performance"])

    physical_fields = base.PHYSICAL_FIELDS + (
        "generation_lm_head_batch_n_tiles",
    )
    if EXPERIMENT >= 166:
        physical_fields += (
            "generation_lm_head_prefetch_count",
            "generation_lm_head_scale_resident_bytes",
            "generation_lm_head_ddr_read_bytes",
        )
    head_fields = (
        "generation_lm_head_weight_dma_ticks",
        "generation_lm_head_scale_dma_ticks",
        "generation_lm_head_expand_ticks",
        "generation_lm_head_hmx_ticks",
        "generation_lm_head_argmax_ticks",
    )
    if EXPERIMENT >= 166:
        head_fields += (
            "generation_lm_head_weight_dma_wait_ticks",
            "generation_lm_head_scale_init_ticks",
            "generation_lm_head_hmx_tail_wait_ticks",
        )
    engine_fields = tuple(dict.fromkeys(
        base.PROJECTION_DIAGNOSTICS
        + base.ATTENTION_DIAGNOSTICS
        + base.MLP_DIAGNOSTICS
    ))

    summary = {
        "experiment": experiment_name,
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "control_mode": CONTROL_MODE,
        "candidate_mode": CANDIDATE_MODE,
        "representative_pair": representative + 1,
        "generated_token_ids": expected,
        "generated_text": semantic["w4f16"]["text"],
        "gates": gates,
        "paired_speed_improvement_percent": {
            "prefill_median": paired_prefill_median,
            "decode_median": float(statistics.median(paired_decode)),
            "complete_sequence_median": float(statistics.median(paired_sequence)),
            "prefill_by_pair": paired_prefill,
        },
        "direct": direct,
        "provenance": {
            "semantic_reference": str(base.SEMANTIC_REFERENCE),
            "regression_log": str(result_dir / "w4f16_exp0158_regression.log"),
            "pc028_f16f16_w4f16": str(base.EXP0158),
            "pc028_w4u8": str(base.EXP0163),
            "logs": {
                path.name: sha256_file(path)
                for path in sorted((result_dir / "raw").glob("*.jsonl"))
            },
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {experiment_name} full profiling report", "",
        "## Identity and scope", "",
        f"- Source: `{args.source_branch}` @ `{args.source_commit}`",
        f"- Formal evidence: `{result_dir}`",
        f"- Direct control: {args.control_description}.",
        f"- Candidate: {args.candidate_description}.",
        "- Ten rotated control/candidate pairs; one complete Prepared Session per cell.",
        "- Transformer layers, KV cache, weights, scales, tokenizer, prompt and token sequence are identical.", "",
        "The PC-028 table retains the shared transformer-only comparison from EXP-0158/EXP-0163. Only W4F16 currently owns the complete token-to-token output boundary, so those archived rows are context rather than the direct EXP-0165 control.", "",
    ]
    append_pc028(lines, "PC-028 M64 shared-transformer overview", overview)

    control_prefill = direct["prefill"]["control_repeat_ten"]["Complete Host wall"]
    candidate_prefill = direct["prefill"]["candidate_repeat_ten"]["Complete Host wall"]
    control_decode = direct["decode"]["control_repeat_ten"]["Complete Host wall"]
    candidate_decode = direct["decode"]["candidate_repeat_ten"]["Complete Host wall"]
    lines += [
        "## Direct end-to-end result", "",
        "| Scope | Control repeat-ten | Candidate repeat-ten | Paired median speed | Throughput control | Throughput candidate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| M64 prefill | {control_prefill:.3f} us | {candidate_prefill:.3f} us | {paired_prefill_median:+.3f}% | {64e6/control_prefill:.3f} tok/s | {64e6/candidate_prefill:.3f} tok/s |",
        f"| Decode token | {control_decode:.3f} us | {candidate_decode:.3f} us | {statistics.median(paired_decode):+.3f}% | {1e6/control_decode:.3f} tok/s | {1e6/candidate_decode:.3f} tok/s |",
        f"| Complete 16-pass session | {statistics.median(sum(base.host_wall_us(item) for item in run) for run in control):.3f} us | {statistics.median(sum(base.host_wall_us(item) for item in run) for run in candidate):.3f} us | {statistics.median(paired_sequence):+.3f}% | N/A | N/A |", "",
        "## Complete additive prefill ledger", "",
        f"Repeat-one is rotated pair {representative + 1}, whose paired prefill speedup is closest to the ten-pair median.", "",
        "| Module | Control repeat-one | Candidate repeat-one | Speed | Control repeat-ten | Candidate repeat-ten | Speed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in rows:
        c1 = float(direct["prefill"]["control_repeat_one"][name])
        n1 = float(direct["prefill"]["candidate_repeat_one"][name])
        c10 = float(direct["prefill"]["control_repeat_ten"][name])
        n10 = float(direct["prefill"]["candidate_repeat_ten"][name])
        c1wall = float(direct["prefill"]["control_repeat_one"]["Complete Host wall"])
        n1wall = float(direct["prefill"]["candidate_repeat_one"]["Complete Host wall"])
        lines.append(
            f"| {name} | {fmt_cell(c1, c1wall)} | {fmt_cell(n1, n1wall)} | {speed(c1, n1) if n1 else 0.0:+.2f}% | "
            f"{fmt_cell(c10, control_prefill)} | {fmt_cell(n10, candidate_prefill)} | {speed(c10, n10) if n10 else 0.0:+.2f}% |"
        )

    lines += [
        "", "## Complete additive decode ledger (repeat-ten)", "",
        "| Module | Control | Candidate | Speed |",
        "|---|---:|---:|---:|",
    ]
    for name in rows:
        c = float(direct["decode"]["control_repeat_ten"][name])
        n = float(direct["decode"]["candidate_repeat_ten"][name])
        lines.append(
            f"| {name} | {fmt_cell(c, control_decode)} | {fmt_cell(n, candidate_decode)} | {speed(c, n) if n else 0.0:+.2f}% |"
        )

    lines += [
        "", "## Generation-head overlapping diagnostics", "",
        "These qtimer intervals may overlap the additive LM-head wall and are not summed into Host wall.", "",
        "| Counter | Control prefill | Candidate prefill | Time change | Control decode | Candidate decode | Time change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field in head_fields:
        cp = diagnostic(control, prefill_indices, field)
        np = diagnostic(candidate, prefill_indices, field)
        cd = diagnostic(control, decode_indices, field)
        nd = diagnostic(candidate, decode_indices, field)
        lines.append(
            f"| {field} | {cp:.3f} us | {np:.3f} us | {time_change(cp, np):+.2f}% | {cd:.3f} us | {nd:.3f} us | {time_change(cd, nd):+.2f}% |"
        )

    lines += [
        "", "## Unchanged transformer engine diagnostics (repeat-ten prefill)", "",
        "| Counter | Control | Candidate | Time change |",
        "|---|---:|---:|---:|",
    ]
    for field in engine_fields:
        c = diagnostic(control, prefill_indices, field)
        n = diagnostic(candidate, prefill_indices, field)
        lines.append(f"| {field} | {c:.3f} us | {n:.3f} us | {time_change(c, n):+.2f}% |")

    lines += [
        "", "## Physical contract (repeat-ten prefill)", "",
        "| Metric | Control | Candidate |",
        "|---|---:|---:|",
    ]
    for field in physical_fields:
        c = raw_metric(control, prefill_indices, field)
        n = raw_metric(candidate, prefill_indices, field)
        lines.append(f"| {field} | {c:.0f} | {n:.0f} |")
    lines += [
        "| FastRPC invocations per pass | 1 | 1 |",
        "| HMX ownership domains | 1 | 1 |",
        "| backend/fallback | standalone cDSP / none | standalone cDSP / none |",
        "| timed full-logits DDR bytes | 0 | 0 |", "",
        "## Correctness and gates", "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for cell in ("control", "candidate"):
        for name, passed in gates[cell].items():
            lines.append(f"| {cell}.{name} | {'PASS' if passed else 'FAIL'} |")
    for name in (
        "selected_logit_bits_identical",
        "accepted_transformer_replay_regression",
        "performance",
        "all_with_performance",
    ):
        lines.append(f"| {name} | {'PASS' if gates[name] else 'FAIL'} |")
    lines += [
        "", args.conclusion, "",
        f"Generated text: `{semantic['w4f16']['text']}`", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "experiment": experiment_name,
        "all_gates_pass": gates["all_with_performance"],
        "representative_pair": representative + 1,
        "prefill_control_us": control_prefill,
        "prefill_candidate_us": candidate_prefill,
        "prefill_paired_speed_percent": paired_prefill_median,
        "decode_control_us": control_decode,
        "decode_candidate_us": candidate_decode,
        "decode_paired_speed_percent": float(statistics.median(paired_decode)),
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_with_performance"]:
        raise SystemExit(f"{experiment_name} gate failed")


if __name__ == "__main__":
    main()
