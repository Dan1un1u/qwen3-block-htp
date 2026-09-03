#!/usr/bin/env python3
"""Validate and summarize EXP-0167 W4U8 token-to-token generation."""

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


EXPERIMENT = 167
GENERATION_MODE = 8
STEPS = base.GENERATION_STEPS
W4F16_SUMMARY = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0166/"
    "20260903T_exp0166_8e0dcf8_formal/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0167-w4u8-greedy-generation",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(result_dir: Path) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob("generation_??.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten timed logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        records = base.read_json_lines(path)
        steps = [item for item in records if "generation_step" in item and "record" not in item]
        profiles = [item for item in records if item.get("record") == "generation_profile"]
        finals = [item for item in records if item.get("generation_sequence_complete") is True]
        if len(steps) != STEPS or len(profiles) != STEPS or len(finals) != 1:
            raise ValueError(f"incomplete generation run: {path}")
        if [int(item["generation_step"]) for item in profiles] != list(range(STEPS)):
            raise ValueError(f"unordered profiles: {path}")
        if int(finals[0].get("generation_mode", -1)) != GENERATION_MODE:
            raise ValueError(f"wrong generation mode: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if (
                int(step.get("experiment", -1)) != EXPERIMENT
                or int(profile.get("experiment", -1)) != EXPERIMENT
                or int(step.get("generation_step", -1)) != index
                or int(profile.get("generation_step", -1)) != index
            ):
                raise ValueError(f"identity mismatch at step {index}: {path}")
            profile["_step_record"] = step
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
    return float(statistics.median(mean_for_run(run, indices, getter) for run in runs))


def validate(runs: list[list[dict[str, object]]], audit: dict[str, object]) -> dict[str, bool]:
    sequences: list[list[int]] = []
    physical = True
    ledger = True
    layers = True
    execution = True
    encoding = True
    for run in runs:
        tokens: list[int] = []
        for index, profile in enumerate(run):
            step = profile["_step_record"]
            tokens.append(int(step["selected_token_id"]))
            execution &= (
                int(step["generation_mode"]) == GENERATION_MODE
                and int(step["rpc_result"]) == 0
                and bool(step["pass"])
                and int(profile["block_invocation_count"]) == base.LAYERS
            )
            encoding &= step.get("selected_logit_encoding") == "u8_code"
            physical &= (
                profile["backend"] == "standalone_fastrpc_dsp"
                and profile["qnn"] == "none"
                and int(profile["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                and int(profile["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                and int(profile["intermediate_ddr_read_bytes"]) == 0
                and int(profile["intermediate_ddr_write_bytes"]) == 0
                and int(profile["intermediate_spill_fill_count"]) == 0
                and int(profile["boundary_ddr_write_bytes"]) == 0
            )
            ledger &= int(profile["ledger_unattributed_ticks"]) == 0
            for layer_index in range(base.LAYERS):
                layer = profile[f"slice_layer_{layer_index}"]
                layers &= (
                    int(layer["layer_index"]) == layer_index
                    and int(layer["hidden_ddr_read_bytes"]) == 0
                    and int(layer["hidden_ddr_write_bytes"]) == 0
                    and int(layer["layer_unattributed_ticks"]) == 0
                )
        sequences.append(tokens)
    stable = all(sequence == sequences[0] for sequence in sequences[1:])
    audit_tokens = [int(item["selected_token_id"]) for item in audit["steps"]]
    return {
        "all_steps_execute": execution,
        "u8_logit_encoding_explicit": encoding,
        "stable_across_ten_sessions": stable,
        "matches_independent_audit_sequence": sequences[0] == audit_tokens,
        "independent_integer_reference": audit.get("implementation_gate") == "pass",
        "physical_contract": physical,
        "complete_ledger": ledger,
        "hidden_residency_structure": layers,
        "quality_gate_disabled": True,
    }


def row_summary(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...], rows: tuple[str, ...],
) -> dict[str, float]:
    return {
        name: median_for_runs(
            runs, indices,
            lambda record, name=name: base.generation_row_us(record, name),
        )
        for name in rows
    }


def speed(reference_us: float, candidate_us: float) -> float:
    return (reference_us / candidate_us - 1.0) * 100.0


def fmt(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = load_runs(result_dir)
    audit = json.loads((result_dir / "audit" / "independent_reference.json").read_text())
    gates = validate(runs, audit)
    gates["all_required"] = all(gates.values())

    prefill_indices = (0,)
    decode_indices = tuple(range(1, STEPS))
    rows = (
        "Token embedding",
        *(name for name, _ in base.BASE_LEDGER[:-2]),
        "Final model RMSNorm",
        "Streaming W4 LM head + greedy argmax",
        base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
        "True Host-DSP boundary", "Complete Host wall",
    )
    direct = {
        "prefill": row_summary(runs, prefill_indices, rows),
        "decode": row_summary(runs, decode_indices, rows),
    }
    prefill_us = direct["prefill"]["Complete Host wall"]
    decode_us = direct["decode"]["Complete Host wall"]
    sequence_us = float(statistics.median(
        sum(base.host_wall_us(item) for item in run) for run in runs
    ))
    token_ids = [int(item["_step_record"]["selected_token_id"]) for item in runs[0]]

    w4f16 = json.loads(W4F16_SUMMARY.read_text())
    w4f16_prefill = float(w4f16["direct"]["prefill"]["candidate_repeat_ten"]["Complete Host wall"])
    w4f16_decode = float(w4f16["direct"]["decode"]["candidate_repeat_ten"]["Complete Host wall"])
    comparison = {
        "w4f16_formal_summary": str(W4F16_SUMMARY),
        "prefill_wall_us": w4f16_prefill,
        "prefill_tok_s": 64e6 / w4f16_prefill,
        "decode_wall_us": w4f16_decode,
        "decode_tok_s": 1e6 / w4f16_decode,
        "w4u8_prefill_speed_percent": speed(w4f16_prefill, prefill_us),
        "w4u8_decode_speed_percent": speed(w4f16_decode, decode_us),
    }
    summary = {
        "experiment": "EXP-0167",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "generated_token_ids": token_ids,
        "quality_gate": "disabled_by_contract",
        "gates": gates,
        "direct": {
            **direct,
            "prefill_wall_us": prefill_us,
            "prefill_tok_s": 64e6 / prefill_us,
            "decode_wall_us_per_token": decode_us,
            "decode_tok_s": 1e6 / decode_us,
            "complete_16_step_session_us": sequence_us,
        },
        "w4f16_comparison": comparison,
        "provenance": {
            "audit_reference": str(result_dir / "audit" / "independent_reference.json"),
            "logs": {
                path.name: sha256_file(path)
                for path in sorted((result_dir / "raw").glob("*.log"))
            },
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# EXP-0167 W4U8 token-to-token generation", "",
        "## Scope", "",
        f"- Source: `{args.source_branch}` @ `{args.source_commit}`",
        "- Ten complete Prepared Sessions; each contains one M64 prefill and fifteen continuous decode steps.",
        "- W4U8 embedding, final RMSNorm output, W4 LM head, and greedy argmax run on the standalone cDSP path.",
        "- Semantic/readable-text quality is intentionally not a gate. The independent integer implementation gate remains mandatory.", "",
        "## Direct end-to-end result", "",
        "| Recipe | M64 prefill wall | Prefill throughput | Decode wall/token | Decode throughput |",
        "|---|---:|---:|---:|---:|",
        f"| W4F16 EXP-0166 | {w4f16_prefill:.3f} us | {64e6/w4f16_prefill:.3f} tok/s | {w4f16_decode:.3f} us | {1e6/w4f16_decode:.3f} tok/s |",
        f"| W4U8 EXP-0167 | {prefill_us:.3f} us | {64e6/prefill_us:.3f} tok/s | {decode_us:.3f} us | {1e6/decode_us:.3f} tok/s |",
        f"| W4U8 speed vs W4F16 | {comparison['w4u8_prefill_speed_percent']:+.3f}% | {comparison['w4u8_prefill_speed_percent']:+.3f}% | {comparison['w4u8_decode_speed_percent']:+.3f}% | {comparison['w4u8_decode_speed_percent']:+.3f}% |", "",
    ]
    for scope, title in (("prefill", "M64 prefill"), ("decode", "Decode token")):
        wall = float(direct[scope]["Complete Host wall"])
        lines += [f"## {title} additive ledger", "", "| Module | Time and share |", "|---|---:|"]
        for name in rows:
            lines.append(f"| {name} | {fmt(float(direct[scope][name]), wall)} |")
        lines.append("")
    lines += [
        "## Required gates", "", "| Gate | Result |", "|---|---:|",
    ]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "The selected tokens are deterministic and exactly reproduced by an independent integer reference over the captured final hidden rows. They are not required to match the W4F16 teacher and are not claimed to be semantically useful.", "",
        f"Generated token IDs: `{token_ids}`", "",
    ]
    (result_dir / "full_profiling_report.md").write_text("\n".join(lines))
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "prefill_wall_us": prefill_us,
        "prefill_tok_s": 64e6 / prefill_us,
        "decode_wall_us_per_token": decode_us,
        "decode_tok_s": 1e6 / decode_us,
        "w4u8_prefill_speed_percent": comparison["w4u8_prefill_speed_percent"],
        "w4u8_decode_speed_percent": comparison["w4u8_decode_speed_percent"],
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0167 required gate failed")


if __name__ == "__main__":
    main()
