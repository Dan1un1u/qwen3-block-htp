#!/usr/bin/env python3
"""Validate and summarize EXP-0196 direct-W4 LM-head batching evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0189 as base


CONTROL = "expanded16"
CANDIDATE = "direct32"
VOCAB_N_TILES = 151936 // 32
K_TILES = 2048 // 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps", type=int, default=193)
    parser.add_argument("--require-audit", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--experiment", type=int, default=196)
    parser.add_argument("--control-group", type=int, default=16)
    parser.add_argument("--candidate-group", type=int, default=32)
    parser.add_argument("--control-mask", type=int, default=7)
    parser.add_argument("--candidate-mask", type=int, default=15)
    parser.add_argument("--control-direct", action="store_true")
    parser.add_argument("--control-cell", default=CONTROL)
    parser.add_argument("--candidate-cell", default=CANDIDATE)
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def signatures(run: dict[str, object]) -> tuple[list[int], list[int], list[str]]:
    summaries = run["summaries"]
    profiles = run["profiles"]
    assert isinstance(summaries, list) and isinstance(profiles, list)
    return (
        [int(value["selected_token_id"]) for value in summaries],
        [int(value["selected_logit_half_bits"]) for value in summaries],
        [str(value["output_hash"]) for value in profiles],
    )


def validate_run(
    run: dict[str, object], cell: str, steps: int, args: argparse.Namespace
) -> None:
    direct = cell == CANDIDATE or args.control_direct
    group_tiles = (
        args.candidate_group if cell == CANDIDATE else args.control_group)
    direct_mask = (
        args.candidate_mask if cell == CANDIDATE else args.control_mask)
    base.validate_run(
        run, 4, steps, experiment=args.experiment,
        direct_mask=direct_mask, lm_head_direct=direct)
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    for index, profile in enumerate(profiles):
        assert isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        base.require_equal(
            int(profile["w4u8_decode_lm_head_group_tiles"]),
            group_tiles, context + " requested LM-head batch")
        if index == 0:
            continue
        base.require_equal(
            int(profile["generation_lm_head_batch_n_tiles"]),
            group_tiles, context + " observed LM-head batch")


def validate_audit(
    result_dir: Path, args: argparse.Namespace
) -> dict[str, bool]:
    runs = {
        cell: base.load_run(result_dir / "raw" / f"audit_{cell}.log", 4)
        for cell in (CONTROL, CANDIDATE)
    }
    for cell, run in runs.items():
        validate_run(run, cell, 4, args)
    result = {
        "tokens_logit_codes_and_output_hashes_equal":
            signatures(runs[CONTROL]) == signatures(runs[CANDIDATE]),
        "audit_hidden_tensors_equal":
            base.audit_tree_hashes(result_dir / "audit" / CONTROL) ==
            base.audit_tree_hashes(result_dir / "audit" / CANDIDATE),
    }
    result["all_pass"] = all(result.values())
    return result


def main() -> None:
    global CONTROL, CANDIDATE
    args = parse_args()
    CONTROL = args.control_cell
    CANDIDATE = args.candidate_cell
    result_dir = args.result_dir.resolve()
    runs: dict[str, list[dict[str, object]]] = {
        CONTROL: [], CANDIDATE: []}
    pair_exact: list[bool] = []
    pair_speedups: list[float] = []
    pair_rows: list[str] = []

    for round_number in range(1, args.rounds + 1):
        pair: dict[str, dict[str, object]] = {}
        for cell in (CONTROL, CANDIDATE):
            path = result_dir / "raw" / f"pair_{round_number:02d}_{cell}.log"
            run = base.load_run(path, args.steps)
            validate_run(run, cell, args.steps, args)
            runs[cell].append(run)
            pair[cell] = run
        exact = signatures(pair[CONTROL]) == signatures(pair[CANDIDATE])
        control_tps = base.decode_tokens_per_second(pair[CONTROL])
        candidate_tps = base.decode_tokens_per_second(pair[CANDIDATE])
        speedup = candidate_tps / control_tps
        pair_exact.append(exact)
        pair_speedups.append(speedup)
        pair_rows.append(
            f"| {round_number} | {control_tps:.3f} | "
            f"{candidate_tps:.3f} | {(speedup - 1.0) * 100.0:+.2f}% | "
            f"{'yes' if exact else 'NO'} |")

    control_tps = median([
        base.decode_tokens_per_second(run) for run in runs[CONTROL]])
    candidate_tps = median([
        base.decode_tokens_per_second(run) for run in runs[CANDIDATE]])
    control_latency = median([
        base.decode_latency_ms(run) for run in runs[CONTROL]])
    candidate_latency = median([
        base.decode_latency_ms(run) for run in runs[CANDIDATE]])
    control_prefill = median([
        base.prefill_tokens_per_second(run) for run in runs[CONTROL]])
    candidate_prefill = median([
        base.prefill_tokens_per_second(run) for run in runs[CANDIDATE]])

    module_summary: dict[str, object] = {}
    module_rows: list[str] = []
    for label, fields in base.MODULES:
        control_us = base.median_module_us(runs[CONTROL], fields)
        candidate_us = base.median_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0
        module_summary[label] = {
            CONTROL: control_us,
            CANDIDATE: candidate_us,
            "candidate_speed_change_percent": speed,
        }
        module_rows.append(
            f"| {label} | {control_us:.1f} "
            f"({control_us / (control_latency * 10.0):.1f}%) | "
            f"{candidate_us:.1f} "
            f"({candidate_us / (candidate_latency * 10.0):.1f}%) | "
            f"{speed:+.1f}% |")
    control_host_us = base.median_host_boundary_us(runs[CONTROL])
    candidate_host_us = base.median_host_boundary_us(runs[CANDIDATE])
    host_speed = (control_host_us / candidate_host_us - 1.0) * 100.0
    module_summary["Host/RPC boundary"] = {
        CONTROL: control_host_us,
        CANDIDATE: candidate_host_us,
        "candidate_speed_change_percent": host_speed,
    }
    module_rows.append(
        f"| Host/RPC boundary | {control_host_us:.1f} "
        f"({control_host_us / (control_latency * 10.0):.1f}%) | "
        f"{candidate_host_us:.1f} "
        f"({candidate_host_us / (candidate_latency * 10.0):.1f}%) | "
        f"{host_speed:+.1f}% |")

    target_fields = {
        "lm_head_us": ("generation_lm_head_ticks",),
        "weight_dma_us": ("generation_lm_head_weight_dma_ticks",),
        "weight_dma_wait_us":
            ("generation_lm_head_weight_dma_wait_ticks",),
        "expand_us": ("generation_lm_head_expand_ticks",),
        "hmx_us": ("generation_lm_head_hmx_ticks",),
        "hmx_tail_wait_us":
            ("generation_lm_head_hmx_tail_wait_ticks",),
        "argmax_us": ("generation_lm_head_argmax_ticks",),
    }
    target = {
        name: {
            CONTROL: base.median_module_us(runs[CONTROL], fields),
            CANDIDATE: base.median_module_us(runs[CANDIDATE], fields),
        }
        for name, fields in target_fields.items()
    }

    control_commands = base.median_counter(
        runs[CONTROL], "generation_lm_head_command_count")
    candidate_commands = base.median_counter(
        runs[CANDIDATE], "generation_lm_head_command_count")
    physical = {
        "control_lm_head_commands": control_commands,
        "candidate_lm_head_commands": candidate_commands,
        "expected_control_lm_head_commands":
            (VOCAB_N_TILES + args.control_group - 1) // args.control_group,
        "expected_candidate_lm_head_commands":
            (VOCAB_N_TILES + args.candidate_group - 1) //
            args.candidate_group,
        "control_direct_n_hmx_commands": base.median_counter(
            runs[CONTROL], "w4u8_decode_direct_n_hmx_command_count"),
        "candidate_direct_n_hmx_commands": base.median_counter(
            runs[CANDIDATE], "w4u8_decode_direct_n_hmx_command_count"),
        "control_hmx_tile_pairs": base.median_counter(
            runs[CONTROL], "hmx_u8s8_tile_pair_count"),
        "candidate_hmx_tile_pairs": base.median_counter(
            runs[CANDIDATE], "hmx_u8s8_tile_pair_count"),
        "control_weight_ddr_read_bytes": base.median_counter(
            runs[CONTROL], "weight_ddr_read_bytes"),
        "candidate_weight_ddr_read_bytes": base.median_counter(
            runs[CANDIDATE], "weight_ddr_read_bytes"),
        "control_lm_head_ddr_read_bytes": base.median_counter(
            runs[CONTROL], "generation_lm_head_ddr_read_bytes"),
        "candidate_lm_head_ddr_read_bytes": base.median_counter(
            runs[CANDIDATE], "generation_lm_head_ddr_read_bytes"),
        "control_dma_descriptors": base.median_counter(
            runs[CONTROL], "weight_dma_descriptor_count"),
        "candidate_dma_descriptors": base.median_counter(
            runs[CANDIDATE], "weight_dma_descriptor_count"),
        "lm_head_hmx_tile_pairs": VOCAB_N_TILES * K_TILES,
        "vtcm_bytes": base.VTCM_BYTES,
        "intermediate_ddr_bytes": 0,
        "spill_fill": 0,
        "fastrpc_calls_per_token": 1,
        "hmx_owners": 1,
    }
    audit = validate_audit(result_dir, args) if args.require_audit else None
    expected_control_commands = (
        VOCAB_N_TILES + args.control_group - 1) // args.control_group
    expected_candidate_commands = (
        VOCAB_N_TILES + args.candidate_group - 1) // args.candidate_group
    control_expansion = base.median_counter(
        runs[CONTROL], "generation_lm_head_expand_ticks")
    gates = {
        "rotated_pairs_present": args.rounds in (5, 10),
        "all_pair_outputs_byte_exact": all(pair_exact),
        "all_pairs_candidate_faster": all(
            value > 1.0 for value in pair_speedups),
        "median_full_stack_decode_faster": candidate_tps > control_tps,
        "lm_head_wall_faster":
            target["lm_head_us"][CANDIDATE] < target["lm_head_us"][CONTROL],
        "lm_head_commands_match_groups":
            control_commands == expected_control_commands and
            candidate_commands == expected_candidate_commands,
        "candidate_lm_head_expansion_zero":
            base.median_counter(
                runs[CANDIDATE], "generation_lm_head_expand_ticks") == 0,
        "control_lm_head_expansion_matches_contract":
            (control_expansion == 0 if args.control_direct
             else control_expansion > 0),
        "hmx_tile_pairs_preserved":
            physical["candidate_hmx_tile_pairs"] ==
            physical["control_hmx_tile_pairs"],
        "weight_bytes_preserved":
            physical["candidate_weight_ddr_read_bytes"] ==
            physical["control_weight_ddr_read_bytes"],
        "lm_head_weight_bytes_preserved":
            physical["candidate_lm_head_ddr_read_bytes"] ==
            physical["control_lm_head_ddr_read_bytes"],
        "transformer_expansion_zero":
            base.median_counter(
                runs[CANDIDATE], "w4u8_qkvo_weight_expand_ticks") == 0 and
            base.median_counter(
                runs[CANDIDATE], "w4u8_mlp_weight_expand_ticks") == 0,
    }
    if audit is not None:
        gates["audit_pass"] = bool(audit["all_pass"])
    gates["all_pass"] = all(gates.values())

    summary = {
        "experiment": f"EXP-{args.experiment:04d}",
        "source_commit": args.source_commit,
        "rounds": args.rounds,
        "generation_steps": args.steps,
        "decode_steps_per_run": args.steps - 1,
        "control_decode_tok_s_median": control_tps,
        "candidate_decode_tok_s_median": candidate_tps,
        "control_decode_latency_ms_median": control_latency,
        "candidate_decode_latency_ms_median": candidate_latency,
        "median_paired_speedup": median(pair_speedups),
        "control_prefill_tok_s_median": control_prefill,
        "candidate_prefill_tok_s_median": candidate_prefill,
        "pair_speedups": pair_speedups,
        "module_us_per_decode_token": module_summary,
        "lm_head_breakdown_us": target,
        "physical": physical,
        "audit": audit,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    target_rows = []
    for label, values in target.items():
        control = values[CONTROL]
        candidate = values[CANDIDATE]
        speed_change = (
            "removed" if candidate == 0.0 and control > 0.0
            else f"{(control / candidate - 1.0) * 100.0:+.1f}%")
        target_rows.append(
            f"| {label} | {control:.1f} | {candidate:.1f} | "
            f"{speed_change} |")
    report = [
        f"# EXP-{args.experiment:04d} LM-head batch report",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        f"The control uses {'direct packed W4' if args.control_direct else 'Expanded-S8'} "
        f"LM-head groups of {args.control_group} tiles. The candidate keeps "
        f"every other path identical and uses direct packed-W4 groups of "
        f"{args.candidate_group} tiles in phase-dead VTCM buffers.",
        "",
        "## End-to-end gate",
        "",
        f"| Metric | {'Direct-W4' if args.control_direct else 'Expanded-S8'} "
        f"batch{args.control_group} | Direct-W4 batch{args.candidate_group} | Change |",
        "|---|---:|---:|---:|",
        f"| Decode throughput | {control_tps:.3f} tok/s | "
        f"{candidate_tps:.3f} tok/s | "
        f"{(candidate_tps / control_tps - 1.0) * 100.0:+.2f}% |",
        f"| Decode latency | {control_latency:.3f} ms/token | "
        f"{candidate_latency:.3f} ms/token | "
        f"{(candidate_latency / control_latency - 1.0) * 100.0:+.2f}% |",
        f"| M64 prefill | {control_prefill:.3f} tok/s | "
        f"{candidate_prefill:.3f} tok/s | unchanged implementation |",
        "",
        "## Rotated pairs",
        "",
        "| Pair | Control tok/s | Candidate tok/s | Change | Exact token/code/hash |",
        "|---:|---:|---:|---:|:---:|",
        *pair_rows,
        "",
        "## Decode module wall attribution",
        "",
        f"| Module | {'Direct-W4' if args.control_direct else 'Expanded-S8'} "
        f"batch{args.control_group} | Direct-W4 batch{args.candidate_group} | Candidate speed change |",
        "|---|---:|---:|---:|",
        *module_rows,
        f"| Complete decode host wall | {control_latency * 1000.0:.1f} "
        f"(100.0%) | {candidate_latency * 1000.0:.1f} (100.0%) | "
        f"{(control_latency / candidate_latency - 1.0) * 100.0:+.1f}% |",
        "",
        "## LM-head target breakdown",
        "",
        f"| Counter | {'Direct-W4' if args.control_direct else 'Expanded-S8'} "
        f"batch{args.control_group} us/token | Direct-W4 "
        f"batch{args.candidate_group} us/token | Speed change |",
        "|---|---:|---:|---:|",
        *target_rows,
        "",
        f"Physical counters: `{json.dumps(physical, sort_keys=True)}`.",
        "",
        f"Audit: `{json.dumps(audit, sort_keys=True) if audit else 'formal reuses short audit'}`.",
        "",
        f"Overall gate: `{'PASS' if gates['all_pass'] else 'FAIL'}`.",
    ]
    (result_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "result_dir": str(result_dir),
        "gates": gates,
        "control_decode_tok_s": control_tps,
        "candidate_decode_tok_s": candidate_tps,
        "candidate_speed_percent":
            (candidate_tps / control_tps - 1.0) * 100.0,
    }, sort_keys=True))
    if args.gate and not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
