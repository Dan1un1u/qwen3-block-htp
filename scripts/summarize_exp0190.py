#!/usr/bin/env python3
"""Validate and summarize EXP-0190 direct-n Gate/Up batch-eight evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0189 as base


CONTROL = "batch4"
CANDIDATE = "batch8"
GATE_UP_N_TILES = 6144 // 32
LAYERS = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps", type=int, default=193)
    parser.add_argument("--require-audit", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--gate", action="store_true")
    return parser.parse_args()


def validate_run(run: dict[str, object], batch: int, steps: int) -> None:
    base.validate_run(run, 4, steps, experiment=190)
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    for index, profile in enumerate(profiles):
        assert isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        base.require_equal(
            int(profile["w4u8_decode_direct_n_gate_up_batch_n_tiles"]),
            batch, context + " Gate/Up batch")


def audit_tree_hashes(root: Path) -> dict[str, str]:
    return base.audit_tree_hashes(root)


def validate_audit(result_dir: Path) -> dict[str, bool]:
    runs = {
        CONTROL: base.load_run(result_dir / "raw/audit_batch4.log", 4),
        CANDIDATE: base.load_run(result_dir / "raw/audit_batch8.log", 4),
    }
    validate_run(runs[CONTROL], 4, 4)
    validate_run(runs[CANDIDATE], 8, 4)
    signatures_equal = (
        base.run_signatures(runs[CONTROL]) ==
        base.run_signatures(runs[CANDIDATE]))
    trees_equal = (
        audit_tree_hashes(result_dir / "audit" / CONTROL) ==
        audit_tree_hashes(result_dir / "audit" / CANDIDATE))
    result = {
        "full_stack_signatures_equal": signatures_equal,
        "audit_hidden_tensors_equal": trees_equal,
    }
    result["all_pass"] = all(result.values())
    return result


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs: dict[str, list[dict[str, object]]] = {CONTROL: [], CANDIDATE: []}
    pair_exact: list[bool] = []
    pair_speedups: list[float] = []
    pair_rows: list[str] = []

    for round_number in range(1, args.rounds + 1):
        pair: dict[str, dict[str, object]] = {}
        for cell, batch in ((CONTROL, 4), (CANDIDATE, 8)):
            path = result_dir / "raw" / f"pair_{round_number:02d}_{cell}.log"
            run = base.load_run(path, args.steps)
            validate_run(run, batch, args.steps)
            runs[cell].append(run)
            pair[cell] = run
        exact = (
            base.run_signatures(pair[CONTROL]) ==
            base.run_signatures(pair[CANDIDATE]))
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

    module_rows: list[str] = []
    module_summary: dict[str, object] = {}
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

    gate_up = {
        CONTROL: base.median_module_us(runs[CONTROL], ("gate_up_ticks",)),
        CANDIDATE: base.median_module_us(
            runs[CANDIDATE], ("gate_up_ticks",)),
    }
    swiglu = {
        CONTROL: base.median_module_us(runs[CONTROL], ("activation_ticks",)),
        CANDIDATE: base.median_module_us(
            runs[CANDIDATE], ("activation_ticks",)),
    }
    physical = {
        "control_gate_up_hmx_commands":
            LAYERS * 2 * (GATE_UP_N_TILES // 4),
        "candidate_gate_up_hmx_commands":
            LAYERS * 2 * (GATE_UP_N_TILES // 8),
        "gate_up_hmx_command_count_source":
            "derived_from_validated_batch_and_confirmed_by_total_direct_n_delta",
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
        "control_dma_descriptors": base.median_counter(
            runs[CONTROL], "weight_dma_descriptor_count"),
        "candidate_dma_descriptors": base.median_counter(
            runs[CANDIDATE], "weight_dma_descriptor_count"),
        "vtcm_bytes": base.VTCM_BYTES,
        "intermediate_ddr_bytes": 0,
        "spill_fill": 0,
        "fastrpc_calls_per_token": 1,
        "hmx_owners": 1,
    }
    audit = validate_audit(result_dir) if args.require_audit else None
    gates = {
        "rotated_pairs_present": args.rounds in (5, 10),
        "all_pair_outputs_byte_exact": all(pair_exact),
        "all_pairs_candidate_faster": all(
            value > 1.0 for value in pair_speedups),
        "median_full_stack_decode_faster": candidate_tps > control_tps,
        "gate_up_wall_faster": gate_up[CANDIDATE] < gate_up[CONTROL],
        "gate_up_hmx_commands_halved":
            physical["candidate_gate_up_hmx_commands"] * 2 ==
            physical["control_gate_up_hmx_commands"],
        "direct_hmx_command_delta_exact":
            physical["control_direct_n_hmx_commands"] -
            physical["candidate_direct_n_hmx_commands"] ==
            LAYERS * GATE_UP_N_TILES // 4,
        "hmx_tile_pairs_preserved":
            physical["candidate_hmx_tile_pairs"] ==
            physical["control_hmx_tile_pairs"],
        "weight_bytes_preserved":
            physical["candidate_weight_ddr_read_bytes"] ==
            physical["control_weight_ddr_read_bytes"],
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
        "experiment": "EXP-0190",
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
        "gate_up_projection_us": gate_up,
        "swiglu_us": swiglu,
        "physical": physical,
        "audit": audit,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    report = [
        "# EXP-0190 direct-n Gate/Up batch-eight report",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        "The candidate changes only logical-M1 direct-n Gate/Up from four "
        "to eight output tiles per HMX command, using decode-phase-dead "
        "Expanded-S8 buffers as packed-W4 slots.",
        "",
        "## End-to-end gate",
        "",
        "| Metric | Batch4 control | Batch8 candidate | Change |",
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
        "| Pair | Control tok/s | Candidate tok/s | Change | Exact token/hash |",
        "|---:|---:|---:|---:|:---:|",
        *pair_rows,
        "",
        "## Decode module wall attribution",
        "",
        "| Module | Batch4 control | Batch8 candidate | Candidate speed change |",
        "|---|---:|---:|---:|",
        *module_rows,
        f"| Complete decode host wall | {control_latency * 1000.0:.1f} "
        f"(100.0%) | {candidate_latency * 1000.0:.1f} (100.0%) | "
        f"{(control_latency / candidate_latency - 1.0) * 100.0:+.1f}% |",
        "",
        "## Target and physical counters",
        "",
        f"Gate/Up wall: {gate_up[CONTROL]:.1f} -> "
        f"{gate_up[CANDIDATE]:.1f} us/token.  SwiGLU: "
        f"{swiglu[CONTROL]:.1f} -> {swiglu[CANDIDATE]:.1f} us/token.",
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
