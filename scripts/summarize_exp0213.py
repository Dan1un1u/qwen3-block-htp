#!/usr/bin/env python3
"""Validate EXP-0213 deferred completion of Down8-tail next-Q32 prefetch."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0189 as base


CONTROL = "control"
CANDIDATE = "next_q"
EXPERIMENT = 213
LAYERS = 28
TRANSITIONS = LAYERS - 1


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


def validate_run(run: dict[str, object], cell: str, steps: int) -> None:
    enabled = 1 if cell == CANDIDATE else 0
    base.validate_run(
        run, 4, steps, experiment=EXPERIMENT,
        direct_mask=15, lm_head_direct=True)
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    for index, profile in enumerate(profiles):
        assert isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        expected_o_batch = 4 if index == 0 else 16
        expected_o_commands = 448 if index == 0 else 112
        checks = {
            "O-to-Gate selector": (
                "w4u8_decode_direct_n_o_gate_prefetch", 1),
            "Gate/Up stream selector": (
                "w4u8_decode_direct_n_gate_up_swiglu_stream", 1),
            "QKV parent batch": (
                "w4u8_decode_direct_n_qkv_batch_n_tiles", 16),
            "Q-only batch": (
                "w4u8_decode_direct_n_q_batch_n_tiles", 32),
            "Gate-to-Up parent": (
                "w4u8_decode_direct_n_gate_up_continuous", 1),
            "Down batch": (
                "w4u8_decode_direct_n_down_batch_n_tiles", 8),
            "Down single-DMA": (
                "w4u8_decode_direct_n_down_single_dma", 1),
            "O single-DMA": (
                "w4u8_decode_direct_n_o_single_dma", 1),
            "next-Q selector": (
                "w4u8_decode_direct_n_next_q_prefetch", enabled),
            "observed O batch": (
                "w4u8_o_batch_n_tiles_observed", expected_o_batch),
            "O command count": (
                "w4u8_o_batch_count", expected_o_commands),
        }
        for label, (field, expected) in checks.items():
            base.require_equal(int(profile[field]), expected, context + " " + label)
        if index > 0:
            base.require_equal(
                int(profile["w4u8_mlp_down_hmx_command_count"]),
                224, context + " Down command count")
            expected_transition_count = TRANSITIONS if enabled else 0
            for field in (
                "w4u8_next_q_prefetch_start_count",
                "w4u8_next_q_prefetch_consume_count",
                "w4u8_next_q_prefetch_overlap_count",
            ):
                base.require_equal(
                    int(profile[field]), expected_transition_count,
                    context + " " + field)


def validate_audit(result_dir: Path) -> dict[str, bool]:
    runs = {
        cell: base.load_run(result_dir / "raw" / f"audit_{cell}.log", 4)
        for cell in (CONTROL, CANDIDATE)
    }
    for cell, run in runs.items():
        validate_run(run, cell, 4)
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
    args = parse_args()
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
            validate_run(run, cell, args.steps)
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

    qkv = {
        CONTROL: base.median_module_us(
            runs[CONTROL], ("qkv_projection_ticks", "qk_norm_rope_ticks")),
        CANDIDATE: base.median_module_us(
            runs[CANDIDATE], ("qkv_projection_ticks", "qk_norm_rope_ticks")),
    }
    next_q = {
        "control_start_count": base.median_counter(
            runs[CONTROL], "w4u8_next_q_prefetch_start_count"),
        "control_consume_count": base.median_counter(
            runs[CONTROL], "w4u8_next_q_prefetch_consume_count"),
        "start_count": base.median_counter(
            runs[CANDIDATE], "w4u8_next_q_prefetch_start_count"),
        "consume_count": base.median_counter(
            runs[CANDIDATE], "w4u8_next_q_prefetch_consume_count"),
        "overlap_count": base.median_counter(
            runs[CANDIDATE], "w4u8_next_q_prefetch_overlap_count"),
        "wait_us": base.median_module_us(
            runs[CANDIDATE], ("w4u8_next_q_prefetch_wait_ticks",)),
        "lifetime_us": base.median_module_us(
            runs[CANDIDATE], ("w4u8_next_q_prefetch_lifetime_ticks",)),
    }
    physical = {
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
        "control_down_commands": base.median_counter(
            runs[CONTROL], "w4u8_mlp_down_hmx_command_count"),
        "candidate_down_commands": base.median_counter(
            runs[CANDIDATE], "w4u8_mlp_down_hmx_command_count"),
        "control_qkv_commands": base.median_counter(
            runs[CONTROL], "w4u8_qkv_ring_batch_count"),
        "candidate_qkv_commands": base.median_counter(
            runs[CANDIDATE], "w4u8_qkv_ring_batch_count"),
        "control_o_commands": base.median_counter(
            runs[CONTROL], "w4u8_o_batch_count"),
        "candidate_o_commands": base.median_counter(
            runs[CANDIDATE], "w4u8_o_batch_count"),
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
        "qkv_wall_faster": qkv[CANDIDATE] < qkv[CONTROL],
        "control_prefetch_disabled":
            next_q["control_start_count"] == 0 and
            next_q["control_consume_count"] == 0,
        "one_prefetch_per_layer_transition":
            next_q["start_count"] == TRANSITIONS and
            next_q["consume_count"] == TRANSITIONS and
            next_q["overlap_count"] == TRANSITIONS,
        "prefetch_wait_below_lifetime":
            next_q["wait_us"] < next_q["lifetime_us"],
        "hmx_commands_preserved":
            physical["candidate_direct_n_hmx_commands"] ==
            physical["control_direct_n_hmx_commands"] == 989,
        "hmx_tile_pairs_preserved":
            physical["candidate_hmx_tile_pairs"] ==
            physical["control_hmx_tile_pairs"],
        "weight_bytes_preserved":
            physical["candidate_weight_ddr_read_bytes"] ==
            physical["control_weight_ddr_read_bytes"],
        "dma_descriptors_preserved":
            physical["candidate_dma_descriptors"] ==
            physical["control_dma_descriptors"] == 1830,
        "down_commands_preserved_at_224":
            physical["control_down_commands"] == 224 and
            physical["candidate_down_commands"] == 224,
        "qkv_commands_preserved_at_168":
            physical["control_qkv_commands"] == 168 and
            physical["candidate_qkv_commands"] == 168,
        "o_commands_preserved_at_112":
            physical["control_o_commands"] == 112 and
            physical["candidate_o_commands"] == 112,
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
        "experiment": "EXP-0213",
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
        "qkv_us": qkv,
        "next_q_prefetch": next_q,
        "physical": physical,
        "audit": audit,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    report = [
        "# EXP-0213 deferred next-Q32 prefetch completion report",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        "Both cells inherit the EXP-0211 fastest enabling parent. The "
        "candidate starts each next layer's first Q32 packed-W4 weight DMA "
        "under the current layer's final Down8 HMX command, deliberately "
        "leaves it active across the layer boundary, and completes it at the "
        "next QKV consumer before fetching bias. Math, bytes, descriptors, "
        "HMX commands and tile pairs remain unchanged.",
        "",
        "## End-to-end gate",
        "",
        "| Metric | Control | Next-Q candidate | Change |",
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
        "| Module | Control | Candidate | Candidate speed change |",
        "|---|---:|---:|---:|",
        *module_rows,
        f"| Complete decode host wall | {control_latency * 1000.0:.1f} "
        f"(100.0%) | {candidate_latency * 1000.0:.1f} (100.0%) | "
        f"{(control_latency / candidate_latency - 1.0) * 100.0:+.1f}% |",
        "",
        "## Target and physical counters",
        "",
        f"QKV plus Q/K preparation wall: {qkv[CONTROL]:.1f} -> "
        f"{qkv[CANDIDATE]:.1f} us/token. Next-Q telemetry: "
        f"`{json.dumps(next_q, sort_keys=True)}`.",
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
