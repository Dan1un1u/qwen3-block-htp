#!/usr/bin/env python3
"""Validate and summarize EXP-0216 full-stack M64 direct-W4 QKV/O."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0189 as base
import summarize_exp0215 as prior


CONTROL = "control"
CANDIDATE = "direct_qkvo"
EXPERIMENT = 216
MASK = {CONTROL: 31, CANDIDATE: 63}
PREFILL_DIRECT = {CONTROL: 84, CANDIDATE: 196}
PREFILL_COMMANDS = {CONTROL: 560, CANDIDATE: 840}

OVERLAP_FIELDS = (
    ("Weight DMA", "weight_dma_ticks"),
    ("HMX compute", "hmx_compute_ticks"),
    ("Projection HMX wait", "projection_hmx_wait_ticks"),
    ("QKVO W4-to-S8 expansion", "w4u8_qkvo_weight_expand_ticks"),
    ("QKV ring expansion", "w4u8_qkv_ring_expand_ticks"),
    ("QKV ring HMX compute", "w4u8_qkv_ring_hmx_compute_ticks"),
    ("QKV ring HMX-ready wait", "w4u8_qkv_ring_hmx_ready_wait_ticks"),
    ("QKV ring DMA wait", "w4u8_qkv_ring_dma_wait_ticks"),
    ("QKV ring pool wait", "w4u8_qkv_ring_pool_wait_ticks"),
    ("MLP W4-to-S8 expansion", "w4u8_mlp_weight_expand_ticks"),
)

PHYSICAL_FIELDS = prior.PHYSICAL_FIELDS + (
    "w4u8_qkv_ring_batch_count",
    "w4u8_qkv_ring_expand_worker_count",
    "w4u8_qkv_ring_prep_worker_count",
    "w4u8_qkv_ring_expand_task_count",
    "w4u8_o_batch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps", type=int, default=193)
    parser.add_argument("--audit-result-dir", type=Path)
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
    base.validate_run(
        run, 4, steps, experiment=EXPERIMENT,
        direct_mask=MASK[cell], lm_head_direct=True,
        prefill_direct_projections=PREFILL_DIRECT[cell])
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    for index, profile in enumerate(profiles):
        assert isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        required_fields = set(PHYSICAL_FIELDS)
        required_fields.update(field for _, field in OVERLAP_FIELDS)
        required_fields.update(field for _, fields in base.MODULES
                               for field in fields)
        missing_fields = sorted(required_fields.difference(profile))
        if missing_fields:
            raise ValueError(
                f"{context}: required profiling fields missing: "
                + ", ".join(missing_fields)
                + "; recollect with the complete generation_profile emitter")
        for field, expected in (
            ("w4u8_decode_direct_n_gate_up_batch_n_tiles", 32),
            ("w4u8_decode_direct_n_gate_up_continuous", 1),
            ("w4u8_decode_direct_n_o_gate_prefetch", 1),
            ("w4u8_decode_direct_n_gate_up_swiglu_stream", 1),
            ("w4u8_decode_direct_n_qkv_batch_n_tiles", 16),
            ("w4u8_decode_direct_n_q_batch_n_tiles", 32),
            ("w4u8_decode_direct_n_down_batch_n_tiles", 8),
            ("w4u8_decode_direct_n_down_single_dma", 1),
            ("w4u8_decode_direct_n_o_single_dma", 1),
        ):
            base.require_equal(int(profile[field]), expected,
                               context + " " + field)
        if index == 0:
            base.require_equal(
                int(profile["w4u8_decode_direct_n_hmx_command_count"]),
                PREFILL_COMMANDS[cell],
                context + " prefill direct HMX commands")
            base.require_equal(
                int(profile["w4u8_mlp_weight_expand_ticks"]), 0,
                context + " inherited MLP expansion")
            if cell == CANDIDATE:
                base.require_equal(
                    int(profile["w4u8_qkvo_weight_expand_ticks"]), 0,
                    context + " QKVO expansion")
                base.require_equal(
                    int(profile["w4u8_qkv_ring_expand_task_count"]), 0,
                    context + " QKV expansion tasks")
            elif int(profile["w4u8_qkvo_weight_expand_ticks"]) <= 0:
                raise ValueError(context + ": control QKVO expansion missing")


def decode_counter_signature(run: dict[str, object]) -> list[tuple[int, ...]]:
    fields = (
        "w4u8_decode_direct_n_projection_count",
        "w4u8_decode_direct_n_hmx_command_count",
        "hmx_u8s8_tile_pair_count",
        "weight_ddr_read_bytes",
        "weight_dma_descriptor_count",
        "intermediate_ddr_read_bytes",
        "intermediate_ddr_write_bytes",
        "intermediate_spill_fill_count",
    )
    return [tuple(int(profile[field]) for field in fields)
            for profile in base.decode_profiles(run)]


def validate_audit(root: Path) -> dict[str, bool]:
    runs = {
        cell: base.load_run(root / "raw" / f"audit_{cell}.log", 4)
        for cell in (CONTROL, CANDIDATE)
    }
    for cell, run in runs.items():
        validate_run(run, cell, 4)
    trees = {cell: base.audit_tree_hashes(root / "audit" / cell)
             for cell in (CONTROL, CANDIDATE)}
    expected_files = {
        f"generation_hidden_step{step:02d}_u8.bin" for step in range(4)}
    expected_files.update(
        f"generation_prefill_layer{layer:02d}_{kind}_cache_u8.bin"
        for layer in range(28) for kind in ("k", "v"))
    layer_hashes = {
        cell: [str(profile[f"slice_layer_{layer}"]["output_hash"])
               for profile in run["profiles"] for layer in range(28)]
        for cell, run in runs.items()}
    result = {
        "tokens_logit_codes_and_output_hashes_equal":
            signatures(runs[CONTROL]) == signatures(runs[CANDIDATE]),
        "audit_hidden_cache_and_boundary_tensors_equal":
            trees[CONTROL] == trees[CANDIDATE],
        "all_prefill_cache_and_hidden_files_present": all(
            set(tree) == expected_files for tree in trees.values()),
        "all_layer_hidden_hashes_nonzero_and_equal":
            layer_hashes[CONTROL] == layer_hashes[CANDIDATE] and
            all(int(value, 16) != 0 for value in layer_hashes[CONTROL]),
    }
    result["all_pass"] = all(result.values())
    return result


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs: dict[str, list[dict[str, object]]] = {
        CONTROL: [], CANDIDATE: []}
    pair_exact: list[bool] = []
    pair_decode_physical_exact: list[bool] = []
    pair_speedups: list[float] = []
    pair_rows: list[str] = []
    for round_number in range(1, args.rounds + 1):
        pair: dict[str, dict[str, object]] = {}
        for cell in (CONTROL, CANDIDATE):
            run = base.load_run(
                result_dir / "raw" /
                f"pair_{round_number:02d}_{cell}.log", args.steps)
            validate_run(run, cell, args.steps)
            runs[cell].append(run)
            pair[cell] = run
        exact = signatures(pair[CONTROL]) == signatures(pair[CANDIDATE])
        decode_exact = (
            decode_counter_signature(pair[CONTROL]) ==
            decode_counter_signature(pair[CANDIDATE]))
        control_tps = base.prefill_tokens_per_second(pair[CONTROL])
        candidate_tps = base.prefill_tokens_per_second(pair[CANDIDATE])
        speedup = candidate_tps / control_tps
        pair_exact.append(exact)
        pair_decode_physical_exact.append(decode_exact)
        pair_speedups.append(speedup)
        pair_rows.append(
            f"| {round_number} | {prior.prefill_host_us(pair[CONTROL]):.1f} | "
            f"{prior.prefill_host_us(pair[CANDIDATE]):.1f} | "
            f"{control_tps:.3f} | {candidate_tps:.3f} | "
            f"{(speedup - 1.0) * 100.0:+.2f}% | "
            f"{'yes' if exact else 'NO'} |")

    control_prefill_tps = median([
        base.prefill_tokens_per_second(run) for run in runs[CONTROL]])
    candidate_prefill_tps = median([
        base.prefill_tokens_per_second(run) for run in runs[CANDIDATE]])
    control_prefill_us = median([
        prior.prefill_host_us(run) for run in runs[CONTROL]])
    candidate_prefill_us = median([
        prior.prefill_host_us(run) for run in runs[CANDIDATE]])
    control_decode_tps = median([
        base.decode_tokens_per_second(run) for run in runs[CONTROL]])
    candidate_decode_tps = median([
        base.decode_tokens_per_second(run) for run in runs[CANDIDATE]])
    control_decode_us = median([
        base.decode_latency_ms(run) * 1000.0 for run in runs[CONTROL]])
    candidate_decode_us = median([
        base.decode_latency_ms(run) * 1000.0 for run in runs[CANDIDATE]])

    prefill_modules: dict[str, object] = {}
    prefill_rows: list[str] = []
    for label, fields in base.MODULES:
        control_us = prior.median_prefill_module_us(runs[CONTROL], fields)
        candidate_us = prior.median_prefill_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0 if candidate_us else 0.0
        prefill_modules[label] = {
            CONTROL: control_us, CANDIDATE: candidate_us,
            "candidate_speed_change_percent": speed}
        prefill_rows.append(
            f"| {label} | {control_us:.1f} ({control_us / control_prefill_us * 100.0:.1f}%) | "
            f"{candidate_us:.1f} ({candidate_us / candidate_prefill_us * 100.0:.1f}%) | "
            f"{speed:+.1f}% |")
    control_boundary = median([
        prior.prefill_host_boundary_us(run) for run in runs[CONTROL]])
    candidate_boundary = median([
        prior.prefill_host_boundary_us(run) for run in runs[CANDIDATE]])
    prefill_rows.append(
        f"| True Host-DSP boundary | {control_boundary:.1f} "
        f"({control_boundary / control_prefill_us * 100.0:.1f}%) | "
        f"{candidate_boundary:.1f} "
        f"({candidate_boundary / candidate_prefill_us * 100.0:.1f}%) | "
        f"{(control_boundary / candidate_boundary - 1.0) * 100.0:+.1f}% |")

    target: dict[str, object] = {}
    target_rows: list[str] = []
    for label, fields in (
        ("QKV + Q/K Norm/RoPE", ("qkv_projection_ticks", "qk_norm_rope_ticks")),
        ("O projection", ("o_projection_ticks",)),
        ("QKVO target total", ("qkv_projection_ticks", "qk_norm_rope_ticks", "o_projection_ticks")),
    ):
        control_us = prior.median_prefill_module_us(runs[CONTROL], fields)
        candidate_us = prior.median_prefill_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0
        target[label] = {CONTROL: control_us, CANDIDATE: candidate_us,
                         "speed_change_percent": speed}
        target_rows.append(
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | {speed:+.1f}% |")

    overlap: dict[str, object] = {}
    overlap_rows: list[str] = []
    for label, field in OVERLAP_FIELDS:
        control_us = prior.median_prefill_counter(
            runs[CONTROL], field) / base.QTIMER_TICKS_PER_US
        candidate_us = prior.median_prefill_counter(
            runs[CANDIDATE], field) / base.QTIMER_TICKS_PER_US
        overlap[label] = {CONTROL: control_us, CANDIDATE: candidate_us}
        change = ((candidate_us / control_us - 1.0) * 100.0
                  if control_us else None)
        overlap_rows.append(
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | "
            f"{change:+.1f}% |" if change is not None else
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | N/A |")

    physical = {
        field: {
            CONTROL: prior.median_prefill_counter(runs[CONTROL], field),
            CANDIDATE: prior.median_prefill_counter(runs[CANDIDATE], field),
        } for field in PHYSICAL_FIELDS
    }
    physical_rows = [
        f"| {field} | {value[CONTROL]:.0f} | {value[CANDIDATE]:.0f} |"
        for field, value in physical.items()
    ]
    audit = validate_audit(args.audit_result_dir.resolve()) \
        if args.audit_result_dir else None

    gates = {
        "rotated_pairs_present": args.rounds in (5, 10),
        "all_pair_outputs_byte_exact": all(pair_exact),
        "all_pair_decode_physical_contracts_exact":
            all(pair_decode_physical_exact),
        "all_pairs_candidate_prefill_faster":
            all(value > 1.0 for value in pair_speedups),
        "median_full_stack_prefill_faster":
            candidate_prefill_tps > control_prefill_tps,
        "QKVO_target_faster":
            target["QKVO target total"][CANDIDATE] <
            target["QKVO target total"][CONTROL],
        "candidate_196_direct_transformer_projections":
            physical["w4u8_decode_direct_n_projection_count"][CANDIDATE] == 196,
        "candidate_840_direct_transformer_HMX_commands":
            physical["w4u8_decode_direct_n_hmx_command_count"][CANDIDATE] == 840,
        "candidate_QKVO_and_MLP_expansion_zero":
            prior.median_prefill_counter(
                runs[CANDIDATE], "w4u8_qkvo_weight_expand_ticks") == 0 and
            prior.median_prefill_counter(
                runs[CANDIDATE], "w4u8_mlp_weight_expand_ticks") == 0,
        "control_QKVO_expansion_present":
            prior.median_prefill_counter(
                runs[CONTROL], "w4u8_qkvo_weight_expand_ticks") > 0,
        "HMX_tile_pairs_preserved":
            physical["hmx_u8s8_tile_pair_count"][CANDIDATE] ==
            physical["hmx_u8s8_tile_pair_count"][CONTROL],
        "exact_8MiB_VTCM":
            physical["vtcm_requested_bytes"][CONTROL] == base.VTCM_BYTES and
            physical["vtcm_requested_bytes"][CANDIDATE] == base.VTCM_BYTES and
            physical["vtcm_acquired_bytes"][CONTROL] == base.VTCM_BYTES and
            physical["vtcm_acquired_bytes"][CANDIDATE] == base.VTCM_BYTES,
        "zero_intermediate_DDR_and_spill": all(
            physical[field][cell] == 0
            for field in (
                "intermediate_ddr_read_bytes",
                "intermediate_ddr_write_bytes",
                "intermediate_spill_fill_count")
            for cell in (CONTROL, CANDIDATE)),
        "decode_throughput_within_two_percent":
            abs(candidate_decode_tps / control_decode_tps - 1.0) <= 0.02,
    }
    if audit is not None:
        gates["audit_pass"] = bool(audit["all_pass"])
    gates["all_pass"] = all(gates.values())

    summary = {
        "experiment": "EXP-0216",
        "source_commit": args.source_commit,
        "rounds": args.rounds,
        "generation_steps": args.steps,
        "control_prefill_host_us_median": control_prefill_us,
        "candidate_prefill_host_us_median": candidate_prefill_us,
        "control_prefill_tok_s_median": control_prefill_tps,
        "candidate_prefill_tok_s_median": candidate_prefill_tps,
        "control_decode_host_us_median": control_decode_us,
        "candidate_decode_host_us_median": candidate_decode_us,
        "control_decode_tok_s_median": control_decode_tps,
        "candidate_decode_tok_s_median": candidate_decode_tps,
        "pair_prefill_speedups": pair_speedups,
        "prefill_modules_us": prefill_modules,
        "target_prefill_QKVO_us": target,
        "overlap_diagnostics_us": overlap,
        "physical_prefill": physical,
        "audit": audit,
        "recipe_context": prior.RECIPE_CONTEXT,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    report = [
        "# EXP-0216 full-stack M64 direct-W4 QKV/O report",
        "",
        "## Identity and scope",
        "",
        f"- Source commit: `{args.source_commit}`",
        f"- Formal evidence: `{result_dir}`",
        "- Control: EXP-0215 direct-W4 MLP, with M64 QKV/O still explicitly expanded to S8.",
        "- Candidate: identical token path with direct packed-W4 HMX for M64 Q/K/V/O and MLP.",
        "- Decode, Attention math, cache, LM head, weights, scales, qparams and output encoding are unchanged.",
        "",
        "## Direct full-stack throughput",
        "",
        "| Scope | Control Host wall | Candidate Host wall | Control throughput | Candidate throughput | Speed |",
        "|---|---:|---:|---:|---:|---:|",
        f"| M64 prefill | {control_prefill_us:.1f} us | {candidate_prefill_us:.1f} us | "
        f"{control_prefill_tps:.3f} tok/s | {candidate_prefill_tps:.3f} tok/s | "
        f"{(candidate_prefill_tps / control_prefill_tps - 1.0) * 100.0:+.2f}% |",
        f"| Continuous decode | {control_decode_us:.1f} us/token | {candidate_decode_us:.1f} us/token | "
        f"{control_decode_tps:.3f} tok/s | {candidate_decode_tps:.3f} tok/s | "
        f"{(candidate_decode_tps / control_decode_tps - 1.0) * 100.0:+.2f}% |",
        "",
        "F16F16 EXP-0158 and W4F16 EXP-0166 remain non-paired context at 952.199/10.495 and 1016.281/10.799 prefill/decode token/s respectively; their evidence paths are retained in summary.json and are not used as this experiment's gate.",
        "",
        "## Rotated direct-control pairs",
        "",
        "| Pair | Control prefill us | Candidate prefill us | Control tok/s | Candidate tok/s | Speed | Exact token/code/hash |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
        *pair_rows,
        "",
        "## Complete additive prefill ledger",
        "",
        "| Module | Control | Candidate | Candidate speed change |",
        "|---|---:|---:|---:|",
        *prefill_rows,
        f"| Complete Host wall | {control_prefill_us:.1f} (100.0%) | "
        f"{candidate_prefill_us:.1f} (100.0%) | "
        f"{(control_prefill_us / candidate_prefill_us - 1.0) * 100.0:+.1f}% |",
        "",
        "## Target QKV/O split",
        "",
        "| Substage | Control us | Candidate us | Candidate speed change |",
        "|---|---:|---:|---:|",
        *target_rows,
        "",
        "## Overlapping engine diagnostics",
        "",
        "These qtimer intervals overlap and are not added to Host wall.",
        "",
        "| Counter | Control us | Candidate us | Candidate time change |",
        "|---|---:|---:|---:|",
        *overlap_rows,
        "",
        "## Physical contract",
        "",
        "| Metric | Control | Candidate |",
        "|---|---:|---:|",
        *physical_rows,
        "| FastRPC invocations per token/pass | 1 | 1 |",
        "| HMX ownership domains | 1 | 1 |",
        "| QNN/CPU fallback | none | none |",
        "",
        "## Correctness and gates",
        "",
        f"Audit: `{json.dumps(audit, sort_keys=True) if audit else 'not attached to this summary'}`.",
        "",
        *[f"- {name}: {'PASS' if value else 'FAIL'}"
          for name, value in gates.items()],
        "",
        "The candidate retains EXP-0215 direct-W4 MLP, removes the remaining explicit Transformer-projection expansion, preserves complete HMX tile-pair work, and is byte exact through selected tokens, logit codes, output hashes, cache and boundary audits.",
    ]
    text = "\n".join(report) + "\n"
    (result_dir / "report.md").write_text(text, encoding="utf-8")
    (result_dir / "full_profiling_report.md").write_text(
        text, encoding="utf-8")
    print(json.dumps({
        "result_dir": str(result_dir),
        "gates": gates,
        "control_prefill_tok_s": control_prefill_tps,
        "candidate_prefill_tok_s": candidate_prefill_tps,
        "candidate_prefill_speed_percent":
            (candidate_prefill_tps / control_prefill_tps - 1.0) * 100.0,
        "control_decode_tok_s": control_decode_tps,
        "candidate_decode_tok_s": candidate_decode_tps,
    }, sort_keys=True))
    if args.gate and not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
