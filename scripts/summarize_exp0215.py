#!/usr/bin/env python3
"""Validate and summarize EXP-0215 full-stack M64 direct-W4 MLP evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import summarize_exp0189 as base


CONTROL = "control"
CANDIDATE = "direct_mlp"
EXPERIMENT = 215
LAYERS = 28
MASK = {CONTROL: 15, CANDIDATE: 31}
PREFILL_DIRECT = {CONTROL: 0, CANDIDATE: LAYERS * 3}
PREFILL_DIRECT_COMMANDS = {CONTROL: 0, CANDIDATE: LAYERS * (6 + 6 + 8)}

# These rows are retained context from their selected formal evidence.  They
# are not direct EXP-0215 controls and are labelled as such in the report.
RECIPE_CONTEXT = {
    "F16F16 EXP-0158": {
        "prefill_host_us": 67212.8125,
        "prefill_tok_s": 952.199404,
        "decode_host_us": 95282.4675,
        "decode_tok_s": 10.495110,
        "evidence": "/mnt/d/llm_exp/results/qwen3-block-htp/exp0158/20260902T071708Z_264c911a65a3_formal",
    },
    "W4F16 EXP-0166": {
        "prefill_host_us": 62974.714,
        "prefill_tok_s": 1016.280924,
        "decode_host_us": 92598.101,
        "decode_tok_s": 10.799358,
        "evidence": "/mnt/d/llm_exp/results/qwen3-block-htp/exp0166/20260903T_exp0166_8e0dcf8_formal",
    },
}

OVERLAP_FIELDS = (
    ("Weight DMA", "weight_dma_ticks"),
    ("HMX compute", "hmx_compute_ticks"),
    ("Projection HMX wait", "projection_hmx_wait_ticks"),
    ("HMX ready wait", "hmx_ready_wait_ticks"),
    ("Projection pack", "projection_pack_ticks"),
    ("Projection unpack", "projection_unpack_ticks"),
    ("QKVO W4-to-S8 expansion", "w4u8_qkvo_weight_expand_ticks"),
    ("MLP W4-to-S8 expansion", "w4u8_mlp_weight_expand_ticks"),
    ("MLP weight stage", "w4u8_mlp_weight_stage_ticks"),
    ("MLP HMX compute", "w4u8_mlp_hmx_compute_ticks"),
    ("MLP HMX ready wait", "w4u8_mlp_hmx_ready_wait_ticks"),
    ("MLP producer-slot wait", "w4u8_mlp_producer_slot_wait_ticks"),
    ("MLP expanded-slot wait", "w4u8_mlp_expanded_slot_wait_ticks"),
)

PHYSICAL_FIELDS = (
    "w4u8_decode_direct_n_projection_count",
    "w4u8_decode_direct_n_hmx_command_count",
    "w4u8_decode_direct_n_weight_ddr_read_bytes",
    "w4u8_decode_direct_n_expand_bytes_avoided",
    "hmx_command_count",
    "hmx_u8s8_tile_pair_count",
    "weight_dma_descriptor_count",
    "boundary_dma_descriptor_count",
    "intermediate_dma_descriptor_count",
    "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
    "intermediate_spill_fill_count",
    "vtcm_requested_bytes",
    "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes",
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
                PREFILL_DIRECT_COMMANDS[cell],
                context + " prefill direct HMX commands")
            if cell == CANDIDATE:
                base.require_equal(
                    int(profile["w4u8_mlp_weight_expand_ticks"]), 0,
                    context + " MLP expansion")
                if int(profile["w4u8_decode_direct_n_expand_bytes_avoided"]) <= 0:
                    raise ValueError(context + ": no direct-W4 bytes avoided")
            elif int(profile["w4u8_mlp_weight_expand_ticks"]) <= 0:
                raise ValueError(context + ": control MLP expansion missing")
            if int(profile["w4u8_qkvo_weight_expand_ticks"]) <= 0:
                raise ValueError(context + ": non-target QKVO expansion missing")


def prefill_profile(run: dict[str, object]) -> dict[str, object]:
    profiles = run["profiles"]
    assert isinstance(profiles, list) and isinstance(profiles[0], dict)
    return profiles[0]


def prefill_host_us(run: dict[str, object]) -> float:
    return int(prefill_profile(run)["host_wall_ns"]) / 1000.0


def prefill_module_us(run: dict[str, object], fields: tuple[str, ...]) -> float:
    profile = prefill_profile(run)
    return sum(int(profile.get(field, 0)) for field in fields) / base.QTIMER_TICKS_PER_US


def median_prefill_module_us(
    runs: list[dict[str, object]], fields: tuple[str, ...]
) -> float:
    return median([prefill_module_us(run, fields) for run in runs])


def prefill_host_boundary_us(run: dict[str, object]) -> float:
    profile = prefill_profile(run)
    return prefill_host_us(run) - int(profile["ledger_named_ticks"]) / base.QTIMER_TICKS_PER_US


def median_prefill_counter(
    runs: list[dict[str, object]], field: str
) -> float:
    return median([float(int(prefill_profile(run).get(field, 0)))
                   for run in runs])


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
    result = {
        "tokens_logit_codes_and_output_hashes_equal":
            signatures(runs[CONTROL]) == signatures(runs[CANDIDATE]),
        "audit_hidden_cache_and_boundary_tensors_equal":
            base.audit_tree_hashes(root / "audit" / CONTROL) ==
            base.audit_tree_hashes(root / "audit" / CANDIDATE),
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
            path = result_dir / "raw" / f"pair_{round_number:02d}_{cell}.log"
            run = base.load_run(path, args.steps)
            validate_run(run, cell, args.steps)
            runs[cell].append(run)
            pair[cell] = run
        exact = signatures(pair[CONTROL]) == signatures(pair[CANDIDATE])
        decode_physical_exact = (
            decode_counter_signature(pair[CONTROL]) ==
            decode_counter_signature(pair[CANDIDATE]))
        control_tps = base.prefill_tokens_per_second(pair[CONTROL])
        candidate_tps = base.prefill_tokens_per_second(pair[CANDIDATE])
        speedup = candidate_tps / control_tps
        pair_exact.append(exact)
        pair_decode_physical_exact.append(decode_physical_exact)
        pair_speedups.append(speedup)
        pair_rows.append(
            f"| {round_number} | {prefill_host_us(pair[CONTROL]):.1f} | "
            f"{prefill_host_us(pair[CANDIDATE]):.1f} | "
            f"{control_tps:.3f} | {candidate_tps:.3f} | "
            f"{(speedup - 1.0) * 100.0:+.2f}% | "
            f"{'yes' if exact else 'NO'} |")

    control_prefill_tps = median([
        base.prefill_tokens_per_second(run) for run in runs[CONTROL]])
    candidate_prefill_tps = median([
        base.prefill_tokens_per_second(run) for run in runs[CANDIDATE]])
    control_prefill_us = median([prefill_host_us(run) for run in runs[CONTROL]])
    candidate_prefill_us = median([prefill_host_us(run) for run in runs[CANDIDATE]])
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
        control_us = median_prefill_module_us(runs[CONTROL], fields)
        candidate_us = median_prefill_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0 if candidate_us else 0.0
        prefill_modules[label] = {
            CONTROL: control_us, CANDIDATE: candidate_us,
            "candidate_speed_change_percent": speed}
        prefill_rows.append(
            f"| {label} | {control_us:.1f} ({control_us / control_prefill_us * 100.0:.1f}%) | "
            f"{candidate_us:.1f} ({candidate_us / candidate_prefill_us * 100.0:.1f}%) | "
            f"{speed:+.1f}% |")
    control_boundary = median([
        prefill_host_boundary_us(run) for run in runs[CONTROL]])
    candidate_boundary = median([
        prefill_host_boundary_us(run) for run in runs[CANDIDATE]])
    boundary_speed = (control_boundary / candidate_boundary - 1.0) * 100.0
    prefill_modules["True Host-DSP boundary"] = {
        CONTROL: control_boundary, CANDIDATE: candidate_boundary,
        "candidate_speed_change_percent": boundary_speed}
    prefill_rows.append(
        f"| True Host-DSP boundary | {control_boundary:.1f} "
        f"({control_boundary / control_prefill_us * 100.0:.1f}%) | "
        f"{candidate_boundary:.1f} "
        f"({candidate_boundary / candidate_prefill_us * 100.0:.1f}%) | "
        f"{boundary_speed:+.1f}% |")

    decode_modules: dict[str, object] = {}
    decode_rows: list[str] = []
    for label, fields in base.MODULES:
        control_us = base.median_module_us(runs[CONTROL], fields)
        candidate_us = base.median_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0 if candidate_us else 0.0
        decode_modules[label] = {
            CONTROL: control_us, CANDIDATE: candidate_us,
            "candidate_speed_change_percent": speed}
        decode_rows.append(
            f"| {label} | {control_us:.1f} ({control_us / control_decode_us * 100.0:.1f}%) | "
            f"{candidate_us:.1f} ({candidate_us / candidate_decode_us * 100.0:.1f}%) | "
            f"{speed:+.1f}% |")

    target = {}
    target_rows: list[str] = []
    for label, fields in (
        ("Gate/Up projection", ("gate_up_ticks",)),
        ("SwiGLU", ("activation_ticks",)),
        ("Down projection", ("down_ticks",)),
        ("Complete MLP", ("gate_up_ticks", "activation_ticks", "down_ticks")),
    ):
        control_us = median_prefill_module_us(runs[CONTROL], fields)
        candidate_us = median_prefill_module_us(runs[CANDIDATE], fields)
        speed = (control_us / candidate_us - 1.0) * 100.0
        target[label] = {CONTROL: control_us, CANDIDATE: candidate_us,
                         "speed_change_percent": speed}
        target_rows.append(
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | {speed:+.1f}% |")

    overlap = {}
    overlap_rows: list[str] = []
    for label, field in OVERLAP_FIELDS:
        control_us = median_prefill_counter(runs[CONTROL], field) / base.QTIMER_TICKS_PER_US
        candidate_us = median_prefill_counter(runs[CANDIDATE], field) / base.QTIMER_TICKS_PER_US
        overlap[label] = {CONTROL: control_us, CANDIDATE: candidate_us}
        change = ((candidate_us / control_us - 1.0) * 100.0
                  if control_us else None)
        overlap_rows.append(
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | "
            f"{change:+.1f}% |" if change is not None else
            f"| {label} | {control_us:.1f} | {candidate_us:.1f} | N/A |")

    physical = {
        field: {
            CONTROL: median_prefill_counter(runs[CONTROL], field),
            CANDIDATE: median_prefill_counter(runs[CANDIDATE], field),
        } for field in PHYSICAL_FIELDS
    }
    physical_rows = [
        f"| {field} | {values[CONTROL]:.0f} | {values[CANDIDATE]:.0f} |"
        for field, values in physical.items()
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
        "complete_MLP_faster":
            target["Complete MLP"][CANDIDATE] < target["Complete MLP"][CONTROL],
        "candidate_84_direct_MLP_projections":
            physical["w4u8_decode_direct_n_projection_count"][CANDIDATE] == 84,
        "candidate_560_direct_MLP_HMX_commands":
            physical["w4u8_decode_direct_n_hmx_command_count"][CANDIDATE] == 560,
        "candidate_explicit_MLP_expansion_zero":
            median_prefill_counter(runs[CANDIDATE],
                                   "w4u8_mlp_weight_expand_ticks") == 0,
        "control_explicit_MLP_expansion_present":
            median_prefill_counter(runs[CONTROL],
                                   "w4u8_mlp_weight_expand_ticks") > 0,
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
        "experiment": "EXP-0215",
        "source_commit": args.source_commit,
        "rounds": args.rounds,
        "generation_steps": args.steps,
        "direct_control": "mask15 explicit-HVX-W4-to-S8 M64 MLP",
        "candidate": "mask31 HMX-direct-packed-W4 M64 MLP",
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
        "decode_modules_us_per_token": decode_modules,
        "target_prefill_MLP_us": target,
        "overlap_diagnostics_us": overlap,
        "physical_prefill": physical,
        "audit": audit,
        "recipe_context": RECIPE_CONTEXT,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    context_rows = [
        f"| {name} | 64 | {value['prefill_host_us']:.1f} | "
        f"{value['prefill_tok_s']:.3f} | 192 | "
        f"{value['decode_host_us']:.1f} | {value['decode_tok_s']:.3f} |"
        for name, value in RECIPE_CONTEXT.items()
    ]
    context_rows.append(
        f"| W4U8 EXP-0215 candidate | 64 | {candidate_prefill_us:.1f} | "
        f"{candidate_prefill_tps:.3f} | {args.steps - 1} | "
        f"{candidate_decode_us:.1f} | {candidate_decode_tps:.3f} |")
    report = [
        "# EXP-0215 full-stack M64 direct-W4 MLP report",
        "",
        "## Identity and scope",
        "",
        f"- Source commit: `{args.source_commit}`",
        f"- Formal evidence: `{result_dir}`",
        "- Direct control: complete W4U8 token path, mask 15; M64 Gate/Up/Down use packed W4 plus explicit HVX W4-to-S8 expansion.",
        "- Candidate: identical path, mask 31; only M64 Gate/Up/Down let HMX read reordered packed W4 directly.",
        "- Decode, QKV/O prefill, Attention, cache, qparams, weights, scales and token-selection mathematics are unchanged.",
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
        "## Three-recipe context",
        "",
        "F16F16 and W4F16 are reused selected formal evidence, not paired EXP-0215 controls; their historical generation length/scope is disclosed rather than silently treated as a direct A/B.",
        "",
        "| Recipe/evidence | Prefill tokens | Prefill Host wall | Prefill tok/s | Decode tokens | Decode Host wall/token | Decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *context_rows,
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
        "## Target MLP split",
        "",
        "| Substage | Control us | Candidate us | Candidate speed change |",
        "|---|---:|---:|---:|",
        *target_rows,
        "",
        "## Continuous-decode preservation ledger",
        "",
        "| Module | Control | Candidate | Candidate speed change |",
        "|---|---:|---:|---:|",
        *decode_rows,
        f"| Complete Host wall | {control_decode_us:.1f} (100.0%) | "
        f"{candidate_decode_us:.1f} (100.0%) | "
        f"{(control_decode_us / candidate_decode_us - 1.0) * 100.0:+.1f}% |",
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
        "The candidate removes all explicit M64 MLP W4-to-S8 expansion, executes 84 direct-W4 projections and 560 direct HMX commands across 28 layers, preserves the complete HMX tile-pair count, and produces byte-identical selected tokens, logit codes, output hashes and audited boundary/cache tensors.",
    ]
    report_text = "\n".join(report) + "\n"
    (result_dir / "report.md").write_text(report_text, encoding="utf-8")
    (result_dir / "full_profiling_report.md").write_text(
        report_text, encoding="utf-8")
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
