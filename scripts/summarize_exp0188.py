#!/usr/bin/env python3
"""Validate and summarize EXP-0188 full-stack decode direct-n evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


QTIMER_TICKS_PER_US = 19.2
VTCM_BYTES = 8 * 1024 * 1024
CONTROL = "expanded_s8"
CANDIDATE = "direct_n"
DIRECT_MASK = 7
DIRECT_PROJECTIONS_PER_DECODE = 28 * 7

MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("I/O and metadata", (
        "input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks")),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm/RoPE", (
        "qkv_projection_ticks", "qk_norm_rope_ticks")),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-attn residual + RMSNorm", (
        "post_attention_residual_ticks", "post_attention_norm_ticks")),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
    ("KV-cache pack + append", (
        "scan_cache_pack_ticks", "scan_cache_append_ticks")),
    ("Embedding/final norm/LM head", (
        "generation_embedding_ticks", "generation_final_norm_ticks",
        "generation_lm_head_ticks")),
    ("DSP runtime/orchestration", (
        "runtime_setup_ticks", "runtime_teardown_ticks",
        "stage_boundary_ticks", "block_orchestration_ticks",
        "layer_bookkeeping_ticks")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=193)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def load_json_objects(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def load_run(path: Path, steps: int) -> dict[str, object]:
    records = load_json_objects(path)
    summaries = [record for record in records
                 if "generation_step" in record and "selected_token_id" in record]
    profiles = [record for record in records
                if record.get("record") == "generation_profile"]
    summaries.sort(key=lambda value: int(value["generation_step"]))
    profiles.sort(key=lambda value: int(value["generation_step"]))
    if len(summaries) != steps or len(profiles) != steps:
        raise ValueError(
            f"{path}: expected {steps} summaries/profiles, got "
            f"{len(summaries)}/{len(profiles)}")
    expected_steps = list(range(steps))
    if [int(value["generation_step"]) for value in summaries] != expected_steps:
        raise ValueError(f"{path}: non-contiguous summary steps")
    if [int(value["generation_step"]) for value in profiles] != expected_steps:
        raise ValueError(f"{path}: non-contiguous profile steps")
    return {"path": path, "summaries": summaries, "profiles": profiles}


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def validate_run(run: dict[str, object], cell: str, steps: int) -> None:
    summaries = run["summaries"]
    profiles = run["profiles"]
    assert isinstance(summaries, list) and isinstance(profiles, list)
    candidate = cell == CANDIDATE
    for index, (summary, profile) in enumerate(zip(summaries, profiles)):
        assert isinstance(summary, dict) and isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        require_equal(int(summary["experiment"]), 188, context + " experiment")
        require_equal(int(profile["experiment"]), 188, context + " profile experiment")
        require_equal(int(summary["rpc_result"]), 0, context + " RPC")
        require_equal(int(summary["dsp_status"]), 3, context + " DSP status")
        require_equal(bool(summary["pass"]), True, context + " pass")
        require_equal(int(profile["dsp_status"]), 3, context + " profile DSP")
        require_equal(int(profile["numerical_status"]), 1, context + " numerical")
        require_equal(int(profile["intermediate_ddr_read_bytes"]), 0,
                      context + " intermediate DDR read")
        require_equal(int(profile["intermediate_ddr_write_bytes"]), 0,
                      context + " intermediate DDR write")
        require_equal(int(profile["intermediate_spill_fill_count"]), 0,
                      context + " spill/fill")
        require_equal(int(profile["vtcm_acquired_bytes"]), VTCM_BYTES,
                      context + " VTCM acquired")
        if int(profile["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
            raise ValueError(f"{context}: VTCM plan exceeds 8 MiB")
        require_equal(int(profile["ledger_unattributed_ticks"]), 0,
                      context + " unattributed ledger")
        require_equal(int(profile["ledger_named_ticks"]),
                      int(profile["invocation_ticks"]), context + " ledger closure")
        require_equal(int(profile.get("cache_mismatches", 0)), 0,
                      context + " cache mismatch")
        require_equal(int(profile.get("cache_prefix_mismatches", 0)), 0,
                      context + " cache prefix mismatch")
        require_equal(int(profile.get("cache_structure_mismatches", 0)), 0,
                      context + " cache structure mismatch")
        expected_mode = "prefill" if index == 0 else "decode"
        require_equal(summary["mode"], expected_mode, context + " summary mode")
        require_equal(profile["mode"], expected_mode, context + " profile mode")
        expected_projection_mode = 1 if candidate else 0
        expected_mask = DIRECT_MASK if candidate else 0
        require_equal(int(profile["w4u8_decode_projection_mode"]),
                      expected_projection_mode, context + " direct-n mode")
        require_equal(int(profile["w4u8_decode_direct_n_mask"]),
                      expected_mask, context + " direct-n mask")
        direct_count = int(profile["w4u8_decode_direct_n_projection_count"])
        if index == 0 or not candidate:
            require_equal(direct_count, 0, context + " direct projection count")
        else:
            require_equal(direct_count, DIRECT_PROJECTIONS_PER_DECODE,
                          context + " direct projection count")
            if int(profile["w4u8_decode_direct_n_hmx_command_count"]) <= 0:
                raise ValueError(f"{context}: no direct-n HMX commands")
            if int(profile["w4u8_decode_direct_n_weight_ddr_read_bytes"]) <= 0:
                raise ValueError(f"{context}: no direct-n weight reads")
            if int(profile["w4u8_decode_direct_n_expand_bytes_avoided"]) <= 0:
                raise ValueError(f"{context}: no expansion bytes avoided")
            require_equal(int(profile["w4u8_qkvo_weight_expand_ticks"]), 0,
                          context + " QKVO expansion")
            require_equal(int(profile["w4u8_mlp_weight_expand_ticks"]), 0,
                          context + " MLP expansion")
            if int(profile["generation_lm_head_expand_ticks"]) <= 0:
                raise ValueError(f"{context}: LM-head fallback expansion missing")


def validate_pair(control: dict[str, object], candidate: dict[str, object]) -> bool:
    control_summaries = control["summaries"]
    candidate_summaries = candidate["summaries"]
    control_profiles = control["profiles"]
    candidate_profiles = candidate["profiles"]
    assert isinstance(control_summaries, list) and isinstance(candidate_summaries, list)
    assert isinstance(control_profiles, list) and isinstance(candidate_profiles, list)
    token_equal = [int(value["selected_token_id"]) for value in control_summaries] == [
        int(value["selected_token_id"]) for value in candidate_summaries]
    hash_equal = [str(value["output_hash"]) for value in control_profiles] == [
        str(value["output_hash"]) for value in candidate_profiles]
    return token_equal and hash_equal


def decode_profiles(run: dict[str, object]) -> list[dict[str, object]]:
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    return [profile for profile in profiles if profile["mode"] == "decode"]


def decode_latency_ms(run: dict[str, object]) -> float:
    profiles = decode_profiles(run)
    return sum(int(profile["host_wall_ns"]) for profile in profiles) / len(profiles) / 1e6


def decode_tokens_per_second(run: dict[str, object]) -> float:
    return 1000.0 / decode_latency_ms(run)


def prefill_tokens_per_second(run: dict[str, object]) -> float:
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    return 64e9 / int(profiles[0]["host_wall_ns"])


def median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def per_token_ticks(run: dict[str, object], fields: tuple[str, ...]) -> float:
    profiles = decode_profiles(run)
    return sum(sum(int(profile.get(field, 0)) for field in fields)
               for profile in profiles) / len(profiles)


def median_module_us(runs: list[dict[str, object]], fields: tuple[str, ...]) -> float:
    return median(per_token_ticks(run, fields) / QTIMER_TICKS_PER_US for run in runs)


def median_host_boundary_us(runs: list[dict[str, object]]) -> float:
    values: list[float] = []
    for run in runs:
        profiles = decode_profiles(run)
        host_us = sum(int(profile["host_wall_ns"]) for profile in profiles) \
            / len(profiles) / 1000.0
        dsp_us = sum(int(profile["ledger_named_ticks"]) for profile in profiles) \
            / len(profiles) / QTIMER_TICKS_PER_US
        values.append(host_us - dsp_us)
    return median(values)


def median_counter(runs: list[dict[str, object]], field: str) -> float:
    values: list[float] = []
    for run in runs:
        profiles = decode_profiles(run)
        values.append(sum(int(profile[field]) for profile in profiles) / len(profiles))
    return median(values)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs: dict[str, list[dict[str, object]]] = {CONTROL: [], CANDIDATE: []}
    pair_exact: list[bool] = []
    pair_speedups: list[float] = []
    pair_rows: list[str] = []

    for round_number in range(1, args.rounds + 1):
        pair: dict[str, dict[str, object]] = {}
        for cell in (CONTROL, CANDIDATE):
            path = result_dir / "raw" / f"pair_{round_number:02d}_{cell}.log"
            if not path.is_file():
                raise ValueError(f"missing formal log: {path}")
            run = load_run(path, args.steps)
            validate_run(run, cell, args.steps)
            runs[cell].append(run)
            pair[cell] = run
        exact = validate_pair(pair[CONTROL], pair[CANDIDATE])
        control_tps = decode_tokens_per_second(pair[CONTROL])
        candidate_tps = decode_tokens_per_second(pair[CANDIDATE])
        speedup = candidate_tps / control_tps
        pair_exact.append(exact)
        pair_speedups.append(speedup)
        pair_rows.append(
            f"| {round_number} | {control_tps:.3f} | {candidate_tps:.3f} | "
            f"{(speedup - 1.0) * 100.0:+.2f}% | {'yes' if exact else 'NO'} |")

    control_tps = median(decode_tokens_per_second(run) for run in runs[CONTROL])
    candidate_tps = median(decode_tokens_per_second(run) for run in runs[CANDIDATE])
    control_latency = median(decode_latency_ms(run) for run in runs[CONTROL])
    candidate_latency = median(decode_latency_ms(run) for run in runs[CANDIDATE])
    control_prefill = median(prefill_tokens_per_second(run) for run in runs[CONTROL])
    candidate_prefill = median(prefill_tokens_per_second(run) for run in runs[CANDIDATE])
    median_paired_speedup = median(pair_speedups)

    module_rows: list[str] = []
    module_summary: dict[str, object] = {}
    for label, fields in MODULES:
        control_us = median_module_us(runs[CONTROL], fields)
        candidate_us = median_module_us(runs[CANDIDATE], fields)
        module_summary[label] = {
            CONTROL: control_us,
            CANDIDATE: candidate_us,
            "candidate_speed_change_percent":
                (control_us / candidate_us - 1.0) * 100.0 if candidate_us else 0.0,
        }
        module_rows.append(
            f"| {label} | {control_us:.1f} ({control_us / (control_latency * 10.0):.1f}%) | "
            f"{candidate_us:.1f} ({candidate_us / (candidate_latency * 10.0):.1f}%) | "
            f"{(control_us / candidate_us - 1.0) * 100.0:+.1f}% |")
    control_host_us = median_host_boundary_us(runs[CONTROL])
    candidate_host_us = median_host_boundary_us(runs[CANDIDATE])
    module_summary["Host/RPC boundary"] = {
        CONTROL: control_host_us,
        CANDIDATE: candidate_host_us,
        "candidate_speed_change_percent":
            (control_host_us / candidate_host_us - 1.0) * 100.0
            if candidate_host_us else 0.0,
    }
    module_rows.append(
        f"| Host/RPC boundary | {control_host_us:.1f} "
        f"({control_host_us / (control_latency * 10.0):.1f}%) | "
        f"{candidate_host_us:.1f} "
        f"({candidate_host_us / (candidate_latency * 10.0):.1f}%) | "
        f"{(control_host_us / candidate_host_us - 1.0) * 100.0:+.1f}% |")

    gates = {
        "ten_rotated_pairs_present": args.rounds == 10,
        "all_pair_outputs_byte_exact": all(pair_exact),
        "all_pairs_candidate_faster": all(speedup > 1.0 for speedup in pair_speedups),
        "median_full_stack_decode_faster": candidate_tps > control_tps,
        "candidate_uses_196_transformer_direct_n_projections":
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_direct_n_projection_count")
            == DIRECT_PROJECTIONS_PER_DECODE,
        "candidate_transformer_expansion_zero":
            median_counter(runs[CANDIDATE], "w4u8_qkvo_weight_expand_ticks") == 0
            and median_counter(runs[CANDIDATE], "w4u8_mlp_weight_expand_ticks") == 0,
        "lm_head_expanded_s8_fallback_retained":
            median_counter(runs[CANDIDATE], "generation_lm_head_expand_ticks") > 0,
    }
    gates["all_pass"] = all(gates.values())

    summary = {
        "experiment": "EXP-0188",
        "source_commit": args.source_commit,
        "rounds": args.rounds,
        "generation_steps": args.steps,
        "decode_steps_per_run": args.steps - 1,
        "candidate_direct_n_mask": DIRECT_MASK,
        "candidate_scope": "QKV,O,Gate,Up,Down; LM-head remains Expanded-S8",
        "control_decode_tok_s_median": control_tps,
        "candidate_decode_tok_s_median": candidate_tps,
        "control_decode_latency_ms_median": control_latency,
        "candidate_decode_latency_ms_median": candidate_latency,
        "median_paired_speedup": median_paired_speedup,
        "control_prefill_tok_s_median": control_prefill,
        "candidate_prefill_tok_s_median": candidate_prefill,
        "pair_speedups": pair_speedups,
        "module_us_per_decode_token": module_summary,
        "candidate_direct_weight_mib_per_token":
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_direct_n_weight_ddr_read_bytes") / 2**20,
        "candidate_expansion_mib_avoided_per_token":
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_direct_n_expand_bytes_avoided") / 2**20,
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# EXP-0188 W4U8 full-stack decode direct-n formal report",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        "The candidate uses HMX weight.n packed-W4 input for all 196 transformer "
        "projections per decode token (Q/K/V/O and Gate/Up/Down across 28 layers). "
        "The LM head remains on the accepted group16 Expanded-S8 path because its "
        "direct-n diagnostic changed U8 argmax tie-breaking and was slower.",
        "",
        "## End-to-end gate",
        "",
        "| Metric | Expanded-S8 control | Transformer direct-n | Change |",
        "|---|---:|---:|---:|",
        f"| Decode throughput | {control_tps:.3f} tok/s | {candidate_tps:.3f} tok/s | "
        f"{(candidate_tps / control_tps - 1.0) * 100.0:+.2f}% |",
        f"| Decode latency | {control_latency:.3f} ms/token | "
        f"{candidate_latency:.3f} ms/token | "
        f"{(candidate_latency / control_latency - 1.0) * 100.0:+.2f}% |",
        f"| Prefill throughput (unchanged kernel) | {control_prefill:.3f} tok/s | "
        f"{candidate_prefill:.3f} tok/s | measurement only |",
        "",
        "## Ten rotated pairs",
        "",
        "| Pair | Control tok/s | Candidate tok/s | Candidate change | Exact token/hash |",
        "|---:|---:|---:|---:|:---:|",
        *pair_rows,
        "",
        f"Median paired decode change: `{(median_paired_speedup - 1.0) * 100.0:+.2f}%`.",
        "",
        "## Decode wall attribution",
        "",
        "Times are median microseconds per generated token. Percentages use the "
        "corresponding measured host wall time; component timers are QTimer-based.",
        "",
        "| Module | Expanded-S8 | Transformer direct-n | Candidate speed change |",
        "|---|---:|---:|---:|",
        *module_rows,
        f"| Complete decode host wall | {control_latency * 1000.0:.1f} (100.0%) | "
        f"{candidate_latency * 1000.0:.1f} (100.0%) | "
        f"{(control_latency / candidate_latency - 1.0) * 100.0:+.1f}% |",
        "",
        "## Physical contract",
        "",
        f"Candidate direct weight traffic: "
        f"`{summary['candidate_direct_weight_mib_per_token']:.2f} MiB/token`; "
        f"expanded-S8 carrier writes avoided: "
        f"`{summary['candidate_expansion_mib_avoided_per_token']:.2f} MiB/token`.",
        "",
        "Both cells use one FastRPC invocation per token, one HMX owner, an 8 MiB "
        "VTCM allocation, zero timed intermediate DDR, zero spill/fill and a closed "
        "DSP attribution ledger. Candidate QKVO and MLP expansion ticks are zero; "
        "LM-head expansion is intentionally retained and reported.",
        "",
        f"Overall formal gate: `{'PASS' if gates['all_pass'] else 'FAIL'}`.",
    ]
    (result_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "gates": gates,
                      "control_decode_tok_s": control_tps,
                      "candidate_decode_tok_s": candidate_tps,
                      "median_paired_speedup": median_paired_speedup}, sort_keys=True))
    if args.formal and not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
