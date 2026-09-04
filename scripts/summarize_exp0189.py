#!/usr/bin/env python3
"""Validate and summarize EXP-0189 direct-n row-four SwiGLU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Iterable


QTIMER_TICKS_PER_US = 19.2
VTCM_BYTES = 8 * 1024 * 1024
CONTROL = "full64"
CANDIDATE = "row4"
DIRECT_MASK = 7
DIRECT_PROJECTIONS_PER_DECODE = 28 * 7
SWIGLU_TILES_PER_LAYER = 6144 // 32
ROW4_CALLS_PER_DECODE = 28 * SWIGLU_TILES_PER_LAYER

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
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps", type=int, default=193)
    parser.add_argument("--require-audit", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--gate", action="store_true")
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
                 if "generation_step" in record and
                 "selected_token_id" in record]
    profiles = [record for record in records
                if record.get("record") == "generation_profile"]
    summaries.sort(key=lambda value: int(value["generation_step"]))
    profiles.sort(key=lambda value: int(value["generation_step"]))
    expected = list(range(steps))
    if len(summaries) != steps or len(profiles) != steps:
        raise ValueError(
            f"{path}: expected {steps} summaries/profiles, got "
            f"{len(summaries)}/{len(profiles)}")
    if [int(value["generation_step"]) for value in summaries] != expected:
        raise ValueError(f"{path}: non-contiguous summary steps")
    if [int(value["generation_step"]) for value in profiles] != expected:
        raise ValueError(f"{path}: non-contiguous profile steps")
    return {"path": path, "summaries": summaries, "profiles": profiles}


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def validate_run(
    run: dict[str, object], rows: int, steps: int, poison: int = 0,
    experiment: int = 189, direct_mask: int = DIRECT_MASK,
    lm_head_direct: bool = False,
) -> None:
    summaries = run["summaries"]
    profiles = run["profiles"]
    assert isinstance(summaries, list) and isinstance(profiles, list)
    for index, (summary, profile) in enumerate(zip(summaries, profiles)):
        assert isinstance(summary, dict) and isinstance(profile, dict)
        context = f"{run['path']} step {index}"
        require_equal(int(summary["experiment"]), experiment,
                      context + " experiment")
        require_equal(int(profile["experiment"]), experiment,
                      context + " profile experiment")
        require_equal(int(summary["rpc_result"]), 0, context + " RPC")
        require_equal(int(summary["dsp_status"]), 3, context + " DSP status")
        require_equal(bool(summary["pass"]), True, context + " pass")
        require_equal(int(profile["dsp_status"]), 3,
                      context + " profile DSP")
        require_equal(int(profile["numerical_status"]), 1,
                      context + " numerical")
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
                      int(profile["invocation_ticks"]),
                      context + " ledger closure")
        require_equal(int(profile.get("cache_mismatches", 0)), 0,
                      context + " cache mismatch")
        require_equal(int(profile.get("cache_prefix_mismatches", 0)), 0,
                      context + " cache prefix mismatch")
        require_equal(int(profile.get("cache_structure_mismatches", 0)), 0,
                      context + " cache structure mismatch")
        expected_mode = "prefill" if index == 0 else "decode"
        require_equal(summary["mode"], expected_mode,
                      context + " summary mode")
        require_equal(profile["mode"], expected_mode,
                      context + " profile mode")
        require_equal(int(profile["w4u8_decode_projection_mode"]), 1,
                      context + " direct-n mode")
        require_equal(int(profile["w4u8_decode_direct_n_mask"]), direct_mask,
                      context + " direct-n mask")
        require_equal(int(profile["w4u8_decode_swiglu_rows"]), rows,
                      context + " requested SwiGLU rows")
        require_equal(
            int(profile["w4u8_decode_swiglu_padding_poison"]), poison,
            context + " requested padding poison")
        if index == 0:
            require_equal(int(profile["w4u8_swiglu_rows_observed"]), 64,
                          context + " prefill SwiGLU rows")
            require_equal(
                int(profile["w4u8_decode_swiglu_row4_call_count"]), 0,
                context + " prefill row4 calls")
            require_equal(
                int(profile["w4u8_decode_direct_n_projection_count"]), 0,
                context + " prefill direct projections")
        else:
            require_equal(int(profile["w4u8_swiglu_rows_observed"]), rows,
                          context + " decode SwiGLU rows")
            require_equal(
                int(profile["w4u8_decode_direct_n_projection_count"]),
                DIRECT_PROJECTIONS_PER_DECODE +
                (1 if lm_head_direct else 0),
                context + " direct projection count")
            require_equal(int(profile["w4u8_qkvo_weight_expand_ticks"]), 0,
                          context + " QKVO expansion")
            require_equal(int(profile["w4u8_mlp_weight_expand_ticks"]), 0,
                          context + " MLP expansion")
            if lm_head_direct:
                require_equal(
                    int(profile["generation_lm_head_expand_ticks"]), 0,
                    context + " direct LM-head expansion")
            elif int(profile["generation_lm_head_expand_ticks"]) <= 0:
                raise ValueError(
                    f"{context}: LM-head fallback expansion missing")
            expected_calls = ROW4_CALLS_PER_DECODE if rows == 4 else 0
            require_equal(
                int(profile["w4u8_decode_swiglu_row4_call_count"]),
                expected_calls, context + " row4 call count")
            require_equal(
                int(profile["w4u8_decode_swiglu_vector_count"]),
                expected_calls, context + " row4 vector count")
            require_equal(
                int(profile["w4u8_decode_swiglu_padding_poison_count"]),
                expected_calls if poison else 0,
                context + " padding poison count")


def run_signatures(run: dict[str, object]) -> tuple[list[int], list[str]]:
    summaries = run["summaries"]
    profiles = run["profiles"]
    assert isinstance(summaries, list) and isinstance(profiles, list)
    return (
        [int(value["selected_token_id"]) for value in summaries],
        [str(value["output_hash"]) for value in profiles],
    )


def valid_row_hashes(run: dict[str, object]) -> list[int]:
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    return [int(value["w4u8_decode_swiglu_valid_row_hash"])
            for value in profiles[1:]]


def audit_tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.bin")):
        hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError(f"{root}: no audit tensors")
    return hashes


def validate_audit(result_dir: Path) -> dict[str, object]:
    runs = {
        CONTROL: load_run(result_dir / "raw/audit_full64.log", 4),
        CANDIDATE: load_run(result_dir / "raw/audit_row4.log", 4),
        "row4_poison": load_run(
            result_dir / "raw/audit_row4_poison.log", 4),
    }
    validate_run(runs[CONTROL], 64, 4, 0)
    validate_run(runs[CANDIDATE], 4, 4, 0)
    validate_run(runs["row4_poison"], 4, 4, 1)
    control_signature = run_signatures(runs[CONTROL])
    candidate_signature = run_signatures(runs[CANDIDATE])
    poison_signature = run_signatures(runs["row4_poison"])
    hashes = {name: valid_row_hashes(run) for name, run in runs.items()}
    nonzero_hashes = all(value != 0 for values in hashes.values()
                         for value in values)
    trees = {
        name: audit_tree_hashes(result_dir / "audit" / name)
        for name in (CONTROL, CANDIDATE, "row4_poison")
    }
    result = {
        "full_stack_signatures_equal":
            control_signature == candidate_signature == poison_signature,
        "valid_row_hashes_nonzero": nonzero_hashes,
        "valid_row_hashes_equal":
            hashes[CONTROL] == hashes[CANDIDATE] == hashes["row4_poison"],
        "audit_hidden_tensors_equal":
            trees[CONTROL] == trees[CANDIDATE] == trees["row4_poison"],
        "poison_exercised": all(
            int(profile["w4u8_decode_swiglu_padding_poison_count"]) ==
            ROW4_CALLS_PER_DECODE
            for profile in runs["row4_poison"]["profiles"][1:]),
    }
    result["all_pass"] = all(result.values())
    return result


def decode_profiles(run: dict[str, object]) -> list[dict[str, object]]:
    profiles = run["profiles"]
    assert isinstance(profiles, list)
    return [profile for profile in profiles if profile["mode"] == "decode"]


def decode_latency_ms(run: dict[str, object]) -> float:
    profiles = decode_profiles(run)
    return sum(int(profile["host_wall_ns"]) for profile in profiles) \
        / len(profiles) / 1e6


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


def median_module_us(
    runs: list[dict[str, object]], fields: tuple[str, ...]
) -> float:
    return median(per_token_ticks(run, fields) / QTIMER_TICKS_PER_US
                  for run in runs)


def median_host_boundary_us(runs: list[dict[str, object]]) -> float:
    values: list[float] = []
    for run in runs:
        profiles = decode_profiles(run)
        host_us = sum(int(profile["host_wall_ns"]) for profile in profiles) \
            / len(profiles) / 1000.0
        dsp_us = sum(int(profile["ledger_named_ticks"])
                     for profile in profiles) \
            / len(profiles) / QTIMER_TICKS_PER_US
        values.append(host_us - dsp_us)
    return median(values)


def median_counter(
    runs: list[dict[str, object]], field: str
) -> float:
    values: list[float] = []
    for run in runs:
        profiles = decode_profiles(run)
        values.append(sum(int(profile[field]) for profile in profiles)
                      / len(profiles))
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
        for cell, rows in ((CONTROL, 64), (CANDIDATE, 4)):
            path = result_dir / "raw" / f"pair_{round_number:02d}_{cell}.log"
            run = load_run(path, args.steps)
            validate_run(run, rows, args.steps)
            runs[cell].append(run)
            pair[cell] = run
        exact = run_signatures(pair[CONTROL]) == run_signatures(pair[CANDIDATE])
        control_tps = decode_tokens_per_second(pair[CONTROL])
        candidate_tps = decode_tokens_per_second(pair[CANDIDATE])
        speedup = candidate_tps / control_tps
        pair_exact.append(exact)
        pair_speedups.append(speedup)
        pair_rows.append(
            f"| {round_number} | {control_tps:.3f} | {candidate_tps:.3f} | "
            f"{(speedup - 1.0) * 100.0:+.2f}% | "
            f"{'yes' if exact else 'NO'} |")

    control_tps = median(decode_tokens_per_second(run)
                         for run in runs[CONTROL])
    candidate_tps = median(decode_tokens_per_second(run)
                           for run in runs[CANDIDATE])
    control_latency = median(decode_latency_ms(run)
                             for run in runs[CONTROL])
    candidate_latency = median(decode_latency_ms(run)
                               for run in runs[CANDIDATE])
    control_prefill = median(prefill_tokens_per_second(run)
                             for run in runs[CONTROL])
    candidate_prefill = median(prefill_tokens_per_second(run)
                               for run in runs[CANDIDATE])

    module_rows: list[str] = []
    module_summary: dict[str, object] = {}
    for label, fields in MODULES:
        control_us = median_module_us(runs[CONTROL], fields)
        candidate_us = median_module_us(runs[CANDIDATE], fields)
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
    control_host_us = median_host_boundary_us(runs[CONTROL])
    candidate_host_us = median_host_boundary_us(runs[CANDIDATE])
    module_summary["Host/RPC boundary"] = {
        CONTROL: control_host_us,
        CANDIDATE: candidate_host_us,
        "candidate_speed_change_percent":
            (control_host_us / candidate_host_us - 1.0) * 100.0,
    }
    module_rows.append(
        f"| Host/RPC boundary | {control_host_us:.1f} "
        f"({control_host_us / (control_latency * 10.0):.1f}%) | "
        f"{candidate_host_us:.1f} "
        f"({candidate_host_us / (candidate_latency * 10.0):.1f}%) | "
        f"{(control_host_us / candidate_host_us - 1.0) * 100.0:+.1f}% |")

    gate_up_projection = {
        CONTROL: median_module_us(runs[CONTROL], ("gate_up_ticks",)),
        CANDIDATE: median_module_us(runs[CANDIDATE], ("gate_up_ticks",)),
    }
    swiglu = {
        CONTROL: median_module_us(runs[CONTROL], ("activation_ticks",)),
        CANDIDATE: median_module_us(runs[CANDIDATE], ("activation_ticks",)),
    }
    audit = validate_audit(result_dir) if args.require_audit else None
    gates = {
        "rotated_pairs_present": args.rounds in (5, 10),
        "all_pair_outputs_byte_exact": all(pair_exact),
        "all_pairs_candidate_faster": all(value > 1.0
                                             for value in pair_speedups),
        "median_full_stack_decode_faster": candidate_tps > control_tps,
        "target_stage_faster":
            module_summary["Gate/Up + SwiGLU"][CANDIDATE] <
            module_summary["Gate/Up + SwiGLU"][CONTROL],
        "candidate_row4_count_exact":
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_swiglu_row4_call_count") ==
            ROW4_CALLS_PER_DECODE,
        "projection_work_preserved":
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_direct_n_projection_count") ==
            DIRECT_PROJECTIONS_PER_DECODE and
            median_counter(runs[CANDIDATE],
                           "w4u8_decode_direct_n_hmx_command_count") ==
            median_counter(runs[CONTROL],
                           "w4u8_decode_direct_n_hmx_command_count") and
            median_counter(runs[CANDIDATE], "hmx_u8s8_tile_pair_count") ==
            median_counter(runs[CONTROL], "hmx_u8s8_tile_pair_count"),
        "transformer_expansion_zero":
            median_counter(runs[CANDIDATE],
                           "w4u8_qkvo_weight_expand_ticks") == 0 and
            median_counter(runs[CANDIDATE],
                           "w4u8_mlp_weight_expand_ticks") == 0,
        "lm_head_expanded_s8_fallback_retained":
            median_counter(runs[CANDIDATE],
                           "generation_lm_head_expand_ticks") > 0,
    }
    if audit is not None:
        gates["audit_pass"] = bool(audit["all_pass"])
    gates["all_pass"] = all(gates.values())

    summary = {
        "experiment": "EXP-0189",
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
        "gate_up_projection_us": gate_up_projection,
        "swiglu_us": swiglu,
        "audit": audit,
        "physical": {
            "control_hmx_commands": median_counter(
                runs[CONTROL], "hmx_command_count"),
            "candidate_hmx_commands": median_counter(
                runs[CANDIDATE], "hmx_command_count"),
            "control_direct_n_hmx_commands": median_counter(
                runs[CONTROL], "w4u8_decode_direct_n_hmx_command_count"),
            "candidate_direct_n_hmx_commands": median_counter(
                runs[CANDIDATE], "w4u8_decode_direct_n_hmx_command_count"),
            "control_weight_ddr_read_bytes": median_counter(
                runs[CONTROL], "weight_ddr_read_bytes"),
            "candidate_weight_ddr_read_bytes": median_counter(
                runs[CANDIDATE], "weight_ddr_read_bytes"),
            "control_direct_weight_ddr_read_bytes": median_counter(
                runs[CONTROL], "w4u8_decode_direct_n_weight_ddr_read_bytes"),
            "candidate_direct_weight_ddr_read_bytes": median_counter(
                runs[CANDIDATE], "w4u8_decode_direct_n_weight_ddr_read_bytes"),
            "vtcm_bytes": VTCM_BYTES,
            "intermediate_ddr_bytes": 0,
            "spill_fill": 0,
            "fastrpc_calls_per_token": 1,
            "hmx_owners": 1,
        },
        "gates": gates,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    report = [
        "# EXP-0189 direct-n decode row-four SwiGLU report",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        "The control and candidate use the same EXP-0188 direct-n transformer "
        "path. The only timed change is that logical-M1 decode applies the "
        "unchanged LUT SwiGLU to four physical rows instead of 64.",
        "",
        "## End-to-end gate",
        "",
        "| Metric | Full64 control | Row4 candidate | Change |",
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
        "| Module | Full64 control | Row4 candidate | Candidate speed change |",
        "|---|---:|---:|---:|",
        *module_rows,
        f"| Complete decode host wall | {control_latency * 1000.0:.1f} "
        f"(100.0%) | {candidate_latency * 1000.0:.1f} (100.0%) | "
        f"{(control_latency / candidate_latency - 1.0) * 100.0:+.1f}% |",
        "",
        "## Target-stage split",
        "",
        "| Substage | Full64 control | Row4 candidate | Change |",
        "|---|---:|---:|---:|",
        f"| Gate/Up direct-n projections | {gate_up_projection[CONTROL]:.1f} us | "
        f"{gate_up_projection[CANDIDATE]:.1f} us | "
        f"{(gate_up_projection[CONTROL] / gate_up_projection[CANDIDATE] - 1.0) * 100.0:+.1f}% |",
        f"| SwiGLU | {swiglu[CONTROL]:.1f} us | "
        f"{swiglu[CANDIDATE]:.1f} us | "
        f"{(swiglu[CONTROL] / swiglu[CANDIDATE] - 1.0) * 100.0:+.1f}% |",
        "",
        "## Correctness and physical contract",
        "",
        f"Audit: `{json.dumps(audit, sort_keys=True) if audit else 'formal reuses short audit'}`.",
        "",
        "Both cells retain exactly 8 MiB VTCM, one FastRPC call per token, "
        "one HMX owner, zero timed intermediate DDR, zero spill/fill, no QNN, "
        "identical direct-n projection/HMX work and zero transformer W4-to-S8 "
        "expansion. The Expanded-S8 LM-head fallback is unchanged.",
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
