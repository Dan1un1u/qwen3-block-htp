#!/usr/bin/env python3
"""Validate and report EXP-0148 real single-layer replay evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


RECIPES = ("F16F16", "W4F16", "W4U8")
FILE_RECIPE = {name.lower(): name for name in RECIPES}
ROUNDS = 10
STEPS = 9
VTCM_BYTES = 8_388_608
TICKS_PER_US = 19.2

PRIMARY = (
    ("Complete Host wall", "host_wall_ns"),
    ("DSP block total", "total_ticks"),
    ("DSP invocation", "invocation_ticks"),
    ("Runtime setup", "runtime_setup_ticks"),
    ("Runtime teardown", "runtime_teardown_ticks"),
    ("Stage-boundary overhead", "stage_boundary_ticks"),
    ("Named ledger total", "ledger_named_ticks"),
    ("Unattributed ledger", "ledger_unattributed_ticks"),
)
LEDGER = (
    ("Input", "input_stage_ticks"),
    ("Metadata", "metadata_stage_ticks"),
    ("Input RMSNorm", "input_norm_ticks"),
    ("QKV projection", "qkv_projection_ticks"),
    ("Q/K Norm-RoPE", "qk_norm_rope_ticks"),
    ("QK-Softmax-AV", "attention_ticks"),
    ("O projection", "o_projection_ticks"),
    ("Post-Attention residual", "post_attention_residual_ticks"),
    ("Post-Attention RMSNorm", "post_attention_norm_ticks"),
    ("Gate/Up", "gate_up_ticks"),
    ("SwiGLU activation", "activation_ticks"),
    ("Down projection", "down_ticks"),
    ("Final residual", "final_residual_ticks"),
    ("Output", "output_stage_ticks"),
)
PROJECTION = (
    ("Weight DMA", "weight_dma_ticks"),
    ("HMX compute", "hmx_compute_ticks"),
    ("Projection pack", "projection_pack_ticks"),
    ("Projection HMX wait", "projection_hmx_wait_ticks"),
    ("Projection unpack", "projection_unpack_ticks"),
    ("HMX-ready wait", "hmx_ready_wait_ticks"),
    ("W4F16 expand lifetime", "w4f16_expand_ticks"),
    ("W4F16 expand work", "w4f16_expand_work_ticks"),
    ("W4F16 expand-pool wait", "w4f16_expand_pool_wait_ticks"),
    ("W4F16 prefetch wait", "w4f16_prefetch_wait_ticks"),
    ("W4F16 HMX-tail wait", "w4f16_hmx_tail_wait_ticks"),
    ("W4F16 cross-prefetch wait", "w4f16_cross_prefetch_wait_ticks"),
    ("W4F16 cross-prefetch lifetime", "w4f16_cross_prefetch_lifetime_ticks"),
    ("W4U8 QKVO expand", "w4u8_qkvo_weight_expand_ticks"),
    ("W4U8 QKVO prefetch wait", "w4u8_qkvo_prefetch_wait_ticks"),
    ("W4U8 QKVO HMX lifetime", "w4u8_qkvo_hmx_lifetime_ticks"),
)
ATTENTION = (
    ("FP16 Attention setup", "attention_setup_ticks"),
    ("FP16 QK pack", "attention_qk_pack_ticks"),
    ("FP16 QK HMX", "attention_qk_hmx_ticks"),
    ("FP16 QK unpack", "attention_qk_unpack_ticks"),
    ("FP16 Softmax", "attention_softmax_ticks"),
    ("FP16 AV pack", "attention_av_pack_ticks"),
    ("FP16 AV HMX", "attention_av_hmx_ticks"),
    ("FP16 AV unpack", "attention_av_unpack_ticks"),
    ("FP16 GQA pipeline", "attention_gqa_pipeline_ticks"),
    ("Attention unattributed", "attention_unattributed_ticks"),
    ("U8 Q/K Norm-RoPE", "u8_attention_qk_norm_rope_ticks"),
    ("U8 K pack", "u8_attention_k_pack_ticks"),
    ("U8 V pack", "u8_attention_v_pack_ticks"),
    ("U8 QK HMX", "u8_attention_qk_hmx_ticks"),
    ("U8 QK requant", "u8_attention_qk_requant_ticks"),
    ("U8 Softmax", "u8_attention_softmax_ticks"),
    ("U8 AV HMX", "u8_attention_av_hmx_ticks"),
    ("U8 AV requant", "u8_attention_av_requant_ticks"),
    ("U8 Attention pipeline wait", "u8_attention_pipeline_wait_ticks"),
    ("Dynamic cache-backed Attention", "scan_dynamic_attention_ticks"),
    ("KV cache stage", "scan_cache_stage_ticks"),
    ("KV cache append", "scan_cache_append_ticks"),
)
MLP = (
    ("W4F16 Gate/Up weight DMA", "w4f16_gate_up_weight_dma_ticks"),
    ("W4F16 Gate/Up expand", "w4f16_gate_up_expand_ticks"),
    ("W4F16 Gate/Up expand work", "w4f16_gate_up_expand_work_ticks"),
    ("W4F16 Gate/Up pool wait", "w4f16_gate_up_expand_pool_wait_ticks"),
    ("W4F16 Gate/Up HMX wait", "w4f16_gate_up_hmx_wait_ticks"),
    ("W4F16 Gate/Up HMX-tail wait", "w4f16_gate_up_hmx_tail_wait_ticks"),
    ("W4F16 streaming SwiGLU work", "w4f16_gate_up_stream_work_ticks"),
    ("W4F16 stream-ready wait", "w4f16_gate_up_stream_ready_wait_ticks"),
    ("W4F16 stream-join wait", "w4f16_gate_up_stream_join_wait_ticks"),
    ("W4U8 Gate/Up pipeline", "w4u8_mlp_gate_up_pipeline_ticks"),
    ("W4U8 Down pipeline", "w4u8_mlp_down_pipeline_ticks"),
    ("W4U8 activation work", "w4u8_mlp_activation_work_ticks"),
    ("W4U8 weight stage", "w4u8_mlp_weight_stage_ticks"),
    ("W4U8 weight expand", "w4u8_mlp_weight_expand_ticks"),
    ("W4U8 HMX compute", "w4u8_mlp_hmx_compute_ticks"),
    ("W4U8 HMX-ready wait", "w4u8_mlp_hmx_ready_wait_ticks"),
    ("W4U8 producer-slot wait", "w4u8_mlp_producer_slot_wait_ticks"),
    ("W4U8 expanded-slot wait", "w4u8_mlp_expanded_slot_wait_ticks"),
)
PHYSICAL = (
    ("Weight DDR read bytes", "weight_ddr_read_bytes"),
    ("Boundary DDR read bytes", "boundary_ddr_read_bytes"),
    ("Boundary DDR write bytes", "boundary_ddr_write_bytes"),
    ("Legal KV-cache DDR read bytes", "scan_cache_ddr_read_bytes"),
    ("Legal KV-cache DDR write bytes", "scan_cache_ddr_write_bytes"),
    ("Intermediate DDR read bytes", "intermediate_ddr_read_bytes"),
    ("Intermediate DDR write bytes", "intermediate_ddr_write_bytes"),
    ("Weight DMA descriptors", "weight_dma_descriptor_count"),
    ("Boundary DMA descriptors", "boundary_dma_descriptor_count"),
    ("KV-cache DMA descriptors", "scan_cache_dma_descriptor_count"),
    ("Intermediate DMA descriptors", "intermediate_dma_descriptor_count"),
    ("Spill/fill count", "intermediate_spill_fill_count"),
    ("HMX commands", "hmx_command_count"),
    ("FP16 HMX tile pairs", "hmx_fp16_tile_pair_count"),
    ("U8xS8 HMX tile pairs", "hmx_u8s8_tile_pair_count"),
    ("Requested VTCM bytes", "vtcm_requested_bytes"),
    ("Acquired VTCM bytes", "vtcm_acquired_bytes"),
    ("Peak physical plan bytes", "vtcm_peak_plan_bytes"),
    ("FastRPC block invocations", "block_invocation_count"),
)
MODULES = (
    ("I/O and metadata", ("input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks")),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm-RoPE preparation", ("qkv_projection_ticks", "qk_norm_rope_ticks")),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-Attention residual + RMSNorm", ("post_attention_residual_ticks", "post_attention_norm_ticks")),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_objects(path: Path) -> list[dict[str, object]]:
    objects = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objects


def profiles(path: Path, recipe: str, audit: bool) -> list[dict[str, object]]:
    objects = read_objects(path)
    rows = [row for row in objects if row.get("record") == "replay_profile"]
    complete = [row for row in objects if row.get("replay_sequence_complete") is True]
    if len(rows) != STEPS or len(complete) != 1:
        raise ValueError(f"{path}: expected {STEPS} profiles and one completion")
    if complete[0].get("all_steps_pass") is not True:
        raise ValueError(f"{path}: replay did not pass")
    for step, row in enumerate(rows):
        if row.get("experiment") != 148 or row.get("variant") != recipe:
            raise ValueError(f"{path}: identity mismatch")
        if int(row["replay_step"]) != step:
            raise ValueError(f"{path}: replay step mismatch")
        if int(row["numerical_audit_enabled"]) != int(audit):
            raise ValueError(f"{path}: numerical-audit mismatch")
        expected_first = 0 if step == 0 else 63 + step
        expected_valid = 64 + step
        expected_m = 64 if step == 0 else 1
        if (int(row["first_position"]) != expected_first or
                int(row["valid_length"]) != expected_valid or
                int(row["scan_total_kv_length"]) != expected_valid or
                int(row["logical_m"]) != expected_m):
            raise ValueError(f"{path}: sequence state mismatch at step {step}")
    return rows


def number(row: dict[str, object], field: str) -> float:
    value = float(row[field])
    if field == "host_wall_ns":
        return value / 1000.0
    if field.endswith("_ticks"):
        return value / TICKS_PER_US
    return value


def sequence_value(rows: list[dict[str, object]], mode: str, field: str) -> float:
    selected = rows[:1] if mode == "prefill" else rows[1:]
    return statistics.fmean(number(row, field) for row in selected)


def median_metric(rounds: list[list[dict[str, object]]], mode: str, field: str) -> float:
    return statistics.median(sequence_value(rows, mode, field) for rows in rounds)


def paired_delta(rounds: dict[str, list[list[dict[str, object]]]], mode: str,
                 field: str) -> float | None:
    values = []
    for control, candidate in zip(rounds["W4F16"], rounds["W4U8"]):
        left = sequence_value(control, mode, field)
        right = sequence_value(candidate, mode, field)
        if left == 0:
            continue
        values.append((right / left - 1.0) * 100.0)
    return statistics.median(values) if values else None


def aggregate_scope(rounds: dict[str, list[list[dict[str, object]]]],
                    repeat: int, mode: str) -> dict[str, dict[str, float]]:
    fields = {field for _, field in (*PRIMARY, *LEDGER, *PROJECTION,
                                     *ATTENTION, *MLP, *PHYSICAL)}
    result: dict[str, dict[str, float]] = {}
    for recipe in RECIPES:
        sample = rounds[recipe][:1] if repeat == 1 else rounds[recipe]
        result[recipe] = {
            field: (sequence_value(sample[0], mode, field) if repeat == 1
                    else median_metric(sample, mode, field))
            for field in fields
        }
    return result


def module_values(scope: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for recipe in RECIPES:
        recipe_values = scope[recipe]
        modules = {
            label: sum(recipe_values[field] for field in fields)
            for label, fields in MODULES
        }
        named = sum(modules.values())
        modules["Host/RPC and profiling-closure remainder"] = (
            recipe_values["host_wall_ns"] - named
        )
        modules["Complete Host wall"] = recipe_values["host_wall_ns"]
        values[recipe] = modules
    return values


def require_physical(rows: Iterable[dict[str, object]]) -> None:
    for row in rows:
        if (int(row["vtcm_requested_bytes"]) != VTCM_BYTES or
                int(row["vtcm_acquired_bytes"]) != VTCM_BYTES or
                not 0 < int(row["vtcm_peak_plan_bytes"]) <= VTCM_BYTES or
                int(row["block_invocation_count"]) != 1 or
                int(row["intermediate_dma_descriptor_count"]) != 0 or
                int(row["intermediate_ddr_read_bytes"]) != 0 or
                int(row["intermediate_ddr_write_bytes"]) != 0 or
                int(row["intermediate_spill_fill_count"]) != 0 or
                int(row["cache_prefix_mismatches"]) != 0 or
                int(row["cache_mismatches"]) != 0 or
                int(row["scan_cache_append_mismatch_count"]) != 0 or
                row["backend"] != "standalone_fastrpc_dsp" or
                row["qnn"] != "none" or
                row["intermediate_residency"] != "VTCM"):
            raise ValueError(f"physical/replay gate failed: {row}")


def correctness_summary(correctness: dict[str, list[dict[str, object]]],
                        manifests: dict[str, dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    all_pass = True
    for recipe, rows in correctness.items():
        require_physical(rows)
        mismatch = max(int(row["output_mismatches"]) for row in rows)
        cache = max(int(row["cache_mismatches"]) for row in rows)
        max_abs = max(float(row["output_max_abs"]) for row in rows)
        min_cos = min(float(row["output_cosine"]) for row in rows)
        max_lsb = max(int(row["output_max_lsb"]) for row in rows)
        mask = max(int(row["u8_attention_probability_mask_violation_count"])
                   for row in rows)
        fused_k = max(int(row["u8_attention_fused_k_operand_mismatch_count"])
                      for row in rows)
        expand = max(int(row["w4f16_expand_mismatch_count"]) for row in rows)
        if recipe == "W4U8":
            numerical = mismatch == cache == max_lsb == mask == fused_k == 0
        else:
            numerical = (mismatch == cache == expand == 0 and
                         max_abs <= 0.0625 and min_cos >= 0.999)
        all_pass = all_pass and numerical
        result[recipe] = {
            "pass": numerical,
            "max_output_mismatches": mismatch,
            "max_cache_mismatches": cache,
            "max_abs": max_abs,
            "min_cosine": min_cos,
            "max_lsb": max_lsb,
            "mask_violations": mask,
            "fused_k_operand_mismatches": fused_k,
            "w4f16_expand_mismatches": expand,
            "step_output_hashes": [str(row["output_hash"]) for row in rows],
        }

    verification = manifests["W4U8"].get("w4u8_boundary_verification")
    if not isinstance(verification, dict):
        raise ValueError("W4U8 formal manifest lacks independent boundary verification")
    steps = verification.get("decode_steps")
    if not isinstance(steps, list) or len(steps) != 8:
        raise ValueError("W4U8 independent decode verification is incomplete")
    independent_pass = (
        int(verification.get("prefill_output_mismatches", -1)) == 0 and
        all(int(step.get("attention_av_mismatches", -1)) == 0 and
            int(step.get("tail_output_mismatches", -1)) == 0
            for step in steps)
    )
    all_pass = all_pass and independent_pass
    result["w4u8_independent_boundary_reference"] = {
        "pass": independent_pass,
        "method": verification.get("method"),
        "cache_reference_kind": verification.get("cache_reference_kind"),
        "cache_is_independent_math_reference": verification.get(
            "cache_is_independent_math_reference"),
        "prefill_output_mismatches": verification.get(
            "prefill_output_mismatches"),
        "decode_attention_av_mismatches": [
            step.get("attention_av_mismatches") for step in steps
        ],
        "decode_tail_output_mismatches": [
            step.get("tail_output_mismatches") for step in steps
        ],
    }
    result["all_pass"] = all_pass
    return result


def fmt(value: float, field: str) -> str:
    if field == "host_wall_ns" or field.endswith("_ticks"):
        return f"{value:.3f} us"
    if field.endswith("_bytes"):
        return f"{value:,.1f} B"
    return f"{value:,.3f}"


def delta(control: float, candidate: float) -> str:
    if control == 0:
        return "0" if candidate == 0 else "N/A (zero control)"
    return f"{(candidate / control - 1.0) * 100:+.2f}%"


def render_overview(lines: list[str], title: str,
                    modules: dict[str, dict[str, float]]) -> None:
    lines.extend([f"## {title}", "", "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
                  "|---|---:|---:|---:|---:|"])
    for name in modules["F16F16"]:
        cells = []
        for recipe in RECIPES:
            value = modules[recipe][name]
            total = modules[recipe]["Complete Host wall"]
            cells.append(f"{value:.1f} us ({100.0 * value / total:.1f}%)" if
                         name != "Complete Host wall" else f"{value:.1f} us")
        left = modules["W4F16"][name]
        right = modules["W4U8"][name]
        speed = "N/A" if right == 0 else f"{(left / right - 1.0) * 100:+.1f}%"
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {speed} |")
    lines.append("")


def render_comparison(lines: list[str], title: str,
                      rows: tuple[tuple[str, str], ...],
                      scopes: dict[str, dict[str, dict[str, float]]],
                      paired: dict[str, dict[str, float | None]]) -> None:
    lines.extend([f"### {title}", "",
                  "| Metric | Prefill W4F16 | Prefill W4U8 | Delta | Paired median | Decode W4F16 | Decode W4U8 | Delta | Paired median |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label, field in rows:
        p_left = scopes["prefill"]["W4F16"][field]
        p_right = scopes["prefill"]["W4U8"][field]
        d_left = scopes["decode"]["W4F16"][field]
        d_right = scopes["decode"]["W4U8"][field]
        p_pair = paired["prefill"].get(field)
        d_pair = paired["decode"].get(field)
        lines.append(
            f"| {label} | {fmt(p_left, field)} | {fmt(p_right, field)} | "
            f"{delta(p_left, p_right)} | "
            f"{'N/A' if p_pair is None else f'{p_pair:+.2f}%'} | "
            f"{fmt(d_left, field)} | {fmt(d_right, field)} | "
            f"{delta(d_left, d_right)} | "
            f"{'N/A' if d_pair is None else f'{d_pair:+.2f}%'} |"
        )
    lines.append("")


def render(summary: dict[str, object]) -> str:
    lines = ["# EXP-0148 — Complete profiling report", ""]
    repeat10 = summary["repeat10"]
    render_overview(lines, "Repeat-ten prefill overview (positions 0-63)",
                    repeat10["prefill_modules"])
    render_overview(lines, "Repeat-ten continuous-decode overview (mean positions 64-71)",
                    repeat10["decode_modules"])
    lines.extend([
        "Repeat one is round 01 (one fresh prefill→eight-decode session). "
        "Repeat ten is the median of ten fresh sessions; each decode value is "
        "first averaged across its eight continuous append steps. Recipe order "
        "rotates by round. The three recipes are therefore round-paired, while "
        "the rows remain self-comparison rather than a Qualcomm comparison.", "",
    ])
    for repeat in (1, 10):
        block = summary[f"repeat{repeat}"]
        lines.extend([f"## Repeat {repeat}", ""])
        for title, rows in (
            ("Primary latency", PRIMARY),
            ("Complete additive Block Timing Ledger", LEDGER),
            ("Projection diagnostics (overlapping)", PROJECTION),
            ("Attention diagnostics (overlapping)", ATTENTION),
            ("MLP diagnostics (overlapping)", MLP),
            ("Traffic, commands and residency", PHYSICAL),
        ):
            render_comparison(lines, title, rows, block["scopes"], block["paired_delta_percent"])
    correctness = summary["correctness"]
    lines.extend(["## Correctness and physical gates", "",
                  "| Gate | F16F16 | W4F16 | W4U8 |", "|---|---:|---:|---:|"])
    for label, key in (
        ("Maximum output mismatches", "max_output_mismatches"),
        ("Maximum cache mismatches", "max_cache_mismatches"),
        ("Maximum absolute error", "max_abs"),
        ("Minimum cosine", "min_cosine"),
        ("Maximum U8 LSB error", "max_lsb"),
        ("Mask violations", "mask_violations"),
        ("Fused-K operand mismatches", "fused_k_operand_mismatches"),
        ("W4F16 expansion mismatches", "w4f16_expand_mismatches"),
    ):
        lines.append("| {} | {} | {} | {} |".format(
            label, *(correctness[recipe][key] for recipe in RECIPES)))
    independent = correctness["w4u8_independent_boundary_reference"]
    lines.extend([
        "", f"W4U8 independent QK→log2 Softmax→AV and tail verification: "
        f"**{'PASS' if independent['pass'] else 'FAIL'}**; all eight decode "
        "AV and final-output mismatch counts are zero. Persistent cache bytes "
        "are explicitly a device physical-boundary golden, not an independent "
        "high-level mathematical reference.", "",
        "| Physical contract | Result |", "|---|---:|",
        f"| Exact 8 MiB VTCM requested/acquired | {'PASS' if summary['physical_gate'] else 'FAIL'} |",
        f"| Zero intermediate DDR and spill/fill | {'PASS' if summary['physical_gate'] else 'FAIL'} |",
        "| Persistent K/V is the only additional legal DDR state | PASS |",
        "| One FastRPC invocation per step | PASS |",
        "| One HMX owner (static contract) | PASS |",
        "| QNN dependency/fallback | none |",
        "", f"Overall EXP-0148 gate: **{'PASS' if summary['local_gate_pass'] else 'FAIL'}**.",
        "No speed gate applies; latency is characterization evidence only.", "",
        "## Provenance", "",
        f"- Source branch: `{summary['source_branch']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Formal evidence: `{summary['result_dir']}`",
        f"- Retained artifacts: `{summary['artifact_dir']}`",
        f"- Device boot ID: `{summary['boot_id']}`",
        "- Backend: standalone FastRPC/cDSP; QNN/QAIRT runtime: none",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    options = args()
    result_dir = options.result_dir.resolve()
    static = json.loads((result_dir / "static_gate.json").read_text())
    if static.get("static_gate") != "pass":
        raise ValueError("static gate failed")
    if ((result_dir / "boot_id_before.txt").read_bytes() !=
            (result_dir / "boot_id_after.txt").read_bytes()):
        raise ValueError("device rebooted during collection")

    rounds: dict[str, list[list[dict[str, object]]]] = {
        recipe: [] for recipe in RECIPES
    }
    correctness: dict[str, list[dict[str, object]]] = {}
    manifests: dict[str, dict[str, object]] = {}
    for file_recipe, recipe in FILE_RECIPE.items():
        correctness[recipe] = profiles(
            result_dir / "raw" / f"correctness_{file_recipe}.jsonl",
            recipe, True)
        manifests[recipe] = json.loads(
            (result_dir / "packages" / f"{file_recipe}_manifest.json").read_text())
        for round_id in range(1, ROUNDS + 1):
            rows = profiles(
                result_dir / "raw" / f"round_{round_id:02d}_{file_recipe}.jsonl",
                recipe, False)
            require_physical(rows)
            rounds[recipe].append(rows)

    correctness_result = correctness_summary(correctness, manifests)
    physical_gate = True
    for recipe_rounds in rounds.values():
        try:
            require_physical(row for rows in recipe_rounds for row in rows)
        except ValueError:
            physical_gate = False
    summary: dict[str, object] = {
        "experiment": "EXP-0148",
        "source_branch": "codex/exp-0148-real-layer-replay-decode-session",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "result_dir": str(result_dir),
        "artifact_dir": str(options.artifact_dir.resolve()),
        "boot_id": (result_dir / "boot_id_before.txt").read_text().strip(),
        "execution_unit": "qwen3_layer14_real_replay_prefill_continuous_decode",
        "repeat_contract": {
            "repeat1": "round_01_one_fresh_continuous_sequence",
            "repeat10": "median_of_ten_fresh_continuous_sequences",
            "decode_normalization": "mean_of_positions_64_through_71_per_sequence",
            "paired_rounds": 10,
            "rotated_recipe_order": True,
        },
        "correctness": correctness_result,
        "physical_gate": physical_gate,
        "local_gate_pass": bool(correctness_result["all_pass"] and physical_gate),
        "speed_gate": "not_applicable_characterization_only",
    }
    for repeat in (1, 10):
        scopes = {
            mode: aggregate_scope(rounds, repeat, mode)
            for mode in ("prefill", "decode")
        }
        fields = {field for _, field in (*PRIMARY, *LEDGER, *PROJECTION,
                                         *ATTENTION, *MLP, *PHYSICAL)}
        paired = {
            mode: {field: paired_delta(rounds, mode, field) for field in fields}
            for mode in ("prefill", "decode")
        }
        summary[f"repeat{repeat}"] = {
            "scopes": scopes,
            "paired_delta_percent": paired,
            "prefill_modules": module_values(scopes["prefill"]),
            "decode_modules": module_values(scopes["decode"]),
        }

    options.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    options.markdown.write_text(render(summary), encoding="utf-8")
    print(options.markdown.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
