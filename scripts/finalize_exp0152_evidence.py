#!/usr/bin/env python3
"""Create the formal EXP-0153 profiling-closure evidence and report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from pathlib import Path


RECIPES = ("f16f16", "w4f16", "w4u8")
DISPLAY = {"f16f16": "W16A16", "w4f16": "W4A16", "w4u8": "W4A8"}
QTIMER_MHZ = 19.2
MODULES = (
    ("I/O 与 metadata", ("input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks")),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm-RoPE", ("qkv_projection_ticks", "qk_norm_rope_ticks")),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-attn residual + RMSNorm", (
        "post_attention_residual_ticks", "post_attention_norm_ticks",
    )),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
    ("KV-cache carrier conversion", ("scan_cache_pack_ticks",)),
    ("KV-cache append DMA", ("scan_cache_append_ticks",)),
    ("Block internal orchestration", ("block_orchestration_ticks",)),
    ("Layer bookkeeping", ("layer_bookkeeping_ticks",)),
    ("DSP unattributed residual", ("stage_boundary_ticks",)),
    ("DSP runtime setup/teardown", (
        "runtime_setup_ticks", "runtime_teardown_ticks",
    )),
)
PRIMARY = (
    ("Host wall", "host_wall_ns", "host_ns"),
    ("DSP block total", "total_ticks", "ticks"),
    ("DSP invocation", "invocation_ticks", "ticks"),
    ("Runtime setup", "runtime_setup_ticks", "ticks"),
    ("Runtime teardown", "runtime_teardown_ticks", "ticks"),
    ("KV-cache carrier conversion", "scan_cache_pack_ticks", "ticks"),
    ("KV-cache append DMA", "scan_cache_append_ticks", "ticks"),
    ("Block internal orchestration", "block_orchestration_ticks", "ticks"),
    ("Layer bookkeeping", "layer_bookkeeping_ticks", "ticks"),
    ("DSP unattributed residual", "stage_boundary_ticks", "ticks"),
    ("Named ledger total", "ledger_named_ticks", "ticks"),
    ("Unattributed ledger", "ledger_unattributed_ticks", "ticks"),
)
LEDGER = (
    ("Input", "input_stage_ticks", "ticks"),
    ("Metadata", "metadata_stage_ticks", "ticks"),
    ("Input RMSNorm", "input_norm_ticks", "ticks"),
    ("QKV", "qkv_projection_ticks", "ticks"),
    ("Q/K Norm-RoPE", "qk_norm_rope_ticks", "ticks"),
    ("Attention", "attention_ticks", "ticks"),
    ("O projection", "o_projection_ticks", "ticks"),
    ("Post-attn residual", "post_attention_residual_ticks", "ticks"),
    ("Post-attn RMSNorm", "post_attention_norm_ticks", "ticks"),
    ("Gate/Up", "gate_up_ticks", "ticks"),
    ("Activation", "activation_ticks", "ticks"),
    ("Down", "down_ticks", "ticks"),
    ("Final residual", "final_residual_ticks", "ticks"),
    ("Output", "output_stage_ticks", "ticks"),
    ("KV-cache carrier conversion", "scan_cache_pack_ticks", "ticks"),
    ("KV-cache append DMA", "scan_cache_append_ticks", "ticks"),
    ("Block internal orchestration", "block_orchestration_ticks", "ticks"),
    ("Layer bookkeeping", "layer_bookkeeping_ticks", "ticks"),
    ("DSP unattributed residual", "stage_boundary_ticks", "ticks"),
)
PROJECTION_DIAGNOSTICS = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "hmx_ready_wait_ticks", "w4f16_expand_ticks",
    "w4f16_expand_work_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_prefetch_wait_ticks", "w4f16_hmx_tail_wait_ticks",
    "w4f16_cross_prefetch_wait_ticks", "w4f16_cross_prefetch_lifetime_ticks",
    "w4u8_qkvo_weight_expand_ticks", "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks",
)
ATTENTION_DIAGNOSTICS = (
    "attention_setup_ticks", "attention_qk_pack_ticks",
    "attention_qk_hmx_ticks", "attention_qk_unpack_ticks",
    "attention_softmax_ticks", "attention_av_pack_ticks",
    "attention_av_hmx_ticks", "attention_av_unpack_ticks",
    "attention_gqa_pipeline_ticks", "attention_unattributed_ticks",
    "u8_attention_qk_norm_rope_ticks", "u8_attention_k_pack_ticks",
    "u8_attention_v_pack_ticks", "u8_attention_qk_hmx_ticks",
    "u8_attention_qk_requant_ticks", "u8_attention_softmax_ticks",
    "u8_attention_av_hmx_ticks", "u8_attention_av_requant_ticks",
    "u8_attention_pipeline_wait_ticks", "scan_dynamic_attention_ticks",
)
MLP_DIAGNOSTICS = (
    "w4f16_gate_up_weight_dma_ticks", "w4f16_gate_up_expand_ticks",
    "w4f16_gate_up_expand_work_ticks", "w4f16_gate_up_expand_pool_wait_ticks",
    "w4f16_gate_up_hmx_wait_ticks", "w4f16_gate_up_hmx_tail_wait_ticks",
    "w4f16_gate_up_stream_work_ticks", "w4f16_gate_up_stream_ready_wait_ticks",
    "w4f16_gate_up_stream_join_wait_ticks", "w4u8_mlp_gate_up_pipeline_ticks",
    "w4u8_mlp_down_pipeline_ticks", "w4u8_mlp_activation_work_ticks",
    "w4u8_mlp_weight_stage_ticks", "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_hmx_compute_ticks", "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks", "w4u8_mlp_expanded_slot_wait_ticks",
)
PHYSICAL = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    "block_invocation_count", "hmx_command_count", "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count", "weight_dma_descriptor_count",
    "boundary_dma_descriptor_count", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "weight_ddr_read_bytes",
    "boundary_ddr_read_bytes", "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes", "intermediate_ddr_write_bytes",
    "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
)
CORRECTNESS = (
    "output_mismatches", "output_max_lsb", "output_nonfinite_count",
    "cache_prefix_mismatches", "cache_mismatches",
    "cache_structure_mismatches", "intermediate_spill_fill_count",
    "u8_attention_probability_mask_violation_count",
    "u8_attention_fused_k_operand_mismatch_count", "w4f16_expand_mismatch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--static-gate", type=Path, required=True)
    parser.add_argument("--fp16-evidence", type=Path, required=True)
    parser.add_argument("--w4u8-exact-report", type=Path, required=True)
    parser.add_argument("--w4u8-reference-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--w4u8-package", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def parse_log(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    profiles: list[dict[str, object]] = []
    completion: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("record") == "replay_profile":
            profiles.append(record)
        if record.get("replay_sequence_complete") is True:
            completion = record
    if len(profiles) != 9 or completion is None:
        raise ValueError(f"{path}: incomplete replay log")
    if any(int(record.get("experiment", -1)) != 153 for record in profiles):
        raise ValueError(f"{path}: replay profile is not EXP-0153")
    if int(completion.get("experiment", -1)) != 153:
        raise ValueError(f"{path}: replay completion is not EXP-0153")
    return sorted(profiles, key=lambda row: int(row["replay_step"])), completion


def samples(
    runs: dict[str, list[list[dict[str, object]]]], recipe: str,
    phase: str, scope: str,
) -> list[dict[str, object]]:
    selected = runs[recipe][:1] if scope == "repeat_one" else runs[recipe]
    if phase == "prefill":
        return [run[0] for run in selected]
    return [record for run in selected for record in run[1:]]


def numeric_median(records: list[dict[str, object]], field: str) -> float:
    return median([float(record[field]) for record in records])


def time_us(records: list[dict[str, object]], field: str, kind: str) -> float:
    value = numeric_median(records, field)
    return value / 1000.0 if kind == "host_ns" else value / QTIMER_MHZ


def improvement(control: float, candidate: float) -> float:
    return (control / candidate - 1.0) * 100.0 if candidate else 0.0


def markdown_time_table(
    title: str, specs: tuple[tuple[str, str, str], ...],
    runs: dict[str, list[list[dict[str, object]]]], phase: str, scope: str,
) -> list[str]:
    lines = [
        f"### {title}", "",
        "| Metric | W16A16 us | W4A16 us | W4A8 us | W4A16 vs W16A16 | W4A8 vs W4A16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, field, kind in specs:
        values = {
            recipe: time_us(samples(runs, recipe, phase, scope), field, kind)
            for recipe in RECIPES
        }
        lines.append(
            f"| {label} | {values['f16f16']:.3f} | {values['w4f16']:.3f} | "
            f"{values['w4u8']:.3f} | "
            f"{improvement(values['f16f16'], values['w4f16']):+.2f}% | "
            f"{improvement(values['w4f16'], values['w4u8']):+.2f}% |"
        )
    lines.append("")
    return lines


def module_values(
    records: list[dict[str, object]],
) -> tuple[list[tuple[str, float]], float]:
    values: list[tuple[str, float]] = []
    for label, fields in MODULES:
        per_record = [
            sum(float(record[field]) for field in fields) / QTIMER_MHZ
            for record in records
        ]
        values.append((label, median(per_record)))
    host = numeric_median(records, "host_wall_ns") / 1000.0
    host_boundary = median([
        float(record["host_wall_ns"]) / 1000.0 -
        float(record["invocation_ticks"]) / QTIMER_MHZ
        for record in records
    ])
    values.append(("Host-DSP boundary", host_boundary))
    return values, host


def module_overview(
    runs: dict[str, list[list[dict[str, object]]]], phase: str,
) -> list[str]:
    data = {
        recipe: module_values(samples(runs, recipe, phase, "repeat_ten"))
        for recipe in RECIPES
    }
    lines = [
        f"## PC-028 three-recipe module overview — {phase}", "",
        "| Module | W16A16 | W4A16 | W4A8 | W4A16 speed vs W16A16 | W4A8 speed vs W4A16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, (label, _) in enumerate(data["f16f16"][0]):
        values = {recipe: data[recipe][0][index][1] for recipe in RECIPES}
        cells = {
            recipe: f"{value:.1f} us ({value / data[recipe][1] * 100.0:.1f}%)"
            for recipe, value in values.items()
        }
        lines.append(
            f"| {label} | {cells['f16f16']} | {cells['w4f16']} | {cells['w4u8']} | "
            f"{improvement(values['f16f16'], values['w4f16']):+.2f}% | "
            f"{improvement(values['w4f16'], values['w4u8']):+.2f}% |"
        )
    walls = {recipe: data[recipe][1] for recipe in RECIPES}
    logical_m = 64 if phase == "prefill" else 1
    lines.append(
        f"| Complete Host wall | {walls['f16f16']:.1f} us (100.0%) | "
        f"{walls['w4f16']:.1f} us (100.0%) | {walls['w4u8']:.1f} us (100.0%) | "
        f"{improvement(walls['f16f16'], walls['w4f16']):+.2f}% | "
        f"{improvement(walls['w4f16'], walls['w4u8']):+.2f}% |"
    )
    lines.append(
        f"| Useful throughput | {logical_m * 1e6 / walls['f16f16']:.3f} tok/s | "
        f"{logical_m * 1e6 / walls['w4f16']:.3f} tok/s | "
        f"{logical_m * 1e6 / walls['w4u8']:.3f} tok/s | N/A | N/A |"
    )
    lines.append("")
    return lines


def diagnostic_table(
    title: str, fields: tuple[str, ...],
    runs: dict[str, list[list[dict[str, object]]]], phase: str, scope: str,
    ticks: bool,
) -> list[str]:
    unit = "us" if ticks else "raw"
    lines = [
        f"### {title}", "",
        f"| Metric ({unit}) | W16A16 | W4A16 | W4A8 |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        values = {}
        for recipe in RECIPES:
            value = numeric_median(samples(runs, recipe, phase, scope), field)
            values[recipe] = value / QTIMER_MHZ if ticks else value
        format_value = (lambda value: f"{value:.3f}") if ticks else (
            lambda value: f"{value:.0f}"
        )
        lines.append(
            f"| {field} | {format_value(values['f16f16'])} | "
            f"{format_value(values['w4f16'])} | {format_value(values['w4u8'])} |"
        )
    lines.append("")
    return lines


def per_layer_table(
    runs: dict[str, list[list[dict[str, object]]]], recipe: str, phase: str,
) -> list[str]:
    records = samples(runs, recipe, phase, "repeat_ten")
    lines = [
        f"### Per-layer ledger — {DISPLAY[recipe]} {phase}", "",
        "| Layer | Total us | QKV us | Attention us | O us | Gate/Up us | Down us | KV pack us | Cache DMA us | Block orchestration us | Bookkeeping us | Unattributed us | Weight DDR MiB | Cache read KiB | Cache write KiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in range(28):
        values = [record[f"slice_layer_{layer}"] for record in records]
        med = lambda field: median([float(value[field]) for value in values])
        lines.append(
            f"| {layer} | {med('layer_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('qkv_projection_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('attention_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('o_projection_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('gate_up_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('down_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('cache_append_pack_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('cache_append_dma_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('block_orchestration_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('layer_bookkeeping_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('layer_unattributed_ticks') / QTIMER_MHZ:.3f} | "
            f"{med('weight_ddr_read_bytes') / (1024.0 * 1024.0):.3f} | "
            f"{med('cache_ddr_read_bytes') / 1024.0:.3f} | "
            f"{med('cache_ddr_write_bytes') / 1024.0:.3f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    runs: dict[str, list[list[dict[str, object]]]] = {recipe: [] for recipe in RECIPES}
    completions: list[dict[str, object]] = []
    log_paths: list[Path] = []
    for round_index in range(1, 11):
        for recipe in RECIPES:
            path = args.logs / f"r{round_index:02d}_{recipe}.log"
            profiles, completion = parse_log(path)
            runs[recipe].append(profiles)
            completions.append(completion)
            log_paths.append(path)

    mapping_records: dict[str, dict[str, object]] = {}
    layout_records: dict[str, dict[str, object]] = {}
    mapping_paths: list[Path] = []
    for recipe in RECIPES:
        for mode, destination in (("layout", layout_records), ("map", mapping_records)):
            path = args.mapping / f"{recipe}_{mode}.log"
            mapping_paths.append(path)
            records = [
                json.loads(line) for line in path.read_text().splitlines()
                if line.startswith("{")
            ]
            if len(records) != 1:
                raise ValueError(f"unexpected {mode} log {path}")
            destination[recipe] = records[0]

    exact = json.loads(args.w4u8_exact_report.read_text())
    f16_composition = json.loads(
        (args.fp16_evidence / "f16f16_composition_audit_final_268a6fe.json").read_text()
    )
    w4f16_composition = json.loads(
        (args.fp16_evidence / "w4f16_composition_audit_final_268a6fe.json").read_text()
    )
    f16_conditional = json.loads(
        (args.fp16_evidence / "f16f16_conditional_trajectory_cache_audit.json").read_text()
    )
    w4f16_conditional = json.loads(
        (args.fp16_evidence / "w4f16_conditional_trajectory_cache_audit.json").read_text()
    )

    all_profiles = [record for recipe_runs in runs.values() for run in recipe_runs for record in run]
    physical_pass = all(
        int(record["vtcm_requested_bytes"]) == 8388608 and
        int(record["vtcm_acquired_bytes"]) == 8388608 and
        int(record["intermediate_ddr_read_bytes"]) == 0 and
        int(record["intermediate_ddr_write_bytes"]) == 0 and
        int(record["intermediate_spill_fill_count"]) == 0 and
        record["backend"] == "standalone_fastrpc_dsp" and
        record["qnn"] == "none"
        for record in all_profiles
    )
    correctness_pass = (
        all(bool(record["all_steps_pass"]) for record in completions) and
        bool(exact["passed"]) and bool(f16_composition["pass"]) and
        bool(w4f16_composition["pass"]) and
        bool(f16_conditional["conditional_gate"]["pass"]) and
        bool(w4f16_conditional["conditional_gate"]["pass"])
    )
    mapping_pass = all(
        bool(layout_records[recipe]["uint32_fit"]) and
        bool(mapping_records[recipe]["gate_pass"])
        for recipe in RECIPES
    )
    profiling_pass = all(
        int(record["ledger_unattributed_ticks"]) == 0 and
        int(record["stage_boundary_ticks"]) <=
            0.001 * int(record["invocation_ticks"]) and
        all(
            int(record[f"slice_layer_{layer}"]["layer_unattributed_ticks"]) <=
                0.001 * int(record[f"slice_layer_{layer}"]["layer_ticks"])
            for layer in range(28)
        )
        for record in all_profiles
    )

    aggregates: dict[str, object] = {}
    for scope in ("repeat_one", "repeat_ten"):
        aggregates[scope] = {}
        for phase in ("prefill", "decode"):
            aggregates[scope][phase] = {}
            for recipe in RECIPES:
                records = samples(runs, recipe, phase, scope)
                numeric_fields = sorted(
                    key for key, value in records[0].items()
                    if isinstance(value, (int, float))
                )
                aggregates[scope][phase][recipe] = {
                    field: numeric_median(records, field)
                    for field in numeric_fields
                }

    package_paths = {
        "f16f16": args.model_root / "f16f16" / "manifest.json",
        "w4f16": args.model_root / "w4f16" / "manifest.json",
        "w4u8": args.w4u8_package / "manifest.json",
    }
    binary_paths = {
        "host_cli": args.source_root /
            "android_ReleaseG_aarch64/ship/qwen3_block_cli",
        "host_stub": args.source_root /
            "android_ReleaseG_aarch64/ship/libqwen3_probe.so",
        "dsp_skel": args.source_root /
            "hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so",
    }
    identity = {
        "experiment": "EXP-0153",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "execution_unit": "Qwen3 layers 0-27, M64 prefill plus 8 continuous decode tokens",
        "project_variants": list(RECIPES),
        "direct_controls": {
            "w4f16": "f16f16",
            "w4u8": "w4f16",
        },
        "paired": True,
        "rotated_rounds": 10,
        "repeat_one_scope": "round 1; decode is median over positions 64-71",
        "repeat_ten_scope": "10 rotated sessions; prefill n=10, decode n=80",
        "artifact_directory": str(args.w4u8_package),
        "package_manifest_sha256": {
            recipe: sha256(path) for recipe, path in package_paths.items()
        },
        "binary_sha256": {
            name: sha256(path) for name, path in binary_paths.items()
        },
    }
    gate_summary = {
        **identity,
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "pass" if (
            mapping_pass and correctness_pass and physical_pass and profiling_pass
        ) else "fail",
        "adoption_status": "not_applicable",
        "mapping_gate_pass": mapping_pass,
        "correctness_gate_pass": correctness_pass,
        "physical_gate_pass": physical_pass,
        "profiling_gate_pass": profiling_pass,
        "speed_gate": "not_applicable",
        "layout": layout_records,
        "mapping": mapping_records,
        "w4u8_exact_summary": exact["summary"],
        "aggregates": aggregates,
    }

    lines = [
        "# EXP-0153 full profiling closure report", "",
        f"Source: `{args.source_branch}` at `{args.source_commit}`. The execution unit and kernels are unchanged from EXP-0152. This instrumentation-only experiment separates KV-cache carrier conversion, cache DMA, block orchestration, layer bookkeeping, DSP unattributed residual and the true Host-DSP boundary.",
        "",
    ]
    lines += module_overview(runs, "prefill")
    lines += module_overview(runs, "decode")
    for scope in ("repeat_one", "repeat_ten"):
        for phase in ("prefill", "decode"):
            lines += [f"## PC-027 {scope} — {phase}", ""]
            lines += markdown_time_table(
                "Primary latency", PRIMARY, runs, phase, scope
            )
            lines += markdown_time_table(
                "Complete additive Block Timing Ledger", LEDGER, runs, phase, scope
            )
            lines += diagnostic_table(
                "Projection diagnostics (overlapping engine/wait work; not additive)",
                PROJECTION_DIAGNOSTICS, runs, phase, scope, True,
            )
            lines += diagnostic_table(
                "Attention diagnostics (overlapping engine/wait work; not additive)",
                ATTENTION_DIAGNOSTICS, runs, phase, scope, True,
            )
            lines += diagnostic_table(
                "MLP diagnostics (overlapping engine/wait work; not additive)",
                MLP_DIAGNOSTICS, runs, phase, scope, True,
            )
            lines += diagnostic_table(
                "Physical contract", PHYSICAL, runs, phase, scope, False,
            )
            lines += diagnostic_table(
                "Correctness counters", CORRECTNESS, runs, phase, scope, False,
            )
    lines += ["## Complete per-layer repeat-ten ledgers", ""]
    for phase in ("prefill", "decode"):
        for recipe in RECIPES:
            lines += per_layer_table(runs, recipe, phase)
    lines += [
        "## Gate closure", "",
        f"- Mapping/layout: {'PASS' if mapping_pass else 'FAIL'}; all three packages fit uint32 and map as one rpcmem_alloc2 arena.",
        f"- Correctness: {'PASS' if correctness_pass else 'FAIL'}; W4U8 prefill, 8 decode outputs and 56 final caches are byte exact (0 LSB).",
        f"- Physical: {'PASS' if physical_pass else 'FAIL'}; every formal record requests/acquires 8 MiB VTCM, has zero intermediate DDR and spill/fill, standalone FastRPC backend, and no QNN.",
        f"- Profiling closure: {'PASS' if profiling_pass else 'FAIL'}; DSP invocation and every per-layer ledger leave at most 0.1% unattributed after explicit KV conversion, cache DMA, block orchestration and bookkeeping intervals.",
        "- Persistent K/V cache DDR and immutable weight DDR are legal declared boundaries and are reported separately.",
        "- Engine work and wait counters overlap; only the Block Timing Ledger is additive.",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        logs_dir = staging / "rotated_logs"
        logs_dir.mkdir()
        for path in log_paths:
            shutil.copy2(path, logs_dir / path.name)
        mapping_dir = staging / "mapping"
        mapping_dir.mkdir()
        for path in mapping_paths:
            shutil.copy2(path, mapping_dir / path.name)
        shutil.copy2(args.static_gate, staging / "static_gate.log")
        evidence_sources = {
            "f16f16_composition_audit.json": args.fp16_evidence /
                "f16f16_composition_audit_final_268a6fe.json",
            "w4f16_composition_audit.json": args.fp16_evidence /
                "w4f16_composition_audit_final_268a6fe.json",
            "f16f16_conditional_audit.json": args.fp16_evidence /
                "f16f16_conditional_trajectory_cache_audit.json",
            "w4f16_conditional_audit.json": args.fp16_evidence /
                "w4f16_conditional_trajectory_cache_audit.json",
            "w4u8_exact_replay_audit.json": args.w4u8_exact_report,
            "w4u8_exact_reference_manifest.json": args.w4u8_reference_manifest,
        }
        for name, path in evidence_sources.items():
            shutil.copy2(path, staging / name)
        (staging / "gate_summary.json").write_text(
            json.dumps(gate_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "full_profiling_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        ledger = {
            str(path.relative_to(staging)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        (staging / "evidence_sha256.json").write_text(
            json.dumps({
                "experiment": "EXP-0153",
                "source_commit": args.source_commit,
                "files": ledger,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(json.dumps({
        "output": str(output),
        "local_gate": gate_summary["local_gate"],
        "mapping_gate_pass": mapping_pass,
        "correctness_gate_pass": correctness_pass,
        "physical_gate_pass": physical_pass,
        "profiling_gate_pass": profiling_pass,
        "full_profiling_report_sha256": sha256(output / "full_profiling_report.md"),
        "gate_summary_sha256": sha256(output / "gate_summary.json"),
        "evidence_sha256_ledger_sha256": sha256(output / "evidence_sha256.json"),
    }, indent=2, sort_keys=True))
    if gate_summary["local_gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
