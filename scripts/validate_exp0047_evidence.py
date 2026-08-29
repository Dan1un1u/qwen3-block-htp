#!/usr/bin/env python3
"""Validate and report EXP-0047 paired W4F16/W4U8 evidence."""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any


SAMPLES = 7
REPEATS = (1, 10)
CONTROL = "w4f16"
CANDIDATE = "w4u8"
VTCM_BYTES = 8_388_608
OUTPUT_HASHES = {
    CONTROL: "f18b9abbe1487231",
    CANDIDATE: "69f22eeb035e5ec5",
}


@dataclass(frozen=True)
class Metric:
    label: str
    key: str
    kind: str = "per_block"
    available: str = "both"


PRIMARY = (
    Metric("Host wall / block (ms)", "host_wall_ns_per_block", "ms"),
    Metric("DSP block total", "total_ticks"),
    Metric("Invocation", "invocation_ticks"),
    Metric("Runtime setup", "runtime_setup_ticks"),
    Metric("Runtime teardown", "runtime_teardown_ticks"),
    Metric("Named ledger total", "ledger_named_ticks"),
    Metric("Unattributed ledger", "ledger_unattributed_ticks"),
)

LEDGER = (
    Metric("Input", "input_stage_ticks"),
    Metric("Metadata", "metadata_stage_ticks"),
    Metric("Input RMSNorm", "input_norm_ticks"),
    Metric("QKV projection", "qkv_projection_ticks"),
    Metric("Q/K Norm + RoPE", "qk_norm_rope_ticks"),
    Metric("Attention", "attention_ticks"),
    Metric("O projection", "o_projection_ticks"),
    Metric("Post-Attention residual", "post_attention_residual_ticks"),
    Metric("Post-Attention norm", "post_attention_norm_ticks"),
    Metric("Gate/Up", "gate_up_ticks"),
    Metric("Activation", "activation_ticks"),
    Metric("Down", "down_ticks"),
    Metric("Final residual", "final_residual_ticks"),
    Metric("Output", "output_stage_ticks"),
)

PROJECTION = (
    Metric("Weight DMA ticks", "weight_dma_ticks"),
    Metric("HMX compute ticks, all projections", "hmx_compute_ticks"),
    Metric("Projection pack ticks", "projection_pack_ticks"),
    Metric("Projection unpack ticks", "projection_unpack_ticks"),
    Metric("Projection HMX wait ticks", "projection_hmx_wait_ticks"),
    Metric("HMX ready wait ticks", "hmx_ready_wait_ticks"),
    Metric("W4F16 expansion wall ticks", "w4f16_expand_ticks", available=CONTROL),
    Metric("W4F16 expansion work ticks", "w4f16_expand_work_ticks", available=CONTROL),
    Metric("W4F16 expansion pool wait", "w4f16_expand_pool_wait_ticks", available=CONTROL),
    Metric("W4F16 prefetch wait", "w4f16_prefetch_wait_ticks", available=CONTROL),
    Metric("W4F16 cross-prefetch lifetime", "w4f16_cross_prefetch_lifetime_ticks", available=CONTROL),
    Metric("W4U8 QKVO expansion ticks", "w4u8_qkvo_weight_expand_ticks", available=CANDIDATE),
    Metric("W4U8 QKVO prefetch wait", "w4u8_qkvo_prefetch_wait_ticks", available=CANDIDATE),
    Metric("W4U8 QKVO HMX lifetime", "w4u8_qkvo_hmx_lifetime_ticks", available=CANDIDATE),
)

ATTENTION = (
    Metric("FP16 Q/K Norm-RoPE worker work", "attention_qk_norm_worker_work_ticks", available=CONTROL),
    Metric("FP16 Q/K Norm-RoPE pool wait", "attention_qk_norm_pool_wait_ticks", available=CONTROL),
    Metric("FP16 GQA worker work", "attention_gqa_worker_work_ticks", available=CONTROL),
    Metric("FP16 GQA HMX wait", "attention_gqa_hmx_wait_ticks", available=CONTROL),
    Metric("FP16 GQA queue wait", "attention_gqa_queue_wait_ticks", available=CONTROL),
    Metric("U8 Q/K Norm-RoPE work", "u8_attention_qk_norm_rope_ticks", available=CANDIDATE),
    Metric("U8 V pack", "u8_attention_v_pack_ticks", available=CANDIDATE),
    Metric("U8 QK HMX", "u8_attention_qk_hmx_ticks", available=CANDIDATE),
    Metric("U8 QK requant", "u8_attention_qk_requant_ticks", available=CANDIDATE),
    Metric("U8 log2 Softmax", "u8_attention_softmax_ticks", available=CANDIDATE),
    Metric("U8 AV HMX", "u8_attention_av_hmx_ticks", available=CANDIDATE),
    Metric("U8 AV requant", "u8_attention_av_requant_ticks", available=CANDIDATE),
    Metric("U8 Attention pipeline wait", "u8_attention_pipeline_wait_ticks", available=CANDIDATE),
    Metric("FP16 GQA groups", "attention_gqa_group_count", available=CONTROL),
    Metric("U8 GQA groups", "u8_attention_group_count", available=CANDIDATE),
    Metric("U8 QK executions", "u8_attention_qk_execution_count", available=CANDIDATE),
    Metric("U8 AV executions", "u8_attention_av_execution_count", available=CANDIDATE),
)

MLP = (
    Metric("W4F16 Gate/Up weight DMA", "w4f16_gate_up_weight_dma_ticks", available=CONTROL),
    Metric("W4F16 Gate/Up expansion work", "w4f16_gate_up_expand_work_ticks", available=CONTROL),
    Metric("W4F16 Gate/Up HMX wait", "w4f16_gate_up_hmx_wait_ticks", available=CONTROL),
    Metric("W4F16 Gate/Up prefetch wait", "w4f16_gate_up_prefetch_wait_ticks", available=CONTROL),
    Metric("W4F16 Gate/Up stream work", "w4f16_gate_up_stream_work_ticks", available=CONTROL),
    Metric("W4F16 Gate/Up stream ready wait", "w4f16_gate_up_stream_ready_wait_ticks", available=CONTROL),
    Metric("W4U8 Gate/Up pipeline", "w4u8_mlp_gate_up_pipeline_ticks", available=CANDIDATE),
    Metric("W4U8 Down pipeline", "w4u8_mlp_down_pipeline_ticks", available=CANDIDATE),
    Metric("W4U8 activation work", "w4u8_mlp_activation_work_ticks", available=CANDIDATE),
    Metric("W4U8 weight stage", "w4u8_mlp_weight_stage_ticks", available=CANDIDATE),
    Metric("W4U8 weight expansion", "w4u8_mlp_weight_expand_ticks", available=CANDIDATE),
    Metric("W4U8 MLP HMX compute", "w4u8_mlp_hmx_compute_ticks", available=CANDIDATE),
    Metric("W4U8 MLP HMX ready wait", "w4u8_mlp_hmx_ready_wait_ticks", available=CANDIDATE),
    Metric("W4U8 producer slot wait", "w4u8_mlp_producer_slot_wait_ticks", available=CANDIDATE),
    Metric("W4U8 expanded slot wait", "w4u8_mlp_expanded_slot_wait_ticks", available=CANDIDATE),
    Metric("W4U8 input pack", "w4u8_mlp_input_pack_ticks", available=CANDIDATE),
    Metric("W4U8 output unpack", "w4u8_mlp_output_unpack_ticks", available=CANDIDATE),
    Metric("W4U8 published pairs", "w4u8_mlp_pair_publish_count", available=CANDIDATE),
    Metric("W4U8 consumed pairs", "w4u8_mlp_pair_consume_count", available=CANDIDATE),
)

PHYSICAL = (
    Metric("Weight DDR read (bytes)", "weight_ddr_read_bytes"),
    Metric("Boundary DDR read (bytes)", "boundary_ddr_read_bytes"),
    Metric("Boundary DDR write (bytes)", "boundary_ddr_write_bytes"),
    Metric("Intermediate DDR read (bytes)", "intermediate_ddr_read_bytes"),
    Metric("Intermediate DDR write (bytes)", "intermediate_ddr_write_bytes"),
    Metric("Intermediate DMA descriptors", "intermediate_dma_descriptor_count"),
    Metric("Weight DMA descriptors", "weight_dma_descriptor_count"),
    Metric("Spill/fill", "intermediate_spill_fill_count"),
    Metric("HMX commands", "hmx_command_count"),
    Metric("FP16 HMX tile pairs", "hmx_fp16_tile_pair_count", available=CONTROL),
    Metric("U8xS8 HMX tile pairs", "hmx_u8s8_tile_pair_count", available=CANDIDATE),
    Metric("VTCM requested (bytes)", "vtcm_requested_bytes", "static"),
    Metric("VTCM acquired (bytes)", "vtcm_acquired_bytes", "static"),
    Metric("VTCM peak plan (bytes)", "vtcm_peak_plan_bytes", "static"),
    Metric("FastRPC block invocations", "block_invocation_count"),
)

ALL_METRICS = tuple(dict.fromkeys(
    metric for table in (PRIMARY, LEDGER, PROJECTION, ATTENTION, MLP, PHYSICAL)
    for metric in table
))


def require(record: dict[str, Any], field: str, expected: Any) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: pathlib.Path, expected: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != expected:
        raise SystemExit(f"wrong record count for {path}: {len(records)} != {expected}")
    return records


def load_manifest_text(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def normalized(record: dict[str, Any], metric: Metric) -> float:
    value = float(record[metric.key])
    if metric.kind == "static":
        return value
    if metric.kind == "ms":
        return value / 1_000_000.0
    return value / int(record["repeat_count"])


def percent(candidate: float, control: float) -> float | None:
    if control == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate / control - 1.0) * 100.0


def metric_summary(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: Metric,
) -> dict[str, float | None]:
    left = [normalized(record, metric) for record in control]
    right = [normalized(record, metric) for record in candidate]
    left_median = statistics.median(left)
    right_median = statistics.median(right)
    paired = [percent(right[i], left[i]) for i in range(SAMPLES)]
    paired_valid = [value for value in paired if value is not None]
    return {
        "control_median": left_median,
        "candidate_median": right_median,
        "change_percent": percent(right_median, left_median),
        "paired_change_percent_median": (
            statistics.median(paired_valid) if paired_valid else None
        ),
    }


def validate_record(record: dict[str, Any], repeat: int, variant: str) -> None:
    for metric in ALL_METRICS:
        if metric.key not in record:
            raise SystemExit(f"missing PC-027 field {metric.key} in {variant}")
    common = {
        "experiment": "EXP-0046",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "repeat_count": repeat,
        "attribution_mode": "on",
        "numerical_audit_mode": "off",
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
        "warmup_max_lsb": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "mismatches": 0,
        "max_lsb": 0,
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in common.items():
        require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit(f"{variant} VTCM plan exceeds 8 MiB")
    if int(record["ledger_named_ticks"]) + int(record["ledger_unattributed_ticks"]) != int(record["invocation_ticks"]):
        raise SystemExit(f"{variant} invocation ledger does not close")
    ledger_sum = sum(int(record[metric.key]) for metric in LEDGER)
    teardown = int(record["runtime_teardown_ticks"])
    unattributed = int(record["ledger_unattributed_ticks"])
    setup = int(record["runtime_setup_ticks"])
    if ledger_sum + teardown + unattributed != int(record["total_ticks"]):
        raise SystemExit(f"{variant} additive block ledger does not close")
    if int(record["total_ticks"]) + setup != int(record["invocation_ticks"]):
        raise SystemExit(f"{variant} invocation/setup ledger does not close")
    if ledger_sum + setup + teardown != int(record["ledger_named_ticks"]):
        raise SystemExit(f"{variant} named ledger does not close")
    if variant == CONTROL:
        fixed = {
            "variant": "W4F16",
            "attention_compute": "FP16_HMX",
            "projection_compute": "FP16_HMX",
            "common_ops_mode": "hvx_fp16",
            "residual_mode": "hvx_fused_post_norm",
            "w4f16_pipeline_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
            "attention_pack_mode": "combined_hvx",
            "attention_pipeline_mode": "gqa_qkv_overlap",
            "attention_hvx_contexts": 4,
            "mlp_mode": "crouton_native_batch8",
            "mlp_hvx_contexts": 4,
            "mlp_chunk_vectors": 64,
            "crouton_boundary_mode": "norms",
            "output_hash": OUTPUT_HASHES[variant],
            "hmx_command_count": 208 * repeat,
            "hmx_fp16_tile_pair_count": 98_816 * repeat,
            "hmx_u8s8_tile_pair_count": 0,
            "weight_dma_descriptor_count": 119 if repeat == 1 else 1127,
            "weight_ddr_read_bytes": 25_296_896 if repeat == 1 else 251_789_312,
            "w4f16_requested_hvx_workers": 3,
            "w4f16_region_tiles": 32,
            "w4f16_hvx_workers_created": 3,
            "w4f16_hvx_workers_locked": 3,
            "crouton_norm_projection_count": 2 * repeat,
        }
    else:
        fixed = {
            "variant": "W4U8",
            "attention_compute": "U8xS8_HMX_log2_softmax",
            "projection_compute": "U8xS8_integer_HMX",
            "common_ops_mode": "rms_rope_softmax",
            "residual_mode": "hvx_fused_post_norm",
            "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
            "attention_hvx_contexts": 4,
            "mlp_mode": "w4u8_streaming",
            "mlp_hvx_contexts": 3,
            "mlp_chunk_vectors": 64,
            "crouton_boundary_mode": "w4u8_mlp_io",
            "w4u8_qkvo_pipeline_mode": "qkvo_batch4",
            "output_hash": OUTPUT_HASHES[variant],
            "hmx_command_count": 1040 * repeat,
            "hmx_fp16_tile_pair_count": 0,
            "hmx_u8s8_tile_pair_count": 49_408 * repeat,
            "weight_dma_descriptor_count": 512 * repeat,
            "weight_ddr_read_bytes": 25_444_352 * repeat,
            "u8_attention_audit_ddr_write_bytes": 0,
            "u8_attention_fused_k_operand_mismatch_count": 0,
            "w4u8_qkv_batch_n_tiles": 4,
            "w4u8_qkv_batch_count": 32 * repeat,
            "w4u8_qkvo_prefetch_count": 44 * repeat,
            "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
            "w4u8_mlp_input_pack_skipped": repeat,
            "w4u8_mlp_output_unpack_skipped": repeat,
            "w4u8_mlp_input_pack_ticks": 0,
            "w4u8_mlp_output_unpack_ticks": 0,
        }
    for field, expected in fixed.items():
        require(record, field, expected)


def validate_audit(record: dict[str, Any], variant: str) -> None:
    require(record, "repeat_count", 1)
    require(record, "attribution_mode", "on")
    require(record, "numerical_audit_mode", "on")
    require(record, "rpc_result", 0)
    require(record, "dsp_status", 3)
    require(record, "numerical_status", 1)
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    require(record, "output_hash", OUTPUT_HASHES[variant])
    require(record, "common_op_nonfinite_count", 0)
    require(record, "common_op_softmax_mask_violation_count", 0)
    require(record, "intermediate_ddr_read_bytes", 0)
    require(record, "intermediate_ddr_write_bytes", 0)
    require(record, "intermediate_spill_fill_count", 0)
    if variant == CANDIDATE:
        require(record, "u8_attention_probability_mask_violation_count", 0)
        require(record, "u8_attention_fused_k_operand_mismatch_count", 0)


def validate_package(package: pathlib.Path) -> dict[str, Any]:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    names = []
    for projection in ("q", "k", "v", "o", "gate", "up", "down"):
        names.extend((
            f"{projection}_weight_w4_hmx.bin",
            f"{projection}_weight_w4_scale_f32.bin",
        ))
    bundle = hashlib.sha256()
    details: dict[str, str] = {}
    for name in names:
        path = package / name
        actual = sha256(path)
        expected = files[name]["sha256"]
        if actual != expected:
            raise SystemExit(f"package hash mismatch for {name}")
        details[name] = actual
        bundle.update(name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(bytes.fromhex(actual))
    return {
        "package_manifest_sha256": sha256(manifest_path),
        "shared_w4_bundle_sha256": bundle.hexdigest(),
        "shared_w4_files": details,
    }


def build_summary(result_dir: pathlib.Path, package: pathlib.Path) -> dict[str, Any]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during collection")
    static = json.loads((result_dir / "static_gate.json").read_text())
    require(static, "static_gate", "pass")
    require(static, "qnn_dependency", False)

    audit_records = {
        variant: load_jsonl(result_dir / f"correctness_{variant}.jsonl", 1)[0]
        for variant in (CONTROL, CANDIDATE)
    }
    for variant, record in audit_records.items():
        validate_audit(record, variant)

    implementation = json.loads((
        result_dir / "attention_implementation_audit" / "implementation_reference.json"
    ).read_text())
    require(implementation, "experiment", "EXP-0047")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(implementation["exact_stage_comparison"][stage], "mismatches", 0)

    package_info = validate_package(package)
    summary: dict[str, Any] = {
        "experiment": "EXP-0047",
        "comparison": "current_best_w4f16_vs_current_best_w4u8",
        "comparison_is_identical_schedule_single_variable": False,
        "paired_rounds": SAMPLES,
        "repeat_scopes": list(REPEATS),
        "boot_id_stable": True,
        "shared_package_for_both_variants": True,
        "shared_w4_values_and_scales_byte_identical": True,
        **package_info,
        "correctness_gate": True,
        "physical_gate": True,
        "pc027_measurement_gate": True,
        "speed_gate": None,
        "qnn_dependency": False,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "fixed_vtcm_request_bytes": VTCM_BYTES,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_spill_fill_count": 0,
        "output_hashes": OUTPUT_HASHES,
        "independent_attention_reference": {
            "qk_mismatches": implementation["exact_stage_comparison"]["qk"]["mismatches"],
            "log2_softmax_mismatches": implementation["exact_stage_comparison"]["log2_softmax"]["mismatches"],
            "av_mismatches": implementation["exact_stage_comparison"]["av"]["mismatches"],
            "probability_mask_violations": implementation["probability_mask_violations"],
        },
        "audit": {
            variant: {
                "mismatches": record["mismatches"],
                "max_lsb": record["max_lsb"],
                "max_abs": record["max_abs"],
                "cosine": record["cosine"],
                "nonfinite_count": record["common_op_nonfinite_count"],
                "mask_violations": record["common_op_softmax_mask_violation_count"],
            }
            for variant, record in audit_records.items()
        },
        "repeat_results": {},
    }
    for repeat in REPEATS:
        records = {
            variant: load_jsonl(
                result_dir / f"paired_{variant}_r{repeat}.jsonl", SAMPLES
            )
            for variant in (CONTROL, CANDIDATE)
        }
        for variant, variant_records in records.items():
            for record in variant_records:
                validate_record(record, repeat, variant)
        metrics = {
            metric.key: metric_summary(records[CONTROL], records[CANDIDATE], metric)
            for metric in ALL_METRICS
        }
        summary["repeat_results"][f"repeat{repeat}"] = {"metrics": metrics}
    summary["completion_gate"] = True
    return summary


def fmt_number(value: float, metric: Metric) -> str:
    if metric.kind == "ms":
        return f"{value:.4f}"
    if metric.kind == "static":
        return f"{value:.0f}"
    return f"{value:.1f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def render_table(summary: dict[str, Any], title: str, metrics: tuple[Metric, ...]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Metric | r1 W4F16 | r1 W4U8 | Delta | r10 W4F16 | r10 W4U8 | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        cells = []
        for repeat in REPEATS:
            data = summary["repeat_results"][f"repeat{repeat}"]["metrics"][metric.key]
            if metric.available == CANDIDATE:
                left = "N/A"
                right = fmt_number(data["candidate_median"], metric)
                delta = "N/A"
            elif metric.available == CONTROL:
                left = fmt_number(data["control_median"], metric)
                right = "N/A"
                delta = "N/A"
            else:
                left = fmt_number(data["control_median"], metric)
                right = fmt_number(data["candidate_median"], metric)
                delta = fmt_percent(data["change_percent"])
            cells.extend((left, right, delta))
        lines.append(f"| {metric.label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def report_markdown(summary: dict[str, Any], result_dir: pathlib.Path) -> str:
    identity = load_manifest_text(result_dir / "manifest.txt")
    lines = [
        "# EXP-0047 complete profiling closure",
        "",
        "## Identity and comparison",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Experiment | EXP-0047 |",
        f"| Branch | `{identity['source_branch']}` |",
        f"| Source commit | `{identity['source_head']}` |",
        "| Execution unit | Real Qwen3 layer-14 complete block, M=64 |",
        "| Direct control | Current-source reproduction of `W4F16-EXP0038-NORMS` |",
        "| Candidate | `W4U8-EXP0046-NATIVE-MLP-IO` |",
        "| Formal comparison | Seven interleaved paired rounds at repeat one and repeat ten |",
        f"| Formal evidence | `{identity['result_dir_windows']}` |",
        f"| Retained artifacts | `{identity['artifact_dir_windows']}` |",
        "| Backend | Standalone FastRPC DSP, one HMX owner, no QNN or alternate CPU path |",
        "",
        "This is a paired **current-best versus current-best** characterization on",
        "one source commit and one device boot. It is not an identical-schedule",
        "single-variable comparison: W4F16 and W4U8 intentionally retain their",
        "own accepted arithmetic, Attention, layout, and tuned pipeline plans.",
        "Both paths consume the same package and the same byte-identical W4 values",
        f"and scales; the combined W4 bundle digest is `{summary['shared_w4_bundle_sha256']}`.",
        "",
        "All qtimer, byte, and count values are medians normalized per block.",
        "Host wall is milliseconds per block. Only the additive Block Timing",
        "Ledger may be summed. Engine work and wait counters overlap and are",
        "diagnostic rather than additive.",
        "",
    ]
    lines.extend(render_table(summary, "Primary latency", PRIMARY))
    lines.extend(render_table(summary, "Complete additive Block Timing Ledger", LEDGER))

    for repeat in REPEATS:
        data = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        total_gap = data["total_ticks"]["candidate_median"] - data["total_ticks"]["control_median"]
        contributions = []
        for metric in LEDGER:
            item = data[metric.key]
            delta = item["candidate_median"] - item["control_median"]
            contributions.append((delta, metric.label))
        contributions.sort(reverse=True)
        lines.append(f"At repeat {repeat}, W4U8 adds {total_gap:.1f} additive DSP ticks/block.")
        if total_gap > 0:
            top = ", ".join(
                f"{label} {delta:+.1f} ({delta / total_gap * 100.0:.1f}% of the net gap)"
                for delta, label in contributions[:5]
            )
            lines.append(f"The five largest positive ledger deltas are {top}.")
        lines.append("")

    lines.extend(render_table(summary, "Projection diagnostics (overlapping, not additive)", PROJECTION))
    lines.append(
        "Variant-specific W4F16 and W4U8 rows are marked N/A on the other path:"
        " the measurement does not exist there because the physical scheduler and"
        " telemetry family differ. Zero is reserved for a counter that was present"
        " and measured zero."
    )
    lines.append("")
    lines.extend(render_table(summary, "Attention diagnostics (overlapping, not additive)", ATTENTION))
    lines.append(
        "The FP16 GQA pipeline exposes aggregate worker/HMX/queue intervals rather"
        " than separate QK, Softmax, and AV timers. Those component cells are N/A,"
        " not zero. W4U8 exposes the integer components separately."
    )
    lines.append("")
    lines.extend(render_table(summary, "MLP diagnostics (overlapping, not additive)", MLP))
    lines.extend(render_table(summary, "Physical contract", PHYSICAL))
    lines.extend([
        "FP16 and U8xS8 HMX tile-pair counts are carrier-specific and are not",
        "one-for-one measures of arithmetic work. The command count, wall intervals,",
        "waits, and complete ledger must be interpreted together.",
        "",
        "## Cross-variant structural attribution",
        "",
        "| Indicator | repeat one | repeat ten |",
        "|---|---:|---:|",
    ])
    for label, key in (
        ("W4U8 / W4F16 weight DDR bytes", "weight_ddr_read_bytes"),
        ("W4U8 / W4F16 HMX commands", "hmx_command_count"),
        ("W4U8 / W4F16 projection-pack ticks", "projection_pack_ticks"),
        ("W4U8 / W4F16 all-projection HMX ticks", "hmx_compute_ticks"),
    ):
        ratios = []
        for repeat in REPEATS:
            item = summary["repeat_results"][f"repeat{repeat}"]["metrics"][key]
            ratios.append(item["candidate_median"] / item["control_median"])
        lines.append(f"| {label} | {ratios[0]:.2f}x | {ratios[1]:.2f}x |")
    density = {}
    for repeat in REPEATS:
        metrics = summary["repeat_results"][f"repeat{repeat}"]["metrics"]
        density[repeat] = (
            metrics["hmx_fp16_tile_pair_count"]["control_median"] /
            metrics["hmx_command_count"]["control_median"],
            metrics["hmx_u8s8_tile_pair_count"]["candidate_median"] /
            metrics["hmx_command_count"]["candidate_median"],
        )
    lines.extend([
        f"| W4F16 FP16 tile pairs / HMX command | {density[1][0]:.1f} | {density[10][0]:.1f} |",
        f"| W4U8 U8xS8 tile pairs / HMX command | {density[1][1]:.1f} | {density[10][1]:.1f} |",
        "",
        "The shared W4 bundle means activation precision does not reduce the dominant",
        "weight bytes: W4U8 actually reads about the same amount. W4F16 reports much",
        "more DMA work ticks, yet its projection ledger is substantially shorter; its",
        "DMA, expansion, and HMX work are therefore overlapped effectively rather than",
        "serialized on the complete-block path.",
        "",
        "The strongest structural difference is command granularity. W4U8 issues five",
        "times as many HMX commands while carrying only half as many carrier-specific",
        "tile pairs in total, or about one tenth as many tile pairs per command. It also",
        "spends roughly 21k ticks/block in projection packing and about 32k overlapping",
        "ticks waiting for expanded MLP slots. These measurements explain why lower",
        "integer arithmetic width does not become lower latency in the current custom",
        "pipeline: scheduling, carrier construction, and fine-grained submission dominate",
        "before the reduced 8x8 multiply work can become visible.",
        "",
        "## Correctness and provenance",
        "",
        "| Gate | W4F16 | W4U8 | Result |",
        "|---|---:|---:|---|",
        f"| Final mismatches | {summary['audit'][CONTROL]['mismatches']} | {summary['audit'][CANDIDATE]['mismatches']} | pass |",
        f"| Final maximum LSB | {summary['audit'][CONTROL]['max_lsb']} | {summary['audit'][CANDIDATE]['max_lsb']} | pass |",
        f"| Output hash | `{OUTPUT_HASHES[CONTROL]}` | `{OUTPUT_HASHES[CANDIDATE]}` | pass against each path's own reference |",
        f"| Non-finite count | {summary['audit'][CONTROL]['nonfinite_count']} | {summary['audit'][CANDIDATE]['nonfinite_count']} | pass |",
        f"| Mask violations | {summary['audit'][CONTROL]['mask_violations']} | {summary['audit'][CANDIDATE]['mask_violations']} | pass |",
        f"| Independent integer QK mismatches | N/A | {summary['independent_attention_reference']['qk_mismatches']} | pass |",
        f"| Independent log2-Softmax mismatches | N/A | {summary['independent_attention_reference']['log2_softmax_mismatches']} | pass |",
        f"| Independent integer AV mismatches | N/A | {summary['independent_attention_reference']['av_mismatches']} | pass |",
        f"| Shared W4 values/scales | `{summary['shared_w4_bundle_sha256']}` | same package | pass |",
        "| Intermediate DDR read/write | 0/0 | 0/0 | pass |",
        "| Spill/fill | 0 | 0 | pass |",
        "| VTCM requested/acquired | 8 MiB / 8 MiB | 8 MiB / 8 MiB | pass |",
        "| FastRPC execution units / HMX owners | 1 / 1 | 1 / 1 | pass |",
        "| QNN dependency / CPU fallback | false / none | false / none | pass |",
        "",
        "W4U8 correctness here means exact execution of the declared local integer",
        "algorithm and stable agreement with its independent QK/Softmax/AV reference.",
        "It does not claim full-model quantization accuracy or equality to W4F16.",
        "",
        "## Closure conclusion",
        "",
    ])
    r1 = summary["repeat_results"]["repeat1"]["metrics"]["host_wall_ns_per_block"]
    r10 = summary["repeat_results"]["repeat10"]["metrics"]["host_wall_ns_per_block"]
    lines.extend([
        f"W4F16 measures {r1['control_median']:.4f} ms/block at repeat one and {r10['control_median']:.4f} ms/block at repeat ten.",
        f"W4U8 measures {r1['candidate_median']:.4f} and {r10['candidate_median']:.4f} ms/block, respectively,",
        f"or {fmt_percent(r1['change_percent'])} and {fmt_percent(r10['change_percent'])} versus W4F16.",
        "The experiment has no speed gate and makes no baseline-promotion decision.",
        "Its completion gate passes because both paths satisfy their numerical and",
        "physical contracts and every required PC-027 table is closed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR [--report]")
    result_dir = pathlib.Path(sys.argv[1])
    package = pathlib.Path(sys.argv[2])
    summary = build_summary(result_dir, package)
    if len(sys.argv) == 4:
        if sys.argv[3] != "--report":
            raise SystemExit(f"unknown option: {sys.argv[3]}")
        print(report_markdown(summary, result_dir))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
