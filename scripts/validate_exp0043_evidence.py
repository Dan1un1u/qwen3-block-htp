#!/usr/bin/env python3
"""Validate EXP-0043 W4U8 complete-block timing attribution."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import sys

SAMPLES = 11
REPEATS = (1, 10)
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
MAX_UNATTRIBUTED_FRACTION = 0.001
MAX_REPEAT10_OVERHEAD_PERCENT = 1.0

TOP_LEVEL_FIELDS = (
    "runtime_setup_ticks",
    "metadata_stage_ticks",
    "input_stage_ticks",
    "input_norm_ticks",
    "qkv_projection_ticks",
    "qk_norm_rope_ticks",
    "attention_ticks",
    "o_projection_ticks",
    "post_attention_residual_ticks",
    "post_attention_norm_ticks",
    "gate_up_ticks",
    "activation_ticks",
    "down_ticks",
    "w4u8_mlp_input_pack_ticks",
    "w4u8_mlp_output_unpack_ticks",
    "w4u8_mlp_control_ticks",
    "final_residual_ticks",
    "output_stage_ticks",
    "runtime_teardown_ticks",
)

PHYSICAL_COUNTER_FIELDS = (
    "output_hash",
    "vtcm_requested_bytes",
    "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes",
    "block_invocation_count",
    "weight_ddr_read_bytes",
    "weight_dma_descriptor_count",
    "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
    "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count",
    "hmx_command_count",
    "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count",
    "u8_attention_group_count",
    "u8_attention_qk_execution_count",
    "u8_attention_av_execution_count",
    "u8_attention_direct_o_tile_count",
    "u8_attention_qkv_unpack_skipped",
)


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def require_positive(record: dict[str, object], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid positive metric {field}: {value!r}")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_package(package: pathlib.Path) -> str:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest, "experiment", "EXP-0042")
    require(manifest, "execution_unit", "qwen3_layer14_complete_block_m64")
    for name, entry in manifest["files"].items():
        path = package / entry.get("file", name)
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise SystemExit(f"invalid package tensor: {path.name}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"package digest mismatch: {path.name}")
    return sha256(manifest_path)


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != SAMPLES:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def validate_common(record: dict[str, object], repeat: int) -> None:
    fixed = {
        "experiment": "EXP-0043",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "common_ops_mode": "rms_rope_softmax",
        "residual_mode": "hvx_fused_post_norm",
        "attention_pack_mode": "combined_hvx",
        "attention_pipeline_mode": "u8_log2_gqa",
        "attention_hvx_contexts": 4,
        "mlp_mode": "w4u8_streaming",
        "repeat_count": repeat,
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
        "warmup_max_lsb": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "intermediate_residency": "VTCM",
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "u8_attention_audit_ddr_write_bytes": 0,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "u8_attention_group_count": 8 * repeat,
        "u8_attention_qk_execution_count": 32 * repeat,
        "u8_attention_av_execution_count": 64 * repeat,
        "u8_attention_direct_o_tile_count": 64 * repeat,
        "u8_attention_qkv_unpack_skipped": 128 * repeat,
        "u8_attention_probability_mask_violation_count": 0,
        "hmx_command_count": 1_184 * repeat,
        "hmx_fp16_tile_pair_count": 0,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if record["vtcm_peak_plan_bytes"] > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds exact 8 MiB contract")
    for field in ("host_wall_ns", "host_wall_ns_per_block", "total_ticks"):
        require_positive(record, field)


def validate_control(record: dict[str, object], repeat: int) -> None:
    validate_common(record, repeat)
    require(record, "attribution_mode", "off")
    for field in (
        "invocation_ticks",
        "runtime_setup_ticks",
        "runtime_teardown_ticks",
        "ledger_named_ticks",
        "ledger_unattributed_ticks",
        "w4u8_mlp_boundary_ticks",
        "w4u8_mlp_input_pack_ticks",
        "w4u8_mlp_output_unpack_ticks",
        "w4u8_mlp_control_ticks",
    ):
        require(record, field, 0)


def validate_instrumented(record: dict[str, object], repeat: int) -> None:
    validate_common(record, repeat)
    require(record, "attribution_mode", "on")
    for field in (
        "invocation_ticks",
        "ledger_named_ticks",
        "w4u8_mlp_boundary_ticks",
        "w4u8_mlp_input_pack_ticks",
        "w4u8_mlp_output_unpack_ticks",
        "w4u8_mlp_control_ticks",
    ):
        require_positive(record, field)
    if record["ledger_named_ticks"] != sum(record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("named top-level ledger does not equal its exclusive fields")
    if record["invocation_ticks"] != (
        record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]
    ):
        raise SystemExit("top-level ledger does not close")
    if record["w4u8_mlp_boundary_ticks"] != (
        record["w4u8_mlp_input_pack_ticks"]
        + record["w4u8_mlp_output_unpack_ticks"]
        + record["w4u8_mlp_control_ticks"]
    ):
        raise SystemExit("W4U8 MLP boundary ledger does not close")
    if record["ledger_unattributed_ticks"] / record["invocation_ticks"] > MAX_UNATTRIBUTED_FRACTION:
        raise SystemExit("top-level unattributed gap exceeds 0.1 percent")


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def per_block(record: dict[str, object], field: str) -> float:
    return float(record[field]) / int(record["repeat_count"])


def summarize(
    control: list[dict[str, object]],
    instrumented: list[dict[str, object]],
    repeat: int,
) -> dict[str, object]:
    control_host = [float(record["host_wall_ns_per_block"]) for record in control]
    instrumented_host = [
        float(record["host_wall_ns_per_block"]) for record in instrumented
    ]
    paired_overhead = [
        percent(instrumented_host[index], control_host[index])
        for index in range(SAMPLES)
    ]
    result: dict[str, object] = {
        "samples": SAMPLES,
        "control_host_wall_ns_per_block_median": statistics.median(control_host),
        "instrumented_host_wall_ns_per_block_median": statistics.median(instrumented_host),
        "instrumentation_overhead_percent": percent(
            statistics.median(instrumented_host), statistics.median(control_host)
        ),
        "paired_instrumentation_overhead_percent_median": statistics.median(
            paired_overhead
        ),
    }
    fields = (
        "total_ticks",
        "invocation_ticks",
        "ledger_named_ticks",
        "ledger_unattributed_ticks",
        "w4u8_mlp_boundary_ticks",
        "w4u8_mlp_input_pack_ticks",
        "w4u8_mlp_output_unpack_ticks",
        "w4u8_mlp_control_ticks",
    )
    for field in fields:
        result[f"instrumented_{field}_per_block_median"] = statistics.median(
            per_block(record, field) for record in instrumented
        )
    boundary = float(result["instrumented_w4u8_mlp_boundary_ticks_per_block_median"])
    packing = (
        float(result["instrumented_w4u8_mlp_input_pack_ticks_per_block_median"])
        + float(result["instrumented_w4u8_mlp_output_unpack_ticks_per_block_median"])
    )
    unattributed = float(
        result["instrumented_ledger_unattributed_ticks_per_block_median"]
    )
    result["w4u8_mlp_pack_unpack_percent_of_boundary"] = packing / boundary * 100.0
    result["former_gap_explained_percent"] = boundary / (boundary + unattributed) * 100.0
    result["unattributed_fraction_percent"] = (
        unattributed
        / float(result["instrumented_invocation_ticks_per_block_median"])
        * 100.0
    )
    result["ledger_closure_gate"] = (
        result["unattributed_fraction_percent"]
        <= MAX_UNATTRIBUTED_FRACTION * 100.0
    )
    result["instrumentation_overhead_gate"] = (
        repeat != 10
        or (
            result["instrumentation_overhead_percent"]
            <= MAX_REPEAT10_OVERHEAD_PERCENT
            and result["paired_instrumentation_overhead_percent_median"]
            <= MAX_REPEAT10_OVERHEAD_PERCENT
        )
    )
    return result


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package = pathlib.Path(sys.argv[2])
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed during collection")

    implementation = json.loads(
        (result_dir / "attention_implementation_audit" /
         "implementation_reference.json").read_text(encoding="utf-8")
    )
    require(implementation, "experiment", "EXP-0042")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(
            implementation["exact_stage_comparison"][stage],
            "mismatches",
            0,
        )

    summary: dict[str, object] = {
        "experiment": "EXP-0043",
        "package_reused_from": "EXP-0042",
        "package_manifest_sha256": audit_package(package),
        "implementation_reference_gate": True,
        "optimization_changes_allowed": False,
        "repeat_results": {},
    }
    for repeat in REPEATS:
        control = load_jsonl(result_dir / f"paired_control_r{repeat}.jsonl")
        instrumented = load_jsonl(
            result_dir / f"paired_instrumented_r{repeat}.jsonl"
        )
        for record in control:
            validate_control(record, repeat)
        for record in instrumented:
            validate_instrumented(record, repeat)
        for index in range(SAMPLES):
            for field in PHYSICAL_COUNTER_FIELDS:
                if control[index][field] != instrumented[index][field]:
                    raise SystemExit(
                        f"physical counter changed for paired sample {index}: {field}"
                    )
        summary["repeat_results"][f"repeat{repeat}"] = summarize(
            control, instrumented, repeat
        )

    summary["byte_exact_final_output_gate"] = True
    summary["physical_counter_equivalence_gate"] = True
    summary["zero_intermediate_ddr_gate"] = True
    summary["fixed_8mib_vtcm_gate"] = True
    summary["single_fastrpc_execution_unit"] = True
    summary["single_hmx_owner"] = True
    summary["qnn_dependency"] = False
    summary["ledger_closure_gate"] = all(
        summary["repeat_results"][f"repeat{repeat}"]["ledger_closure_gate"]
        for repeat in REPEATS
    )
    summary["instrumentation_overhead_gate"] = summary["repeat_results"][
        "repeat10"
    ]["instrumentation_overhead_gate"]
    summary["completion_gate_pass"] = (
        summary["ledger_closure_gate"]
        and summary["instrumentation_overhead_gate"]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
