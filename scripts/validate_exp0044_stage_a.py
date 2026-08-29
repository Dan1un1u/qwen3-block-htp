#!/usr/bin/env python3
"""Validate paired EXP-0044 Stage-A complete-block evidence."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import sys

SAMPLES = 7
REPEATS = (1, 10)
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608


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


def audit_package(package: pathlib.Path) -> dict[str, object]:
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
    return {
        "source_package_experiment": manifest["experiment"],
        "manifest_sha256": sha256(manifest_path),
    }


def load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != SAMPLES:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def validate_record(record: dict[str, object], repeat: int, candidate: bool) -> None:
    fixed = {
        "experiment": "EXP-0044",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "common_ops_mode": "rms_rope_softmax",
        "residual_mode": "hvx_fused_post_norm",
        "attention_pack_mode": "combined_hvx",
        "attention_pipeline_mode": (
            "u8_log2_gqa_fused_k" if candidate else "u8_log2_gqa"
        ),
        "attention_hvx_contexts": 4,
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
        "u8_attention_fused_k_operand_mismatch_count": 0,
        "u8_attention_group_count": 8 * repeat,
        "u8_attention_qk_execution_count": 32 * repeat,
        "u8_attention_av_execution_count": 64 * repeat,
        "u8_attention_direct_o_tile_count": 64 * repeat,
        "u8_attention_qkv_unpack_skipped": 128 * repeat,
        "hmx_command_count": 1_184 * repeat,
        "hmx_fp16_tile_pair_count": 0,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if record["vtcm_peak_plan_bytes"] > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds exact 8 MiB contract")
    for field in (
        "host_wall_ns_per_block",
        "total_ticks",
        "attention_ticks",
        "u8_attention_qk_norm_rope_ticks",
        "u8_attention_v_pack_ticks",
    ):
        require_positive(record, field)
    if candidate:
        require(record, "u8_attention_k_pack_ticks", 0)
    else:
        require_positive(record, "u8_attention_k_pack_ticks")


def per_block(record: dict[str, object], field: str) -> float:
    return float(record[field]) / int(record["repeat_count"])


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def summarize(
    control: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    fields = (
        "host_wall_ns_per_block",
        "total_ticks",
        "qkv_projection_ticks",
        "attention_ticks",
        "u8_attention_qk_norm_rope_ticks",
        "u8_attention_k_pack_ticks",
        "u8_attention_v_pack_ticks",
        "u8_attention_qk_hmx_ticks",
        "u8_attention_softmax_ticks",
        "u8_attention_av_hmx_ticks",
        "u8_attention_pipeline_wait_ticks",
    )
    result: dict[str, object] = {"samples": SAMPLES}
    for field in fields:
        if field == "host_wall_ns_per_block":
            left = [float(record[field]) for record in control]
            right = [float(record[field]) for record in candidate]
        else:
            left = [per_block(record, field) for record in control]
            right = [per_block(record, field) for record in candidate]
        result[f"control_{field}_median"] = statistics.median(left)
        result[f"candidate_{field}_median"] = statistics.median(right)
        result[f"{field}_change_percent"] = percent(
            statistics.median(right), statistics.median(left)
        )
        result[f"paired_{field}_change_percent_median"] = statistics.median(
            percent(right[index], left[index]) for index in range(SAMPLES)
        )
    control_prep = [
        per_block(record, "u8_attention_qk_norm_rope_ticks")
        + per_block(record, "u8_attention_k_pack_ticks")
        for record in control
    ]
    candidate_prep = [
        per_block(record, "u8_attention_qk_norm_rope_ticks")
        + per_block(record, "u8_attention_k_pack_ticks")
        for record in candidate
    ]
    result["control_qk_norm_rope_plus_k_pack_median"] = statistics.median(
        control_prep
    )
    result["candidate_qk_norm_rope_plus_k_pack_median"] = statistics.median(
        candidate_prep
    )
    result["qk_norm_rope_plus_k_pack_change_percent"] = percent(
        statistics.median(candidate_prep), statistics.median(control_prep)
    )
    result["paired_qk_norm_rope_plus_k_pack_change_percent_median"] = (
        statistics.median(
            percent(candidate_prep[index], control_prep[index])
            for index in range(SAMPLES)
        )
    )
    result["stage_a_speed_gate_pass"] = all(
        result[key] < 0.0
        for key in (
            "host_wall_ns_per_block_change_percent",
            "paired_host_wall_ns_per_block_change_percent_median",
            "attention_ticks_change_percent",
            "paired_attention_ticks_change_percent_median",
            "qk_norm_rope_plus_k_pack_change_percent",
            "paired_qk_norm_rope_plus_k_pack_change_percent_median",
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

    implementation = load_json(
        result_dir / "attention_implementation_audit" /
        "implementation_reference.json"
    )
    require(implementation, "experiment", "EXP-0044")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(implementation["exact_stage_comparison"][stage], "mismatches", 0)

    device_audit = load_json(
        result_dir / "attention_implementation_audit" / "device_audit.json"
    )
    require(device_audit, "u8_attention_fused_k_operand_mismatch_count", 0)
    require(device_audit, "mismatches", 0)
    require(device_audit, "max_lsb", 0)

    summary: dict[str, object] = {
        "experiment": "EXP-0044",
        "stage": "A",
        "package": audit_package(package),
        "implementation_reference_gate": True,
        "fused_k_carrier_bias_byte_exact_gate": True,
        "byte_exact_final_output_gate": True,
        "zero_intermediate_ddr_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "repeat_results": {},
    }
    for repeat in REPEATS:
        control = load_jsonl(result_dir / f"paired_control_r{repeat}.jsonl")
        candidate = load_jsonl(result_dir / f"paired_candidate_r{repeat}.jsonl")
        for record in control:
            validate_record(record, repeat, False)
        for record in candidate:
            validate_record(record, repeat, True)
        summary["repeat_results"][f"repeat{repeat}"] = summarize(
            control, candidate
        )
    summary["stage_a_gate_pass"] = all(
        summary["repeat_results"][f"repeat{repeat}"]["stage_a_speed_gate_pass"]
        for repeat in REPEATS
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
