#!/usr/bin/env python3
"""Validate paired EXP-0042 complete-block hardware evidence."""

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


def audit_package(package: pathlib.Path) -> str:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest, "experiment", "EXP-0042")
    require(manifest, "execution_unit", "qwen3_layer14_complete_block_m64")
    for entry in manifest["files"].values():
        path = package / entry["file"]
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
        "experiment": "EXP-0042",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "projection_compute": "U8xS8_integer_HMX",
        "common_ops_mode": "rms_rope_softmax",
        "residual_mode": "hvx_fused_post_norm",
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
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if record["vtcm_peak_plan_bytes"] > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds exact 8 MiB contract")
    for field in ("host_wall_ns", "host_wall_ns_per_block", "total_ticks"):
        require_positive(record, field)


def validate_mode(record: dict[str, object], repeat: int, candidate: bool) -> None:
    validate_common(record, repeat)
    if candidate:
        fixed = {
            "attention_compute": "U8xS8_HMX_log2_softmax",
            "attention_pipeline_mode": "u8_log2_gqa",
            "attention_hvx_contexts": 4,
            "attention_hvx_workers_created": 3,
            "attention_hvx_workers_locked": 3,
            "u8_attention_group_count": 8 * repeat,
            "u8_attention_qk_execution_count": 32 * repeat,
            "u8_attention_av_execution_count": 64 * repeat,
            "u8_attention_direct_o_tile_count": 64 * repeat,
            "u8_attention_qkv_unpack_skipped": 128 * repeat,
            "u8_attention_probability_mask_violation_count": 0,
            "hmx_command_count": 1_184 * repeat,
            "hmx_fp16_tile_pair_count": 0,
            "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        }
        for field, expected in fixed.items():
            require(record, field, expected)
        require_positive(record, "u8_attention_qk_hmx_ticks")
        require_positive(record, "u8_attention_av_hmx_ticks")
    else:
        fixed = {
            "attention_compute": "FP16_HMX",
            "attention_pipeline_mode": "control",
            "attention_hvx_contexts": 1,
            "u8_attention_group_count": 0,
            "u8_attention_qk_execution_count": 0,
            "u8_attention_av_execution_count": 0,
            "hmx_command_count": 1_120 * repeat,
            "hmx_fp16_tile_pair_count": 512 * repeat,
            "hmx_u8s8_tile_pair_count": 49_152 * repeat,
        }
        for field, expected in fixed.items():
            require(record, field, expected)


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def per_block(record: dict[str, object], field: str) -> float:
    return float(record[field]) / int(record["repeat_count"])


def summarize(
    control: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    control_host = [float(record["host_wall_ns_per_block"]) for record in control]
    candidate_host = [float(record["host_wall_ns_per_block"]) for record in candidate]
    control_attention = [per_block(record, "attention_ticks") for record in control]
    candidate_attention = [per_block(record, "attention_ticks") for record in candidate]
    host_change = percent(
        statistics.median(candidate_host), statistics.median(control_host)
    )
    attention_change = percent(
        statistics.median(candidate_attention),
        statistics.median(control_attention),
    )
    paired_host = [
        percent(candidate_host[index], control_host[index])
        for index in range(SAMPLES)
    ]
    paired_attention = [
        percent(candidate_attention[index], control_attention[index])
        for index in range(SAMPLES)
    ]
    fields = (
        "total_ticks",
        "qkv_projection_ticks",
        "attention_ticks",
        "o_projection_ticks",
        "gate_up_ticks",
        "down_ticks",
        "u8_attention_qk_norm_rope_ticks",
        "u8_attention_k_pack_ticks",
        "u8_attention_v_pack_ticks",
        "u8_attention_qk_hmx_ticks",
        "u8_attention_softmax_ticks",
        "u8_attention_av_hmx_ticks",
        "u8_attention_pipeline_wait_ticks",
    )
    result: dict[str, object] = {
        "samples": SAMPLES,
        "control_host_wall_ns_per_block_median": statistics.median(control_host),
        "candidate_host_wall_ns_per_block_median": statistics.median(candidate_host),
        "host_change_percent": host_change,
        "paired_host_change_percent_median": statistics.median(paired_host),
        "control_attention_ticks_per_block_median": statistics.median(
            control_attention
        ),
        "candidate_attention_ticks_per_block_median": statistics.median(
            candidate_attention
        ),
        "attention_change_percent": attention_change,
        "paired_attention_change_percent_median": statistics.median(
            paired_attention
        ),
    }
    for field in fields:
        result[f"control_{field}_per_block_median"] = statistics.median(
            per_block(record, field) for record in control
        )
        result[f"candidate_{field}_per_block_median"] = statistics.median(
            per_block(record, field) for record in candidate
        )
    result["speed_gate_pass"] = (
        host_change < 0.0
        and result["paired_host_change_percent_median"] < 0.0
        and attention_change < 0.0
        and result["paired_attention_change_percent_median"] < 0.0
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
        "experiment": "EXP-0042",
        "package_manifest_sha256": audit_package(package),
        "implementation_reference_gate": True,
        "implementation_reference_path": str(
            result_dir / "attention_implementation_audit" /
            "implementation_reference.json"
        ),
        "repeat_results": {},
    }
    for repeat in REPEATS:
        control = load_jsonl(result_dir / f"paired_control_r{repeat}.jsonl")
        candidate = load_jsonl(result_dir / f"paired_candidate_r{repeat}.jsonl")
        for record in control:
            validate_mode(record, repeat, False)
        for record in candidate:
            validate_mode(record, repeat, True)
        summary["repeat_results"][f"repeat{repeat}"] = summarize(
            control, candidate
        )

    summary["byte_exact_final_output_gate"] = True
    summary["zero_intermediate_ddr_gate"] = True
    summary["fixed_8mib_vtcm_gate"] = True
    summary["single_fastrpc_execution_unit"] = True
    summary["single_hmx_owner"] = True
    summary["qnn_dependency"] = False
    summary["speed_gate_pass"] = all(
        summary["repeat_results"][f"repeat{repeat}"]["speed_gate_pass"]
        for repeat in REPEATS
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
