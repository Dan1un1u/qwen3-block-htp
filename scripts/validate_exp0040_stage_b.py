#!/usr/bin/env python3
"""Validate formal EXP-0040 Stage-B complete-block evidence."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import sys


SAMPLES = 7
REPEATS = (1, 10)
W4U8_OUTPUT_HASH = "69f22eeb035e5ec5"
W4F16_OUTPUT_HASH = "f18b9abbe1487231"
VTCM_BYTES = 8_388_608


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def require_positive(record: dict[str, object], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid positive metric {field}: {value!r}")


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    if len(records) != SAMPLES:
        raise SystemExit(f"wrong record count for {path}: {len(records)} != {SAMPLES}")
    return records


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_package(package: pathlib.Path) -> dict[str, object]:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest, "experiment", "EXP-0040")
    require(manifest, "stage", "B")
    require(manifest, "execution_unit", "qwen3_layer14_complete_block_m64")
    contract = manifest["candidate_contract"]
    require(contract, "historical_complete_output_required_byte_exact", True)
    require(contract, "historical_hmx_conversion", True)
    require(contract, "intermediate_ddr_allowed", False)
    require(contract, "vtcm_request_bytes", VTCM_BYTES)
    for name, entry in manifest["files"].items():
        path = package / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise SystemExit(f"invalid package tensor: {name}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"package digest mismatch: {name}")
    return manifest


def validate_common(record: dict[str, object], repeat: int) -> None:
    fixed = {
        "experiment": "EXP-0040",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "repeat_count": repeat,
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
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
        "mismatches": 0,
        "max_lsb": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if record["vtcm_peak_plan_bytes"] > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds the exact 8 MiB contract")
    for field in ("host_wall_ns", "host_wall_ns_per_block", "total_ticks"):
        require_positive(record, field)


def validate_w4u8(
    record: dict[str, object], repeat: int, candidate: bool,
) -> None:
    validate_common(record, repeat)
    fixed = {
        "variant": "W4U8",
        "attention_compute": "FP16_HMX",
        "projection_compute": "U8xS8_integer_HMX",
        "common_ops_mode": "scalar",
        "residual_mode": "scalar",
        "attention_pack_mode": "control",
        "attention_pipeline_mode": "control",
        "attention_hvx_contexts": 1,
        "crouton_boundary_mode": "control",
        "output_hash": W4U8_OUTPUT_HASH,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    require(record, "mlp_mode", "w4u8_streaming" if candidate else "control")
    if candidate:
        candidate_fixed = {
            "mlp_hvx_contexts": 3,
            "activation_ticks": 0,
            "w4u8_mlp_lut_vtcm_bytes": 131_072,
            "w4u8_mlp_gather_scratch_vtcm_bytes": 768,
            "w4u8_mlp_gate_up_hvx_workers": 3,
            "w4u8_mlp_down_hvx_workers": 6,
            "w4u8_mlp_pair_publish_count": 192 * repeat,
            "w4u8_mlp_pair_consume_count": 192 * repeat,
            "w4u8_mlp_gate_up_hvx_hmx_overlap": 1,
            "w4u8_mlp_down_hvx_hmx_overlap": 1,
            "w4u8_mlp_gate_up_hvx_parallel_overlap": 1,
            "w4u8_mlp_down_hvx_parallel_overlap": 1,
        }
        for field, expected in candidate_fixed.items():
            require(record, field, expected)
        base = record["w4u8_mlp_vtcm_base_offset"]
        plan = record["w4u8_mlp_vtcm_plan_bytes"]
        if base + plan > VTCM_BYTES:
            raise SystemExit("Stage-B MLP arena exceeds VTCM")
        for field in (
            "w4u8_mlp_gate_up_pipeline_ticks",
            "w4u8_mlp_down_pipeline_ticks",
            "w4u8_mlp_activation_work_ticks",
            "w4u8_mlp_weight_stage_ticks",
            "w4u8_mlp_weight_expand_ticks",
            "w4u8_mlp_hmx_compute_ticks",
        ):
            require_positive(record, field)
    else:
        require(record, "mlp_hvx_contexts", 1)
        require_positive(record, "activation_ticks")
        require(record, "w4u8_mlp_pair_publish_count", 0)
        require(record, "w4u8_mlp_pair_consume_count", 0)


def validate_w4f16(record: dict[str, object], repeat: int) -> None:
    validate_common(record, repeat)
    fixed = {
        "variant": "W4F16",
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "residual_mode": "hvx_fused_post_norm",
        "attention_pack_mode": "combined_hvx",
        "attention_pipeline_mode": "gqa_qkv_overlap",
        "attention_hvx_contexts": 4,
        "mlp_mode": "crouton_native_batch8",
        "mlp_hvx_contexts": 4,
        "crouton_boundary_mode": "norms",
        "output_hash": W4F16_OUTPUT_HASH,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    cosine = record.get("cosine")
    if not isinstance(cosine, (int, float)) or cosine < 0.999:
        raise SystemExit(f"invalid W4F16 cosine: {cosine!r}")


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def median_per_block(records: list[dict[str, object]], field: str) -> float:
    return statistics.median(record[field] / record["repeat_count"] for record in records)


def summarize_repeat(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
    w4f16: list[dict[str, object]],
) -> dict[str, object]:
    control_host = [record["host_wall_ns_per_block"] for record in control]
    candidate_host = [record["host_wall_ns_per_block"] for record in candidate]
    w4f16_host = [record["host_wall_ns_per_block"] for record in w4f16]
    paired_deltas = [
        percent(candidate_host[index], control_host[index])
        for index in range(SAMPLES)
    ]
    control_median = statistics.median(control_host)
    candidate_median = statistics.median(candidate_host)
    candidate_change = percent(candidate_median, control_median)
    paired_change = statistics.median(paired_deltas)
    return {
        "samples": SAMPLES,
        "control_host_wall_ns_per_block_median": control_median,
        "candidate_host_wall_ns_per_block_median": candidate_median,
        "candidate_host_change_percent": candidate_change,
        "paired_host_change_percent_median": paired_change,
        "pass_strict_complete_block_speed": (
            candidate_change < 0.0 and paired_change < 0.0
        ),
        "w4f16_host_wall_ns_per_block_median": statistics.median(w4f16_host),
        "candidate_vs_w4f16_host_change_percent": percent(
            candidate_median, statistics.median(w4f16_host)
        ),
        "control_total_ticks_per_block_median": median_per_block(
            control, "total_ticks"
        ),
        "candidate_total_ticks_per_block_median": median_per_block(
            candidate, "total_ticks"
        ),
        "control_gate_up_ticks_per_block_median": median_per_block(
            control, "gate_up_ticks"
        ),
        "candidate_gate_up_ticks_per_block_median": median_per_block(
            candidate, "gate_up_ticks"
        ),
        "control_activation_ticks_per_block_median": median_per_block(
            control, "activation_ticks"
        ),
        "candidate_activation_work_ticks_per_block_median": median_per_block(
            candidate, "w4u8_mlp_activation_work_ticks"
        ),
        "control_down_ticks_per_block_median": median_per_block(
            control, "down_ticks"
        ),
        "candidate_down_ticks_per_block_median": median_per_block(
            candidate, "down_ticks"
        ),
        "candidate_hmx_ready_wait_ticks_per_block_median": median_per_block(
            candidate, "w4u8_mlp_hmx_ready_wait_ticks"
        ),
        "candidate_expanded_slot_wait_ticks_per_block_median": median_per_block(
            candidate, "w4u8_mlp_expanded_slot_wait_ticks"
        ),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package = pathlib.Path(sys.argv[2])
    manifest = audit_package(package)
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed during formal collection")

    summary: dict[str, object] = {
        "experiment": "EXP-0040",
        "stage": "B",
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "historical_block_manifest_sha256": manifest[
            "block_source_manifest_sha256"
        ],
        "stage_a_manifest_sha256": manifest["mlp_source_manifest_sha256"],
        "repeat_results": {},
    }
    for repeat in REPEATS:
        control = load_jsonl(
            result_dir / f"paired_w4u8_control_r{repeat}.jsonl"
        )
        candidate = load_jsonl(
            result_dir / f"paired_w4u8_candidate_r{repeat}.jsonl"
        )
        w4f16 = load_jsonl(result_dir / f"w4f16_reference_r{repeat}.jsonl")
        for record in control:
            validate_w4u8(record, repeat, False)
        for record in candidate:
            validate_w4u8(record, repeat, True)
        for record in w4f16:
            validate_w4f16(record, repeat)
        summary["repeat_results"][f"repeat{repeat}"] = summarize_repeat(
            control, candidate, w4f16
        )

    summary["byte_exact_gate"] = True
    summary["zero_intermediate_ddr_gate"] = True
    summary["fixed_8mib_vtcm_gate"] = True
    summary["stage_b_gate_pass"] = all(
        summary["repeat_results"][f"repeat{repeat}"][
            "pass_strict_complete_block_speed"
        ]
        for repeat in REPEATS
    )
    summary["local_adoption_eligible"] = summary["stage_b_gate_pass"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
