#!/usr/bin/env python3
"""Validate paired EXP-0041 complete-block hardware evidence."""

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
    manifest = json.loads(manifest_path.read_text())
    require(manifest, "experiment", "EXP-0040")
    require(manifest, "stage", "B")
    require(manifest, "execution_unit", "qwen3_layer14_complete_block_m64")
    for name, entry in manifest["files"].items():
        path = package / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise SystemExit(f"invalid package tensor: {name}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"package digest mismatch: {name}")
    return sha256(manifest_path)


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != SAMPLES:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def validate_common(record: dict[str, object], repeat: int) -> None:
    fixed = {
        "experiment": "EXP-0041",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "FP16_HMX",
        "projection_compute": "U8xS8_integer_HMX",
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
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "hmx_command_count": 1_120 * repeat,
        "hmx_fp16_tile_pair_count": 512 * repeat,
        "hmx_u8s8_tile_pair_count": 49_152 * repeat,
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
        require(record, "common_ops_mode", "rms_rope_softmax")
        require(record, "residual_mode", "hvx_fused_post_norm")
        post_norm_ticks = record.get("post_attention_norm_ticks")
        if not isinstance(post_norm_ticks, int) or not (
            0 <= post_norm_ticks <= repeat
        ):
            raise SystemExit(
                "fused post-norm attribution exceeds one timer tick per block: "
                f"{post_norm_ticks!r}"
            )
    else:
        require(record, "common_ops_mode", "scalar")
        require(record, "residual_mode", "scalar")
        require_positive(record, "post_attention_norm_ticks")


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def median_per_block(records: list[dict[str, object]], field: str) -> float:
    return statistics.median(record[field] / record["repeat_count"] for record in records)


def summarize(control: list[dict[str, object]], candidate: list[dict[str, object]]) -> dict[str, object]:
    control_host = [record["host_wall_ns_per_block"] for record in control]
    candidate_host = [record["host_wall_ns_per_block"] for record in candidate]
    control_median = statistics.median(control_host)
    candidate_median = statistics.median(candidate_host)
    paired_changes = [
        percent(candidate_host[index], control_host[index])
        for index in range(SAMPLES)
    ]
    fields = (
        "total_ticks",
        "input_norm_ticks",
        "qk_norm_rope_ticks",
        "attention_ticks",
        "attention_softmax_ticks",
        "post_attention_residual_ticks",
        "post_attention_norm_ticks",
        "final_residual_ticks",
    )
    result: dict[str, object] = {
        "samples": SAMPLES,
        "control_host_wall_ns_per_block_median": control_median,
        "candidate_host_wall_ns_per_block_median": candidate_median,
        "host_change_percent": percent(candidate_median, control_median),
        "paired_host_change_percent_median": statistics.median(paired_changes),
    }
    for field in fields:
        result[f"control_{field}_per_block_median"] = median_per_block(control, field)
        result[f"candidate_{field}_per_block_median"] = median_per_block(candidate, field)
    result["speed_gate_pass"] = (
        result["host_change_percent"] < 0.0
        and result["paired_host_change_percent_median"] < 0.0
        and result["candidate_total_ticks_per_block_median"]
        < result["control_total_ticks_per_block_median"]
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

    summary: dict[str, object] = {
        "experiment": "EXP-0041",
        "package_reused_unchanged_from": "EXP-0040-stage-B",
        "package_manifest_sha256": audit_package(package),
        "repeat_results": {},
    }
    for repeat in REPEATS:
        control = load_jsonl(result_dir / f"paired_control_r{repeat}.jsonl")
        candidate = load_jsonl(result_dir / f"paired_candidate_r{repeat}.jsonl")
        for record in control:
            validate_mode(record, repeat, False)
        for record in candidate:
            validate_mode(record, repeat, True)
        summary["repeat_results"][f"repeat{repeat}"] = summarize(control, candidate)

    summary["byte_exact_gate"] = True
    summary["zero_intermediate_ddr_gate"] = True
    summary["fixed_8mib_vtcm_gate"] = True
    summary["unchanged_hmx_work_gate"] = True
    summary["speed_gate_pass"] = all(
        summary["repeat_results"][f"repeat{repeat}"]["speed_gate_pass"]
        for repeat in REPEATS
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
