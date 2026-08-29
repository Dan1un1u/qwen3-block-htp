#!/usr/bin/env python3
"""Validate the formal EXP-0040 Stage-A evidence matrix."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import sys


STORAGES = {
    "packed": "packed_w4_candidate",
    "expanded": "expanded_s8_control",
}
WORKERS = (2, 3)
REPEATS = (1, 10)
SAMPLES = 11


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
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
    require(manifest["shape"], "m", 64)
    require(manifest["shape"], "hidden", 2048)
    require(manifest["shape"], "intermediate", 6144)
    require(manifest["weight_contract"], "packed_candidate",
            "no persistent pre-expanded carrier")
    require(manifest["weight_contract"],
            "bias_and_requant_tables_identical", True)
    if "rounded right shift" not in manifest["activation_contract"][
            "implementation_reference"]:
        raise SystemExit("package does not use the selected rounded reference")
    for name, tensor in manifest["tensors"].items():
        path = package / tensor["file"]
        if not path.is_file() or path.stat().st_size != tensor["bytes"]:
            raise SystemExit(f"invalid package tensor: {name}")
        if sha256(path) != tensor["sha256"]:
            raise SystemExit(f"package digest mismatch: {name}")
    return manifest


def validate_record(
    record: dict[str, object], storage: str, repeat: int, workers: int,
    correctness: bool,
) -> None:
    fixed = {
        "experiment": "EXP-0040",
        "stage": "A_latest_layout_real_mlp",
        "weight_storage": STORAGES[storage],
        "hmx_contract": "U8xS8_integer",
        "activation": "asymmetric_U8",
        "silu_contract": "exact_U8_pair_LUT_HVX_vgather",
        "intermediate_residency": "VTCM",
        "repeat_count": repeat,
        "gate_up_worker_count": workers,
        "warmup_result": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "warmup_mismatches": 0,
        "mismatches": 0,
        "activation_self_test_mismatches": 0,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "gate_up_full_tensor_materialized": 0,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "gate_up_hvx_hmx_overlap": 1,
        "down_hvx_hmx_overlap": 1,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    require(record, "activation_self_test_cases", 65_536 if correctness else 0)
    if record["reference_checksum"] != record["output_checksum"]:
        raise SystemExit("output checksum differs from implementation reference")
    if record["warmup_output_checksum"] != record["reference_checksum"]:
        raise SystemExit("warmup checksum differs from implementation reference")
    if record["vtcm_peak_plan_bytes"] > 8_388_608:
        raise SystemExit("VTCM plan exceeds the exact 8 MiB contract")
    for field in ("host_wall_ns", "total_ticks", "gate_up_ticks", "down_ticks"):
        value = record[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive metric {field}: {value!r}")


def percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def summarize_pair(
    packed: list[dict[str, object]], expanded: list[dict[str, object]],
    repeat: int,
) -> dict[str, object]:
    packed_host = [record["host_wall_ns"] / repeat for record in packed]
    expanded_host = [record["host_wall_ns"] / repeat for record in expanded]
    packed_projection = [
        (record["gate_up_ticks"] + record["down_ticks"]) / repeat
        for record in packed
    ]
    expanded_projection = [
        (record["gate_up_ticks"] + record["down_ticks"]) / repeat
        for record in expanded
    ]
    paired_host = [
        percent(packed_host[index], expanded_host[index])
        for index in range(SAMPLES)
    ]
    paired_projection = [
        percent(packed_projection[index], expanded_projection[index])
        for index in range(SAMPLES)
    ]
    host_delta = percent(statistics.median(packed_host),
                         statistics.median(expanded_host))
    projection_delta = percent(statistics.median(packed_projection),
                               statistics.median(expanded_projection))
    paired_host_delta = statistics.median(paired_host)
    paired_projection_delta = statistics.median(paired_projection)
    passed = all(delta < 0.0 for delta in (
        host_delta, projection_delta, paired_host_delta,
        paired_projection_delta,
    ))
    return {
        "samples": SAMPLES,
        "packed_host_wall_ns_per_mlp_median": statistics.median(packed_host),
        "expanded_host_wall_ns_per_mlp_median": statistics.median(expanded_host),
        "packed_host_percent_vs_expanded": host_delta,
        "paired_host_percent_median": paired_host_delta,
        "packed_projection_ticks_per_mlp_median": statistics.median(
            packed_projection),
        "expanded_projection_ticks_per_mlp_median": statistics.median(
            expanded_projection),
        "packed_projection_percent_vs_expanded": projection_delta,
        "paired_projection_percent_median": paired_projection_delta,
        "pass_strict_host_and_projection": passed,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package = pathlib.Path(sys.argv[2])
    manifest = audit_package(package)
    summary: dict[str, object] = {
        "experiment": "EXP-0040",
        "stage": "A",
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "cross_numerical_reference": {},
        "worker_results": {},
    }

    audit = json.loads((result_dir / "package_numerical_audit.json").read_text())
    w4f16_audit = json.loads(
        (result_dir / "w4f16_cross_numerical_audit.json").read_text()
    )
    for name, expected_elements in {
        "gate": 64 * 6144,
        "up": 64 * 6144,
        "middle": 64 * 6144,
        "down": 64 * 2048,
    }.items():
        require(w4f16_audit[name], "elements", expected_elements)
        for field in (
            "maximum_absolute_error", "mean_absolute_error",
            "root_mean_square_error", "cosine_similarity",
        ):
            if not math.isfinite(w4f16_audit[name][field]):
                raise SystemExit(f"non-finite W4F16 cross-reference {name}.{field}")
    summary["cross_numerical_reference"] = {
        "candidate_vs_retained_w4u8_lsb": audit,
        "candidate_dequantized_vs_selected_w4f16": w4f16_audit,
        "gate": "reported_only_no_threshold",
    }
    for workers in WORKERS:
        worker_summary: dict[str, object] = {}
        for storage in STORAGES:
            correctness_path = result_dir / (
                f"correctness_w{workers}_{storage}.jsonl"
            )
            correctness_records = load_jsonl(correctness_path)
            if len(correctness_records) != 1:
                raise SystemExit(f"wrong correctness count: {correctness_path}")
            validate_record(correctness_records[0], storage, 1, workers, True)

        for repeat in REPEATS:
            records = {}
            for storage in STORAGES:
                path = result_dir / f"timing_w{workers}_{storage}_r{repeat}.jsonl"
                records[storage] = load_jsonl(path)
                if len(records[storage]) != SAMPLES:
                    raise SystemExit(f"wrong timing count: {path}")
                for record in records[storage]:
                    validate_record(record, storage, repeat, workers, False)
            worker_summary[f"repeat{repeat}"] = summarize_pair(
                records["packed"], records["expanded"], repeat
            )
        worker_summary["pass_both_repeats"] = all(
            worker_summary[f"repeat{repeat}"][
                "pass_strict_host_and_projection"]
            for repeat in REPEATS
        )
        summary["worker_results"][str(workers)] = worker_summary

    summary["contract_worker_count"] = 2
    summary["contract_stage_a_pass"] = summary["worker_results"]["2"][
        "pass_both_repeats"]
    summary["diagnostic_worker3_pass"] = summary["worker_results"]["3"][
        "pass_both_repeats"]
    summary["stage_b_authorized"] = summary["contract_stage_a_pass"]
    summary["package_factorization"] = manifest["weight_contract"][
        "per_channel_scale_factorization"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
