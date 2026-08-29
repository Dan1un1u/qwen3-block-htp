#!/usr/bin/env python3
"""Validate paired EXP-0045 Stage-B QKV/O pipeline evidence."""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import sys


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("serial", "qkv_batch4", "qkvo_batch4")
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608
MODE_COUNTERS = {
    "serial": (0, 0, 0, 0, 1184, 800),
    "qkv_batch4": (4, 32, 29, 29, 1088, 608),
    "qkvo_batch4": (4, 32, 44, 44, 1040, 512),
}


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != SAMPLES:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def validate_record(record: dict[str, object], repeat: int, mode: str) -> None:
    batch, batches, prefetches, overlaps, commands, descriptors = MODE_COUNTERS[mode]
    fixed = {
        "experiment": "EXP-0045",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "w4u8_qkvo_pipeline_mode": mode,
        "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
        "repeat_count": repeat,
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
        "warmup_max_lsb": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "u8_attention_audit_ddr_write_bytes": 0,
        "u8_attention_fused_k_operand_mismatch_count": 0,
        "u8_attention_qkv_unpack_skipped": 128 * repeat,
        "hmx_command_count": commands * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": descriptors * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": batch,
        "w4u8_qkv_batch_count": batches * repeat,
        "w4u8_qkvo_prefetch_count": prefetches * repeat,
        "w4u8_qkvo_overlap_schedule_count": overlaps * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds exact 8 MiB contract")
    for field in (
        "host_wall_ns_per_block",
        "total_ticks",
        "qkv_projection_ticks",
        "o_projection_ticks",
    ):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def per_block(record: dict[str, object], field: str) -> float:
    if field == "host_wall_ns_per_block":
        return float(record[field])
    return float(record[field]) / int(record["repeat_count"])


def qkvo(record: dict[str, object]) -> float:
    return per_block(record, "qkv_projection_ticks") + per_block(
        record, "o_projection_ticks"
    )


def summarize_field(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
    field: str,
) -> dict[str, float]:
    if field == "qkvo_projection_ticks":
        left = [qkvo(record) for record in control]
        right = [qkvo(record) for record in candidate]
    else:
        left = [per_block(record, field) for record in control]
        right = [per_block(record, field) for record in candidate]
    left_median = statistics.median(left)
    right_median = statistics.median(right)
    paired = [
        (right[index] / left[index] - 1.0) * 100.0
        for index in range(SAMPLES)
        if left[index] != 0.0
    ]
    result = {
        f"control_{field}_median": left_median,
        f"candidate_{field}_median": right_median,
    }
    if left_median != 0.0:
        result[f"{field}_change_percent"] = (
            right_median / left_median - 1.0
        ) * 100.0
    if paired:
        result[f"paired_{field}_change_percent_median"] = statistics.median(
            paired
        )
    return result


def compare(
    control: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {"samples": SAMPLES}
    for field in (
        "host_wall_ns_per_block",
        "total_ticks",
        "qkv_projection_ticks",
        "o_projection_ticks",
        "qkvo_projection_ticks",
        "attention_ticks",
        "attention_qk_norm_pool_wait_ticks",
        "hmx_command_count",
        "weight_dma_descriptor_count",
        "w4u8_qkvo_weight_expand_ticks",
        "w4u8_qkvo_prefetch_wait_ticks",
        "w4u8_qkvo_hmx_lifetime_ticks",
    ):
        result.update(summarize_field(control, candidate, field))
    result["stage_b_speed_gate_pass"] = all(
        float(result[field]) < 0.0
        for field in (
            "host_wall_ns_per_block_change_percent",
            "paired_host_wall_ns_per_block_change_percent_median",
            "o_projection_ticks_change_percent",
            "paired_o_projection_ticks_change_percent_median",
            "qkvo_projection_ticks_change_percent",
            "paired_qkvo_projection_ticks_change_percent_median",
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
        (
            result_dir
            / "attention_implementation_audit"
            / "implementation_reference.json"
        ).read_text()
    )
    require(implementation, "experiment", "EXP-0045")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(implementation["exact_stage_comparison"][stage], "mismatches", 0)

    summary: dict[str, object] = {
        "experiment": "EXP-0045",
        "stage": "B",
        "control": "qkv_batch4",
        "candidate": "qkvo_batch4",
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "implementation_reference_gate": True,
        "byte_exact_final_output_gate": True,
        "zero_intermediate_ddr_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "repeat_results": {},
    }
    stage_b_gates: list[bool] = []
    final_vs_serial_gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load_jsonl(result_dir / f"paired_{mode}_r{repeat}.jsonl")
            for mode in MODES
        }
        for mode, mode_records in records.items():
            for record in mode_records:
                validate_record(record, repeat, mode)
        stage_b = compare(records["qkv_batch4"], records["qkvo_batch4"])
        final_vs_serial = compare(records["serial"], records["qkvo_batch4"])
        stage_b_gates.append(bool(stage_b["stage_b_speed_gate_pass"]))
        final_vs_serial_gates.append(
            all(
                float(final_vs_serial[field]) < 0.0
                for field in (
                    "host_wall_ns_per_block_change_percent",
                    "paired_host_wall_ns_per_block_change_percent_median",
                    "qkvo_projection_ticks_change_percent",
                    "paired_qkvo_projection_ticks_change_percent_median",
                )
            )
        )
        summary["repeat_results"][f"repeat{repeat}"] = {
            "qkvo_batch4_vs_qkv_batch4": stage_b,
            "qkvo_batch4_vs_serial": final_vs_serial,
        }

    summary["stage_b_speed_gate_pass"] = all(stage_b_gates)
    summary["final_candidate_vs_serial_gate_pass"] = all(final_vs_serial_gates)
    summary["stage_b_gate_pass"] = all(stage_b_gates) and all(
        final_vs_serial_gates
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
