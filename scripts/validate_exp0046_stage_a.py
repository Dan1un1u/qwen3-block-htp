#!/usr/bin/env python3
"""Validate paired EXP-0046 Stage-A native MLP-input evidence."""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import sys


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "w4u8_mlp_input")
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608


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
    fixed = {
        "experiment": "EXP-0046",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "crouton_boundary_mode": mode,
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4",
        "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
        "mlp_mode": "w4u8_streaming",
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
        "hmx_command_count": 1040 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": 4,
        "w4u8_qkv_batch_count": 32 * repeat,
        "w4u8_qkvo_prefetch_count": 44 * repeat,
        "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
        "w4u8_mlp_input_pack_skipped": repeat if mode != "control" else 0,
        "w4u8_mlp_output_unpack_skipped": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds exact 8 MiB contract")
    input_pack_ticks = int(record["w4u8_mlp_input_pack_ticks"])
    if mode == "control" and input_pack_ticks <= 0:
        raise SystemExit("control did not execute the W4U8 MLP input pack")
    if mode != "control" and input_pack_ticks != 0:
        raise SystemExit("candidate retained W4U8 MLP input pack work")
    if int(record["w4u8_mlp_output_unpack_ticks"]) <= 0:
        raise SystemExit("Stage A must retain the output unpack control")
    for field in (
        "host_wall_ns_per_block",
        "total_ticks",
        "post_attention_residual_ticks",
        "gate_up_ticks",
        "down_ticks",
        "final_residual_ticks",
    ):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def per_block(record: dict[str, object], field: str) -> float:
    if field == "host_wall_ns_per_block":
        return float(record[field])
    return float(record[field]) / int(record["repeat_count"])


def derived(record: dict[str, object], field: str) -> float:
    if field == "mlp_input_boundary_ticks":
        return sum(
            per_block(record, part)
            for part in (
                "post_attention_residual_ticks",
                "post_attention_norm_ticks",
                "w4u8_mlp_input_pack_ticks",
                "gate_up_ticks",
            )
        )
    return per_block(record, field)


def summarize_field(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
    field: str,
) -> dict[str, float]:
    left = [derived(record, field) for record in control]
    right = [derived(record, field) for record in candidate]
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
        f"paired_{field}_change_percent_median": statistics.median(paired),
    }
    if left_median != 0.0:
        result[f"{field}_change_percent"] = (
            right_median / left_median - 1.0
        ) * 100.0
    return result


def compare(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {"samples": SAMPLES}
    fields = (
        "host_wall_ns_per_block",
        "total_ticks",
        "mlp_input_boundary_ticks",
        "post_attention_residual_ticks",
        "post_attention_norm_ticks",
        "w4u8_mlp_input_pack_ticks",
        "gate_up_ticks",
        "attention_ticks",
        "qkv_projection_ticks",
        "o_projection_ticks",
        "down_ticks",
        "w4u8_mlp_output_unpack_ticks",
        "final_residual_ticks",
        "projection_pack_ticks",
        "projection_unpack_ticks",
        "w4u8_mlp_weight_expand_ticks",
        "w4u8_mlp_hmx_compute_ticks",
        "w4u8_mlp_hmx_ready_wait_ticks",
        "w4u8_mlp_expanded_slot_wait_ticks",
        "attention_qk_norm_pool_wait_ticks",
        "hmx_command_count",
        "weight_dma_descriptor_count",
    )
    for field in fields:
        result.update(summarize_field(control, candidate, field))
    result["stage_a_speed_gate_pass"] = all(
        float(result[field]) < 0.0
        for field in (
            "host_wall_ns_per_block_change_percent",
            "paired_host_wall_ns_per_block_change_percent_median",
            "mlp_input_boundary_ticks_change_percent",
            "paired_mlp_input_boundary_ticks_change_percent_median",
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
         "implementation_reference.json").read_text()
    )
    require(implementation, "experiment", "EXP-0046")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(implementation["exact_stage_comparison"][stage], "mismatches", 0)

    summary: dict[str, object] = {
        "experiment": "EXP-0046",
        "stage": "A",
        "control": "control",
        "candidate": "w4u8_mlp_input",
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
    gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load_jsonl(result_dir / f"paired_{mode}_r{repeat}.jsonl")
            for mode in MODES
        }
        for mode, mode_records in records.items():
            for record in mode_records:
                validate_record(record, repeat, mode)
        comparison = compare(records["control"], records["w4u8_mlp_input"])
        gates.append(bool(comparison["stage_a_speed_gate_pass"]))
        summary["repeat_results"][f"repeat{repeat}"] = comparison

    summary["stage_a_speed_gate_pass"] = all(gates)
    summary["stage_a_gate_pass"] = all(gates)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
