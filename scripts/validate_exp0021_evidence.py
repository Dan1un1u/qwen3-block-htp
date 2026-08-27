#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


def load_jsonl(path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def require(record, field, value):
    if record.get(field) != value:
        raise SystemExit(
            f"wrong {field}: {record.get(field)!r} != {value!r}")


def validate_record(record, repeat, expect_self_test):
    fixed = {
        "experiment": "EXP-0021",
        "implementation": "vtcm_resident_mlp_paired_tile_pipeline",
        "execution_unit": "qwen3_middle_block_mlp",
        "resource_lifetime_mode": "prepared_session",
        "intermediate_residency": "VTCM",
        "weight_storage": "packed_signed_w4_per_channel_integer_scale",
        "activation_approximation": "fixed_point_clipped_linear_silu",
        "warmup_rpc_calls": 1,
        "measured_rpc_calls": 1,
        "warmup_repeat_count": 1,
        "warmup_rpc_result": 0,
        "warmup_prepared_session_run_index": 1,
        "warmup_mismatches": 0,
        "repeat_count": repeat,
        "prepared_session_run_index": 2,
        "rpc_result": 0,
        "dsp_status": 3,
        "mismatches": 0,
        "vtcm_requested_bytes": 2097152,
        "vtcm_acquired_bytes": 2097152,
        "gate_up_output_vtcm_bytes": 32768,
        "middle_vtcm_bytes": 393216,
        "final_output_vtcm_bytes": 131072,
        "gate_up_pair_slot_count": 8,
        "gate_up_pair_publish_count": 192 * repeat,
        "gate_up_pair_consume_count": 192 * repeat,
        "gate_up_full_tensor_materialized": 0,
        "activation_self_test_cases": 65536 if expect_self_test else 0,
        "activation_self_test_mismatches": 0,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "gate_up_output_dma_descriptor_count": 0,
        "middle_dma_descriptor_count": 0,
        "final_output_dma_descriptor_count": 64,
        "final_output_dma_timeout_count": 0,
        "gate_up_hvx_hmx_overlap": 1,
        "down_hvx_hmx_overlap": 1,
    }
    for field, value in fixed.items():
        require(record, field, value)
    if record["warmup_output_checksum"] != record["reference_checksum"]:
        raise SystemExit("warm-up output checksum differs from Host reference")
    if record["output_checksum"] != record["reference_checksum"]:
        raise SystemExit("measured output checksum differs from Host reference")
    if not 0 < record["vtcm_peak_plan_bytes"] <= 2097152:
        raise SystemExit("VTCM plan exceeds the fixed 2 MiB allocation")
    for field in (
        "warmup_host_wall_ns", "host_wall_ns", "reference_wall_ns",
        "gate_up_ticks", "activation_ticks", "down_ticks", "total_ticks",
    ):
        if record[field] <= 0:
            raise SystemExit(f"missing positive timing field: {field}")


def medians(records, repeat):
    fields = (
        "host_wall_ns", "total_ticks", "gate_up_ticks",
        "activation_ticks", "down_ticks",
    )
    return {
        f"{field}_median": statistics.median(
            record[field] / repeat for record in records)
        for field in fields
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing1 = load_jsonl(result_dir / "timing_repeat1.jsonl")
    timing10 = load_jsonl(result_dir / "timing_repeat10.jsonl")
    if len(correctness) != 1 or len(timing1) != 10 or len(timing10) != 10:
        raise SystemExit("wrong evidence matrix size")
    for record in correctness:
        validate_record(record, 1, True)
    for record in timing1:
        validate_record(record, 1, False)
    for record in timing10:
        validate_record(record, 10, False)

    print(json.dumps({
        "experiment": "EXP-0021",
        "records": len(correctness) + len(timing1) + len(timing10),
        "execution_unit": "qwen3_middle_block_mlp",
        "exact_host_reference": True,
        "exhaustive_u8_pair_activation_self_test": True,
        "single_prepared_measured_fastrpc": True,
        "gate_up_full_tensor_materialized": False,
        "intermediate_residency": "VTCM",
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "zero_intermediate_ddr_gate": True,
        "fixed_vtcm_request_bytes": 2097152,
        "device_reboot": False,
        "speed_gate_required": False,
        "timing_repeat1_per_mlp": medians(timing1, 1),
        "timing_repeat10_per_mlp": medians(timing10, 10),
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
