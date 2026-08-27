#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = ("F16F16", "W4F16", "W4U8")
EXPECTED = {
    "F16F16": {
        "projection_compute": "FP16_HMX",
        "vtcm_peak_plan_bytes": 5794048,
        "weight_ddr_read_bytes_per_block": 100663296,
        "hmx_fp16_tile_pair_count_per_block": 98816,
        "hmx_u8s8_tile_pair_count_per_block": 0,
    },
    "W4F16": {
        "projection_compute": "FP16_HMX",
        "vtcm_peak_plan_bytes": 5794048,
        "weight_ddr_read_bytes_per_block": 25247744,
        "hmx_fp16_tile_pair_count_per_block": 98816,
        "hmx_u8s8_tile_pair_count_per_block": 0,
    },
    "W4U8": {
        "projection_compute": "U8xS8_integer_HMX",
        "vtcm_peak_plan_bytes": 4090112,
        "weight_ddr_read_bytes_per_block": 25329664,
        "hmx_fp16_tile_pair_count_per_block": 512,
        "hmx_u8s8_tile_pair_count_per_block": 49152,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_record(record, variant, repeat):
    expected = EXPECTED[variant]
    fixed = {
        "experiment": "EXP-0022",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": expected["projection_compute"],
        "intermediate_residency": "VTCM",
        "warmup_rpc_result": 0,
        "warmup_prepared_session_run_index": 1,
        "repeat_count": repeat,
        "prepared_session_run_index": 2,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "projection_failure_result": 0,
        "projection_failure_index": 0,
        "projection_failure_n_tile": 0,
        "projection_failure_step": 0,
        "vtcm_requested_bytes": 8388608,
        "vtcm_acquired_bytes": 8388608,
        "vtcm_peak_plan_bytes": expected["vtcm_peak_plan_bytes"],
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": 672 * repeat,
        "hmx_fp16_tile_pair_count":
            expected["hmx_fp16_tile_pair_count_per_block"] * repeat,
        "hmx_u8s8_tile_pair_count":
            expected["hmx_u8s8_tile_pair_count_per_block"] * repeat,
        "weight_ddr_read_bytes":
            expected["weight_ddr_read_bytes_per_block"] * repeat,
        "release_result": 0,
        "close_result": 0,
    }
    for field, value in fixed.items():
        require(record, field, value)

    for field in (
        "warmup_host_wall_ns", "host_wall_ns", "host_wall_ns_per_block",
        "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
        "attention_ticks", "o_projection_ticks", "post_attention_norm_ticks",
        "gate_up_ticks", "activation_ticks", "down_ticks", "total_ticks",
        "hmx_compute_ticks", "attention_qk_max_abs",
        "attention_probability_max_abs", "attention_av_max_abs",
    ):
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"missing positive finite field {variant}.{field}: {value!r}")

    for prefix in ("warmup_", ""):
        cosine = record[f"{prefix}cosine"]
        max_abs = record[f"{prefix}max_abs"]
        if not math.isfinite(cosine) or cosine < 0.99:
            raise SystemExit(f"invalid {variant} {prefix}cosine: {cosine}")
        if not math.isfinite(max_abs):
            raise SystemExit(f"invalid {variant} {prefix}max_abs: {max_abs}")
    if variant == "W4U8":
        for prefix in ("warmup_", ""):
            require(record, f"{prefix}mismatches", 0)
            require(record, f"{prefix}max_lsb", 0)


def audit_package(package_dir):
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fixed = {
        "experiment": "EXP-0022",
        "package_abi": 1,
        "layer": 14,
        "sequence_length": 64,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    for field, value in fixed.items():
        require(manifest, field, value)
    require(manifest["w4_contract"], "qmin", -7)
    require(manifest["w4_contract"], "qmax", 7)
    require(manifest["w4_contract"], "scale_granularity", "per_output_channel")
    require(manifest["w4_contract"], "shared_carrier_between", ["W4F16", "W4U8"])
    if len(manifest["tensors"]) != 118:
        raise SystemExit("Block Package tensor count changed")
    total_bytes = 0
    for name, tensor in manifest["tensors"].items():
        path = package_dir / tensor["file"]
        if not path.is_file():
            raise SystemExit(f"missing package tensor: {name}: {path}")
        if path.stat().st_size != tensor["bytes"]:
            raise SystemExit(f"wrong byte count for package tensor: {name}")
        if sha256_file(path) != tensor["sha256"]:
            raise SystemExit(f"wrong digest for package tensor: {name}")
        total_bytes += tensor["bytes"]
    return manifest, total_bytes


def medians(records):
    fields = (
        "host_wall_ns_per_block", "total_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "o_projection_ticks", "post_attention_norm_ticks", "gate_up_ticks",
        "activation_ticks", "down_ticks", "hmx_compute_ticks",
    )
    repeat = records[0]["repeat_count"]
    result = {}
    for field in fields:
        divisor = 1 if field == "host_wall_ns_per_block" else repeat
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / divisor for record in records)
    result["weight_ddr_read_bytes_per_block"] = (
        records[0]["weight_ddr_read_bytes"] // repeat)
    result["vtcm_peak_plan_bytes"] = records[0]["vtcm_peak_plan_bytes"]
    return result


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR [PACKAGE_DIR]")
    result_dir = pathlib.Path(sys.argv[1])
    package_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else pathlib.Path(
        "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
        "block_package_layer14_m64")
    manifest, package_tensor_bytes = audit_package(package_dir)

    all_records = 0
    timing = {"repeat1": {}, "repeat10": {}}
    for variant in VARIANTS:
        correctness = load_jsonl(result_dir / f"correctness_{variant}.jsonl")
        repeat1 = load_jsonl(result_dir / f"timing_{variant}_repeat1.jsonl")
        repeat10 = load_jsonl(result_dir / f"timing_{variant}_repeat10.jsonl")
        if len(correctness) != 1 or len(repeat1) != 5 or len(repeat10) != 5:
            raise SystemExit(f"wrong evidence matrix size for {variant}")
        for record in correctness + repeat1:
            validate_record(record, variant, 1)
        for record in repeat10:
            validate_record(record, variant, 10)
        timing["repeat1"][variant] = medians(repeat1)
        timing["repeat10"][variant] = medians(repeat10)
        all_records += len(correctness) + len(repeat1) + len(repeat10)

    comparisons = {}
    for family in ("repeat1", "repeat10"):
        baseline = timing[family]["F16F16"][
            "host_wall_ns_per_block_median_per_block"]
        comparisons[family] = {}
        for variant in ("W4F16", "W4U8"):
            latency = timing[family][variant][
                "host_wall_ns_per_block_median_per_block"]
            comparisons[family][variant] = {
                "latency_ratio_vs_F16F16": latency / baseline,
                "speedup_vs_F16F16": baseline / latency,
                "latency_change_percent": (latency / baseline - 1.0) * 100.0,
            }

    summary = {
        "experiment": "EXP-0022",
        "records": all_records,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "source_model": manifest["source_model"],
        "block_package_tensor_count": len(manifest["tensors"]),
        "block_package_tensor_bytes": package_tensor_bytes,
        "block_package_hash_audit": True,
        "w4_carrier_shared_between_W4F16_W4U8": True,
        "single_prepared_measured_fastrpc": True,
        "fixed_vtcm_request_bytes": 8388608,
        "intermediate_residency": "VTCM",
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "zero_intermediate_ddr_gate": True,
        "speed_gate_required": False,
        "teacher_reference_metrics": manifest["reference_metrics"],
        "timing": timing,
        "comparisons": comparisons,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
