#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = ("F16F16", "W4F16")
EXPECTED = {
    "F16F16": {
        "vtcm": 6_286_080,
        "hmx_commands_per_block": 672,
        "weight_bytes": lambda repeat: 100_663_296 * repeat,
        "weight_descriptors": lambda repeat: 640 * repeat,
    },
    "W4F16": {
        "vtcm": 7_744_512,
        "hmx_commands_per_block": 352,
        "weight_bytes": lambda repeat: 25_165_824 * repeat + 81_920,
        "weight_descriptors": lambda repeat: 160 * repeat + 7,
    },
}


def require(record, field, expected):
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


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


def validate_record(record, variant, repeat):
    expected = EXPECTED[variant]
    fixed = {
        "experiment": "EXP-0023",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
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
        "w4f16_expand_mismatch_count": 0,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": expected["vtcm"],
        "block_invocation_count": repeat,
        "weight_ddr_read_bytes": expected["weight_bytes"](repeat),
        "weight_dma_descriptor_count": expected["weight_descriptors"](repeat),
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": expected["hmx_commands_per_block"] * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
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
            raise SystemExit(f"invalid positive finite {variant}.{field}: {value!r}")

    for prefix in ("warmup_", ""):
        cosine = record[f"{prefix}cosine"]
        max_abs = record[f"{prefix}max_abs"]
        if not math.isfinite(cosine) or cosine < 0.99999:
            raise SystemExit(f"invalid {variant} {prefix}cosine: {cosine}")
        if not math.isfinite(max_abs) or max_abs > 0.0625:
            raise SystemExit(f"invalid {variant} {prefix}max_abs: {max_abs}")

    if variant == "W4F16":
        require(record, "w4f16_scale_placement", "hmx_output_per_channel")
        require(record, "w4f16_hvx_workers_created", 2)
        require(record, "w4f16_hvx_workers_locked", 2)
        require(record, "w4f16_requested_hvx_workers", 2)
        require(record, "w4f16_region_tiles", 16)
        require(record, "w4f16_pool_status", 0)
        require(record, "w4f16_streamed_command_count", 320 * repeat)
        for field in (
            "w4f16_expand_ticks", "w4f16_expand_work_ticks",
            "w4f16_expand_region_count", "w4f16_prefetch_count",
        ):
            if record.get(field, 0) <= 0:
                raise SystemExit(f"missing W4F16 pipeline evidence: {field}")
    else:
        require(record, "w4f16_scale_placement", "not_applicable")
        require(record, "w4f16_hvx_workers_created", 0)
        require(record, "w4f16_hvx_workers_locked", 0)
        require(record, "w4f16_streamed_command_count", 0)
        require(record, "w4f16_expand_ticks", 0)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_package(package_dir):
    manifest = json.loads((package_dir / "manifest.json").read_text())
    require(manifest, "experiment", "EXP-0022")
    require(manifest, "layer", 14)
    require(manifest, "sequence_length", 64)
    require(manifest, "hidden_size", 2048)
    require(manifest, "intermediate_size", 6144)
    require(manifest["w4_contract"], "qmin", -7)
    require(manifest["w4_contract"], "qmax", 7)
    require(manifest["w4_contract"], "scale_granularity", "per_output_channel")
    if len(manifest["tensors"]) != 118:
        raise SystemExit("parent Block Package tensor count changed")
    for name, tensor in manifest["tensors"].items():
        path = package_dir / tensor["file"]
        if not path.is_file() or path.stat().st_size != tensor["bytes"]:
            raise SystemExit(f"invalid package tensor: {name}")
        if sha256_file(path) != tensor["sha256"]:
            raise SystemExit(f"package digest mismatch: {name}")
    return manifest


def medians(records):
    repeat = records[0]["repeat_count"]
    result = {}
    for field in (
        "host_wall_ns_per_block", "total_ticks", "qkv_projection_ticks",
        "o_projection_ticks", "gate_up_ticks", "down_ticks",
        "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
        "w4f16_expand_ticks", "projection_hmx_wait_ticks",
        "projection_unpack_ticks", "w4f16_expand_work_ticks",
        "w4f16_prefetch_wait_ticks",
    ):
        divisor = 1 if field == "host_wall_ns_per_block" else repeat
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / divisor for record in records)
    projection_fields = (
        "qkv_projection_ticks", "o_projection_ticks",
        "gate_up_ticks", "down_ticks",
    )
    result["projection_ticks_median_per_block"] = statistics.median(
        sum(record[field] for field in projection_fields) / repeat
        for record in records)
    result["weight_ddr_read_bytes_per_block"] = statistics.median(
        record["weight_ddr_read_bytes"] / repeat for record in records)
    result["weight_dma_descriptors_per_block"] = statistics.median(
        record["weight_dma_descriptor_count"] / repeat for record in records)
    result["vtcm_peak_plan_bytes"] = records[0]["vtcm_peak_plan_bytes"]
    return result


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package_dir = pathlib.Path(sys.argv[2])
    manifest = audit_package(package_dir)
    timing = {"repeat1": {}, "repeat10": {}}
    record_count = 0

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
        record_count += len(correctness) + len(repeat1) + len(repeat10)

    comparisons = {}
    for family in ("repeat1", "repeat10"):
        f16 = timing[family]["F16F16"]
        w4 = timing[family]["W4F16"]
        wall_ratio = (w4["host_wall_ns_per_block_median_per_block"] /
                      f16["host_wall_ns_per_block_median_per_block"])
        projection_ratio = (w4["projection_ticks_median_per_block"] /
                            f16["projection_ticks_median_per_block"])
        comparisons[family] = {
            "whole_block_latency_change_percent": (wall_ratio - 1.0) * 100.0,
            "whole_block_speedup": 1.0 / wall_ratio,
            "projection_latency_change_percent": (projection_ratio - 1.0) * 100.0,
            "projection_speedup": 1.0 / projection_ratio,
        }

    strict_speed_gate = all(
        comparisons[family]["whole_block_latency_change_percent"] < 0.0
        for family in ("repeat1", "repeat10"))
    repeat10_target = (
        comparisons["repeat10"]["whole_block_latency_change_percent"] <= -15.0)
    summary = {
        "experiment": "EXP-0023",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "single_prepared_measured_fastrpc": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "intermediate_residency": "VTCM",
        "zero_intermediate_ddr_gate": True,
        "numerical_gate": True,
        "w4f16_hmx_batch_n_tiles": 2,
        "w4f16_dma_batch_n_tiles": 4,
        "timing": timing,
        "comparisons": comparisons,
        "strict_speed_gate": strict_speed_gate,
        "repeat10_improvement_target_percent": 15.0,
        "repeat10_target_met": repeat10_target,
    }
    if not strict_speed_gate:
        raise SystemExit("W4F16 did not pass the strict whole-block speed gate")
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
