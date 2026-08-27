#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = ("F16F16", "W4F16")
FORMAL_MODES = ("scalar", "hvx")
MICRO_MODES = ("rms", "rope", "softmax", "silu")
MODE_NAMES = {
    "scalar": "scalar",
    "hvx": "hvx_fp16",
    "rms": "rms",
    "rope": "rope",
    "softmax": "softmax",
    "silu": "silu",
}
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


def validate_record(record, variant, repeat, mode):
    expected = EXPECTED[variant]
    fixed = {
        "experiment": "EXP-0024",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": MODE_NAMES[mode],
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
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
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

    positive_fields = (
        "warmup_host_wall_ns", "host_wall_ns", "host_wall_ns_per_block",
        "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
        "attention_ticks", "o_projection_ticks", "post_attention_norm_ticks",
        "gate_up_ticks", "activation_ticks", "down_ticks", "total_ticks",
        "hmx_compute_ticks", "attention_qk_max_abs",
        "attention_probability_max_abs", "attention_av_max_abs",
    )
    for field in positive_fields:
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive finite {mode}.{variant}.{field}: {value!r}")

    for prefix in ("warmup_", ""):
        cosine = record[f"{prefix}cosine"]
        max_abs = record[f"{prefix}max_abs"]
        if not math.isfinite(cosine) or cosine < 0.99999:
            raise SystemExit(f"invalid {mode}.{variant} {prefix}cosine: {cosine}")
        if not math.isfinite(max_abs) or max_abs > 0.0625:
            raise SystemExit(f"invalid {mode}.{variant} {prefix}max_abs: {max_abs}")

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
    fields = (
        "host_wall_ns_per_block", "total_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "o_projection_ticks", "post_attention_norm_ticks", "gate_up_ticks",
        "activation_ticks", "down_ticks", "weight_dma_ticks",
        "hmx_compute_ticks", "projection_pack_ticks", "w4f16_expand_ticks",
        "projection_hmx_wait_ticks", "projection_unpack_ticks",
        "w4f16_expand_work_ticks", "w4f16_prefetch_wait_ticks",
    )
    result = {}
    for field in fields:
        divisor = 1 if field == "host_wall_ns_per_block" else repeat
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / divisor for record in records)
    result["rmsnorm_ticks_median_per_block"] = statistics.median(
        (record["input_norm_ticks"] + record["post_attention_norm_ticks"]) /
        repeat for record in records)
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


def percent_change(candidate, control):
    return (candidate / control - 1.0) * 100.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package_dir = pathlib.Path(sys.argv[2])
    manifest = audit_package(package_dir)
    timing = {"repeat1": {}, "repeat10": {}}
    record_count = 0

    for mode in FORMAL_MODES:
        for variant in VARIANTS:
            correctness = load_jsonl(
                result_dir / f"correctness_{mode}_{variant}.jsonl")
            repeat1 = load_jsonl(
                result_dir / f"timing_{mode}_{variant}_repeat1.jsonl")
            repeat10 = load_jsonl(
                result_dir / f"timing_{mode}_{variant}_repeat10.jsonl")
            if len(correctness) != 1 or len(repeat1) != 5 or len(repeat10) != 5:
                raise SystemExit(f"wrong evidence matrix size for {mode}.{variant}")
            for record in correctness + repeat1:
                validate_record(record, variant, 1, mode)
            for record in repeat10:
                validate_record(record, variant, 10, mode)
            timing["repeat1"][f"{mode}_{variant}"] = medians(repeat1)
            timing["repeat10"][f"{mode}_{variant}"] = medians(repeat10)
            record_count += len(correctness) + len(repeat1) + len(repeat10)

    microgates = {}
    for mode in MICRO_MODES:
        microgates[mode] = {}
        for variant in VARIANTS:
            records = load_jsonl(
                result_dir / f"microgate_{mode}_{variant}.jsonl")
            if len(records) != 1:
                raise SystemExit(f"wrong microgate size for {mode}.{variant}")
            validate_record(records[0], variant, 1, mode)
            microgates[mode][variant] = {
                "max_abs": records[0]["max_abs"],
                "cosine": records[0]["cosine"],
            }
            record_count += 1

    comparisons = {}
    stage_fields = {
        "whole_block": "host_wall_ns_per_block_median_per_block",
        "rmsnorm": "rmsnorm_ticks_median_per_block",
        "qk_norm_rope": "qk_norm_rope_ticks_median_per_block",
        "attention_including_softmax": "attention_ticks_median_per_block",
        "silu_by_up": "activation_ticks_median_per_block",
    }
    vectorization_gate = True
    final_w4_gate = True
    for family in ("repeat1", "repeat10"):
        comparisons[family] = {"hvx_vs_scalar": {}, "w4f16_vs_f16f16": {}}
        for variant in VARIANTS:
            scalar = timing[family][f"scalar_{variant}"]
            hvx = timing[family][f"hvx_{variant}"]
            stage_changes = {
                stage: percent_change(hvx[field], scalar[field])
                for stage, field in stage_fields.items()
            }
            comparisons[family]["hvx_vs_scalar"][variant] = stage_changes
            vectorization_gate = vectorization_gate and all(
                change < 0.0 for change in stage_changes.values())
        for mode in FORMAL_MODES:
            f16 = timing[family][f"{mode}_F16F16"]
            w4 = timing[family][f"{mode}_W4F16"]
            wall_change = percent_change(
                w4["host_wall_ns_per_block_median_per_block"],
                f16["host_wall_ns_per_block_median_per_block"])
            projection_change = percent_change(
                w4["projection_ticks_median_per_block"],
                f16["projection_ticks_median_per_block"])
            comparisons[family]["w4f16_vs_f16f16"][mode] = {
                "whole_block_latency_change_percent": wall_change,
                "whole_block_speedup": 1.0 / (1.0 + wall_change / 100.0),
                "projection_latency_change_percent": projection_change,
                "projection_speedup": 1.0 / (1.0 + projection_change / 100.0),
            }
            if mode == "hvx":
                final_w4_gate = final_w4_gate and wall_change < 0.0

    local_gate = vectorization_gate and final_w4_gate
    summary = {
        "experiment": "EXP-0024",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "common_ops": [
            "rmsnorm", "qk_norm_rope", "stable_causal_softmax", "silu_by_up"
        ],
        "softmax_algorithm": "standard_stable_natural_exp",
        "single_prepared_measured_fastrpc": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "intermediate_residency": "VTCM",
        "zero_intermediate_ddr_gate": True,
        "numerical_gate": True,
        "microgates": microgates,
        "timing": timing,
        "comparisons": comparisons,
        "vectorization_non_regression_gate": vectorization_gate,
        "final_w4f16_faster_than_f16f16_gate": final_w4_gate,
        "local_gate": "pass" if local_gate else "fail",
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if local_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
