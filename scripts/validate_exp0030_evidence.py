#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "f16_control": ("F16F16", "control"),
    "f16_parallel": ("F16F16", "multi_worker_silu"),
    "f16_streaming": ("F16F16", "streaming"),
    "w4_control": ("W4F16", "control"),
    "w4_parallel": ("W4F16", "multi_worker_silu"),
    "w4_streaming": ("W4F16", "streaming"),
}

TOP_LEVEL_FIELDS = (
    "runtime_setup_ticks", "metadata_stage_ticks", "input_stage_ticks",
    "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
    "attention_ticks", "o_projection_ticks",
    "post_attention_residual_ticks", "post_attention_norm_ticks",
    "gate_up_ticks", "activation_ticks", "down_ticks",
    "final_residual_ticks", "output_stage_ticks", "runtime_teardown_ticks",
)
AUDIT_FIELDS = (
    "input_norm_audit_ticks", "qkv_audit_ticks",
    "qk_norm_rope_audit_ticks", "o_projection_audit_ticks",
    "post_attention_residual_audit_ticks",
    "post_attention_norm_audit_ticks", "gate_up_audit_ticks",
    "activation_audit_ticks", "down_audit_ticks",
    "final_residual_audit_ticks", "attention_qk_audit_ticks",
    "attention_softmax_audit_ticks", "attention_av_audit_ticks",
)
ATTENTION_FIELDS = (
    "attention_setup_ticks", "attention_qk_pack_ticks",
    "attention_qk_hmx_ticks", "attention_qk_unpack_ticks",
    "attention_qk_audit_ticks", "attention_softmax_ticks",
    "attention_softmax_audit_ticks", "attention_av_pack_ticks",
    "attention_av_hmx_ticks", "attention_av_unpack_ticks",
    "attention_av_audit_ticks", "attention_unattributed_ticks",
)
TIMING_FIELDS = (
    "total_ticks", "invocation_ticks", "input_stage_ticks",
    "metadata_stage_ticks", "input_norm_ticks", "qkv_projection_ticks",
    "qk_norm_rope_ticks", "attention_ticks", "o_projection_ticks",
    "post_attention_residual_ticks", "post_attention_norm_ticks",
    "gate_up_ticks", "activation_ticks", "down_ticks",
    "final_residual_ticks", "output_stage_ticks", "weight_dma_ticks",
    "hmx_compute_ticks", "projection_pack_ticks", "w4f16_expand_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks",
    "ledger_named_ticks", "ledger_unattributed_ticks",
    "hmx_ready_wait_ticks", "w4f16_expand_work_ticks",
    "w4f16_expand_region_count", "w4f16_prefetch_count",
    "w4f16_prefetch_wait_ticks", "f16f16_prefetch_count",
    "f16f16_prefetch_wait_ticks", "w4f16_first_expand_ticks",
    "w4f16_steady_expand_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_hmx_tail_wait_ticks", "w4f16_early_region_command_count",
    "w4f16_cross_prefetch_count", "w4f16_cross_prefetch_wait_ticks",
    "w4f16_cross_prefetch_lifetime_ticks",
    "mlp_silu_main_work_ticks", "mlp_silu_worker_work_ticks",
    "mlp_silu_pool_wait_ticks", "mlp_stream_worker_work_ticks",
    "mlp_stream_main_work_ticks", "mlp_stream_ready_wait_ticks",
    "mlp_stream_join_wait_ticks",
) + AUDIT_FIELDS + ATTENTION_FIELDS


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


def expected_weight_bytes(variant, repeat):
    if variant == "F16F16":
        return 100_663_296 * repeat
    return 25_165_824 * repeat + 81_920


def validate_record(record, config_name, repeat, audit_mode):
    variant, mlp_mode = CONFIGS[config_name]
    is_f16 = variant == "F16F16"
    is_control = mlp_mode == "control"
    is_parallel = mlp_mode == "multi_worker_silu"
    is_streaming = mlp_mode == "streaming"
    fixed = {
        "experiment": "EXP-0030",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "f16f16_projection_mode":
            "double_buffer_batch2" if is_f16 else "not_applicable",
        "w4f16_pipeline_mode":
            "not_applicable" if is_f16
            else "adaptive_down96_cross_prefetch",
        "attention_pack_mode": "combined_hvx",
        "mlp_mode": mlp_mode,
        "mlp_hvx_contexts": 1 if is_control else 4,
        "mlp_chunk_vectors": 64,
        "w4f16_scale_placement":
            "not_applicable" if is_f16 else "hmx_output_per_channel",
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
        "w4f16_hvx_workers_created": 0 if is_f16 else 3,
        "w4f16_hvx_workers_locked": 0 if is_f16 else 3,
        "w4f16_requested_hvx_workers": 2 if is_f16 else 3,
        "w4f16_region_tiles": 32,
        "w4f16_pool_status": 0,
        "f16f16_weight_batch_n_tiles": 2 if is_f16 else 0,
        "w4f16_active_worker_min": 0 if is_f16 else 2,
        "w4f16_active_worker_max": 0 if is_f16 else 3,
        "w4f16_effective_region_min": 0 if is_f16 else 32,
        "w4f16_effective_region_max": 0 if is_f16 else 96,
        "mlp_hvx_workers_created": 0 if is_control else 3,
        "mlp_hvx_workers_locked": 0 if is_control else 3,
        "mlp_pool_status": 0,
        "mlp_silu_chunk_count": 96 * repeat if is_parallel else 0,
        "mlp_stream_group_count": 96 * repeat if is_streaming else 0,
        "mlp_down_pack_skipped": repeat if is_streaming else 0,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": 7_072_512 if is_f16 else 7_744_512,
        "block_invocation_count": repeat,
        "weight_ddr_read_bytes": expected_weight_bytes(variant, repeat),
        "weight_dma_descriptor_count":
            320 * repeat if is_f16 else 160 * repeat + 7,
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": 352 * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "f16f16_prefetch_count": 313 * repeat if is_f16 else 0,
        "w4f16_prefetch_count": 0 if is_f16 else 153 * repeat,
        "w4f16_cross_prefetch_count":
            0 if is_f16 else (5 if is_streaming else 6) * repeat,
        "w4f16_early_region_command_count": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, value in fixed.items():
        require(record, field, value)

    require(record, "output_hash",
            "704252c89780e695" if is_f16 else "f18b9abbe1487231")
    if is_f16:
        require(record, "w4f16_streamed_command_count", 0)
        require(record, "w4f16_expand_ticks", 0)
        require(record, "w4f16_prefetch_wait_ticks", 0)
        require(record, "w4f16_cross_prefetch_wait_ticks", 0)
        require(record, "w4f16_cross_prefetch_lifetime_ticks", 0)
    else:
        require(record, "w4f16_streamed_command_count", 320 * repeat)
        require(record, "f16f16_prefetch_wait_ticks", 0)
        if record["w4f16_expand_ticks"] <= 0:
            raise SystemExit("W4 expansion was not observed")
        if record["w4f16_cross_prefetch_lifetime_ticks"] <= 0:
            raise SystemExit("cross-projection prefetch lifetime missing")

    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {config_name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {config_name} {prefix}max_abs")
    for field in TIMING_FIELDS:
        value = record.get(field)
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"invalid timing field {field}: {value!r}")
    positive_fields = (
        "host_wall_ns", "host_wall_ns_per_block", "total_ticks",
        "invocation_ticks", "gate_up_ticks", "down_ticks",
        "weight_dma_ticks", "hmx_compute_ticks",
    )
    if not is_streaming:
        positive_fields += ("activation_ticks",)
    for field in positive_fields:
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {config_name}.{field}: {value!r}")
    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("top-level named timing sum is inconsistent")
    if record["ledger_unattributed_ticks"] / record["invocation_ticks"] > 0.01:
        raise SystemExit("top-level unattributed gap exceeds one percent")
    if sum(record[field] for field in ATTENTION_FIELDS) != record["attention_ticks"]:
        raise SystemExit("nested Attention ledger does not close")
    if audit_mode == "on":
        for field in AUDIT_FIELDS:
            if record[field] <= 0:
                raise SystemExit(f"missing audit interval: {field}")
    else:
        for field in AUDIT_FIELDS:
            require(record, field, 0)
    if is_parallel:
        if record["mlp_silu_worker_work_ticks"] <= 0:
            raise SystemExit("parallel SiLU worker work missing")
    else:
        require(record, "mlp_silu_worker_work_ticks", 0)
    if is_streaming:
        if record["mlp_stream_worker_work_ticks"] <= 0:
            raise SystemExit("streaming worker work missing")
        if is_f16:
            require(record, "mlp_stream_main_work_ticks", 0)
        elif record["mlp_stream_main_work_ticks"] <= 0:
            raise SystemExit("W4 streaming main-thread tail work missing")
    else:
        require(record, "mlp_stream_worker_work_ticks", 0)
        require(record, "mlp_stream_main_work_ticks", 0)


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
    result = {
        "host_wall_ns_per_block_median": statistics.median(
            record["host_wall_ns_per_block"] for record in records),
        "output_hash": records[0]["output_hash"],
    }
    for field in TIMING_FIELDS:
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / repeat for record in records)
    result["complete_mlp_ticks_median_per_block"] = sum(
        result[f"{field}_median_per_block"]
        for field in ("gate_up_ticks", "activation_ticks", "down_ticks"))
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
    for name in CONFIGS:
        correctness = load_jsonl(result_dir / f"correctness_{name}.jsonl")
        repeat1 = load_jsonl(result_dir / f"timing_{name}_repeat1.jsonl")
        repeat10 = load_jsonl(result_dir / f"timing_{name}_repeat10.jsonl")
        if len(correctness) != 1 or len(repeat1) != 5 or len(repeat10) != 5:
            raise SystemExit(f"wrong evidence matrix size for {name}")
        for record in correctness:
            validate_record(record, name, 1, "on")
        for record in repeat1:
            validate_record(record, name, 1, "off")
        for record in repeat10:
            validate_record(record, name, 10, "off")
        timing["repeat1"][name] = medians(repeat1)
        timing["repeat10"][name] = medians(repeat10)
        record_count += 11

    for family in ("repeat1", "repeat10"):
        for variant in ("f16", "w4"):
            control = timing[family][f"{variant}_control"]
            parallel = timing[family][f"{variant}_parallel"]
            streaming = timing[family][f"{variant}_streaming"]
            if parallel["activation_ticks_median_per_block"] >= \
                    control["activation_ticks_median_per_block"]:
                raise SystemExit(f"{family} {variant} parallel SiLU did not improve")
            if parallel["host_wall_ns_per_block_median"] >= \
                    control["host_wall_ns_per_block_median"]:
                raise SystemExit(f"{family} {variant} parallel Host wall regressed")
            if streaming["complete_mlp_ticks_median_per_block"] >= \
                    control["complete_mlp_ticks_median_per_block"]:
                raise SystemExit(f"{family} {variant} streaming MLP did not improve")
            if streaming["host_wall_ns_per_block_median"] >= \
                    control["host_wall_ns_per_block_median"]:
                raise SystemExit(f"{family} {variant} streaming Host wall regressed")
        if timing[family]["w4_streaming"]["host_wall_ns_per_block_median"] >= \
                timing[family]["f16_streaming"]["host_wall_ns_per_block_median"]:
            raise SystemExit(f"{family} W4F16 is not faster than F16F16")

    summary = {
        "experiment": "EXP-0030",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "zero_intermediate_ddr_gate": True,
        "variant_output_hash_stability_gate": True,
        "multi_worker_silu_gate": True,
        "interleaved_streaming_mlp_gate": True,
        "audit_free_performance_mode_gate": True,
        "top_level_ledger_closure_gate": True,
        "timing": timing,
        "repeat1_f16_streaming_host_change_percent": percent_change(
            timing["repeat1"]["f16_streaming"]["host_wall_ns_per_block_median"],
            timing["repeat1"]["f16_control"]["host_wall_ns_per_block_median"]),
        "repeat1_w4_streaming_host_change_percent": percent_change(
            timing["repeat1"]["w4_streaming"]["host_wall_ns_per_block_median"],
            timing["repeat1"]["w4_control"]["host_wall_ns_per_block_median"]),
        "repeat10_f16_streaming_host_change_percent": percent_change(
            timing["repeat10"]["f16_streaming"]["host_wall_ns_per_block_median"],
            timing["repeat10"]["f16_control"]["host_wall_ns_per_block_median"]),
        "repeat10_w4_streaming_host_change_percent": percent_change(
            timing["repeat10"]["w4_streaming"]["host_wall_ns_per_block_median"],
            timing["repeat10"]["w4_control"]["host_wall_ns_per_block_median"]),
        "repeat10_w4_vs_f16_streaming_percent": percent_change(
            timing["repeat10"]["w4_streaming"]["host_wall_ns_per_block_median"],
            timing["repeat10"]["f16_streaming"]["host_wall_ns_per_block_median"]),
        "local_gate": "pass",
    }
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
