#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


MODES = {
    "control": ("control", 1),
    "qk": ("parallel_qk_norm_rope", 4),
    "softmax": ("parallel_softmax", 4),
    "parallel": ("parallel_hvx", 4),
    "gqa": ("gqa_pipeline", 4),
    "overlap": ("gqa_qkv_overlap", 4),
}
CONFIGS = {
    f"{variant}_{name}": (variant, mode, contexts)
    for variant in ("f16", "w4")
    for name, (mode, contexts) in MODES.items()
}

TOP_LEVEL_FIELDS = (
    "runtime_setup_ticks", "metadata_stage_ticks", "input_stage_ticks",
    "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
    "attention_ticks", "o_projection_ticks",
    "post_attention_residual_ticks", "post_attention_norm_ticks",
    "gate_up_ticks", "activation_ticks", "down_ticks",
    "final_residual_ticks", "output_stage_ticks", "runtime_teardown_ticks",
)
STAGE_AUDIT_FIELDS = (
    "input_norm_audit_ticks", "qkv_audit_ticks",
    "o_projection_audit_ticks", "post_attention_residual_audit_ticks",
    "post_attention_norm_audit_ticks", "gate_up_audit_ticks",
    "activation_audit_ticks", "down_audit_ticks",
    "final_residual_audit_ticks",
)
ATTENTION_AUDIT_FIELDS = (
    "attention_qk_audit_ticks", "attention_softmax_audit_ticks",
    "attention_av_audit_ticks",
)
ATTENTION_FIELDS = (
    "attention_setup_ticks", "attention_qk_pack_ticks",
    "attention_qk_hmx_ticks", "attention_qk_unpack_ticks",
    "attention_qk_audit_ticks", "attention_softmax_ticks",
    "attention_softmax_audit_ticks", "attention_av_pack_ticks",
    "attention_av_hmx_ticks", "attention_av_unpack_ticks",
    "attention_av_audit_ticks", "attention_gqa_pipeline_ticks",
    "attention_unattributed_ticks",
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
    "mlp_stream_worker_work_ticks", "mlp_stream_main_work_ticks",
    "mlp_stream_ready_wait_ticks", "mlp_stream_join_wait_ticks",
    "attention_qk_norm_main_work_ticks",
    "attention_qk_norm_worker_work_ticks",
    "attention_qk_norm_pool_wait_ticks",
    "attention_softmax_main_work_ticks",
    "attention_softmax_worker_work_ticks",
    "attention_softmax_pool_wait_ticks",
    "attention_gqa_worker_work_ticks", "attention_gqa_hmx_wait_ticks",
    "attention_gqa_queue_wait_ticks",
) + STAGE_AUDIT_FIELDS + ("qk_norm_rope_audit_ticks",) + \
    ATTENTION_AUDIT_FIELDS + ATTENTION_FIELDS


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
    family, mode, contexts = CONFIGS[config_name]
    variant = "F16F16" if family == "f16" else "W4F16"
    is_f16 = family == "f16"
    is_control = mode == "control"
    is_overlap = mode == "gqa_qkv_overlap"
    is_qk = mode in ("parallel_qk_norm_rope", "parallel_hvx") or is_overlap
    is_softmax = mode in ("parallel_softmax", "parallel_hvx")
    is_gqa = mode in ("gqa_pipeline", "gqa_qkv_overlap")

    fixed = {
        "experiment": "EXP-0031",
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
        "attention_pipeline_mode": mode,
        "attention_hvx_contexts": contexts,
        "mlp_mode": "streaming",
        "mlp_hvx_contexts": 4,
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
        "mlp_hvx_workers_created": 3,
        "mlp_hvx_workers_locked": 3,
        "mlp_pool_status": 0,
        "mlp_silu_chunk_count": 0,
        "mlp_stream_group_count": 96 * repeat,
        "mlp_down_pack_skipped": repeat,
        "attention_hvx_workers_created": 0 if is_control else 3,
        "attention_hvx_workers_locked": 0 if is_control else 3,
        "attention_pool_status": 0,
        "attention_qk_norm_task_count": 24 * repeat if is_qk else 0,
        "attention_softmax_task_count": 16 * repeat if is_softmax else 0,
        "attention_gqa_group_count": 8 * repeat if is_gqa else 0,
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
        "w4f16_cross_prefetch_count": 0 if is_f16 else 5 * repeat,
        "w4f16_early_region_command_count": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, value in fixed.items():
        require(record, field, value)

    require(record, "output_hash",
            "704252c89780e695" if is_f16 else "f18b9abbe1487231")
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
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
    for field in (
            "host_wall_ns", "host_wall_ns_per_block", "total_ticks",
            "invocation_ticks", "qkv_projection_ticks", "attention_ticks",
            "gate_up_ticks", "down_ticks", "weight_dma_ticks",
            "hmx_compute_ticks"):
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
        for field in STAGE_AUDIT_FIELDS:
            if record[field] <= 0:
                raise SystemExit(f"missing audit interval: {field}")
        if is_gqa:
            require(record, "qk_norm_rope_audit_ticks", 0)
            for field in ATTENTION_AUDIT_FIELDS:
                require(record, field, 0)
            if not (0 < record["attention_qk_max_abs"] < math.inf):
                raise SystemExit("GQA raw QK audit is invalid")
        else:
            if record["qk_norm_rope_audit_ticks"] <= 0:
                raise SystemExit("missing QK norm audit interval")
            for field in ATTENTION_AUDIT_FIELDS:
                if record[field] <= 0:
                    raise SystemExit(f"missing Attention audit interval: {field}")
    else:
        for field in STAGE_AUDIT_FIELDS + ("qk_norm_rope_audit_ticks",) + \
                ATTENTION_AUDIT_FIELDS:
            require(record, field, 0)

    if is_overlap:
        require(record, "attention_qk_norm_main_work_ticks", 0)
        if record["attention_qk_norm_worker_work_ticks"] <= 0:
            raise SystemExit("streamed QK norm/RoPE work is missing")
    elif is_qk:
        if record["attention_qk_norm_main_work_ticks"] <= 0 or \
                record["attention_qk_norm_worker_work_ticks"] <= 0:
            raise SystemExit("parallel QK norm/RoPE work is missing")
    else:
        require(record, "attention_qk_norm_main_work_ticks", 0)
        require(record, "attention_qk_norm_worker_work_ticks", 0)
    if is_softmax:
        if record["attention_softmax_main_work_ticks"] <= 0 or \
                record["attention_softmax_worker_work_ticks"] <= 0:
            raise SystemExit("parallel Softmax work is missing")
    else:
        require(record, "attention_softmax_main_work_ticks", 0)
        require(record, "attention_softmax_worker_work_ticks", 0)
    if is_gqa:
        if record["attention_gqa_pipeline_ticks"] <= 0 or \
                record["attention_gqa_worker_work_ticks"] <= 0 or \
                record["attention_gqa_hmx_wait_ticks"] <= 0 or \
                record["attention_gqa_queue_wait_ticks"] <= 0:
            raise SystemExit("GQA pipeline telemetry is missing")
        for field in (
                "attention_qk_pack_ticks", "attention_qk_hmx_ticks",
                "attention_qk_unpack_ticks", "attention_softmax_ticks",
                "attention_av_pack_ticks", "attention_av_hmx_ticks",
                "attention_av_unpack_ticks"):
            require(record, field, 0)
    else:
        require(record, "attention_gqa_pipeline_ticks", 0)
        require(record, "attention_gqa_worker_work_ticks", 0)
        require(record, "attention_gqa_hmx_wait_ticks", 0)
        require(record, "attention_gqa_queue_wait_ticks", 0)


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
    result["qknorm_attention_ticks_median_per_block"] = (
        result["qk_norm_rope_ticks_median_per_block"] +
        result["attention_ticks_median_per_block"])
    result["qkv_qknorm_attention_ticks_median_per_block"] = (
        result["qkv_projection_ticks_median_per_block"] +
        result["qknorm_attention_ticks_median_per_block"])
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
            qk = timing[family][f"{variant}_qk"]
            softmax = timing[family][f"{variant}_softmax"]
            if qk["qk_norm_rope_ticks_median_per_block"] >= \
                    control["qk_norm_rope_ticks_median_per_block"]:
                raise SystemExit("parallel QK norm/RoPE did not reduce its stage")
            if softmax["attention_softmax_ticks_median_per_block"] >= \
                    control["attention_softmax_ticks_median_per_block"]:
                raise SystemExit("parallel Softmax did not reduce its stage")
            accepted = timing[family][f"{variant}_gqa"]
            overlap = timing[family][f"{variant}_overlap"]
            if overlap["host_wall_ns_per_block_median"] > \
                    accepted["host_wall_ns_per_block_median"]:
                raise SystemExit("QKV overlap regressed accepted GQA Host wall")
            if overlap["qkv_qknorm_attention_ticks_median_per_block"] >= \
                    accepted["qkv_qknorm_attention_ticks_median_per_block"]:
                raise SystemExit("QKV overlap did not reduce its full scope")

    eligible = []
    scores = {}
    for candidate in ("parallel", "gqa", "overlap"):
        passes = True
        score = 0.0
        for family in ("repeat1", "repeat10"):
            for variant in ("f16", "w4"):
                control = timing[family][f"{variant}_control"]
                trial = timing[family][f"{variant}_{candidate}"]
                host_ratio = (trial["host_wall_ns_per_block_median"] /
                              control["host_wall_ns_per_block_median"])
                score += host_ratio
                if host_ratio > 1.0 or \
                        trial["qknorm_attention_ticks_median_per_block"] >= \
                        control["qknorm_attention_ticks_median_per_block"]:
                    passes = False
        scores[candidate] = score
        if passes:
            eligible.append(candidate)
    if not eligible:
        raise SystemExit("no Attention candidate passed the local gate")
    selected = min(eligible, key=lambda name: scores[name])
    for family in ("repeat1", "repeat10"):
        if timing[family][f"w4_{selected}"]["host_wall_ns_per_block_median"] >= \
                timing[family][f"f16_{selected}"]["host_wall_ns_per_block_median"]:
            raise SystemExit("selected W4F16 is not faster than F16F16")

    comparisons = {}
    for family in ("repeat1", "repeat10"):
        comparisons[family] = {}
        for variant in ("f16", "w4"):
            control = timing[family][f"{variant}_control"]
            comparisons[family][variant] = {}
            for candidate in ("qk", "softmax", "parallel", "gqa", "overlap"):
                trial = timing[family][f"{variant}_{candidate}"]
                comparisons[family][variant][candidate] = {
                    "host_wall_percent_vs_control": percent_change(
                        trial["host_wall_ns_per_block_median"],
                        control["host_wall_ns_per_block_median"]),
                    "qknorm_attention_percent_vs_control": percent_change(
                        trial["qknorm_attention_ticks_median_per_block"],
                        control["qknorm_attention_ticks_median_per_block"]),
                }
        comparisons[family]["selected_w4_vs_f16_percent"] = percent_change(
            timing[family][f"w4_{selected}"]["host_wall_ns_per_block_median"],
            timing[family][f"f16_{selected}"]["host_wall_ns_per_block_median"])

    summary = {
        "experiment": "EXP-0031",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "pass",
        "adoption_status": "pending",
        "accepted_checkpoint": "gqa",
        "continuation_adoption_status": "pending_user_decision",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "selected_candidate": selected,
        "eligible_candidates": eligible,
        "timing": timing,
        "comparisons": comparisons,
        "correctness": {
            "paired_output_byte_exact": True,
            "f16_output_hash": "704252c89780e695",
            "w4_output_hash": "f18b9abbe1487231",
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
