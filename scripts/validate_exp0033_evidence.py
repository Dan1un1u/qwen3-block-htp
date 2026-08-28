#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "f16_control": {
        "variant": "F16F16", "attention": "gqa_qkv_overlap",
        "f16_mode": "gate_up_batch4", "w4_mode": "not_applicable",
        "workers": 2, "peak": 7_072_512, "commands": 256,
        "descriptors": 224, "prefetch": 217, "dynamic": False,
    },
    "w4_control": {
        "variant": "W4F16", "attention": "gqa_qkv_overlap",
        "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
        "workers": 3, "peak": 7_843_328, "commands": 256,
        "descriptors": 112, "prefetch": 105, "dynamic": False,
    },
    "w4_dynamic": {
        "variant": "W4F16", "attention": "gqa_qkv_dynamic",
        "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
        "workers": 3, "peak": 7_843_328, "commands": 256,
        "descriptors": 112, "prefetch": 105, "dynamic": True,
    },
}

TOP_LEVEL_FIELDS = (
    "runtime_setup_ticks", "metadata_stage_ticks", "input_stage_ticks",
    "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
    "attention_ticks", "o_projection_ticks",
    "post_attention_residual_ticks", "post_attention_norm_ticks",
    "gate_up_ticks", "activation_ticks", "down_ticks",
    "final_residual_ticks", "output_stage_ticks", "runtime_teardown_ticks",
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
STAGE_AUDIT_FIELDS = (
    "input_norm_audit_ticks", "qkv_audit_ticks", "o_projection_audit_ticks",
    "post_attention_residual_audit_ticks", "post_attention_norm_audit_ticks",
    "gate_up_audit_ticks", "activation_audit_ticks", "down_audit_ticks",
    "final_residual_audit_ticks",
)


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


def expected_weight_bytes(config, repeat):
    if config["variant"] == "F16F16":
        return 100_663_296 * repeat
    return 25_165_824 * repeat + 131_072


def validate_record(record, name, repeat, audit_mode):
    config = CONFIGS[name]
    is_f16 = config["variant"] == "F16F16"
    fixed = {
        "experiment": "EXP-0033",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": config["variant"],
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "f16f16_projection_mode": config["f16_mode"],
        "w4f16_pipeline_mode": config["w4_mode"],
        "attention_pack_mode": "combined_hvx",
        "attention_pipeline_mode": config["attention"],
        "attention_hvx_contexts": 4,
        "mlp_mode": "streaming",
        "mlp_hvx_contexts": 4,
        "mlp_chunk_vectors": 64,
        "intermediate_residency": "VTCM",
        "warmup_rpc_result": 0,
        "warmup_prepared_session_run_index": 1,
        "repeat_count": repeat,
        "prepared_session_run_index": 2,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "projection_failure_result": 0,
        "w4f16_expand_mismatch_count": 0,
        "w4f16_requested_hvx_workers": config["workers"],
        "w4f16_region_tiles": 32,
        "w4f16_pool_status": 0,
        "mlp_hvx_workers_created": 3,
        "mlp_hvx_workers_locked": 3,
        "mlp_pool_status": 0,
        "mlp_stream_group_count": 96 * repeat,
        "mlp_down_pack_skipped": repeat,
        "attention_hvx_workers_created": 3,
        "attention_hvx_workers_locked": 3,
        "attention_pool_status": 0,
        "attention_qk_norm_task_count": 24 * repeat,
        "attention_softmax_task_count": 0,
        "attention_gqa_group_count": 8 * repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": config["peak"],
        "block_invocation_count": repeat,
        "weight_ddr_read_bytes": expected_weight_bytes(config, repeat),
        "weight_dma_descriptor_count":
            config["descriptors"] * repeat + (0 if is_f16 else 7),
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": config["commands"] * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "w4f16_streamed_command_count": 0 if is_f16 else 224 * repeat,
        "w4f16_prefetch_count": 0 if is_f16 else config["prefetch"] * repeat,
        "f16f16_prefetch_count": config["prefetch"] * repeat if is_f16 else 0,
        "w4f16_cross_prefetch_count": 0 if is_f16 else 5 * repeat,
        "w4f16_early_region_command_count": 0,
        "w4f16_gate_up_hmx_command_count": 0 if is_f16 else 96 * repeat,
        "w4f16_gate_up_scale_cache_bytes": 0 if is_f16 else 98_304,
        "qkv_dynamic_enabled": 1 if config["dynamic"] else 0,
        "qkv_dynamic_rebalanced_expand_calls": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    require(record, "output_hash",
            "704252c89780e695" if is_f16 else "f18b9abbe1487231")
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {name} {prefix}max_abs")
    for field in ("host_wall_ns", "host_wall_ns_per_block", "total_ticks",
                  "invocation_ticks", "qkv_projection_ticks",
                  "attention_ticks", "gate_up_ticks", "down_ticks"):
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {name}.{field}: {value!r}")
    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("top-level named timing sum is inconsistent")
    if sum(record[field] for field in ATTENTION_FIELDS) != record["attention_ticks"]:
        raise SystemExit("nested Attention ledger does not close")
    if audit_mode == "on":
        for field in STAGE_AUDIT_FIELDS:
            if record[field] <= 0:
                raise SystemExit(f"missing audit interval: {field}")
    else:
        for field in STAGE_AUDIT_FIELDS:
            require(record, field, 0)

    dynamic_fields = (
        "qkv_dynamic_steal_attempts", "qkv_dynamic_steal_successes",
        "qkv_dynamic_main_norm_tasks", "qkv_dynamic_main_norm_ticks",
    )
    if config["dynamic"]:
        if record["qkv_dynamic_steal_attempts"] <= 0:
            raise SystemExit("dynamic scheduler did not attempt work stealing")
        if record["qkv_dynamic_steal_successes"] <= 0:
            raise SystemExit("dynamic scheduler did not steal a ready task")
        require(record, "qkv_dynamic_main_norm_tasks",
                record["qkv_dynamic_steal_successes"])
        if record["qkv_dynamic_main_norm_ticks"] <= 0:
            raise SystemExit("dynamic main-context Norm work was not timed")
    else:
        for field in dynamic_fields:
            require(record, field, 0)


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


def summarize(records):
    repeat = records[0]["repeat_count"]
    fields = (
        "total_ticks", "qkv_projection_ticks", "attention_ticks",
        "gate_up_ticks", "down_ticks", "projection_hmx_wait_ticks",
        "w4f16_expand_pool_wait_ticks",
        "attention_qk_norm_pool_wait_ticks",
        "qkv_dynamic_steal_attempts", "qkv_dynamic_steal_successes",
        "qkv_dynamic_main_norm_ticks",
    )
    result = {
        "host_wall_ns_per_block_median": statistics.median(
            record["host_wall_ns_per_block"] for record in records),
        "output_hash": records[0]["output_hash"],
    }
    for field in fields:
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / repeat for record in records)
    return result


def pct(candidate, control):
    return (candidate / control - 1.0) * 100.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    manifest = audit_package(pathlib.Path(sys.argv[2]))
    timing = {"repeat1": {}, "repeat10": {}}
    record_count = 0

    for name in CONFIGS:
        correctness = load_jsonl(result_dir / f"correctness_{name}.jsonl")
        repeat1 = load_jsonl(result_dir / f"timing_{name}_repeat1.jsonl")
        repeat10 = load_jsonl(result_dir / f"timing_{name}_repeat10.jsonl")
        if len(correctness) != 1 or len(repeat1) != 5 or len(repeat10) != 5:
            raise SystemExit(f"wrong evidence matrix size for {name}")
        validate_record(correctness[0], name, 1, "on")
        for record in repeat1:
            validate_record(record, name, 1, "off")
        for record in repeat10:
            validate_record(record, name, 10, "off")
        timing["repeat1"][name] = summarize(repeat1)
        timing["repeat10"][name] = summarize(repeat10)
        record_count += 11

    comparisons = {}
    gate_pass = True
    for family in ("repeat1", "repeat10"):
        candidate = timing[family]["w4_dynamic"]
        w4_control = timing[family]["w4_control"]
        f16_control = timing[family]["f16_control"]
        scope_pass = (
            candidate["qkv_projection_ticks_median_per_block"] <
                w4_control["qkv_projection_ticks_median_per_block"] and
            candidate["host_wall_ns_per_block_median"] <
                w4_control["host_wall_ns_per_block_median"] and
            candidate["host_wall_ns_per_block_median"] <
                f16_control["host_wall_ns_per_block_median"]
        )
        gate_pass = gate_pass and scope_pass
        comparisons[family] = {
            "scope_gate_pass": scope_pass,
            "qkv_percent_vs_w4_control": pct(
                candidate["qkv_projection_ticks_median_per_block"],
                w4_control["qkv_projection_ticks_median_per_block"]),
            "host_percent_vs_w4_control": pct(
                candidate["host_wall_ns_per_block_median"],
                w4_control["host_wall_ns_per_block_median"]),
            "host_percent_vs_f16_control": pct(
                candidate["host_wall_ns_per_block_median"],
                f16_control["host_wall_ns_per_block_median"]),
            "qk_norm_join_wait_change_ticks": (
                candidate["attention_qk_norm_pool_wait_ticks_median_per_block"] -
                w4_control["attention_qk_norm_pool_wait_ticks_median_per_block"]),
            "projection_hmx_wait_change_ticks": (
                candidate["projection_hmx_wait_ticks_median_per_block"] -
                w4_control["projection_hmx_wait_ticks_median_per_block"]),
        }

    print(json.dumps({
        "experiment": "EXP-0033",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "pass" if gate_pass else "fail",
        "adoption_status": "pending" if gate_pass else "rejected",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "candidate": "w4_dynamic",
        "timing": timing,
        "comparisons": comparisons,
        "correctness": {
            "candidate_byte_exact": True,
            "w4_output_hash": "f18b9abbe1487231",
            "f16_output_hash": "704252c89780e695",
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "candidate_peak_vtcm_bytes": 7_843_328,
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
