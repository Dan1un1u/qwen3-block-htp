#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "control": {
        "qkv_mode": "batch2_control",
        "hmx_commands": 208,
        "streamed_commands": 176,
        "q_batch": 2,
        "q_commands": 32,
    },
    "q_batch4": {
        "qkv_mode": "q_batch4",
        "hmx_commands": 192,
        "streamed_commands": 160,
        "q_batch": 4,
        "q_commands": 16,
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
STAGE_AUDIT_FIELDS = (
    "input_norm_audit_ticks", "qkv_audit_ticks", "o_projection_audit_ticks",
    "post_attention_residual_audit_ticks", "post_attention_norm_audit_ticks",
    "gate_up_audit_ticks", "activation_audit_ticks", "down_audit_ticks",
    "final_residual_audit_ticks",
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
POSITIVE_TIMING_FIELDS = (
    "host_wall_ns", "host_wall_ns_per_block", "total_ticks",
    "invocation_ticks", "qkv_projection_ticks", "q_projection_wall_ticks",
    "k_projection_wall_ticks", "v_projection_wall_ticks",
    "q_weight_dma_ticks", "k_weight_dma_ticks", "v_weight_dma_ticks",
    "q_expand_ticks", "k_expand_ticks", "v_expand_ticks",
    "q_expand_work_ticks", "k_expand_work_ticks", "v_expand_work_ticks",
    "q_hmx_wait_ticks", "k_hmx_wait_ticks", "v_hmx_wait_ticks",
    "q_unpack_ticks", "k_unpack_ticks", "v_unpack_ticks",
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


def validate_record(record, name, repeat, audit_mode):
    config = CONFIGS[name]
    fixed = {
        "experiment": "EXP-0037",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4F16",
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "f16f16_projection_mode": "not_applicable",
        "qkv_batch_mode": config["qkv_mode"],
        "w4f16_pipeline_mode":
            "adaptive_down96_gate4_dma8_cross_prefetch",
        "attention_pack_mode": "combined_hvx",
        "attention_pipeline_mode": "gqa_qkv_overlap",
        "attention_hvx_contexts": 4,
        "mlp_mode": "crouton_native_batch8",
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
        "w4f16_requested_hvx_workers": 3,
        "w4f16_region_tiles": 32,
        "w4f16_pool_status": 0,
        "f16f16_weight_batch_n_tiles": 0,
        "w4f16_active_worker_min": 2,
        "w4f16_active_worker_max": 3,
        "w4f16_effective_region_min": 32,
        "w4f16_effective_region_max": 96,
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
        "q_hmx_batch_n_tiles": config["q_batch"],
        "k_hmx_batch_n_tiles": 2,
        "v_hmx_batch_n_tiles": 2,
        "q_hmx_command_count": config["q_commands"] * repeat,
        "k_hmx_command_count": 16 * repeat,
        "v_hmx_command_count": 16 * repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": 8_171_008,
        "block_invocation_count": repeat,
        "weight_ddr_read_bytes": 25_165_824 * repeat + 131_072,
        "weight_dma_descriptor_count": 112 * repeat + 7,
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": config["hmx_commands"] * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "w4f16_streamed_command_count":
            config["streamed_commands"] * repeat,
        "w4f16_prefetch_count": 105 * repeat,
        "f16f16_prefetch_count": 0,
        "w4f16_cross_prefetch_count": 5 * repeat,
        "w4f16_early_region_command_count": 0,
        "w4f16_gate_up_hmx_command_count": 48 * repeat,
        "w4f16_gate_up_scale_cache_bytes": 98_304,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    require(record, "output_hash", "f18b9abbe1487231")
    require(record, "mlp_down_input_hash",
            "b0d16655c164a22b" if audit_mode == "on"
            else "0000000000000000")
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {name} {prefix}max_abs")
    for field in POSITIVE_TIMING_FIELDS:
        value = record.get(field)
        if (not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise SystemExit(f"invalid positive {name}.{field}: {value!r}")
    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(
            record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("top-level named timing sum is inconsistent")
    if sum(record[field] for field in ATTENTION_FIELDS) != record["attention_ticks"]:
        raise SystemExit("nested Attention ledger does not close")
    if audit_mode == "off":
        for field in STAGE_AUDIT_FIELDS:
            require(record, field, 0)
    else:
        for field in STAGE_AUDIT_FIELDS:
            if field in ("gate_up_audit_ticks", "activation_audit_ticks"):
                require(record, field, 0)
            elif record[field] <= 0:
                raise SystemExit(f"missing audit interval: {name}.{field}")
    if record["w4f16_gate_up_scale_init_ticks"] > 32 * repeat:
        raise SystemExit("cached Gate/Up scales were rebuilt in the hot path")


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
    fields = (
        "total_ticks", "invocation_ticks", "qkv_projection_ticks",
        "q_projection_wall_ticks", "k_projection_wall_ticks",
        "v_projection_wall_ticks", "q_weight_dma_ticks",
        "q_expand_ticks", "q_expand_work_ticks", "q_hmx_wait_ticks",
        "q_unpack_ticks", "attention_ticks", "gate_up_ticks", "down_ticks",
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
        timing["repeat1"][name] = medians(repeat1)
        timing["repeat10"][name] = medians(repeat10)
        record_count += 11

    comparisons = {}
    gate_failures = []
    for family in ("repeat1", "repeat10"):
        control = timing[family]["control"]
        candidate = timing[family]["q_batch4"]
        if candidate["q_projection_wall_ticks_median_per_block"] >= \
                control["q_projection_wall_ticks_median_per_block"]:
            gate_failures.append(
                f"{family}: Q projection wall did not improve")
        if candidate["host_wall_ns_per_block_median"] >= \
                control["host_wall_ns_per_block_median"]:
            gate_failures.append(
                f"{family}: complete-block Host wall did not improve")
        comparisons[family] = {
            "q_projection_wall_percent_vs_control": pct(
                candidate["q_projection_wall_ticks_median_per_block"],
                control["q_projection_wall_ticks_median_per_block"]),
            "host_wall_percent_vs_control": pct(
                candidate["host_wall_ns_per_block_median"],
                control["host_wall_ns_per_block_median"]),
        }

    print(json.dumps({
        "experiment": "EXP-0037",
        "stage": "A",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "fail" if gate_failures else "pass",
        "adoption_status": "pending",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "tested_candidate": "q_batch4",
        "selected_candidate": None if gate_failures else "q_batch4",
        "gate_failures": gate_failures,
        "timing": timing,
        "comparisons": comparisons,
        "correctness": {
            "control_byte_exact": True,
            "candidate_byte_exact": True,
            "shared_output_hash": "f18b9abbe1487231",
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "peak_vtcm_bytes": 8_171_008,
            "q_hmx_commands_control": 32,
            "q_hmx_commands_candidate": 16,
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
