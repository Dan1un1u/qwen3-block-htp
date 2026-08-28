#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "w4_b4": {
        "variant": "W4F16", "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
        "mlp_mode": "crouton_native", "peak": 7_843_328,
        "commands": 256, "descriptors": 112, "w4_prefetch": 105,
        "f16_prefetch": 0, "streamed": 224, "gate_commands": 96,
        "down_hash": "b0d16655c164a22b",
        "output_hash": "f18b9abbe1487231", "direct": True,
    },
    "w4_b8": {
        "variant": "W4F16", "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
        "mlp_mode": "crouton_native_batch8", "peak": 8_171_008,
        "commands": 208, "descriptors": 112, "w4_prefetch": 105,
        "f16_prefetch": 0, "streamed": 176, "gate_commands": 48,
        "down_hash": "b0d16655c164a22b",
        "output_hash": "f18b9abbe1487231", "direct": True,
    },
    "f16_b4": {
        "variant": "F16F16", "f16_mode": "gate_up_batch4",
        "w4_mode": "not_applicable", "mlp_mode": "streaming",
        "peak": 7_072_512, "commands": 256, "descriptors": 224,
        "w4_prefetch": 0, "f16_prefetch": 217, "streamed": 0,
        "gate_commands": 0, "down_hash": "72076aa944fc97e4",
        "output_hash": "704252c89780e695", "direct": False,
    },
    "f16_b8": {
        "variant": "F16F16", "f16_mode": "gate_up_batch8",
        "w4_mode": "not_applicable", "mlp_mode": "crouton_native_batch8",
        "peak": 6_875_904, "commands": 208, "descriptors": 176,
        "w4_prefetch": 0, "f16_prefetch": 169, "streamed": 0,
        "gate_commands": 0, "down_hash": "72076aa944fc97e4",
        "output_hash": "704252c89780e695", "direct": True,
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
    is_w4 = config["variant"] == "W4F16"
    fixed = {
        "experiment": "EXP-0036",
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
        "attention_pipeline_mode": "gqa_qkv_overlap",
        "attention_hvx_contexts": 4,
        "mlp_mode": config["mlp_mode"],
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
        "w4f16_requested_hvx_workers": 3 if is_w4 else 2,
        "w4f16_region_tiles": 32,
        "w4f16_pool_status": 0,
        "f16f16_weight_batch_n_tiles": 0 if is_w4 else 2,
        "w4f16_active_worker_min": 2 if is_w4 else 0,
        "w4f16_active_worker_max": 3 if is_w4 else 0,
        "w4f16_effective_region_min": 32 if is_w4 else 0,
        "w4f16_effective_region_max": 96 if is_w4 else 0,
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
            config["descriptors"] * repeat + (7 if is_w4 else 0),
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": config["commands"] * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "w4f16_streamed_command_count": config["streamed"] * repeat,
        "w4f16_prefetch_count": config["w4_prefetch"] * repeat,
        "f16f16_prefetch_count": config["f16_prefetch"] * repeat,
        "w4f16_cross_prefetch_count": 5 * repeat if is_w4 else 0,
        "w4f16_early_region_command_count": 0,
        "w4f16_gate_up_hmx_command_count":
            config["gate_commands"] * repeat,
        "w4f16_gate_up_scale_cache_bytes": 98_304 if is_w4 else 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    require(record, "output_hash", config["output_hash"])
    require(record, "mlp_down_input_hash",
            config["down_hash"] if audit_mode == "on"
            else "0000000000000000")
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {name} {prefix}max_abs")
    for field in ("host_wall_ns", "host_wall_ns_per_block", "total_ticks",
                  "invocation_ticks", "gate_up_ticks", "down_ticks",
                  "weight_dma_ticks", "hmx_compute_ticks",
                  "mlp_stream_worker_work_ticks"):
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
    if audit_mode == "off":
        for field in STAGE_AUDIT_FIELDS:
            require(record, field, 0)
    else:
        for field in STAGE_AUDIT_FIELDS:
            if config["direct"] and field in (
                    "gate_up_audit_ticks", "activation_audit_ticks"):
                require(record, field, 0)
            elif record[field] <= 0:
                raise SystemExit(f"missing audit interval: {name}.{field}")
    if is_w4 and record["w4f16_gate_up_scale_init_ticks"] > 32 * repeat:
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
        "total_ticks", "invocation_ticks", "gate_up_ticks",
        "qkv_projection_ticks", "attention_ticks", "o_projection_ticks",
        "down_ticks", "projection_hmx_wait_ticks", "weight_dma_ticks",
        "w4f16_gate_up_prefetch_wait_ticks",
        "w4f16_gate_up_expand_ticks", "w4f16_gate_up_hmx_wait_ticks",
        "w4f16_gate_up_stream_work_ticks", "f16f16_prefetch_wait_ticks",
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
    fastest_f16 = {}
    for family in ("repeat1", "repeat10"):
        w4_b4 = timing[family]["w4_b4"]
        w4_b8 = timing[family]["w4_b8"]
        if w4_b8["gate_up_ticks_median_per_block"] >= \
                w4_b4["gate_up_ticks_median_per_block"]:
            raise SystemExit(f"{family}: W4 batch8 Gate/Up did not improve")
        if w4_b8["host_wall_ns_per_block_median"] >= \
                w4_b4["host_wall_ns_per_block_median"]:
            raise SystemExit(f"{family}: W4 batch8 Host wall did not improve")
        fastest_name = min(
            ("f16_b4", "f16_b8"),
            key=lambda name: timing[family][name]["host_wall_ns_per_block_median"])
        fastest_f16[family] = fastest_name
        f16 = timing[family][fastest_name]
        comparisons[family] = {
            "w4_b8_gate_up_percent_vs_w4_b4": pct(
                w4_b8["gate_up_ticks_median_per_block"],
                w4_b4["gate_up_ticks_median_per_block"]),
            "w4_b8_host_percent_vs_w4_b4": pct(
                w4_b8["host_wall_ns_per_block_median"],
                w4_b4["host_wall_ns_per_block_median"]),
            "w4_b8_gate_up_percent_vs_fastest_f16": pct(
                w4_b8["gate_up_ticks_median_per_block"],
                f16["gate_up_ticks_median_per_block"]),
            "w4_b8_host_percent_vs_fastest_f16": pct(
                w4_b8["host_wall_ns_per_block_median"],
                f16["host_wall_ns_per_block_median"]),
            "f16_b8_gate_up_percent_vs_f16_b4": pct(
                timing[family]["f16_b8"]["gate_up_ticks_median_per_block"],
                timing[family]["f16_b4"]["gate_up_ticks_median_per_block"]),
            "f16_b8_host_percent_vs_f16_b4": pct(
                timing[family]["f16_b8"]["host_wall_ns_per_block_median"],
                timing[family]["f16_b4"]["host_wall_ns_per_block_median"]),
        }

    print(json.dumps({
        "experiment": "EXP-0036",
        "stage": "B",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "pass",
        "adoption_status": "pending",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "selected_w4_candidate": "w4_b8",
        "fastest_f16_by_repeat": fastest_f16,
        "timing": timing,
        "comparisons": comparisons,
        "correctness": {
            "w4_down_input_byte_exact": True,
            "w4_output_byte_exact": True,
            "f16_down_input_byte_exact": True,
            "f16_output_byte_exact": True,
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "w4_b8_peak_vtcm_bytes": 8_171_008,
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
