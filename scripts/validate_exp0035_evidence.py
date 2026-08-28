#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "f16_control": {
        "variant": "F16F16",
        "f16_mode": "gate_up_batch4",
        "w4_mode": "not_applicable",
        "workers": 2,
        "peak": 7_072_512,
        "output_hash": "704252c89780e695",
        "qkv_weight_bytes": [8_388_608, 4_194_304, 4_194_304],
        "qkv_descriptors": [32, 16, 16],
        "w4": False,
        "q_inbound": False,
        "cross_prefetches": 0,
    },
    "w4_control": {
        "variant": "W4F16",
        "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_prefetch",
        "workers": 3,
        "peak": 7_843_328,
        "output_hash": "f18b9abbe1487231",
        "qkv_weight_bytes": [2_228_224, 1_048_576, 1_048_576],
        "qkv_descriptors": [17, 8, 8],
        "w4": True,
        "q_inbound": False,
        "cross_prefetches": 5,
    },
    "w4_q_inbound_prefetch": {
        "variant": "W4F16",
        "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_gate4_dma8_cross_q_prefetch",
        "workers": 3,
        "peak": 7_843_328,
        "output_hash": "f18b9abbe1487231",
        "qkv_weight_bytes": [2_097_152, 1_048_576, 1_048_576],
        "qkv_descriptors": [16, 8, 8],
        "w4": True,
        "q_inbound": True,
        "cross_prefetches": 6,
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
ARRAY_FIELDS = (
    "qkv_projection_wall_ticks", "qkv_weight_dma_ticks", "qkv_pack_ticks",
    "qkv_w4_expand_ticks", "qkv_w4_expand_work_ticks",
    "qkv_w4_expand_pool_wait_ticks", "qkv_hmx_compute_ticks",
    "qkv_hmx_wait_ticks", "qkv_hmx_ready_wait_ticks",
    "qkv_prefetch_wait_ticks", "qkv_unpack_ticks",
    "qkv_weight_ddr_read_bytes", "qkv_weight_dma_descriptor_count",
    "qkv_hmx_command_count", "qkv_hmx_fp16_tile_pair_count",
    "qkv_w4_streamed_command_count", "qkv_w4_expand_region_count",
    "qkv_w4_prefetch_count", "qkv_f16_prefetch_count",
)
Q_INBOUND_FIELDS = (
    "q_inbound_prefetch_enabled",
    "q_inbound_prefetch_start_count",
    "q_inbound_prefetch_completion_count",
    "q_inbound_prefetch_consume_count",
    "q_inbound_prefetch_bytes",
    "q_inbound_prefetch_descriptor_count",
    "q_inbound_prefetch_lifetime_ticks",
    "q_inbound_prefetch_wait_ticks",
    "q_inbound_prefetch_overlap_window_ticks",
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


def scale(values, repeat):
    return [value * repeat for value in values]


def validate_record(record, name, repeat, audit_mode):
    config = CONFIGS[name]
    is_w4 = config["w4"]
    fixed = {
        "experiment": "EXP-0035",
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
        "mlp_mode": "streaming",
        "mlp_hvx_contexts": 4,
        "mlp_chunk_vectors": 64,
        "intermediate_residency": "VTCM",
        "repeat_count": repeat,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "projection_failure_result": 0,
        "w4f16_expand_mismatch_count": 0,
        "w4f16_requested_hvx_workers": config["workers"],
        "w4f16_region_tiles": 32,
        "w4f16_pool_status": 0,
        "attention_qk_norm_task_count": 24 * repeat,
        "attention_softmax_task_count": 0,
        "attention_gqa_group_count": 8 * repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": config["peak"],
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": 256 * repeat,
        "qkv_attribution_version": 1,
        "release_result": 0,
        "close_result": 0,
        "output_hash": config["output_hash"],
        "mismatches": 0,
        "max_lsb": 0,
        "w4f16_cross_prefetch_count": config["cross_prefetches"] * repeat,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {name} {prefix}max_abs")

    for field in ARRAY_FIELDS:
        values = record.get(field)
        if not isinstance(values, list) or len(values) != 3:
            raise SystemExit(f"invalid {name}.{field} shape")
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise SystemExit(f"invalid {name}.{field} values: {values!r}")
    if any(value <= 0 for value in record["qkv_projection_wall_ticks"]):
        raise SystemExit("missing positive Q/K/V wall interval")
    if (sum(record["qkv_projection_wall_ticks"]) +
            record["qkv_projection_unattributed_ticks"] !=
            record["qkv_projection_ticks"]):
        raise SystemExit("QKV wall ledger does not close")

    require(record, "qkv_hmx_command_count", scale([32, 16, 16], repeat))
    require(record, "qkv_hmx_fp16_tile_pair_count",
            scale([8192, 4096, 4096], repeat))
    require(record, "qkv_weight_ddr_read_bytes",
            scale(config["qkv_weight_bytes"], repeat))
    require(record, "qkv_weight_dma_descriptor_count",
            scale(config["qkv_descriptors"], repeat))

    if is_w4:
        require(record, "weight_ddr_read_bytes",
                131_072 + 25_165_824 * repeat)
        require(record, "weight_dma_descriptor_count", 7 + 112 * repeat)
        require(record, "qkv_w4_streamed_command_count",
                scale([32, 16, 16], repeat))
        require(record, "qkv_w4_expand_region_count",
                scale([64, 32, 32], repeat))
        require(record, "qkv_w4_prefetch_count", scale([15, 7, 7], repeat))
        require(record, "qkv_f16_prefetch_count", [0, 0, 0])
        for field in ("qkv_w4_expand_ticks", "qkv_w4_expand_work_ticks",
                      "qkv_hmx_wait_ticks"):
            if any(value <= 0 for value in record[field]):
                raise SystemExit(f"missing W4 work in {field}")
    else:
        for field in ("qkv_w4_streamed_command_count",
                      "qkv_w4_expand_region_count", "qkv_w4_prefetch_count",
                      "qkv_w4_expand_ticks", "qkv_w4_expand_work_ticks",
                      "qkv_w4_expand_pool_wait_ticks",
                      "qkv_prefetch_wait_ticks"):
            require(record, field, [0, 0, 0])
        require(record, "qkv_f16_prefetch_count", scale([31, 15, 15], repeat))

    if config["q_inbound"]:
        require(record, "q_inbound_prefetch_enabled", 1)
        for field in ("q_inbound_prefetch_start_count",
                      "q_inbound_prefetch_completion_count",
                      "q_inbound_prefetch_consume_count",
                      "q_inbound_prefetch_descriptor_count"):
            require(record, field, repeat)
        require(record, "q_inbound_prefetch_bytes", 131_072 * repeat)
        lifetime = record["q_inbound_prefetch_lifetime_ticks"]
        wait = record["q_inbound_prefetch_wait_ticks"]
        overlap = record["q_inbound_prefetch_overlap_window_ticks"]
        if lifetime <= 0 or overlap <= 0 or wait < 0:
            raise SystemExit("missing Q inbound prefetch timing evidence")
        if lifetime < overlap or lifetime < wait:
            raise SystemExit("invalid Q inbound prefetch timing relationship")
    else:
        for field in Q_INBOUND_FIELDS:
            require(record, field, 0)

    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(
            record[field] for field in TOP_LEVEL_FIELDS):
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
    for field in ("host_wall_ns_per_block", "total_ticks",
                  "qkv_projection_ticks"):
        value = record[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {name}.{field}")


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


def median_per_block(records, field):
    repeat = records[0]["repeat_count"]
    return statistics.median(record[field] / repeat for record in records)


def median_vector(records, field):
    repeat = records[0]["repeat_count"]
    return [statistics.median(record[field][index] / repeat
                              for record in records)
            for index in range(3)]


def summarize(records):
    result = {
        "host_wall_ns_per_block_median": statistics.median(
            record["host_wall_ns_per_block"] for record in records),
        "input_norm_ticks_median_per_block": median_per_block(
            records, "input_norm_ticks"),
        "qkv_projection_ticks_median_per_block": median_per_block(
            records, "qkv_projection_ticks"),
        "qkv_projection_unattributed_ticks_median_per_block": median_per_block(
            records, "qkv_projection_unattributed_ticks"),
        "q_inbound_prefetch_lifetime_ticks_median_per_block": median_per_block(
            records, "q_inbound_prefetch_lifetime_ticks"),
        "q_inbound_prefetch_wait_ticks_median_per_block": median_per_block(
            records, "q_inbound_prefetch_wait_ticks"),
        "q_inbound_prefetch_overlap_window_ticks_median_per_block": median_per_block(
            records, "q_inbound_prefetch_overlap_window_ticks"),
    }
    for field in ARRAY_FIELDS:
        result[f"{field}_median_per_block"] = median_vector(records, field)
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
        f16 = timing[family]["f16_control"]
        control = timing[family]["w4_control"]
        candidate = timing[family]["w4_q_inbound_prefetch"]
        f16_host = f16["host_wall_ns_per_block_median"]
        control_host = control["host_wall_ns_per_block_median"]
        candidate_host = candidate["host_wall_ns_per_block_median"]
        control_q = control["qkv_projection_wall_ticks_median_per_block"][0]
        candidate_q = candidate["qkv_projection_wall_ticks_median_per_block"][0]
        q_reduced = candidate_q < control_q
        host_reduced = candidate_host < control_host
        faster_than_f16 = candidate_host < f16_host
        family_pass = q_reduced and host_reduced and faster_than_f16
        gate_pass = gate_pass and family_pass
        comparisons[family] = {
            "candidate_minus_control_q_ticks": candidate_q - control_q,
            "candidate_q_percent_vs_control": pct(candidate_q, control_q),
            "candidate_minus_control_host_ns": candidate_host - control_host,
            "candidate_host_percent_vs_control": pct(candidate_host, control_host),
            "candidate_host_percent_vs_f16": pct(candidate_host, f16_host),
            "q_wall_reduced": q_reduced,
            "complete_block_host_wall_reduced": host_reduced,
            "candidate_faster_than_f16": faster_than_f16,
            "family_gate_pass": family_pass,
        }

    print(json.dumps({
        "experiment": "EXP-0035",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gate": "pass" if gate_pass else "fail",
        "adoption_status": "pending_user_decision" if gate_pass else "rejected",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "candidate": "q_inbound_prefetch_over_input_rmsnorm",
        "timing": timing,
        "comparisons": comparisons,
        "correctness": {
            "candidate_byte_identical_to_w4_control": True,
            "w4_output_hash": "f18b9abbe1487231",
            "f16_output_hash": "704252c89780e695",
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "w4_peak_vtcm_bytes": 7_843_328,
            "candidate_weight_bytes_equal_control": True,
            "candidate_weight_descriptors_equal_control": True,
            "candidate_hmx_commands_equal_control": True,
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
