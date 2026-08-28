#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "f16_batch2": {
        "variant": "F16F16",
        "f16_mode": "double_buffer_batch2",
        "w4_mode": "not_applicable",
        "workers": 0,
        "requested_workers": 2,
        "requested_region": 32,
        "region_min": 0,
        "region_max": 0,
        "vtcm": 7_072_512,
        "descriptors": lambda repeat: 320 * repeat,
        "f16_prefetches": lambda repeat: 313 * repeat,
        "w4_prefetches": lambda repeat: 0,
        "cross_prefetches": lambda repeat: 0,
        "active_min": 0,
        "active_max": 0,
    },
    "w4_exp27": {
        "variant": "W4F16",
        "f16_mode": "not_applicable",
        "w4_mode": "control",
        "workers": 2,
        "requested_workers": 2,
        "requested_region": 16,
        "region_min": 16,
        "region_max": 16,
        "vtcm": 7_744_512,
        "descriptors": lambda repeat: 160 * repeat + 7,
        "f16_prefetches": lambda repeat: 0,
        "w4_prefetches": lambda repeat: 153 * repeat,
        "cross_prefetches": lambda repeat: 0,
        "active_min": 2,
        "active_max": 2,
    },
    "w4_tuned32": {
        "variant": "W4F16",
        "f16_mode": "not_applicable",
        "w4_mode": "control",
        "workers": 2,
        "requested_workers": 2,
        "requested_region": 32,
        "region_min": 32,
        "region_max": 32,
        "vtcm": 7_744_512,
        "descriptors": lambda repeat: 160 * repeat + 7,
        "f16_prefetches": lambda repeat: 0,
        "w4_prefetches": lambda repeat: 153 * repeat,
        "cross_prefetches": lambda repeat: 0,
        "active_min": 2,
        "active_max": 2,
    },
    "w4_adaptive96": {
        "variant": "W4F16",
        "f16_mode": "not_applicable",
        "w4_mode": "adaptive_down96_cross_prefetch",
        "workers": 3,
        "requested_workers": 3,
        "requested_region": 32,
        "region_min": 32,
        "region_max": 96,
        "vtcm": 7_744_512,
        "descriptors": lambda repeat: 160 * repeat + 7,
        "f16_prefetches": lambda repeat: 0,
        "w4_prefetches": lambda repeat: 153 * repeat,
        "cross_prefetches": lambda repeat: 6 * repeat,
        "active_min": 2,
        "active_max": 3,
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
    config = CONFIGS[config_name]
    variant = config["variant"]
    fixed = {
        "experiment": "EXP-0028",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "f16f16_projection_mode": config["f16_mode"],
        "w4f16_pipeline_mode": config["w4_mode"],
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
        "w4f16_hvx_workers_created": config["workers"],
        "w4f16_hvx_workers_locked": config["workers"],
        "w4f16_requested_hvx_workers": config["requested_workers"],
        "w4f16_region_tiles": config["requested_region"],
        "w4f16_pool_status": 0,
        "w4f16_active_worker_min": config["active_min"],
        "w4f16_active_worker_max": config["active_max"],
        "w4f16_effective_region_min": config["region_min"],
        "w4f16_effective_region_max": config["region_max"],
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": config["vtcm"],
        "block_invocation_count": repeat,
        "weight_ddr_read_bytes": expected_weight_bytes(variant, repeat),
        "weight_dma_descriptor_count": config["descriptors"](repeat),
        "boundary_ddr_read_bytes": 262_144 * repeat + 41_472,
        "boundary_ddr_write_bytes": 262_144,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "hmx_command_count": 352 * repeat,
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "f16f16_weight_batch_n_tiles": 2 if variant == "F16F16" else 0,
        "f16f16_prefetch_count": config["f16_prefetches"](repeat),
        "w4f16_prefetch_count": config["w4_prefetches"](repeat),
        "w4f16_cross_prefetch_count":
            config["cross_prefetches"](repeat),
        "w4f16_early_region_command_count": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, value in fixed.items():
        require(record, field, value)

    if variant == "W4F16":
        require(record, "w4f16_scale_placement", "hmx_output_per_channel")
        require(record, "w4f16_streamed_command_count", 320 * repeat)
        require(record, "f16f16_prefetch_wait_ticks", 0)
        if record["w4f16_expand_ticks"] <= 0:
            raise SystemExit("W4 expansion was not observed")
    else:
        require(record, "w4f16_scale_placement", "not_applicable")
        require(record, "w4f16_streamed_command_count", 0)
        require(record, "w4f16_expand_ticks", 0)
        require(record, "w4f16_prefetch_wait_ticks", 0)

    if config_name == "w4_adaptive96":
        if record["w4f16_cross_prefetch_lifetime_ticks"] <= 0:
            raise SystemExit("cross-projection prefetch lifetime missing")
    else:
        require(record, "w4f16_cross_prefetch_wait_ticks", 0)
        require(record, "w4f16_cross_prefetch_lifetime_ticks", 0)

    positive = (
        "warmup_host_wall_ns", "host_wall_ns", "host_wall_ns_per_block",
        "total_ticks", "invocation_ticks", "input_stage_ticks",
        "metadata_stage_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "o_projection_ticks", "post_attention_residual_ticks",
        "gate_up_ticks", "activation_ticks", "down_ticks",
        "final_residual_ticks", "output_stage_ticks", "weight_dma_ticks",
        "hmx_compute_ticks", "projection_hmx_wait_ticks",
        "projection_unpack_ticks",
    )
    for field in positive:
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {config_name}.{field}: {value!r}")

    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {config_name} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {config_name} {prefix}max_abs")
    output_hash = record.get("output_hash")
    if not isinstance(output_hash, str) or len(output_hash) != 16:
        raise SystemExit(f"invalid output hash: {output_hash!r}")

    for field in TIMING_FIELDS:
        value = record.get(field)
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"invalid timing field {field}: {value!r}")
    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] + record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(
            record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("top-level named timing sum is inconsistent")
    if record["ledger_unattributed_ticks"] / record["invocation_ticks"] > 0.01:
        raise SystemExit("top-level unattributed gap exceeds one percent")
    if sum(record[field] for field in ATTENTION_FIELDS) != record["attention_ticks"]:
        raise SystemExit("nested Attention ledger does not close")
    if record["attention_unattributed_ticks"] / record["attention_ticks"] > 0.01:
        raise SystemExit("Attention unattributed gap exceeds one percent")

    if audit_mode == "on":
        for field in AUDIT_FIELDS:
            if record[field] <= 0:
                raise SystemExit(f"missing audit interval: {field}")
    else:
        for field in AUDIT_FIELDS:
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
    result["projection_ticks_median_per_block"] = sum(
        result[f"{field}_median_per_block"] for field in (
            "qkv_projection_ticks", "o_projection_ticks",
            "gate_up_ticks", "down_ticks"))
    return result


def percent_change(candidate, control):
    return (candidate / control - 1.0) * 100.0


def faster_in_both(candidate, control, timing):
    return all(
        timing[family][candidate]["host_wall_ns_per_block_median"] <
        timing[family][control]["host_wall_ns_per_block_median"]
        for family in ("repeat1", "repeat10"))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package_dir = pathlib.Path(sys.argv[2])
    manifest = audit_package(package_dir)
    timing = {"repeat1": {}, "repeat10": {}}
    hashes = {"F16F16": set(), "W4F16": set()}
    record_count = 0

    for name, config in CONFIGS.items():
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
        for record in correctness + repeat1 + repeat10:
            hashes[config["variant"]].add(record["output_hash"])
        timing["repeat1"][name] = medians(repeat1)
        timing["repeat10"][name] = medians(repeat10)
        record_count += len(correctness) + len(repeat1) + len(repeat10)

    if len(hashes["F16F16"]) != 1 or len(hashes["W4F16"]) != 1:
        raise SystemExit(f"variant output hashes are unstable: {hashes}")

    eligible = [name for name in ("w4_tuned32", "w4_adaptive96")
                if faster_in_both(name, "w4_exp27", timing)]
    selected = min(
        eligible,
        key=lambda name: timing["repeat10"][name][
            "host_wall_ns_per_block_median"],
        default=None)
    if selected is None:
        raise SystemExit("no W4 candidate improves both timing families")

    local_gate = all(
        timing[family][selected]["host_wall_ns_per_block_median"] <
            timing[family]["f16_batch2"]["host_wall_ns_per_block_median"] and
        timing[family][selected]["projection_ticks_median_per_block"] <
            timing[family]["f16_batch2"]["projection_ticks_median_per_block"]
        for family in ("repeat1", "repeat10"))
    selected10 = timing["repeat10"][selected]
    f16_10 = timing["repeat10"]["f16_batch2"]
    w4_control10 = timing["repeat10"]["w4_exp27"]
    projection_change = percent_change(
        selected10["projection_ticks_median_per_block"],
        f16_10["projection_ticks_median_per_block"])
    host_change = percent_change(
        selected10["host_wall_ns_per_block_median"],
        f16_10["host_wall_ns_per_block_median"])

    summary = {
        "experiment": "EXP-0028",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "zero_intermediate_ddr_gate": True,
        "variant_output_hash_stability_gate": True,
        "audit_free_performance_mode_gate": True,
        "top_level_ledger_closure_gate": True,
        "attention_ledger_closure_gate": True,
        "static_w4_pipeline_gate": True,
        "selected_w4_plan": selected,
        "selected_w4_improves_exp0027_control_both_repeats": True,
        "selected_w4_repeat10_control_host_change_percent": percent_change(
            selected10["host_wall_ns_per_block_median"],
            w4_control10["host_wall_ns_per_block_median"]),
        "selected_w4_repeat10_control_projection_change_percent":
            percent_change(
                selected10["projection_ticks_median_per_block"],
                w4_control10["projection_ticks_median_per_block"]),
        "selected_w4_repeat10_f16_host_change_percent": host_change,
        "selected_w4_repeat10_f16_projection_change_percent":
            projection_change,
        "stretch_projection_20_percent_gate": projection_change <= -20.0,
        "stretch_complete_block_5_percent_gate": host_change <= -5.0,
        "local_gate": "pass" if local_gate else "fail",
        "timing": timing,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if local_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
