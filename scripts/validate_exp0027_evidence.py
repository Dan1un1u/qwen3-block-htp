#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


CONFIGS = {
    "f16_serial": {
        "variant": "F16F16", "mode": "serial", "vtcm": 6_286_080,
        "batch": 1, "descriptors": lambda repeat: 640 * repeat,
        "commands": lambda repeat: 672 * repeat,
        "prefetches": lambda repeat: 0,
    },
    "f16_async": {
        "variant": "F16F16", "mode": "async_single",
        "vtcm": 6_286_080, "batch": 1,
        "descriptors": lambda repeat: 640 * repeat,
        "commands": lambda repeat: 672 * repeat,
        "prefetches": lambda repeat: 633 * repeat,
    },
    "f16_batch2": {
        "variant": "F16F16", "mode": "double_buffer_batch2",
        "vtcm": 7_072_512, "batch": 2,
        "descriptors": lambda repeat: 320 * repeat,
        "commands": lambda repeat: 352 * repeat,
        "prefetches": lambda repeat: 313 * repeat,
    },
    "w4": {
        "variant": "W4F16", "mode": "not_applicable",
        "vtcm": 7_744_512, "batch": 0,
        "descriptors": lambda repeat: 160 * repeat + 7,
        "commands": lambda repeat: 352 * repeat,
        "prefetches": lambda repeat: 0,
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
    "w4f16_prefetch_wait_ticks", "f16f16_prefetch_wait_ticks",
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
        "experiment": "EXP-0027",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "f16f16_projection_mode": config["mode"],
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
        "hmx_command_count": config["commands"](repeat),
        "hmx_fp16_tile_pair_count": 98_816 * repeat,
        "hmx_u8s8_tile_pair_count": 0,
        "f16f16_weight_batch_n_tiles": config["batch"],
        "f16f16_prefetch_count": config["prefetches"](repeat),
        "release_result": 0,
        "close_result": 0,
    }
    for field, value in fixed.items():
        require(record, field, value)

    if variant == "W4F16":
        require(record, "w4f16_scale_placement", "hmx_output_per_channel")
        require(record, "w4f16_hvx_workers_created", 2)
        require(record, "w4f16_hvx_workers_locked", 2)
        require(record, "w4f16_pool_status", 0)
        require(record, "w4f16_streamed_command_count", 320 * repeat)
        require(record, "f16f16_prefetch_wait_ticks", 0)
    else:
        require(record, "w4f16_scale_placement", "not_applicable")
        require(record, "w4f16_hvx_workers_created", 0)
        require(record, "w4f16_hvx_workers_locked", 0)
        require(record, "w4f16_streamed_command_count", 0)
        require(record, "w4f16_prefetch_count", 0)
        require(record, "w4f16_prefetch_wait_ticks", 0)
        if config_name == "f16_serial":
            require(record, "f16f16_prefetch_wait_ticks", 0)

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
            record["ledger_named_ticks"] +
            record["ledger_unattributed_ticks"]):
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
    for field in (
        "weight_ddr_read_bytes", "weight_dma_descriptor_count",
        "hmx_command_count", "f16f16_prefetch_count",
    ):
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / repeat for record in records)
    result["projection_ticks_median_per_block"] = sum(
        result[f"{field}_median_per_block"] for field in (
            "qkv_projection_ticks", "o_projection_ticks",
            "gate_up_ticks", "down_ticks"))
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
    f16_hashes = set()
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
        if config["variant"] == "F16F16":
            for record in correctness + repeat1 + repeat10:
                f16_hashes.add(record["output_hash"])
        timing["repeat1"][name] = medians(repeat1)
        timing["repeat10"][name] = medians(repeat10)
        record_count += len(correctness) + len(repeat1) + len(repeat10)

    if len(f16_hashes) != 1:
        raise SystemExit(f"F16F16 schedules changed output hash: {f16_hashes}")

    serial1 = timing["repeat1"]["f16_serial"]
    serial10 = timing["repeat10"]["f16_serial"]
    eligible = []
    for name in ("f16_async", "f16_batch2"):
        if (timing["repeat1"][name]["host_wall_ns_per_block_median"] <
                serial1["host_wall_ns_per_block_median"] and
            timing["repeat10"][name]["host_wall_ns_per_block_median"] <
                serial10["host_wall_ns_per_block_median"]):
            eligible.append(name)
    selected = min(
        eligible,
        key=lambda name: timing["repeat10"][name][
            "host_wall_ns_per_block_median"],
        default="f16_serial")
    selected10 = timing["repeat10"][selected]
    w4 = timing["repeat10"]["w4"]
    serial_projection = serial10["projection_ticks_median_per_block"]
    selected_projection = selected10["projection_ticks_median_per_block"]
    w4_projection = w4["projection_ticks_median_per_block"]
    qtimer_hz = 19_200_000.0
    effective_gbps = (
        100_663_296 /
        (selected_projection / qtimer_hz) / 1_000_000_000.0)

    summary = {
        "experiment": "EXP-0027",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "zero_intermediate_ddr_gate": True,
        "f16f16_output_hash_equivalence_gate": True,
        "audit_free_performance_mode_gate": True,
        "top_level_ledger_closure_gate": True,
        "attention_ledger_closure_gate": True,
        "static_f16f16_pipeline_gate": True,
        "selected_f16f16_plan": selected,
        "selected_f16f16_non_regression_gate": selected != "f16_serial",
        "serial_f16f16_projection_ticks_per_block": serial_projection,
        "selected_f16f16_projection_ticks_per_block": selected_projection,
        "selected_f16f16_projection_change_percent": percent_change(
            selected_projection, serial_projection),
        "selected_f16f16_host_change_percent": percent_change(
            selected10["host_wall_ns_per_block_median"],
            serial10["host_wall_ns_per_block_median"]),
        "selected_f16f16_effective_weight_bandwidth_gbps": effective_gbps,
        "serial_f16f16_weight_dma_to_raw_hmx_ratio":
            serial10["weight_dma_ticks_median_per_block"] /
            serial10["hmx_compute_ticks_median_per_block"],
        "w4f16_projection_ticks_per_block": w4_projection,
        "w4f16_projection_change_from_selected_f16f16_percent":
            percent_change(w4_projection, selected_projection),
        "w4f16_host_change_from_selected_f16f16_percent": percent_change(
            w4["host_wall_ns_per_block_median"],
            selected10["host_wall_ns_per_block_median"]),
        "w4f16_faster_than_selected_f16f16":
            w4["host_wall_ns_per_block_median"] <
            selected10["host_wall_ns_per_block_median"],
        "stop_for_w4_pipeline_discussion":
            w4["host_wall_ns_per_block_median"] >=
            selected10["host_wall_ns_per_block_median"],
        "timing": timing,
        "local_gate": "pass" if selected != "f16_serial" else "fail",
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if selected != "f16_serial" else 1


if __name__ == "__main__":
    raise SystemExit(main())
