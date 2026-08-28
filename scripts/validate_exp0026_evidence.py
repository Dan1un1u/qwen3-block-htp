#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = ("F16F16", "W4F16")
MODES = ("scalar", "hvx", "fused")
MODE_NAMES = {
    "scalar": "scalar",
    "hvx": "hvx_fp16",
    "fused": "hvx_fused_post_norm",
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


def validate_record(record, variant, repeat, mode, audit_mode):
    expected = EXPECTED[variant]
    fixed = {
        "experiment": "EXP-0026",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": MODE_NAMES[mode],
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

    positive_timing_fields = list(TIMING_FIELDS[:17])
    if mode == "fused":
        positive_timing_fields.remove("post_attention_norm_ticks")
    for field in (
        "warmup_host_wall_ns", "host_wall_ns", "host_wall_ns_per_block",
    ) + tuple(positive_timing_fields):
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {mode}.{variant}.{field}: {value!r}")

    for prefix in ("warmup_", ""):
        if record[f"{prefix}cosine"] < 0.99999:
            raise SystemExit(f"invalid {mode}.{variant} {prefix}cosine")
        if record[f"{prefix}max_abs"] > 0.0625:
            raise SystemExit(f"invalid {mode}.{variant} {prefix}max_abs")
    output_hash = record.get("output_hash")
    if not isinstance(output_hash, str) or len(output_hash) != 16:
        raise SystemExit(f"invalid output hash: {output_hash!r}")

    if variant == "W4F16":
        require(record, "w4f16_scale_placement", "hmx_output_per_channel")
        require(record, "w4f16_hvx_workers_created", 2)
        require(record, "w4f16_hvx_workers_locked", 2)
        require(record, "w4f16_pool_status", 0)
        require(record, "w4f16_streamed_command_count", 320 * repeat)
    else:
        require(record, "w4f16_scale_placement", "not_applicable")
        require(record, "w4f16_hvx_workers_created", 0)
        require(record, "w4f16_hvx_workers_locked", 0)
        require(record, "w4f16_streamed_command_count", 0)

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
    all_hashes = {variant: set() for variant in VARIANTS}
    record_count = 0

    for mode in MODES:
        for variant in VARIANTS:
            correctness = load_jsonl(
                result_dir / f"correctness_{mode}_{variant}.jsonl")
            repeat1 = load_jsonl(
                result_dir / f"timing_{mode}_{variant}_repeat1.jsonl")
            repeat10 = load_jsonl(
                result_dir / f"timing_{mode}_{variant}_repeat10.jsonl")
            if len(correctness) != 1 or len(repeat1) != 5 or len(repeat10) != 5:
                raise SystemExit(f"wrong evidence matrix size for {mode}.{variant}")
            for record in correctness:
                validate_record(record, variant, 1, mode, "on")
            for record in repeat1:
                validate_record(record, variant, 1, mode, "off")
            for record in repeat10:
                validate_record(record, variant, 10, mode, "off")
            for record in correctness + repeat1 + repeat10:
                all_hashes[variant].add(record["output_hash"])
            timing["repeat1"][f"{mode}_{variant}"] = medians(repeat1)
            timing["repeat10"][f"{mode}_{variant}"] = medians(repeat10)
            record_count += len(correctness) + len(repeat1) + len(repeat10)

    for variant, hashes in all_hashes.items():
        if len(hashes) != 1:
            raise SystemExit(f"residual mode changed {variant} output hash: {hashes}")

    residual_tick_gate = True
    host_gate = True
    fused_host_gate = True
    comparisons = {}
    for family in ("repeat1", "repeat10"):
        comparisons[family] = {}
        for variant in VARIANTS:
            scalar = timing[family][f"scalar_{variant}"]
            hvx = timing[family][f"hvx_{variant}"]
            fused = timing[family][f"fused_{variant}"]
            scalar_residual = (
                scalar["post_attention_residual_ticks_median_per_block"] +
                scalar["final_residual_ticks_median_per_block"])
            hvx_residual = (
                hvx["post_attention_residual_ticks_median_per_block"] +
                hvx["final_residual_ticks_median_per_block"])
            hvx_post_pair = (
                hvx["post_attention_residual_ticks_median_per_block"] +
                hvx["post_attention_norm_ticks_median_per_block"])
            fused_post_pair = (
                fused["post_attention_residual_ticks_median_per_block"] +
                fused["post_attention_norm_ticks_median_per_block"])
            comparisons[family][variant] = {
                "scalar_host_wall_ns_per_block":
                    scalar["host_wall_ns_per_block_median"],
                "hvx_host_wall_ns_per_block":
                    hvx["host_wall_ns_per_block_median"],
                "hvx_host_change_percent": percent_change(
                    hvx["host_wall_ns_per_block_median"],
                    scalar["host_wall_ns_per_block_median"]),
                "scalar_residual_ticks_per_block": scalar_residual,
                "hvx_residual_ticks_per_block": hvx_residual,
                "hvx_residual_change_percent": percent_change(
                    hvx_residual, scalar_residual),
                "hvx_post_residual_norm_ticks_per_block": hvx_post_pair,
                "fused_post_residual_norm_ticks_per_block": fused_post_pair,
                "fused_host_wall_ns_per_block":
                    fused["host_wall_ns_per_block_median"],
                "fused_host_change_from_hvx_percent": percent_change(
                    fused["host_wall_ns_per_block_median"],
                    hvx["host_wall_ns_per_block_median"]),
            }
            if family == "repeat10":
                residual_tick_gate &= hvx_residual < scalar_residual
                host_gate &= (
                    hvx["host_wall_ns_per_block_median"] <
                    scalar["host_wall_ns_per_block_median"])
                fused_host_gate &= (
                    fused["host_wall_ns_per_block_median"] <
                    hvx["host_wall_ns_per_block_median"])

    local_gate = residual_tick_gate and host_gate
    summary = {
        "experiment": "EXP-0026",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "zero_intermediate_ddr_gate": True,
        "output_hash_equivalence_gate": True,
        "audit_free_performance_mode_gate": True,
        "top_level_ledger_closure_gate": True,
        "attention_ledger_closure_gate": True,
        "static_hvx_residual_gate": True,
        "hvx_residual_tick_gate": residual_tick_gate,
        "hvx_complete_block_host_gate": host_gate,
        "fused_complete_block_host_gate": fused_host_gate,
        "selected_residual_mode": "fused" if fused_host_gate else "hvx",
        "timing": timing,
        "comparisons": comparisons,
        "local_gate": "pass" if local_gate else "fail",
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if local_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
