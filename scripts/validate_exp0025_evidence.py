#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = ("F16F16", "W4F16")
MODES = ("off", "on")
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
ATTENTION_FIELDS = (
    "attention_setup_ticks", "attention_qk_pack_ticks",
    "attention_qk_hmx_ticks", "attention_qk_unpack_ticks",
    "attention_qk_audit_ticks", "attention_softmax_ticks",
    "attention_softmax_audit_ticks", "attention_av_pack_ticks",
    "attention_av_hmx_ticks", "attention_av_unpack_ticks",
    "attention_av_audit_ticks", "attention_unattributed_ticks",
)
NON_ATTENTION_AUDIT_FIELDS = (
    "input_norm_audit_ticks", "qkv_audit_ticks",
    "qk_norm_rope_audit_ticks", "o_projection_audit_ticks",
    "post_attention_residual_audit_ticks",
    "post_attention_norm_audit_ticks", "gate_up_audit_ticks",
    "activation_audit_ticks", "down_audit_ticks",
    "final_residual_audit_ticks",
)
ATTENTION_AUDIT_FIELDS = (
    "attention_qk_audit_ticks", "attention_softmax_audit_ticks",
    "attention_av_audit_ticks",
)
ATTRIBUTION_FIELDS = (
    "invocation_ticks", "runtime_setup_ticks", "runtime_teardown_ticks",
    "ledger_named_ticks", "ledger_unattributed_ticks",
) + NON_ATTENTION_AUDIT_FIELDS + ATTENTION_FIELDS


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
        "experiment": "EXP-0025",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant,
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": mode,
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

    for field in (
        "warmup_host_wall_ns", "host_wall_ns", "host_wall_ns_per_block",
        "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "o_projection_ticks", "post_attention_residual_ticks",
        "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
        "down_ticks", "final_residual_ticks", "output_stage_ticks",
        "total_ticks", "hmx_compute_ticks",
    ):
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

    if mode == "off":
        for field in ATTRIBUTION_FIELDS:
            require(record, field, 0)
        return

    for field in ATTRIBUTION_FIELDS:
        value = record.get(field)
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"invalid attribution field {field}: {value!r}")
    for field in NON_ATTENTION_AUDIT_FIELDS + ATTENTION_FIELDS[:-1]:
        if record[field] <= 0:
            raise SystemExit(f"missing attribution field {field}")

    if record["invocation_ticks"] != (
            record["ledger_named_ticks"] +
            record["ledger_unattributed_ticks"]):
        raise SystemExit("top-level timing ledger does not close")
    if record["ledger_named_ticks"] != sum(record[field] for field in TOP_LEVEL_FIELDS):
        raise SystemExit("top-level named timing sum is inconsistent")
    if record["ledger_unattributed_ticks"] / record["invocation_ticks"] > 0.01:
        raise SystemExit("top-level unattributed gap exceeds one percent")
    if sum(record[field] for field in ATTENTION_FIELDS) != record["attention_ticks"]:
        raise SystemExit("nested Attention timing ledger does not close")
    if record["attention_unattributed_ticks"] / record["attention_ticks"] > 0.01:
        raise SystemExit("Attention unattributed gap exceeds one percent")


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
        "total_ticks", "input_stage_ticks", "metadata_stage_ticks",
        "input_norm_ticks", "qkv_projection_ticks", "qk_norm_rope_ticks",
        "attention_ticks", "o_projection_ticks",
        "post_attention_residual_ticks", "post_attention_norm_ticks",
        "gate_up_ticks", "activation_ticks", "down_ticks",
        "final_residual_ticks", "output_stage_ticks", "weight_dma_ticks",
        "hmx_compute_ticks", "projection_pack_ticks", "w4f16_expand_ticks",
        "projection_hmx_wait_ticks", "projection_unpack_ticks",
    ) + ATTRIBUTION_FIELDS
    result = {
        "host_wall_ns_per_block_median": statistics.median(
            record["host_wall_ns_per_block"] for record in records),
        "output_hash": records[0]["output_hash"],
    }
    for field in fields:
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / repeat for record in records)
    return result


def percent_change(candidate, control):
    return (candidate / control - 1.0) * 100.0


def composition(metrics):
    value = lambda field: metrics[f"{field}_median_per_block"]
    total = value("invocation_ticks")
    projection = sum(value(field) for field in (
        "qkv_projection_ticks", "o_projection_ticks",
        "gate_up_ticks", "down_ticks"))
    rmsnorm = value("input_norm_ticks") + value("post_attention_norm_ticks")
    residual = (value("post_attention_residual_ticks") +
                value("final_residual_ticks"))
    attention_audit = sum(value(field) for field in ATTENTION_AUDIT_FIELDS)
    non_attention_audit = sum(
        value(field) for field in NON_ATTENTION_AUDIT_FIELDS)
    audit_total = attention_audit + non_attention_audit
    residual_audit = (value("post_attention_residual_audit_ticks") +
                      value("final_residual_audit_ticks"))
    categories = {
        "projection": projection,
        "attention": value("attention_ticks"),
        "residual_add_including_audit": residual,
        "rmsnorm": rmsnorm,
        "qk_norm_rope": value("qk_norm_rope_ticks"),
        "silu_by_up": value("activation_ticks"),
        "runtime_boundary_and_ledger_gap": sum(value(field) for field in (
            "runtime_setup_ticks", "metadata_stage_ticks", "input_stage_ticks",
            "output_stage_ticks", "runtime_teardown_ticks",
            "ledger_unattributed_ticks")),
    }
    return {
        "invocation_ticks_per_block": total,
        "exclusive_categories": {
            name: {"ticks": ticks, "share_percent": ticks / total * 100.0}
            for name, ticks in categories.items()
        },
        "numerical_audit_ticks": audit_total,
        "numerical_audit_share_percent": audit_total / total * 100.0,
        "residual_add_without_audit_ticks": residual - residual_audit,
        "residual_add_without_audit_share_percent":
            (residual - residual_audit) / total * 100.0,
        "attention_without_audit_ticks":
            value("attention_ticks") - attention_audit,
        "attention_without_audit_share_percent":
            (value("attention_ticks") - attention_audit) / total * 100.0,
        "attention_breakdown": {
            field.removeprefix("attention_").removesuffix("_ticks"): {
                "ticks": value(field),
                "attention_share_percent":
                    value(field) / value("attention_ticks") * 100.0,
            }
            for field in ATTENTION_FIELDS
        },
    }


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
            for record in correctness + repeat1:
                validate_record(record, variant, 1, mode)
            for record in repeat10:
                validate_record(record, variant, 10, mode)
            for record in correctness + repeat1 + repeat10:
                all_hashes[variant].add(record["output_hash"])
            timing["repeat1"][f"{mode}_{variant}"] = medians(repeat1)
            timing["repeat10"][f"{mode}_{variant}"] = medians(repeat10)
            record_count += len(correctness) + len(repeat1) + len(repeat10)

    for variant, hashes in all_hashes.items():
        if len(hashes) != 1:
            raise SystemExit(f"attribution changed {variant} output hash: {hashes}")

    overhead = {}
    overhead_gate = True
    for family in ("repeat1", "repeat10"):
        overhead[family] = {}
        for variant in VARIANTS:
            off = timing[family][f"off_{variant}"]["host_wall_ns_per_block_median"]
            on = timing[family][f"on_{variant}"]["host_wall_ns_per_block_median"]
            change = percent_change(on, off)
            overhead[family][variant] = {
                "off_host_wall_ns_per_block": off,
                "on_host_wall_ns_per_block": on,
                "change_percent": change,
            }
            if family == "repeat10" and change > 2.0:
                overhead_gate = False

    summary = {
        "experiment": "EXP-0025",
        "records": record_count,
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "parent_block_package_experiment": manifest["experiment"],
        "parent_block_package_hash_audit": True,
        "fixed_vtcm_request_bytes": 8_388_608,
        "zero_intermediate_ddr_gate": True,
        "output_hash_equivalence_gate": True,
        "top_level_ledger_closure_gate": True,
        "attention_ledger_closure_gate": True,
        "instrumentation_overhead_gate": overhead_gate,
        "timing": timing,
        "instrumentation_overhead": overhead,
        "composition": {
            family: {
                variant: composition(timing[family][f"on_{variant}"])
                for variant in VARIANTS
            }
            for family in ("repeat1", "repeat10")
        },
        "local_gate": "pass" if overhead_gate else "fail",
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if overhead_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
