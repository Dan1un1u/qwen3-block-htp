#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


VARIANTS = {
    "f16": {
        "variant": "F16F16",
        "output_hash": "704252c89780e695",
        "peak": 6_875_904,
    },
    "w4": {
        "variant": "W4F16",
        "output_hash": "f18b9abbe1487231",
        "peak": 8_171_008,
    },
}

MODES = {
    "control": {
        "runtime": "control",
        "qkv_projection": 0,
        "qkv_unpack": 0,
        "qk_operands": 0,
        "av_weights": 0,
        "av_heads": 0,
        "av_unpack": 0,
        "norm_projections": 0,
    },
    "qkv": {
        "runtime": "qkv",
        "qkv_projection": 3,
        "qkv_unpack": 128,
        "qk_operands": 24,
        "av_weights": 8,
        "av_heads": 0,
        "av_unpack": 0,
        "norm_projections": 0,
    },
    "avo": {
        "runtime": "av_to_o",
        "qkv_projection": 0,
        "qkv_unpack": 0,
        "qk_operands": 0,
        "av_weights": 0,
        "av_heads": 16,
        "av_unpack": 16,
        "norm_projections": 0,
    },
    "input_norm": {
        "runtime": "input_norm",
        "qkv_projection": 0,
        "qkv_unpack": 0,
        "qk_operands": 0,
        "av_weights": 0,
        "av_heads": 0,
        "av_unpack": 0,
        "norm_projections": 1,
    },
    "post_norm": {
        "runtime": "post_norm",
        "qkv_projection": 0,
        "qkv_unpack": 0,
        "qk_operands": 0,
        "av_weights": 0,
        "av_heads": 0,
        "av_unpack": 0,
        "norm_projections": 1,
    },
    "norms": {
        "runtime": "norms",
        "qkv_projection": 0,
        "qkv_unpack": 0,
        "qk_operands": 0,
        "av_weights": 0,
        "av_heads": 0,
        "av_unpack": 0,
        "norm_projections": 2,
    },
}

CONFIGS = tuple(
    f"{variant}_{mode}" for variant in VARIANTS for mode in MODES
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


def split_config(name):
    variant, mode = name.split("_", 1)
    return VARIANTS[variant], MODES[mode]


def validate_record(record, name, repeat, audit_mode):
    variant, mode = split_config(name)
    fixed = {
        "experiment": "EXP-0038",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": variant["variant"],
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "attention_pipeline_mode": "gqa_qkv_overlap",
        "mlp_mode": "crouton_native_batch8",
        "crouton_boundary_mode": mode["runtime"],
        "intermediate_residency": "VTCM",
        "warmup_rpc_result": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "repeat_count": repeat,
        "block_invocation_count": repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": variant["peak"],
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "projection_failure_result": 0,
        "release_result": 0,
        "close_result": 0,
        "output_hash": variant["output_hash"],
        "crouton_qkv_projection_count": mode["qkv_projection"] * repeat,
        "crouton_qkv_unpack_skipped": mode["qkv_unpack"] * repeat,
        "crouton_qk_operand_count": mode["qk_operands"] * repeat,
        "crouton_av_weight_count": mode["av_weights"] * repeat,
        "crouton_av_o_head_count": mode["av_heads"] * repeat,
        "crouton_av_unpack_skipped": mode["av_unpack"] * repeat,
        "crouton_norm_projection_count": mode["norm_projections"] * repeat,
        "crouton_q_operand_mismatch_count": 0,
        "crouton_k_operand_mismatch_count": 0,
        "crouton_v_operand_mismatch_count": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    if record["cosine"] < 0.99999 or record["warmup_cosine"] < 0.99999:
        raise SystemExit(f"invalid cosine for {name}")
    if record["max_abs"] > 0.0625 or record["warmup_max_abs"] > 0.0625:
        raise SystemExit(f"invalid max_abs for {name}")
    for field in (
        "host_wall_ns", "host_wall_ns_per_block", "total_ticks",
        "invocation_ticks", "hmx_compute_ticks", "weight_dma_ticks",
    ):
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {name}.{field}: {value!r}")

    if audit_mode == "on":
        require(record, "mismatches", 0)
        require(record, "max_lsb", 0)
    if mode["runtime"] == "qkv" and record["crouton_qkv_transform_ticks"] <= 0:
        raise SystemExit(f"missing QKV transform attribution for {name}")
    if mode["runtime"] == "av_to_o" and record["crouton_av_o_copy_ticks"] <= 0:
        raise SystemExit(f"missing AV->O copy attribution for {name}")
    if mode["norm_projections"] and record["crouton_norm_store_ticks"] <= 0:
        raise SystemExit(f"missing norm store attribution for {name}")


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
        "total_ticks", "invocation_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
        "o_projection_ticks", "post_attention_norm_ticks", "gate_up_ticks",
        "down_ticks", "weight_dma_ticks", "hmx_compute_ticks",
        "crouton_qkv_transform_ticks", "crouton_av_o_copy_ticks",
        "crouton_norm_store_ticks",
    )
    result = {
        "samples": len(records),
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


def compare(timing, variant, mode):
    output = {}
    passed = True
    for family in ("repeat1", "repeat10"):
        control = timing[family][f"{variant}_control"]
        candidate = timing[family][f"{variant}_{mode}"]
        delta = pct(
            candidate["host_wall_ns_per_block_median"],
            control["host_wall_ns_per_block_median"],
        )
        output[family] = {
            "control_host_wall_ns_per_block_median":
                control["host_wall_ns_per_block_median"],
            "candidate_host_wall_ns_per_block_median":
                candidate["host_wall_ns_per_block_median"],
            "candidate_percent_vs_control": delta,
            "pass_not_slower": delta <= 0.0,
        }
        passed = passed and delta <= 0.0
    output["pass_not_slower_both_repeats"] = passed
    return output


def paired_norms_gate(result_dir):
    result = {}
    passed = True
    for family, repeat in (("repeat1", 1), ("repeat10", 10)):
        control = load_jsonl(result_dir / f"paired_w4_control_{family}.jsonl")
        candidate = load_jsonl(result_dir / f"paired_w4_norms_{family}.jsonl")
        if len(control) != 11 or len(candidate) != 11:
            raise SystemExit(f"wrong paired W4 norms evidence size for {family}")
        for record in control:
            validate_record(record, "w4_control", repeat, "off")
        for record in candidate:
            validate_record(record, "w4_norms", repeat, "off")
        deltas = [
            pct(candidate[index]["host_wall_ns_per_block"],
                control[index]["host_wall_ns_per_block"])
            for index in range(11)
        ]
        median_delta = statistics.median(deltas)
        result[family] = {
            "paired_samples": 11,
            "paired_host_percent_median": median_delta,
            "control_host_wall_ns_per_block_median": statistics.median(
                item["host_wall_ns_per_block"] for item in control),
            "candidate_host_wall_ns_per_block_median": statistics.median(
                item["host_wall_ns_per_block"] for item in candidate),
            "pass_paired_median_not_slower": median_delta <= 0.0,
        }
        passed = passed and median_delta <= 0.0
    result["pass_both_repeats"] = passed
    return result


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
        if len(correctness) != 1 or len(repeat1) != 7 or len(repeat10) != 7:
            raise SystemExit(f"wrong evidence matrix size for {name}")
        validate_record(correctness[0], name, 1, "on")
        for record in repeat1:
            validate_record(record, name, 1, "off")
        for record in repeat10:
            validate_record(record, name, 10, "off")
        timing["repeat1"][name] = medians(repeat1)
        timing["repeat10"][name] = medians(repeat10)
        record_count += 15

    comparisons = {
        mode: {
            variant: compare(timing, variant, mode)
            for variant in VARIANTS
        }
        for mode in ("qkv", "avo", "input_norm", "post_norm", "norms")
    }
    paired_norms = paired_norms_gate(result_dir)
    record_count += 44

    local_gates = {
        "qkv": all(comparisons["qkv"][variant]
                   ["pass_not_slower_both_repeats"] for variant in VARIANTS),
        "av_to_o": all(comparisons["avo"][variant]
                       ["pass_not_slower_both_repeats"] for variant in VARIANTS),
        "input_norm": all(comparisons["input_norm"][variant]
                          ["pass_not_slower_both_repeats"] for variant in VARIANTS),
        "post_norm": all(comparisons["post_norm"][variant]
                         ["pass_not_slower_both_repeats"] for variant in VARIANTS),
        "norms_combined": (
            all(comparisons["norms"][variant]
                ["pass_not_slower_both_repeats"] for variant in VARIANTS)
            and paired_norms["pass_both_repeats"]
        ),
    }

    print(json.dumps({
        "experiment": "EXP-0038",
        "execution_state": "completed",
        "evidence_validity": "valid",
        "local_gates": local_gates,
        "adoption_status": "pending",
        "formal_run_records": record_count,
        "parent_block_package_experiment": manifest["experiment"],
        "timing": timing,
        "comparisons": comparisons,
        "paired_w4_norms_confirmation": paired_norms,
        "correctness": {
            "output_byte_exact_to_variant_control": True,
            "qkv_hmx_operands_byte_exact_to_shadow_path": True,
            "all_modes_numerical_status_pass": True,
        },
        "physical_contract": {
            "fixed_vtcm_bytes": 8_388_608,
            "f16_peak_vtcm_bytes": VARIANTS["f16"]["peak"],
            "w4_peak_vtcm_bytes": VARIANTS["w4"]["peak"],
            "single_fastrpc_per_execution_unit": True,
            "single_hmx_owner": True,
            "intermediate_ddr_bytes": 0,
            "spill_fill_count": 0,
            "qnn_dependency": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
