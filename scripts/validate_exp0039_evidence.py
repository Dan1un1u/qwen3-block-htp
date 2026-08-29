#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import statistics
import sys


MODES = {
    "control": {
        "runtime": "control",
        "phase_overlay_bytes": 0,
        "ring_compaction_bytes": 0,
        "lookahead_bytes": 0,
        "ring_slots": 48,
        "lookahead_per_block": 0,
    },
    "overlay": {
        "runtime": "phase_overlay",
        "phase_overlay_bytes": 0,
        "ring_compaction_bytes": 0,
        "lookahead_bytes": 0,
        "ring_slots": 48,
        "lookahead_per_block": 0,
    },
    "deep": {
        "runtime": "gate_up_deep",
        "phase_overlay_bytes": 524_288,
        "ring_compaction_bytes": 524_288,
        "lookahead_bytes": 1_048_576,
        "ring_slots": 16,
        "lookahead_per_block": 12,
    },
}


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


def validate_record(record, mode_key, repeat, audit_mode):
    mode = MODES[mode_key]
    fixed = {
        "experiment": "EXP-0039",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4F16",
        "attention_compute": "FP16_HMX",
        "projection_compute": "FP16_HMX",
        "common_ops_mode": "hvx_fp16",
        "attribution_mode": "on",
        "numerical_audit_mode": audit_mode,
        "residual_mode": "hvx_fused_post_norm",
        "w4f16_pipeline_mode":
            "adaptive_down96_gate4_dma8_cross_prefetch",
        "attention_pipeline_mode": "gqa_qkv_overlap",
        "mlp_mode": "crouton_native_batch8",
        "crouton_boundary_mode": "norms",
        "vtcm_plan_mode": mode["runtime"],
        "intermediate_residency": "VTCM",
        "warmup_rpc_result": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "repeat_count": repeat,
        "block_invocation_count": repeat,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": 8_171_008,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "projection_failure_result": 0,
        "release_result": 0,
        "close_result": 0,
        "output_hash": "f18b9abbe1487231",
        "vtcm_phase_overlay_bytes": mode["phase_overlay_bytes"],
        "vtcm_ring_compaction_bytes": mode["ring_compaction_bytes"],
        "vtcm_lookahead_bytes": mode["lookahead_bytes"],
        "mlp_crouton_ring_slots": mode["ring_slots"],
        "w4f16_gate_up_lookahead_hmx_count":
            mode["lookahead_per_block"] * repeat,
        "hmx_command_count": 208 * repeat,
        "w4f16_gate_up_hmx_command_count": 48 * repeat,
    }
    for field, expected in fixed.items():
        require(record, field, expected)

    if record["cosine"] < 0.99999 or record["warmup_cosine"] < 0.99999:
        raise SystemExit(f"invalid cosine for {mode_key}")
    if record["max_abs"] > 0.0625 or record["warmup_max_abs"] > 0.0625:
        raise SystemExit(f"invalid max_abs for {mode_key}")
    for field in (
        "host_wall_ns", "host_wall_ns_per_block", "total_ticks",
        "invocation_ticks", "gate_up_ticks", "weight_dma_ticks",
        "w4f16_gate_up_expand_ticks", "w4f16_gate_up_hmx_wait_ticks",
    ):
        value = record.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"invalid positive {mode_key}.{field}: {value!r}")
    if audit_mode == "on":
        require(record, "mismatches", 0)
        require(record, "max_lsb", 0)
        require(record, "mlp_down_input_hash", "b0d16655c164a22b")


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


def median_summary(records):
    repeat = records[0]["repeat_count"]
    fields = (
        "total_ticks", "invocation_ticks", "gate_up_ticks",
        "w4f16_gate_up_weight_dma_ticks",
        "w4f16_gate_up_expand_ticks",
        "w4f16_gate_up_expand_work_ticks",
        "w4f16_gate_up_expand_pool_wait_ticks",
        "w4f16_gate_up_prefetch_wait_ticks",
        "w4f16_gate_up_hmx_wait_ticks",
        "w4f16_gate_up_hmx_tail_wait_ticks",
        "w4f16_gate_up_stream_work_ticks",
        "w4f16_gate_up_stream_ready_wait_ticks",
    )
    result = {
        "samples": len(records),
        "host_wall_ns_per_block_median": statistics.median(
            record["host_wall_ns_per_block"] for record in records),
    }
    for field in fields:
        result[f"{field}_median_per_block"] = statistics.median(
            record[field] / repeat for record in records)
    return result


def percent(candidate, control):
    return (candidate / control - 1.0) * 100.0


def compare(control_records, candidate_records):
    control = median_summary(control_records)
    candidate = median_summary(candidate_records)
    paired_host = [
        percent(cand["host_wall_ns_per_block"],
                ctrl["host_wall_ns_per_block"])
        for ctrl, cand in zip(control_records, candidate_records)
    ]
    paired_gate = [
        percent(cand["gate_up_ticks"], ctrl["gate_up_ticks"])
        for ctrl, cand in zip(control_records, candidate_records)
    ]
    return {
        "control": control,
        "candidate": candidate,
        "host_median_percent_vs_control": percent(
            candidate["host_wall_ns_per_block_median"],
            control["host_wall_ns_per_block_median"]),
        "gate_up_median_percent_vs_control": percent(
            candidate["gate_up_ticks_median_per_block"],
            control["gate_up_ticks_median_per_block"]),
        "paired_host_delta_percent_median": statistics.median(paired_host),
        "paired_gate_up_delta_percent_median": statistics.median(paired_gate),
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_exp0039_evidence.py RESULT_DIR PACKAGE_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    package_dir = pathlib.Path(sys.argv[2])
    audit_package(package_dir)

    correctness = {}
    timing = {"repeat1": {}, "repeat10": {}}
    for mode in MODES:
        records = load_jsonl(result_dir / f"correctness_{mode}.jsonl")
        if len(records) != 1:
            raise SystemExit(f"wrong correctness record count for {mode}")
        validate_record(records[0], mode, 1, "on")
        correctness[mode] = {
            "output_hash": records[0]["output_hash"],
            "max_abs": records[0]["max_abs"],
            "cosine": records[0]["cosine"],
            "intermediate_ddr_read_bytes":
                records[0]["intermediate_ddr_read_bytes"],
            "intermediate_ddr_write_bytes":
                records[0]["intermediate_ddr_write_bytes"],
        }
        for family, repeat in (("repeat1", 1), ("repeat10", 10)):
            records = load_jsonl(result_dir / f"timing_{mode}_{family}.jsonl")
            if len(records) != 11:
                raise SystemExit(f"wrong timing record count for {mode} {family}")
            for record in records:
                validate_record(record, mode, repeat, "off")
            timing[family][mode] = records

    comparisons = {"overlay": {}, "deep": {}}
    for family in ("repeat1", "repeat10"):
        for mode in ("overlay", "deep"):
            comparisons[mode][family] = compare(
                timing[family]["control"], timing[family][mode])

    stage_a_pass = all(
        abs(comparisons["overlay"][family][key]) <= 1.0
        for family in ("repeat1", "repeat10")
        for key in (
            "host_median_percent_vs_control",
            "gate_up_median_percent_vs_control",
        )
    )
    stage_b_pass = all(
        comparisons["deep"][family]["host_median_percent_vs_control"] < 0.0
        and comparisons["deep"][family]["gate_up_median_percent_vs_control"] < 0.0
        for family in ("repeat1", "repeat10")
    )
    output = {
        "experiment": "EXP-0039",
        "correctness": correctness,
        "comparisons": comparisons,
        "stage_a_overlay_gate_pass": stage_a_pass,
        "stage_b_deep_pipeline_gate_pass": stage_b_pass,
        "adoption_recommendation": "accept" if stage_b_pass else "reject",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
