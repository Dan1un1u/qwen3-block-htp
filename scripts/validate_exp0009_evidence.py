#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


VARIANTS = (
    ("expanded_s8_control", "exp0005_full_bundle_control"),
    ("packed_w4_hmx_postscale", "slots4_chunk32"),
    ("packed_w4_hmx_postscale", "slots4_chunk64"),
    ("packed_w4_hmx_postscale", "slots4_chunk96"),
)
PLAN_CONFIG = {
    "exp0005_full_bundle_control": (1, 2, 32, 6),
    "slots4_chunk32": (6, 4, 32, 6),
    "slots4_chunk64": (6, 4, 64, 3),
    "slots4_chunk96": (6, 4, 96, 2),
}
COARSE_PLANS = ("slots4_chunk64", "slots4_chunk96")
PATTERNS = ("identity", "signed", "structured", "boundary")


def load_jsonl(path: pathlib.Path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def key(record):
    return (
        record["weight_storage"],
        record["physical_plan"],
        record["projection"],
        record["pattern"],
    )


def assert_runtime_valid(record, repeat):
    record_key = key(record)
    if record["experiment"] != "EXP-0009":
        raise SystemExit(f"wrong experiment for {record_key}")
    if record["projection"] != "down":
        raise SystemExit(f"non-down projection in {record_key}")
    if record["repeat_count"] != repeat:
        raise SystemExit(f"wrong repeat for {record_key}")
    if record["rpc_result"] != 0 or record["dsp_status"] != 0:
        raise SystemExit(f"runtime failure for {record_key}")
    if record["mismatches"] != 0:
        raise SystemExit(f"output mismatch for {record_key}")
    if record["dma_status"] != 0 or record["sync_status"] != 0:
        raise SystemExit(f"DMA/sync failure for {record_key}")
    if record["dcvs_power_setup_status"] != 0:
        raise SystemExit(f"DCVS setup failure for {record_key}")
    if record["dcvs_power_reset_status"] != 0:
        raise SystemExit(f"DCVS reset failure for {record_key}")
    if record["hmx_execution_count"] != 12288 * repeat:
        raise SystemExit(f"wrong HMX execution count for {record_key}")
    if record["hmx_stream_count"] != 384 * repeat:
        raise SystemExit(f"wrong internal HMX stream count for {record_key}")

    workers, compressed_slots, chunk_tiles, chunks_per_output = PLAN_CONFIG[
        record["physical_plan"]
    ]
    if record["requested_hvx_workers"] != workers:
        raise SystemExit(f"wrong requested workers for {record_key}")
    if record["compressed_slot_count"] != compressed_slots:
        raise SystemExit(f"wrong compressed slots for {record_key}")
    if record["chunk_tiles"] != chunk_tiles:
        raise SystemExit(f"wrong chunk depth for {record_key}")
    if record["chunks_per_output"] != chunks_per_output:
        raise SystemExit(f"wrong publications per output for {record_key}")
    if record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM plan exceeds request for {record_key}")

    if record["weight_storage"] == "packed_w4_hmx_postscale":
        if record["hvx_workers_created"] != workers:
            raise SystemExit(f"worker creation fallback for {record_key}")
        if record["hvx_workers_locked"] != workers:
            raise SystemExit(f"worker lock fallback for {record_key}")
        if record["hvx_hmx_overlap_observed"] != 1:
            raise SystemExit(f"no HVX/HMX overlap for {record_key}")
        if record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"no multi-HVX overlap for {record_key}")
        if record["hmx_carrier_checksum"] != record["packed_w4_checksum"]:
            raise SystemExit(f"HMX carrier is not raw q4 for {record_key}")
    elif record["hmx_carrier_checksum"] != record["expanded_carrier_checksum"]:
        raise SystemExit(f"S8 control carrier mismatch for {record_key}")


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return grouped


def median_metrics(records):
    fields = (
        "host_wall_ns",
        "pipeline_ticks",
        "weight_stage_ticks",
        "weight_expand_ticks",
        "hmx_compute_ticks",
        "hmx_ready_wait_ticks",
        "producer_slot_wait_ticks",
        "expanded_slot_wait_ticks",
        "hvx_max_active_workers",
        "vtcm_plan_bytes",
    )
    return {
        f"{field}_median": statistics.median(
            record[field] for record in records
        )
        for field in fields
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    expected_keys = {
        (storage, plan, "down", pattern)
        for storage, plan in VARIANTS
        for pattern in PATTERNS
    }

    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing = load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = group_records(correctness)
    timing_groups = group_records(timing)
    if set(correctness_groups) != expected_keys:
        raise SystemExit("correctness matrix has missing/unexpected keys")
    if set(timing_groups) != expected_keys:
        raise SystemExit("timing matrix has missing/unexpected keys")
    if any(len(records) != 1 for records in correctness_groups.values()):
        raise SystemExit("correctness matrix must have one record per key")
    if any(len(records) != 3 for records in timing_groups.values()):
        raise SystemExit("timing matrix must have three records per key")

    for record in correctness:
        assert_runtime_valid(record, 1)
    for record in timing:
        assert_runtime_valid(record, 10)

    comparable = (
        "reference_checksum",
        "reference_min",
        "reference_max",
        "expanded_carrier_checksum",
        "packed_w4_checksum",
        "projection_m",
        "projection_k",
        "projection_n",
        "hmx_execution_count",
        "hmx_stream_count",
        "output_tile_count",
    )
    for grouped in (correctness_groups, timing_groups):
        for pattern in PATTERNS:
            peers = [
                record
                for group_key, records in grouped.items()
                if group_key[2:] == ("down", pattern)
                for record in records
            ]
            reference = peers[0]
            for record in peers[1:]:
                for field in comparable:
                    if record[field] != reference[field]:
                        raise SystemExit(
                            f"down/{pattern}: variants differ at {field}"
                        )

    metrics = {}
    for storage, plan in VARIANTS:
        records = [
            record
            for group_key, group in timing_groups.items()
            if group_key[0] == storage and group_key[1] == plan
            for record in group
        ]
        metrics[plan] = median_metrics(records)

    expanded = metrics["exp0005_full_bundle_control"]["host_wall_ns_median"]
    parent = metrics["slots4_chunk32"]["host_wall_ns_median"]
    selected = min(
        COARSE_PLANS,
        key=lambda plan: metrics[plan]["host_wall_ns_median"],
    )
    selected_wall = metrics[selected]["host_wall_ns_median"]
    improves_parent = selected_wall < parent
    beats_expanded = selected_wall < expanded

    packed_sample = correctness_groups[
        ("packed_w4_hmx_postscale", selected, "down", "identity")
    ][0]
    expanded_sample = correctness_groups[
        ("expanded_s8_control", "exp0005_full_bundle_control", "down", "identity")
    ][0]
    if not (
        packed_sample["stored_weight_bytes_per_repeat"]
        < expanded_sample["stored_weight_bytes_per_repeat"]
    ):
        raise SystemExit("packed W4 did not reduce Native DDR weight bytes")

    print(
        json.dumps(
            {
                "experiment": "EXP-0009",
                "records": len(correctness) + len(timing),
                "correctness_records": len(correctness),
                "timing_records": len(timing),
                "exact_reference": True,
                "all_variants_equivalent": True,
                "same_hmx_execution_and_stream_counts": True,
                "worker_fallback": False,
                "dcvs_performance_vote": True,
                "hvx_hmx_overlap": True,
                "selected_candidate_plan": selected,
                "local_performance_gate": improves_parent,
                "beats_expanded_s8": beats_expanded,
                "parent_host_wall_ns_median": parent,
                "selected_host_wall_ns_median": selected_wall,
                "expanded_s8_host_wall_ns_median": expanded,
                "candidate_vs_parent_improvement_percent":
                    (parent - selected_wall) * 100.0 / parent,
                "candidate_vs_expanded_improvement_percent":
                    (expanded - selected_wall) * 100.0 / expanded,
                "metrics": metrics,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
