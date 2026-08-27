#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


PATTERNS = ("identity", "signed", "structured", "boundary")
SHAPES = ("gate_up_pair", "down")
STORAGES = (
    "expanded_s8_control",
    "packed_w4_hvx_prescale",
    "packed_w4_hmx_postscale",
)


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


def key(record):
    return record["projection"], record["weight_storage"], record["pattern"]


def expected_keys():
    return {(shape, storage, pattern) for shape in SHAPES
            for storage in STORAGES for pattern in PATTERNS}


def expected_plan(shape, storage):
    if storage == "expanded_s8_control":
        return ("expanded_s8_dma_chain2" if shape == "gate_up_pair"
                else "exp0005_full_bundle_control")
    return ("slots8e7_chunk64_dma_chain4" if shape == "gate_up_pair"
            else "slots4_chunk96_dma_batch2")


def assert_runtime_valid(record, repeat):
    shape, storage, _ = key(record)
    pair = shape == "gate_up_pair"
    w4 = storage != "expanded_s8_control"
    n_tiles = 384 if pair else 64
    k_tiles = 64 if pair else 192
    hmx_pairs = 24576 if pair else 12288
    hmx_streams = 768 if pair else 384
    chunks = (1 if pair else 2) if w4 else (2 if pair else 6)
    expected_expands = n_tiles * repeat * chunks if w4 else 0

    if record["experiment"] != "EXP-0017" or record["repeat_count"] != repeat:
        raise SystemExit(f"wrong experiment/repeat for {key(record)}")
    if record["physical_plan"] != expected_plan(shape, storage):
        raise SystemExit(f"wrong physical plan for {key(record)}")
    if record["host_invocation_mode"] != "single_invocation" or \
            record["measured_rpc_calls"] != 1 or \
            record["resource_lifetime_mode"] != "prepared_session" or \
            record["output_assembly_mode"] != "linked_2d_dma":
        raise SystemExit(f"wrong execution boundary for {key(record)}")
    if record["rpc_result"] != 0 or record["warmup_rpc_result"] != 0 or \
            record["dsp_status"] != 0 or record["mismatches"] != 0:
        raise SystemExit(f"runtime/correctness failure for {key(record)}")
    if record["warmup_output_checksum"] != record["measured_output_checksum"]:
        raise SystemExit(f"warm-up output changed for {key(record)}")
    for field in (
            "prepare_result", "release_result", "session_close_result",
            "output_dma_status", "dcvs_power_setup_status",
            "dcvs_power_reset_status", "hmx_resource_status",
            "hmx_lock_status", "hmx_unlock_status", "hmx_release_status",
            "hmx_thread_create_status", "hmx_thread_join_status",
            "hvx_lock_status", "hvx_unlock_status", "dma_status",
            "sync_status"):
        if record[field] != 0:
            raise SystemExit(f"{field} failure for {key(record)}")

    expected = {
        "requested_hvx_workers": 6 if w4 else 1,
        "projection_m": 64,
        "projection_k": 2048 if pair else 6144,
        "projection_n": 12288 if pair else 2048,
        "k_tile_count": k_tiles,
        "n_tile_count": n_tiles,
        "hmx_execution_count": hmx_pairs * repeat,
        "hmx_stream_count": hmx_streams * repeat,
        "weight_bundle_stage_count": n_tiles * repeat,
        "output_tile_count": n_tiles * repeat,
        "activation_stage_count": k_tiles,
        "weight_expand_count": expected_expands,
        "chunk_expand_count": expected_expands,
        "output_dma_submit_count": 1,
        "output_dma_wait_count": 2,
        "output_dma_descriptor_count": n_tiles,
        "output_dma_chain_count": 1,
        "output_dma_descriptor_completion_count": n_tiles,
        "output_dma_descriptor_timeout_count": 0,
        "dma_descriptor_timeout_count": 0,
        "resource_setup_in_run": 0,
        "resource_release_in_run": 0,
        "warmup_prepared_session_run_index": 1,
        "prepared_session_run_index": 2,
    }
    for field, value in expected.items():
        if record[field] != value:
            raise SystemExit(
                f"wrong {field} for {key(record)}: {record[field]} != {value}")
    if record["vtcm_requested_bytes"] != 2097152 or \
            record["vtcm_acquired_bytes"] != 2097152 or \
            record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM contract failure for {key(record)}")
    if record["resource_vtcm_address"] == 0 or \
            record["resource_hmx_context_id"] == 0 or \
            record["warmup_resource_vtcm_address"] != record["resource_vtcm_address"] or \
            record["warmup_resource_hmx_context_id"] != record["resource_hmx_context_id"]:
        raise SystemExit(f"resource identity failure for {key(record)}")
    if record["prepare_wall_ns"] <= 0 or record["release_wall_ns"] <= 0 or \
            record["first_call_host_wall_ns"] <= 0 or \
            record["second_call_host_wall_ns"] != 0:
        raise SystemExit(f"missing lifecycle/call timing for {key(record)}")
    if w4:
        if record["hvx_workers_created"] != 6 or \
                record["hvx_workers_locked"] != 6 or \
                record["hvx_hmx_overlap_observed"] != 1 or \
                record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"worker/overlap failure for {key(record)}")
        expected_carrier = (record["expanded_carrier_checksum"]
                            if storage == "packed_w4_hvx_prescale"
                            else record["packed_w4_checksum"])
        if record["hmx_carrier_checksum"] != expected_carrier:
            raise SystemExit(f"W4 carrier failure for {key(record)}")
    elif record["hmx_carrier_checksum"] != record["expanded_carrier_checksum"]:
        raise SystemExit(f"S8 carrier failure for {key(record)}")


def group_records(records):
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def assert_matrix(groups, records_per_key, repeat):
    if set(groups) != expected_keys():
        raise SystemExit(
            f"matrix keys mismatch missing={expected_keys() - set(groups)} "
            f"extra={set(groups) - expected_keys()}")
    if any(len(group) != records_per_key for group in groups.values()):
        raise SystemExit("matrix has wrong records per key")
    for group in groups.values():
        for record in group:
            assert_runtime_valid(record, repeat)


def assert_equivalent(groups):
    for shape in SHAPES:
        for pattern in PATTERNS:
            checksums = set()
            for storage in STORAGES:
                for record in groups[(shape, storage, pattern)]:
                    checksums.add((record["reference_checksum"],
                                   record["measured_output_checksum"]))
            if len(checksums) != 1:
                raise SystemExit(f"variant output mismatch for {shape}/{pattern}")


def median_metrics(records):
    fields = (
        "host_wall_ns", "dsp_total_ticks", "pipeline_ticks",
        "activation_stage_ticks", "weight_stage_ticks",
        "weight_expand_ticks", "hmx_compute_ticks",
        "hmx_ready_wait_ticks", "producer_slot_wait_ticks",
        "expanded_slot_wait_ticks", "output_assembly_ticks",
        "input_cache_ticks", "output_cache_ticks", "vtcm_plan_bytes",
    )
    return {f"{field}_median": statistics.median(
                record[field] for record in records) for field in fields}


def summarize(groups):
    metrics = {}
    comparisons = {}
    prescale_wins = []
    for shape in SHAPES:
        for storage in STORAGES:
            records = [record for (group_shape, group_storage, _), group
                       in groups.items()
                       if group_shape == shape and group_storage == storage
                       for record in group]
            metrics[f"{shape}:{storage}"] = median_metrics(records)
        pre = metrics[f"{shape}:packed_w4_hvx_prescale"]["host_wall_ns_median"]
        post = metrics[f"{shape}:packed_w4_hmx_postscale"]["host_wall_ns_median"]
        s8 = metrics[f"{shape}:expanded_s8_control"]["host_wall_ns_median"]
        selected_name, selected = min(("hvx_prescale", pre),
                                      ("hmx_postscale", post), key=lambda x: x[1])
        comparisons[shape] = {
            "hvx_prescale_host_wall_ns_median": pre,
            "hmx_postscale_host_wall_ns_median": post,
            "prescale_improvement_percent": (post - pre) * 100.0 / post,
            "prescale_is_faster": pre < post,
            "selected_w4_scale_placement": selected_name,
            "selected_w4_host_wall_ns_median": selected,
            "expanded_s8_host_wall_ns_median": s8,
            "selected_w4_vs_s8_improvement_percent":
                (s8 - selected) * 100.0 / s8,
            "selected_w4_is_faster_than_s8": selected < s8,
        }
        prescale_wins.append(pre < post)
    return metrics, comparisons, any(prescale_wins)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing1 = load_jsonl(result_dir / "timing_repeat1.jsonl")
    timing10 = load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = group_records(correctness)
    timing1_groups = group_records(timing1)
    timing10_groups = group_records(timing10)
    assert_matrix(correctness_groups, 1, 1)
    assert_matrix(timing1_groups, 5, 1)
    assert_matrix(timing10_groups, 5, 10)
    for groups in (correctness_groups, timing1_groups, timing10_groups):
        assert_equivalent(groups)
    repeat1_metrics, repeat1_comparisons, repeat1_gate = summarize(timing1_groups)
    repeat10_metrics, repeat10_comparisons, repeat10_gate = summarize(timing10_groups)
    print(json.dumps({
        "experiment": "EXP-0017",
        "records": len(correctness) + len(timing1) + len(timing10),
        "correctness_records": len(correctness),
        "timing_repeat1_records": len(timing1),
        "timing_repeat10_records": len(timing10),
        "exact_reference": True,
        "all_scale_placements_equivalent": True,
        "persistent_resource_identity_stable": True,
        "worker_fallback": False,
        "descriptor_timeouts": 0,
        "device_reboot": False,
        "vtcm_request_bytes": 2097152,
        "local_performance_gate": repeat1_gate or repeat10_gate,
        "repeat1_prescale_gate": repeat1_gate,
        "repeat10_prescale_gate": repeat10_gate,
        "repeat1_comparisons": repeat1_comparisons,
        "repeat10_comparisons": repeat10_comparisons,
        "repeat1_metrics": repeat1_metrics,
        "repeat10_metrics": repeat10_metrics,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
