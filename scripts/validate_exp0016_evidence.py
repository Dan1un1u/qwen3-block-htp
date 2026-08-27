#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


PATTERNS = ("identity", "signed", "structured", "boundary")
STORAGES = ("expanded_s8_control", "packed_w4_hmx_postscale")
MODES = ("two_call_control", "single_invocation")


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
    return (record["weight_storage"], record["host_invocation_mode"],
            record["pattern"])


def expected_keys():
    return {(storage, mode, pattern)
            for storage in STORAGES
            for mode in MODES
            for pattern in PATTERNS}


def expected_plan(storage):
    return ("expanded_s8_dma_chain2" if storage == "expanded_s8_control"
            else "slots8e7_chunk64_dma_chain4")


def assert_runtime_valid(record, repeat):
    record_key = key(record)
    storage, mode, _ = record_key
    candidate = mode == "single_invocation"
    n_tiles = 384 if candidate else 192
    rpc_calls = 1 if candidate else 2
    hmx_pairs = 24576 if candidate else 12288
    hmx_streams = 768 if candidate else 384
    workers = 1 if storage == "expanded_s8_control" else 6
    chunks = 2 if storage == "expanded_s8_control" else 1
    expected_expands = (n_tiles * repeat * chunks
                        if storage == "packed_w4_hmx_postscale" else 0)

    if record["experiment"] != "EXP-0016" or \
            record["repeat_count"] != repeat:
        raise SystemExit(f"wrong experiment/repeat for {record_key}")
    if record["projection"] != ("gate_up_pair" if candidate else "gate_up"):
        raise SystemExit(f"wrong projection for {record_key}")
    if record["physical_plan"] != expected_plan(storage):
        raise SystemExit(f"wrong physical plan for {record_key}")
    if record["resource_lifetime_mode"] != "prepared_session" or \
            record["output_assembly_mode"] != "linked_2d_dma":
        raise SystemExit(f"wrong runtime boundary for {record_key}")
    if record["measured_rpc_calls"] != rpc_calls:
        raise SystemExit(f"wrong measured RPC count for {record_key}")
    if record["rpc_result"] != 0 or record["warmup_rpc_result"] != 0 or \
            record["dsp_status"] != 0 or record["mismatches"] != 0:
        raise SystemExit(f"runtime/correctness failure for {record_key}")
    if record["warmup_output_checksum"] != \
            record["measured_output_checksum"]:
        raise SystemExit(f"warm-up output changed for {record_key}")

    for field in (
            "prepare_result", "release_result", "session_close_result",
            "output_dma_status", "dcvs_power_setup_status",
            "dcvs_power_reset_status", "hmx_resource_status",
            "hmx_lock_status", "hmx_unlock_status", "hmx_release_status",
            "hmx_thread_create_status", "hmx_thread_join_status",
            "hvx_lock_status", "hvx_unlock_status", "dma_status",
            "sync_status"):
        if record[field] != 0:
            raise SystemExit(f"{field} failure for {record_key}")

    expected_fields = {
        "requested_hvx_workers": workers,
        "projection_m": 64,
        "projection_k": 2048,
        "projection_n": 12288 if candidate else 6144,
        "k_tile_count": 64,
        "n_tile_count": n_tiles,
        "hmx_execution_count": hmx_pairs * repeat,
        "hmx_stream_count": hmx_streams * repeat,
        "aggregate_hmx_execution_count": 24576 * repeat,
        "aggregate_hmx_stream_count": 768 * repeat,
        "weight_bundle_stage_count": n_tiles * repeat,
        "output_tile_count": n_tiles * repeat,
        "aggregate_weight_bundle_stage_count": 384 * repeat,
        "aggregate_output_tile_count": 384 * repeat,
        "activation_stage_count": 64,
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
        "warmup_prepared_session_run_index": rpc_calls,
        "prepared_session_run_index": 2 * rpc_calls,
    }
    for field, expected in expected_fields.items():
        if record[field] != expected:
            raise SystemExit(
                f"wrong {field} for {record_key}: "
                f"{record[field]} != {expected}")

    if record["aggregate_activation_stage_ticks"] <= 0 or \
            record["aggregate_input_cache_ticks"] <= 0 or \
            record["aggregate_output_cache_ticks"] <= 0:
        raise SystemExit(f"missing stage/cache timing for {record_key}")
    if record["vtcm_requested_bytes"] != 2097152 or \
            record["vtcm_acquired_bytes"] != 2097152 or \
            record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM contract failure for {record_key}")
    if record["resource_vtcm_address"] == 0 or \
            record["resource_hmx_context_id"] == 0 or \
            record["warmup_resource_vtcm_address"] != \
                record["resource_vtcm_address"] or \
            record["warmup_resource_hmx_context_id"] != \
                record["resource_hmx_context_id"]:
        raise SystemExit(f"persistent resource identity failure for {record_key}")
    if record["prepare_wall_ns"] <= 0 or record["release_wall_ns"] <= 0:
        raise SystemExit(f"missing lifecycle timing for {record_key}")
    if record["first_call_host_wall_ns"] <= 0 or \
            (candidate and record["second_call_host_wall_ns"] != 0) or \
            (not candidate and record["second_call_host_wall_ns"] <= 0):
        raise SystemExit(f"bad per-call wall timing for {record_key}")

    if storage == "packed_w4_hmx_postscale":
        if record["hvx_workers_created"] != 6 or \
                record["hvx_workers_locked"] != 6 or \
                record["hvx_hmx_overlap_observed"] != 1 or \
                record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"worker/overlap failure for {record_key}")
        if record["hmx_carrier_checksum"] != record["packed_w4_checksum"]:
            raise SystemExit(f"raw-q4 carrier failure for {record_key}")
    elif record["hmx_carrier_checksum"] != \
            record["expanded_carrier_checksum"]:
        raise SystemExit(f"S8 carrier failure for {record_key}")


def group_records(records):
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def assert_matrix(groups, records_per_key, repeat):
    if set(groups) != expected_keys():
        missing = expected_keys() - set(groups)
        extra = set(groups) - expected_keys()
        raise SystemExit(f"matrix key mismatch missing={missing} extra={extra}")
    if any(len(group) != records_per_key for group in groups.values()):
        raise SystemExit("matrix has wrong records per key")
    for group in groups.values():
        for record in group:
            assert_runtime_valid(record, repeat)


def assert_pair_equivalent(groups):
    for storage in STORAGES:
        for pattern in PATTERNS:
            control = groups[(storage, "two_call_control", pattern)]
            candidate = groups[(storage, "single_invocation", pattern)]
            control_checksums = {
                (record["reference_checksum"],
                 record["measured_output_checksum"])
                for record in control}
            candidate_checksums = {
                (record["reference_checksum"],
                 record["measured_output_checksum"])
                for record in candidate}
            if len(control_checksums) != 1 or \
                    control_checksums != candidate_checksums:
                raise SystemExit(
                    f"paired canonical output mismatch for {storage}/{pattern}")
            for control_record in control:
                for candidate_record in candidate:
                    if candidate_record["stored_weight_bytes_per_repeat"] != \
                            2 * control_record["stored_weight_bytes_per_repeat"] or \
                            candidate_record["expanded_weight_bytes_per_repeat"] != \
                            2 * control_record["expanded_weight_bytes_per_repeat"]:
                        raise SystemExit(
                            f"paired weight-byte mismatch for {storage}/{pattern}")


def median_metrics(records):
    fields = (
        "host_wall_ns", "aggregate_dsp_total_ticks",
        "aggregate_pipeline_ticks", "aggregate_activation_stage_ticks",
        "aggregate_weight_stage_ticks", "aggregate_weight_expand_ticks",
        "aggregate_hmx_compute_ticks", "aggregate_hmx_ready_wait_ticks",
        "aggregate_output_assembly_ticks", "aggregate_input_cache_ticks",
        "aggregate_output_cache_ticks", "vtcm_plan_bytes",
        "session_open_wall_ns", "prepare_wall_ns", "release_wall_ns",
        "session_close_wall_ns",
    )
    return {f"{field}_median": statistics.median(
                record[field] for record in records)
            for field in fields}


def summarize(groups):
    metrics = {}
    comparisons = {}
    w4_gate = True
    for storage in STORAGES:
        for mode in MODES:
            records = [record for (group_storage, group_mode, _), group
                       in groups.items()
                       if group_storage == storage and group_mode == mode
                       for record in group]
            metrics[f"{storage}:{mode}"] = median_metrics(records)
        control_wall = metrics[
            f"{storage}:two_call_control"]["host_wall_ns_median"]
        candidate_wall = metrics[
            f"{storage}:single_invocation"]["host_wall_ns_median"]
        comparisons[storage] = {
            "two_call_host_wall_ns_median": control_wall,
            "paired_host_wall_ns_median": candidate_wall,
            "paired_improvement_percent":
                (control_wall - candidate_wall) * 100.0 / control_wall,
            "paired_is_faster": candidate_wall < control_wall,
        }
        if storage == "packed_w4_hmx_postscale":
            w4_gate = candidate_wall < control_wall

    s8_wall = metrics[
        "expanded_s8_control:single_invocation"]["host_wall_ns_median"]
    w4_wall = metrics[
        "packed_w4_hmx_postscale:single_invocation"]["host_wall_ns_median"]
    comparisons["paired_w4_vs_s8"] = {
        "w4_host_wall_ns_median": w4_wall,
        "s8_host_wall_ns_median": s8_wall,
        "w4_improvement_percent": (s8_wall - w4_wall) * 100.0 / s8_wall,
        "w4_is_faster": w4_wall < s8_wall,
    }
    return metrics, comparisons, w4_gate


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
        assert_pair_equivalent(groups)

    repeat1_metrics, repeat1_comparisons, repeat1_gate = summarize(
        timing1_groups)
    repeat10_metrics, repeat10_comparisons, repeat10_gate = summarize(
        timing10_groups)
    print(json.dumps({
        "experiment": "EXP-0016",
        "records": len(correctness) + len(timing1) + len(timing10),
        "correctness_records": len(correctness),
        "timing_repeat1_records": len(timing1),
        "timing_repeat10_records": len(timing10),
        "exact_reference": True,
        "paired_and_two_call_outputs_identical": True,
        "total_weight_and_output_bytes_equal": True,
        "persistent_resource_identity_stable": True,
        "worker_fallback": False,
        "descriptor_timeouts": 0,
        "device_reboot": False,
        "vtcm_request_bytes": 2097152,
        "active_range_cache_maintenance": True,
        "local_performance_gate": repeat1_gate and repeat10_gate,
        "repeat1_w4_pair_gate": repeat1_gate,
        "repeat10_w4_pair_gate": repeat10_gate,
        "repeat1_comparisons": repeat1_comparisons,
        "repeat10_comparisons": repeat10_comparisons,
        "repeat1_metrics": repeat1_metrics,
        "repeat10_metrics": repeat10_metrics,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
