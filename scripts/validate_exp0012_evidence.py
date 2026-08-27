#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


PATTERNS = ("identity", "signed", "structured", "boundary")
PROJECTION_CONFIG = {
    "gate_up": {"k_tiles": 64, "n_tiles": 192, "w4_chunk": 64,
                "w4_chunks": 1},
    "down": {"k_tiles": 192, "n_tiles": 64, "w4_chunk": 96,
             "w4_chunks": 2},
}


def variants(projection):
    chunk = PROJECTION_CONFIG[projection]["w4_chunk"]
    return (
        ("expanded_s8_control", "exp0005_full_bundle_control", 1, False),
        ("expanded_s8_control", "expanded_s8_dma_batch2", 2, False),
        ("expanded_s8_control", "expanded_s8_dma_chain2", 2, True),
        ("packed_w4_hmx_postscale", f"slots4_chunk{chunk}", 1, False),
        ("packed_w4_hmx_postscale",
         f"slots4_chunk{chunk}_dma_batch2", 2, False),
        ("packed_w4_hmx_postscale",
         f"slots4_chunk{chunk}_dma_chain2", 2, True),
        ("packed_w4_hmx_postscale",
         f"slots4_chunk{chunk}_dma_chain4", 4, True),
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
    return (record["weight_storage"], record["physical_plan"],
            record["projection"], record["pattern"])


def plan_config(record):
    projection = record["projection"]
    for storage, plan, batch, linked in variants(projection):
        if (record["weight_storage"], record["physical_plan"]) != (
                storage, plan):
            continue
        if storage == "expanded_s8_control":
            return (1, 2, 32, batch,
                    PROJECTION_CONFIG[projection]["k_tiles"] // 32,
                    linked)
        return (6, 4, PROJECTION_CONFIG[projection]["w4_chunk"], batch,
                PROJECTION_CONFIG[projection]["w4_chunks"], linked)
    raise SystemExit(f"unknown variant: {key(record)}")


def assert_runtime_valid(record, repeat):
    record_key = key(record)
    if record["experiment"] != "EXP-0012":
        raise SystemExit(f"wrong experiment for {record_key}")
    if record["repeat_count"] != repeat:
        raise SystemExit(f"wrong repeat for {record_key}")
    if record["rpc_result"] != 0 or record["dsp_status"] != 0:
        raise SystemExit(f"runtime failure for {record_key}")
    if record["mismatches"] != 0:
        raise SystemExit(f"output mismatch for {record_key}")
    if record["dma_status"] != 0 or record["sync_status"] != 0:
        raise SystemExit(f"DMA/sync failure for {record_key}")
    for field in (
            "dcvs_power_setup_status", "dcvs_power_reset_status",
            "hmx_resource_status", "hmx_lock_status", "hmx_unlock_status",
            "hmx_release_status", "hmx_thread_create_status",
            "hmx_thread_join_status", "hvx_lock_status",
            "hvx_unlock_status"):
        if record[field] != 0:
            raise SystemExit(f"{field} failure for {record_key}")

    config = PROJECTION_CONFIG[record["projection"]]
    workers, slots, chunk, batch, chunks, linked = plan_config(record)
    logical_bundles = config["n_tiles"] * repeat
    weight_groups = logical_bundles // batch
    activation_descriptors = config["k_tiles"]
    expected_submits = activation_descriptors + weight_groups
    expected_descriptors = activation_descriptors + (
        logical_bundles if linked else weight_groups)
    expected_chains = weight_groups if linked else 0
    expected_waits = (2 * activation_descriptors + logical_bundles +
                      2 * weight_groups if linked else 2 * expected_submits)

    if record["requested_hvx_workers"] != workers:
        raise SystemExit(f"wrong workers for {record_key}")
    if record["compressed_slot_count"] != slots:
        raise SystemExit(f"wrong slots for {record_key}")
    if record["chunk_tiles"] != chunk:
        raise SystemExit(f"wrong chunk for {record_key}")
    if record["dma_bundle_batch"] != batch:
        raise SystemExit(f"wrong DMA batch for {record_key}")
    if record["chunks_per_output"] != chunks:
        raise SystemExit(f"wrong chunk publications for {record_key}")
    if record["hmx_execution_count"] != 12288 * repeat:
        raise SystemExit(f"wrong HMX execution count for {record_key}")
    if record["hmx_stream_count"] != 384 * repeat:
        raise SystemExit(f"wrong HMX stream count for {record_key}")
    if record["weight_bundle_stage_count"] != logical_bundles:
        raise SystemExit(f"wrong logical bundle count for {record_key}")
    if record["dma_submit_count"] != expected_submits:
        raise SystemExit(f"wrong DMA submit count for {record_key}")
    if record["dma_wait_count"] != expected_waits:
        raise SystemExit(f"wrong DMA wait count for {record_key}")
    if record["dma_descriptor_count"] != expected_descriptors:
        raise SystemExit(f"wrong DMA descriptor count for {record_key}")
    if record["dma_chain_count"] != expected_chains:
        raise SystemExit(f"wrong DMA chain count for {record_key}")
    if record["dma_descriptor_completion_count"] != expected_descriptors:
        raise SystemExit(f"incomplete DMA descriptors for {record_key}")
    if record["dma_descriptor_timeout_count"] != 0:
        raise SystemExit(f"DMA descriptor timeout for {record_key}")
    if record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM plan exceeds request for {record_key}")

    if record["weight_storage"] == "packed_w4_hmx_postscale":
        if record["hvx_workers_created"] != workers or \
                record["hvx_workers_locked"] != workers:
            raise SystemExit(f"worker fallback for {record_key}")
        if record["hvx_hmx_overlap_observed"] != 1 or \
                record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"missing HVX/HMX overlap for {record_key}")
        if record["hmx_carrier_checksum"] != record["packed_w4_checksum"]:
            raise SystemExit(f"HMX carrier is not raw q4 for {record_key}")
    elif record["hmx_carrier_checksum"] != record[
            "expanded_carrier_checksum"]:
        raise SystemExit(f"S8 carrier mismatch for {record_key}")


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return grouped


def median_metrics(records):
    fields = (
        "host_wall_ns", "dsp_total_ticks", "pipeline_ticks",
        "weight_stage_ticks", "weight_expand_ticks", "hmx_compute_ticks",
        "hmx_ready_wait_ticks", "producer_slot_wait_ticks",
        "expanded_slot_wait_ticks", "dma_submit_count", "dma_wait_count",
        "dma_descriptor_count", "dma_chain_count",
        "dma_descriptor_completion_count", "hvx_max_active_workers",
        "vtcm_plan_bytes",
    )
    metrics = {
        f"{field}_median": statistics.median(record[field] for record in records)
        for field in fields
    }
    sample = records[0]
    transferred = sample["stored_weight_bytes_per_repeat"] * sample[
        "repeat_count"]
    ticks = metrics["weight_stage_ticks_median"]
    metrics["effective_weight_gbytes_per_second"] = (
        transferred * 19200000.0 / ticks / 1.0e9 if ticks else 0.0)
    return metrics


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing = load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = group_records(correctness)
    timing_groups = group_records(timing)
    expected_keys = {
        (storage, plan, projection, pattern)
        for projection in PROJECTION_CONFIG
        for storage, plan, _, _ in variants(projection)
        for pattern in PATTERNS
    }
    if set(correctness_groups) != expected_keys:
        raise SystemExit("correctness matrix has missing/unexpected keys")
    if set(timing_groups) != expected_keys:
        raise SystemExit("timing matrix has missing/unexpected keys")
    if any(len(group) != 1 for group in correctness_groups.values()):
        raise SystemExit("correctness matrix must have one record per key")
    if any(len(group) != 3 for group in timing_groups.values()):
        raise SystemExit("timing matrix must have three records per key")
    for record in correctness:
        assert_runtime_valid(record, 1)
    for record in timing:
        assert_runtime_valid(record, 10)

    comparable = (
        "reference_checksum", "reference_min", "reference_max",
        "expanded_carrier_checksum", "packed_w4_checksum", "projection_m",
        "projection_k", "projection_n", "hmx_execution_count",
        "hmx_stream_count", "output_tile_count",
        "weight_bundle_stage_count",
    )
    for grouped in (correctness_groups, timing_groups):
        for projection in PROJECTION_CONFIG:
            for pattern in PATTERNS:
                peers = [record for group_key, group in grouped.items()
                         if group_key[2:] == (projection, pattern)
                         for record in group]
                reference = peers[0]
                for record in peers[1:]:
                    for field in comparable:
                        if record[field] != reference[field]:
                            raise SystemExit(
                                f"{projection}/{pattern}: variants differ at {field}")

    metrics = {}
    selections = {}
    local_gate = True
    stretch_gate = True
    for projection in PROJECTION_CONFIG:
        projection_metrics = {}
        for storage, plan, batch, linked in variants(projection):
            records = [record for group_key, group in timing_groups.items()
                       if group_key[0] == storage and group_key[1] == plan and
                       group_key[2] == projection for record in group]
            variant_key = (storage, plan, batch, linked)
            projection_metrics[variant_key] = median_metrics(records)
            metrics[f"{projection}:{storage}:{plan}"] = projection_metrics[
                variant_key]

        linked_w4 = [(variant_key, value)
                     for variant_key, value in projection_metrics.items()
                     if variant_key[0] == "packed_w4_hmx_postscale" and
                     variant_key[3]]
        all_s8 = [(variant_key, value)
                  for variant_key, value in projection_metrics.items()
                  if variant_key[0] == "expanded_s8_control"]
        parent_plan = f"slots4_chunk{PROJECTION_CONFIG[projection]['w4_chunk']}_dma_batch2"
        parent = next(value for variant_key, value in projection_metrics.items()
                      if variant_key[1] == parent_plan)
        selected_w4 = min(
            linked_w4, key=lambda item: item[1]["host_wall_ns_median"])
        selected_s8 = min(
            all_s8, key=lambda item: item[1]["host_wall_ns_median"])
        w4_wall = selected_w4[1]["host_wall_ns_median"]
        s8_wall = selected_s8[1]["host_wall_ns_median"]
        parent_wall = parent["host_wall_ns_median"]
        improves_parent = w4_wall < parent_wall
        beats_s8 = w4_wall < s8_wall
        local_gate &= improves_parent
        stretch_gate &= beats_s8
        selections[projection] = {
            "selected_linked_w4_plan": selected_w4[0][1],
            "selected_linked_w4_dma_batch": selected_w4[0][2],
            "selected_s8_plan": selected_s8[0][1],
            "selected_s8_dma_batch": selected_s8[0][2],
            "parent_contiguous_w4_plan": parent_plan,
            "parent_contiguous_w4_host_wall_ns_median": parent_wall,
            "selected_linked_w4_host_wall_ns_median": w4_wall,
            "selected_s8_host_wall_ns_median": s8_wall,
            "w4_vs_parent_improvement_percent":
                (parent_wall - w4_wall) * 100.0 / parent_wall,
            "w4_vs_s8_improvement_percent":
                (s8_wall - w4_wall) * 100.0 / s8_wall,
            "improves_contiguous_batch2": improves_parent,
            "beats_fair_s8": beats_s8,
        }

    print(json.dumps({
        "experiment": "EXP-0012",
        "records": len(correctness) + len(timing),
        "correctness_records": len(correctness),
        "timing_records": len(timing),
        "exact_reference": True,
        "all_variants_equivalent": True,
        "same_hmx_execution_and_stream_counts": True,
        "worker_fallback": False,
        "descriptor_timeouts": 0,
        "device_reboot": False,
        "dcvs_performance_vote": True,
        "hvx_hmx_overlap": True,
        "local_performance_gate": local_gate,
        "beats_fairly_searched_expanded_s8": stretch_gate,
        "selections": selections,
        "metrics": metrics,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
