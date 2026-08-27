#!/usr/bin/env python3
import json
import pathlib
import sys

import validate_exp0012_evidence as base


PATTERNS = base.PATTERNS
PROJECTION_CONFIG = base.PROJECTION_CONFIG


def variants(projection):
    chunk = PROJECTION_CONFIG[projection]["w4_chunk"]
    return (
        ("expanded_s8_control", "exp0005_full_bundle_control",
         1, False, 2, 8, False),
        ("expanded_s8_control", "expanded_s8_dma_batch2",
         2, False, 2, 8, False),
        ("expanded_s8_control", "expanded_s8_dma_chain2",
         2, True, 2, 8, False),
        ("packed_w4_hmx_postscale", f"slots4_chunk{chunk}_dma_batch2",
         2, False, 4, 8, False),
        ("packed_w4_hmx_postscale", f"slots4e7_chunk{chunk}_dma_batch2",
         2, False, 4, 7, False),
        ("packed_w4_hmx_postscale", f"slots8e7_chunk{chunk}_dma_batch4",
         4, False, 8, 7, True),
        ("packed_w4_hmx_postscale", f"slots8e7_chunk{chunk}_dma_chain4",
         4, True, 8, 7, True),
    )


def plan_config(record):
    projection = record["projection"]
    for storage, plan, batch, linked, compressed, expanded, candidate in \
            variants(projection):
        if (record["weight_storage"], record["physical_plan"]) != (
                storage, plan):
            continue
        if storage == "expanded_s8_control":
            return (1, compressed, expanded, 32, batch,
                    PROJECTION_CONFIG[projection]["k_tiles"] // 32,
                    linked, candidate)
        return (6, compressed, expanded,
                PROJECTION_CONFIG[projection]["w4_chunk"], batch,
                PROJECTION_CONFIG[projection]["w4_chunks"], linked,
                candidate)
    raise SystemExit(f"unknown variant: {base.key(record)}")


def assert_runtime_valid(record, repeat):
    record_key = base.key(record)
    if record["experiment"] != "EXP-0013":
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
    workers, compressed, expanded, chunk, batch, chunks, linked, _ = \
        plan_config(record)
    logical_bundles = config["n_tiles"] * repeat
    weight_groups = logical_bundles // batch
    activation_descriptors = config["k_tiles"]
    expected_submits = activation_descriptors + weight_groups
    expected_descriptors = activation_descriptors + (
        logical_bundles if linked else weight_groups)
    expected_chains = weight_groups if linked else 0
    expected_waits = (2 * activation_descriptors + logical_bundles +
                      2 * weight_groups if linked else 2 * expected_submits)
    expected_expands = logical_bundles * chunks

    expected_fields = {
        "requested_hvx_workers": workers,
        "compressed_slot_count": compressed,
        "expanded_chunk_slot_count": expanded,
        "chunk_tiles": chunk,
        "dma_bundle_batch": batch,
        "chunks_per_output": chunks,
        "hmx_execution_count": 12288 * repeat,
        "hmx_stream_count": 384 * repeat,
        "weight_bundle_stage_count": logical_bundles,
        "dma_submit_count": expected_submits,
        "dma_wait_count": expected_waits,
        "dma_descriptor_count": expected_descriptors,
        "dma_chain_count": expected_chains,
        "dma_descriptor_completion_count": expected_descriptors,
        "dma_descriptor_timeout_count": 0,
        "weight_slot_reuse_count": max(0, logical_bundles - compressed),
        "expanded_chunk_slot_reuse_count": (
            max(0, expected_expands - expanded)
            if record["weight_storage"] == "packed_w4_hmx_postscale"
            else 0),
    }
    for field, expected in expected_fields.items():
        if record[field] != expected:
            raise SystemExit(
                f"wrong {field} for {record_key}: "
                f"{record[field]} != {expected}")
    if record["vtcm_requested_bytes"] != 2097152 or \
            record["vtcm_acquired_bytes"] != 2097152 or \
            record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM contract failure for {record_key}")

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


def median_metrics(records):
    metrics = base.median_metrics(records)
    sample = records[0]
    metrics["compressed_slot_count"] = sample["compressed_slot_count"]
    metrics["expanded_chunk_slot_count"] = sample[
        "expanded_chunk_slot_count"]
    return metrics


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    correctness = base.load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing = base.load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = base.group_records(correctness)
    timing_groups = base.group_records(timing)
    expected_keys = {
        (storage, plan, projection, pattern)
        for projection in PROJECTION_CONFIG
        for storage, plan, *_ in variants(projection)
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
        for variant in variants(projection):
            storage, plan, batch, linked, compressed, expanded, candidate = \
                variant
            records = [record for group_key, group in timing_groups.items()
                       if group_key[0] == storage and group_key[1] == plan and
                       group_key[2] == projection for record in group]
            projection_metrics[variant] = median_metrics(records)
            metrics[f"{projection}:{storage}:{plan}"] = projection_metrics[
                variant]

        candidates = [(variant, value)
                      for variant, value in projection_metrics.items()
                      if variant[6]]
        s8 = [(variant, value)
              for variant, value in projection_metrics.items()
              if variant[0] == "expanded_s8_control"]
        parent = next((variant, value)
                      for variant, value in projection_metrics.items()
                      if variant[0] == "packed_w4_hmx_postscale" and
                      variant[4:6] == (4, 8))
        isolation = next((variant, value)
                         for variant, value in projection_metrics.items()
                         if variant[0] == "packed_w4_hmx_postscale" and
                         variant[4:6] == (4, 7))
        selected = min(
            candidates, key=lambda item: item[1]["host_wall_ns_median"])
        selected_s8 = min(
            s8, key=lambda item: item[1]["host_wall_ns_median"])
        candidate_wall = selected[1]["host_wall_ns_median"]
        parent_wall = parent[1]["host_wall_ns_median"]
        isolation_wall = isolation[1]["host_wall_ns_median"]
        s8_wall = selected_s8[1]["host_wall_ns_median"]
        improves_parent = candidate_wall < parent_wall
        beats_s8 = candidate_wall < s8_wall
        local_gate &= improves_parent
        stretch_gate &= beats_s8
        selections[projection] = {
            "selected_candidate_plan": selected[0][1],
            "selected_candidate_linked": selected[0][3],
            "selected_s8_plan": selected_s8[0][1],
            "parent_plan": parent[0][1],
            "isolation_plan": isolation[0][1],
            "parent_host_wall_ns_median": parent_wall,
            "isolation_host_wall_ns_median": isolation_wall,
            "candidate_host_wall_ns_median": candidate_wall,
            "selected_s8_host_wall_ns_median": s8_wall,
            "candidate_vs_parent_improvement_percent":
                (parent_wall - candidate_wall) * 100.0 / parent_wall,
            "candidate_vs_s8_improvement_percent":
                (s8_wall - candidate_wall) * 100.0 / s8_wall,
            "expanded_ring_7_vs_8_change_percent":
                (parent_wall - isolation_wall) * 100.0 / parent_wall,
            "improves_parent": improves_parent,
            "beats_fair_s8": beats_s8,
        }

    print(json.dumps({
        "experiment": "EXP-0013",
        "records": len(correctness) + len(timing),
        "correctness_records": len(correctness),
        "timing_records": len(timing),
        "exact_reference": True,
        "all_variants_equivalent": True,
        "same_hmx_execution_and_stream_counts": True,
        "worker_fallback": False,
        "descriptor_timeouts": 0,
        "device_reboot": False,
        "vtcm_request_bytes": 2097152,
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
