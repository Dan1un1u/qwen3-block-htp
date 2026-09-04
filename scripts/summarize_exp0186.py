#!/usr/bin/env python3
"""Validate EXP-0186 session-native direct V-tail evidence."""

from __future__ import annotations

import summarize_exp0185 as exp185


implementation = exp185.implementation
implementation.EXPERIMENT_NUMBER = 186
implementation.EXPERIMENT_NAME = "EXP-0186"
implementation.DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0186-w4u8-decode-vtcm-v-tail-no-journal"
)
implementation.CANDIDATE_LABEL = "session-native K7+V8 direct V tail"
implementation.REPORT_TITLE = (
    "Session-native V tail with direct projection-carrier append"
)
implementation.REPORT_DESCRIPTION = (
    "The candidate retains accepted EXP-0185 K14 and the exact VTCM/DDR "
    "segment ABI, but copies each decode V row directly from the QKV native "
    "carrier into the persistent VTCM tail. It removes the temporary "
    "row-major bounce and all mutable V-tail DDR journal writes."
)

LAYERS = 28
KV_HEADS = 8
CACHED_K_HEADS = 7
HEAD_TILES = 4
APPEND_HEADS = LAYERS * KV_HEADS
CACHED_HEADS = LAYERS * CACHED_K_HEADS
V_ATLAS_BYTES = LAYERS * KV_HEADS * 32 * 128
K_HEAD_BYTES = 4096 + 32 * 4
K_ATLAS_BYTES = LAYERS * CACHED_K_HEADS * K_HEAD_BYTES
CONTROL_V_FORMAT = 12
CANDIDATE_V_FORMAT = 15

DIRECT_TICK_FIELD = "u8_cache_v_vtcm_tail_direct_row_update_ticks"
DIRECT_COUNTER_FIELDS = (
    "u8_cache_v_vtcm_tail_direct_row_update_count",
    "u8_cache_v_vtcm_tail_ddr_write_skip_count",
    "u8_cache_v_vtcm_tail_ddr_write_skip_bytes",
)
implementation.DIAGNOSTICS = implementation.DIAGNOSTICS + (
    DIRECT_TICK_FIELD,
)
implementation.COUNTERS = implementation.COUNTERS + DIRECT_COUNTER_FIELDS


def validate_layout(run: list[dict[str, object]], cell: str) -> bool:
    candidate = cell == "quartet"
    expected_v_format = (
        CANDIDATE_V_FORMAT if candidate else CONTROL_V_FORMAT
    )
    quartet_fields = (
        "u8_cache_v_quartet_append_count",
        "u8_cache_v_quartet_publish_count",
        "u8_cache_v_quartet_attention_publish_count",
        "u8_cache_v_quartet_partial_pack_rows",
        "u8_cache_v_quartet_full_tile_rmw_count",
        "u8_cache_v_quartet_native_load_bytes",
    )
    for index, profile in enumerate(run):
        if not (
            int(profile["kv_cache_k_format"]) == 14
            and int(profile["kv_cache_v_format"]) == expected_v_format
            and all(int(profile.get(field, 0)) == 0
                    for field in quartet_fields)
        ):
            return False

        direct_count = int(profile.get(DIRECT_COUNTER_FIELDS[0], 0))
        direct_ticks = int(profile.get(DIRECT_TICK_FIELD, 0))
        v_skip_count = int(profile.get(DIRECT_COUNTER_FIELDS[1], 0))
        v_skip_bytes = int(profile.get(DIRECT_COUNTER_FIELDS[2], 0))
        if index == 0:
            if not (
                int(profile.get("u8_cache_v_vtcm_tail_init_count", 0)) == 1
                and int(profile.get("u8_cache_v_vtcm_tail_init_bytes", 0))
                    == V_ATLAS_BYTES
                and int(profile.get("u8_cache_k_vtcm_tail_init_count", 0)) == 1
                and int(profile.get("u8_cache_k_vtcm_tail_init_bytes", 0))
                    == K_ATLAS_BYTES
                and direct_count == 0 and direct_ticks == 0
                and v_skip_count == 0 and v_skip_bytes == 0
                and int(profile.get(
                    "u8_cache_k_vtcm_tail_ddr_write_skip_count", 0)) == 0
                and int(profile.get(
                    "u8_cache_k_vtcm_tail_ddr_write_skip_bytes", 0)) == 0
            ):
                return False
            continue

        valid = int(profile["valid_length"])
        tail_rows = valid % 32
        groups = (tail_rows + 3) // 4
        expected_v_publish = APPEND_HEADS if valid % 4 == 0 else 0
        expected_v_partial = APPEND_HEADS * (tail_rows % 4)
        expected_v_load = LAYERS * KV_HEADS * HEAD_TILES * groups * 128
        if not (
            int(profile.get("u8_cache_v_vtcm_tail_init_count", 0)) == 0
            and int(profile.get("u8_cache_v_vtcm_tail_init_bytes", 0)) == 0
            and int(profile.get(
                "u8_cache_v_vtcm_tail_row_update_count", 0)) == APPEND_HEADS
            and int(profile.get(
                "u8_cache_v_vtcm_tail_publish_count", 0)) == expected_v_publish
            and int(profile.get(
                "u8_cache_v_vtcm_tail_seal_count", 0)) == (
                    APPEND_HEADS if tail_rows == 0 else 0)
            and int(profile.get(
                "u8_cache_v_vtcm_tail_partial_pack_rows", 0))
                    == expected_v_partial
            and int(profile.get(
                "u8_cache_v_vtcm_tail_native_load_bytes", 0))
                    == expected_v_load
            and int(profile.get(
                "u8_cache_k_vtcm_tail_row_update_count", 0)) == CACHED_HEADS
            and int(profile.get(
                "u8_cache_k_vtcm_tail_cached_head_count", 0)) == CACHED_HEADS
            and int(profile.get(
                "u8_cache_k_vtcm_tail_fallback_head_count", 0)) == LAYERS
            and int(profile.get(
                "u8_cache_k_vtcm_tail_seal_count", 0)) == (
                    CACHED_HEADS if tail_rows == 0 else 0)
            and int(profile.get(
                "u8_cache_k_vtcm_tail_native_load_bytes", 0)) == (
                    CACHED_HEADS * 4096 if tail_rows != 0 else 0)
            and int(profile.get(
                "u8_cache_k_vtcm_tail_correction_load_bytes", 0))
                    == CACHED_HEADS * tail_rows * 4
            and int(profile.get(
                "u8_cache_k_vtcm_tail_hvx_row_update_count", 0))
                    == CACHED_HEADS
            and int(profile.get(
                "u8_cache_k_vtcm_tail_hvx_row_update_ticks", 0)) > 0
            and int(profile.get(
                "u8_cache_k_vtcm_tail_ddr_write_skip_count", 0))
                    == CACHED_HEADS
            and int(profile.get(
                "u8_cache_k_vtcm_tail_ddr_write_skip_bytes", 0))
                    == CACHED_HEADS * 128
        ):
            return False

        if candidate:
            if not (
                direct_count == APPEND_HEADS
                and direct_ticks > 0
                and v_skip_count == APPEND_HEADS
                and v_skip_bytes == APPEND_HEADS * 128
            ):
                return False
        elif not (
            direct_count == 0 and direct_ticks == 0
            and v_skip_count == 0 and v_skip_bytes == 0
        ):
            return False
    return True


def validate_vtcm_read_contract(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> bool:
    if len(control) != len(candidate):
        return False
    for index, (control_profile, candidate_profile) in enumerate(
            zip(control, candidate)):
        expected_saved = 0 if index == 0 else APPEND_HEADS * 128
        expected_descriptors = 0 if index == 0 else APPEND_HEADS
        if (int(control_profile["scan_cache_ddr_read_bytes"])
                != int(candidate_profile["scan_cache_ddr_read_bytes"])):
            return False
        if (int(control_profile["scan_cache_ddr_write_bytes"])
                - int(candidate_profile["scan_cache_ddr_write_bytes"])
                != expected_saved):
            return False
        if (int(control_profile["scan_cache_dma_descriptor_count"])
                - int(candidate_profile["scan_cache_dma_descriptor_count"])
                != expected_descriptors):
            return False
        if (int(control_profile["vtcm_peak_plan_bytes"])
                != int(candidate_profile["vtcm_peak_plan_bytes"])):
            return False
    return True


def extra_gates(diagnostics, counters, module_rows, runs):
    del counters, module_rows, runs
    return {
        "candidate_strictly_lowers_cache_carrier_update":
            diagnostics["quartet"]["u8_cache_native_append_update_us"]
            < diagnostics["control"]["u8_cache_native_append_update_us"],
        "candidate_strictly_lowers_cache_append_dma":
            diagnostics["quartet"]["scan_cache_append_us"]
            < diagnostics["control"]["scan_cache_append_us"],
    }


implementation.validate_layout = validate_layout
implementation.validate_vtcm_read_contract = validate_vtcm_read_contract
implementation.EXTRA_GATE_BUILDER = extra_gates


if __name__ == "__main__":
    implementation.main()
