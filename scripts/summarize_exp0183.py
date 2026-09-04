#!/usr/bin/env python3
"""Validate EXP-0183 partial prepared-session VTCM K-tail evidence."""

from __future__ import annotations

import summarize_exp0181 as implementation


LAYERS = 28
KV_HEADS = 8
CACHED_HEADS = 7
CANDIDATE_K_FORMAT = 13
HEAD_TILES = 4
V_ATLAS_BYTES = LAYERS * KV_HEADS * 32 * 128
K_HEAD_BYTES = 4096 + 32 * 4
K_ATLAS_BYTES = LAYERS * CACHED_HEADS * K_HEAD_BYTES
K_FIELDS = (
    "u8_cache_k_vtcm_tail_init_count",
    "u8_cache_k_vtcm_tail_row_update_count",
    "u8_cache_k_vtcm_tail_seal_count",
    "u8_cache_k_vtcm_tail_cached_head_count",
    "u8_cache_k_vtcm_tail_fallback_head_count",
    "u8_cache_k_vtcm_tail_init_bytes",
    "u8_cache_k_vtcm_tail_native_load_bytes",
    "u8_cache_k_vtcm_tail_correction_load_bytes",
)


implementation.EXPERIMENT_NUMBER = 183
implementation.EXPERIMENT_NAME = "EXP-0183"
implementation.DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0183-w4u8-decode-vtcm-native-k-tail"
)
implementation.CANDIDATE_FORMAT = 12
implementation.CANDIDATE_LABEL = "VTCM K7+V8 tail"
implementation.REPORT_TITLE = (
    "Partial prepared-session VTCM-native mutable K tail"
)
implementation.REPORT_DESCRIPTION = (
    "The candidate retains the EXP-0182 VTCM-resident V tail and adds an "
    "exact native K carrier plus compact correction vector for seven of "
    "eight KV heads per layer. The eighth head remains an explicit DDR "
    "fallback, while the external segmented-v4 K/V journal and immutable "
    "segment ABI remain byte-identical."
)
implementation.VTCM_TAIL_MODE = False
implementation.COUNTERS = implementation.COUNTERS + K_FIELDS


def validate_layout(run: list[dict[str, object]], cell: str) -> bool:
    candidate = cell == "quartet"
    v_fields = (
        "u8_cache_v_vtcm_tail_init_count",
        "u8_cache_v_vtcm_tail_row_update_count",
        "u8_cache_v_vtcm_tail_publish_count",
        "u8_cache_v_vtcm_tail_seal_count",
        "u8_cache_v_vtcm_tail_partial_pack_rows",
        "u8_cache_v_vtcm_tail_init_bytes",
        "u8_cache_v_vtcm_tail_native_load_bytes",
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
            int(profile["kv_cache_k_format"]) ==
                (CANDIDATE_K_FORMAT if candidate else 8)
            and int(profile["kv_cache_v_format"]) == 12
            and all(int(profile.get(field, 0)) == 0
                    for field in quartet_fields)
        ):
            return False
        if index == 0:
            if not (
                int(profile.get("u8_cache_v_vtcm_tail_init_count", 0)) == 1
                and int(profile.get("u8_cache_v_vtcm_tail_init_bytes", 0))
                    == V_ATLAS_BYTES
                and all(int(profile.get(field, 0)) == 0
                        for field in v_fields
                        if field not in (
                            "u8_cache_v_vtcm_tail_init_count",
                            "u8_cache_v_vtcm_tail_init_bytes"))
            ):
                return False
            if candidate:
                if not (
                    int(profile.get("u8_cache_k_vtcm_tail_init_count", 0))
                        == 1
                    and int(profile.get(
                        "u8_cache_k_vtcm_tail_init_bytes", 0))
                        == K_ATLAS_BYTES
                    and all(int(profile.get(field, 0)) == 0
                            for field in K_FIELDS
                            if field not in (
                                "u8_cache_k_vtcm_tail_init_count",
                                "u8_cache_k_vtcm_tail_init_bytes"))
                ):
                    return False
            elif any(int(profile.get(field, 0)) != 0 for field in K_FIELDS):
                return False
            continue
        valid = int(profile["valid_length"])
        tail_rows = valid % 32
        groups = (tail_rows + 3) // 4
        append_heads = LAYERS * KV_HEADS
        cached_heads = LAYERS * CACHED_HEADS
        expected_v_publish = append_heads if valid % 4 == 0 else 0
        expected_v_partial = append_heads * (tail_rows % 4)
        expected_v_load = LAYERS * KV_HEADS * HEAD_TILES * groups * 128
        if not (
            int(profile.get("u8_cache_v_vtcm_tail_init_count", 0)) == 0
            and int(profile.get("u8_cache_v_vtcm_tail_init_bytes", 0)) == 0
            and int(profile.get("u8_cache_v_vtcm_tail_row_update_count", 0))
                == append_heads
            and int(profile.get("u8_cache_v_vtcm_tail_publish_count", 0))
                == expected_v_publish
            and int(profile.get("u8_cache_v_vtcm_tail_seal_count", 0))
                == (append_heads if tail_rows == 0 else 0)
            and int(profile.get(
                "u8_cache_v_vtcm_tail_partial_pack_rows", 0))
                == expected_v_partial
            and int(profile.get("u8_cache_v_vtcm_tail_native_load_bytes", 0))
                == expected_v_load
        ):
            return False
        if not candidate:
            if any(int(profile.get(field, 0)) != 0 for field in K_FIELDS):
                return False
            continue
        if not (
            int(profile.get("u8_cache_k_vtcm_tail_init_count", 0)) == 0
            and int(profile.get("u8_cache_k_vtcm_tail_init_bytes", 0)) == 0
            and int(profile.get("u8_cache_k_vtcm_tail_row_update_count", 0))
                == cached_heads
            and int(profile.get("u8_cache_k_vtcm_tail_cached_head_count", 0))
                == cached_heads
            and int(profile.get("u8_cache_k_vtcm_tail_fallback_head_count", 0))
                == LAYERS
            and int(profile.get("u8_cache_k_vtcm_tail_seal_count", 0))
                == (cached_heads if tail_rows == 0 else 0)
            and int(profile.get("u8_cache_k_vtcm_tail_native_load_bytes", 0))
                == (cached_heads * 4096 if tail_rows != 0 else 0)
            and int(profile.get(
                "u8_cache_k_vtcm_tail_correction_load_bytes", 0))
                == cached_heads * tail_rows * 4
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
        tail_rows = int(candidate_profile["valid_length"]) % 32
        expected_saved = 0 if index == 0 else LAYERS * CACHED_HEADS * (
            tail_rows * 128 + (4096 if tail_rows == 0 else 0)
        )
        if (int(control_profile["scan_cache_ddr_read_bytes"])
                - int(candidate_profile["scan_cache_ddr_read_bytes"])
                != expected_saved):
            return False
        if (int(candidate_profile["scan_cache_ddr_write_bytes"])
                != int(control_profile["scan_cache_ddr_write_bytes"])):
            return False
        control_peak = int(control_profile["vtcm_peak_plan_bytes"])
        expected_peak = ((control_peak + 127) // 128) * 128 + K_ATLAS_BYTES
        if int(candidate_profile["vtcm_peak_plan_bytes"]) != expected_peak:
            return False
    return True


implementation.validate_layout = validate_layout
implementation.validate_vtcm_read_contract = validate_vtcm_read_contract


if __name__ == "__main__":
    implementation.main()
