#!/usr/bin/env python3
"""Validate EXP-0184 HVX K-tail row-update evidence."""

from __future__ import annotations

import summarize_exp0183 as exp183


implementation = exp183.implementation
prior_validate_layout = exp183.validate_layout
implementation.EXPERIMENT_NUMBER = 184
implementation.EXPERIMENT_NAME = "EXP-0184"
implementation.DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0184-w4u8-decode-hvx-k-tail-update"
)
implementation.CANDIDATE_LABEL = "VTCM K7+V8 with HVX append"
implementation.REPORT_TITLE = "HVX single-row native K-tail update"
implementation.REPORT_DESCRIPTION = (
    "The candidate keeps EXP-0183 partial K-tail residency but replaces each "
    "scalar 128-byte center/scatter with one HVX center, signed reduction and "
    "vscatter at the active HMX output lane."
)
implementation.DIAGNOSTICS = implementation.DIAGNOSTICS + (
    "u8_cache_k_vtcm_tail_hvx_row_update_ticks",
)
implementation.COUNTERS = implementation.COUNTERS + (
    "u8_cache_k_vtcm_tail_hvx_row_update_count",
)


def validate_layout(run: list[dict[str, object]], cell: str) -> bool:
    if not prior_validate_layout(run, cell):
        return False
    candidate = cell == "quartet"
    for index, profile in enumerate(run):
        count = int(profile.get(
            "u8_cache_k_vtcm_tail_hvx_row_update_count", 0))
        ticks = int(profile.get(
            "u8_cache_k_vtcm_tail_hvx_row_update_ticks", 0))
        if candidate and index != 0:
            if count != LAYERS_CACHED_HEADS or ticks <= 0:
                return False
        elif count != 0 or ticks != 0:
            return False
    return True


LAYERS_CACHED_HEADS = 28 * 7
implementation.validate_layout = validate_layout


def extra_gates(diagnostics, counters, module_rows, runs):
    del counters, module_rows, runs
    return {
        "candidate_strictly_lowers_cache_carrier_update":
            diagnostics["quartet"]["u8_cache_native_append_update_us"]
            < diagnostics["control"]["u8_cache_native_append_update_us"],
    }


implementation.EXTRA_GATE_BUILDER = extra_gates


if __name__ == "__main__":
    implementation.main()
