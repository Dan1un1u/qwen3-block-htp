#!/usr/bin/env python3
"""Validate EXP-0185 session-native K-tail evidence."""

from __future__ import annotations

import summarize_exp0183 as exp183
import summarize_exp0184 as exp184


implementation = exp184.implementation
prior_validate_layout = exp184.validate_layout
implementation.EXPERIMENT_NUMBER = 185
implementation.EXPERIMENT_NAME = "EXP-0185"
implementation.DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0185-w4u8-decode-vtcm-k-tail-no-journal"
)
implementation.CANDIDATE_LABEL = "session-native VTCM K7+V8 tail"
implementation.REPORT_TITLE = (
    "Session-native K tail without mutable DDR journal"
)
implementation.REPORT_DESCRIPTION = (
    "Seven cached K heads per layer keep their mutable unsealed tail only "
    "in the prepared session's persistent VTCM atlas. Their redundant "
    "128-byte per-token DDR row journal writes are removed; the eighth "
    "head remains an explicit fallback and every completed 32-token "
    "native segment is still sealed to the unchanged immutable DDR ABI."
)
exp183.CANDIDATE_K_FORMAT = 14

SKIP_FIELDS = (
    "u8_cache_k_vtcm_tail_ddr_write_skip_count",
    "u8_cache_k_vtcm_tail_ddr_write_skip_bytes",
)
implementation.COUNTERS = implementation.COUNTERS + SKIP_FIELDS


def validate_layout(run: list[dict[str, object]], cell: str) -> bool:
    if not prior_validate_layout(run, cell):
        return False
    candidate = cell == "quartet"
    for index, profile in enumerate(run):
        count = int(profile.get(SKIP_FIELDS[0], 0))
        size = int(profile.get(SKIP_FIELDS[1], 0))
        if candidate and index != 0:
            if count != 28 * 7 or size != 28 * 7 * 128:
                return False
        elif count != 0 or size != 0:
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
        expected_read_saved = 0 if index == 0 else 28 * 7 * (
            tail_rows * 128 + (4096 if tail_rows == 0 else 0)
        )
        expected_write_saved = 0 if index == 0 else 28 * 7 * 128
        if (int(control_profile["scan_cache_ddr_read_bytes"])
                - int(candidate_profile["scan_cache_ddr_read_bytes"])
                != expected_read_saved):
            return False
        if (int(control_profile["scan_cache_ddr_write_bytes"])
                - int(candidate_profile["scan_cache_ddr_write_bytes"])
                != expected_write_saved):
            return False
        control_peak = int(control_profile["vtcm_peak_plan_bytes"])
        expected_peak = ((control_peak + 127) // 128) * 128 + exp183.K_ATLAS_BYTES
        if int(candidate_profile["vtcm_peak_plan_bytes"]) != expected_peak:
            return False
    return True


def extra_gates(diagnostics, counters, module_rows, runs):
    del counters, runs
    return {
        "candidate_strictly_lowers_cache_carrier_update":
            diagnostics["quartet"]["u8_cache_native_append_update_us"]
            < diagnostics["control"]["u8_cache_native_append_update_us"],
        "candidate_strictly_lowers_attention_module":
            module_rows["quartet"]["decode"]["QK-Softmax-AV"]
            < module_rows["control"]["decode"]["QK-Softmax-AV"],
    }


implementation.validate_layout = validate_layout
implementation.validate_vtcm_read_contract = validate_vtcm_read_contract
implementation.EXTRA_GATE_BUILDER = extra_gates


if __name__ == "__main__":
    implementation.main()
