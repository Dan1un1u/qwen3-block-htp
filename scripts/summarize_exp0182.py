#!/usr/bin/env python3
"""Validate EXP-0182 prepared-session VTCM V-tail evidence."""

from __future__ import annotations

import summarize_exp0181 as implementation


implementation.EXPERIMENT_NUMBER = 182
implementation.EXPERIMENT_NAME = "EXP-0182"
implementation.DEFAULT_SOURCE_BRANCH = (
    "codex/exp-0182-w4u8-decode-vtcm-resident-v-tail"
)
implementation.CANDIDATE_FORMAT = 12
implementation.CANDIDATE_LABEL = "VTCM-tail-v7"
implementation.REPORT_TITLE = "Prepared-session VTCM-resident mutable V tail"
implementation.VTCM_TAIL_MODE = True


if __name__ == "__main__":
    implementation.main()
