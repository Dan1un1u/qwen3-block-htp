# EXP0258 active checkpoint

Source codex/exp-0258-dense-r3-e2e-diagnostic 7fc5213e436126b5f44086199217b232b2869feb; ABI119,28layer binaries staged under results/binaries/l28_attempt1. Authority PC072 user exception for speed-only fullmodel despite known idealR3 whole gate failure. Do not mislabel numerical or speed eligibility as pass.

/mnt/d/llm_exp/results/qwen3-block-htp/exp0258: export_audit.json, slice_gate.json, smoke_a0/smoke_a1, short_gate.json allcaptured. Correct R3_ALL QK/rotatedofflineU8prefix; noR3 control exactly EXP0257 seed_full_nr64. Every fullmodel step has28R3calls/43008prefill or672decode live rows, no refinement. Prefill196 native directW4 projections,decode197 includingLMhead. All physical/timing ledger checks pass.

Five short rounds complete. Paired repeat10 R3/noR3 latency ratios prefill3.84825,decode1.59523, both fail>10percent speed eligibility. User explicitly authorized diagnostic costs, so next run ten formal rotated pairs: scripts/device_exp0258.py formal. Inspect processes/logs before resuming; do not duplicate active work. Runner writes formal_gate.json only after ten rounds. Capture timings whole M64+15 actualfeedback,repeat1/repeat10 resetsequences.

After formal: scripts/report_exp0258.py writes complete reports, independently rechecks everytimed profile and CI. Copy REPORT/FULL_PROFILING_REPORT into source docs,commitpush; run finalize_exp0258.py with clean source, no redirected logfile inside result duringseal. Register closure and sync memory. Existing RECOVERY.md explains retained routinecollector/export defects. No PPL/newalgorithm/promotion. EXP0257 originallegacyC0 transient remains unresolved and is not our control.
