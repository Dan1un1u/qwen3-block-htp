# EXP-0222 running

Read authority then preflight. Source codex/exp-0222-w4f16-lm-head-ablation parent 08a67eee666fe5ca7335a8150f7207635019b588. Follow docs/experiments/EXP-0222.md. Host-only six-way head comparison; no device work or profiling; discuss result before next direction. No jobs launched yet.

Implementation 61cf9a74a4bf0b2e8bc2da82325dc4a6fa0c7cf6 committed and synced. Host A/B/C sequential process started (tool session63558). It preserves execution logs under results/exp0222/{A,B,C}_execution.log and refuses overwrite. Per-variant summary.json signals complete W4-control reproduction, FP16-head full quality, four-case repeat for both heads, head artifact and provenance. Inspect existing files/process before resuming; do not duplicate. No device jobs. Close after all three summaries; report paired software values, not historical DSP as control.

Checkpoint: A complete, W4 software control exact (NLL difference0); W4 NLL3.874188/9tasks versus FP16-head3.853870/8. Four bilingual repeats exact, non-head hashes unchanged. B running in same63558 sequential process, C follows. Source report/archive helper committed at 412a5b761ca18e064edacef78d5bb5bcf4bd761c; measurement implementation unchanged. Do not duplicate or rerun A.
