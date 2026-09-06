# EXP0228 completed — content-aware short-task scoring

Source branch codex/exp-0228-content-aware-short-task-scoring, HEAD 365d1e2f136125462fa4f7ecbe1abc93b4309ca1. EXP0228 completed, evidence valid, scorer/audit gate pass (not model effectiveness), adoption not applicable. No active experiment/device jobs. Next229 requires chosen user-approved model work. Baselines unchanged.

User explicitly permits correct answer content despite presentation differences. PC048 establishes qbh-content-v2 as short-task content metric with strict-v1 retained. Source scripts/score_short_tasks_v2.py provides reusable API/CLI; docs/SHORT_TASK_SCORING.md explains future scoring. Unknown prose remains unresolved until explicit per-output review; no substring acceptance. Numeric equations verified, JSON keys/types/values exact after fences, case-conversion correctness preserved. No prompt strengthening or model rerun this round.

Retrospective unchanged qbh outputs: BF16/F16 22strict->23content; original RTN8->14; EXP224A0 19->22; EXP227A18->22; EXP225R0 20->22; EXP227R19->22; EXP226selected9->16; frozen provisionalW4U8 0->0. Current A errors42/45/56/58 are correct content with formatting differences;46 gives6 instead of18,59 gives15 instead of5. FP16 actual error45 gives5 instead of7;46 equation6*3=18 is correct. Consequently prior one-task strict losses after reconstruction are not content-accuracy losses. PPL/independent validation and original experimental gates remain unchanged; no incremental rotation advantage established.

Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0228; all264stored grades and token decodes reproduced. Six synthetic behavioral test groups pass. Source input ledgers/tokenizer verified and rechecked unchanged. All per-item output and explicit review retained. Ledger SHA256 c61fcbf4dbd1f9ee5269597c1f6d1276c7a3d45ac4f173176fb983236812339c, closure SHA256 2f09f299891e28a321660a26ad8c3e206de678106789d556c2e0b0db1ea51e5b. Authority report docs/experiments/EXP-0228-RESULTS.md; profiling N/A is explicit in EXP-0228-PROFILE.md. Do not overwrite prior artifacts or silently reinterpret earlier strict scores.

Resolved stale PROJECT_STATUS summary pointers from EXP226 to latest completion while preserving prior detailed evidence. No source/runtime/weight changes beyond new evaluation scripts/docs. No automatic next experiment or promotion.

## Latest user preference and discussion, 2026-09-06

PPL is now primary; short answers reference-only (PC049). No new experiment or baseline accepted. See docs/NEXT_DIRECTION_PPL_ACCEPTANCE.md for proposed independent W4A16 acceptance and subsequent W4U8 reproducibility/activation attribution. Next229 remains unallocated. Original EXP0228 evidence unchanged.
