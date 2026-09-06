# EXP0231 completed

Source b5fd32c8123fae11abb7b472ebcd20c1e9f8a500 on codex/exp-0231-w4f16-group128-software, clean/pushed. No running jobs. Do not rerun export/evaluations/report. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0231; ledger 9dd60d4a855f56499da2d284898e88c5de376d4b680403613b0f1d800ff2333d.

## EXP0231 completed group128 software diagnostic

1024newdocuments/16384targets: F23.961270,C827.644064,C6426.085709,G12826.541898. G128/F+10.770% (95%CI+9.141..12.506%) fails fixed acceptance. Matched8K calibration G128/C8 improves3.987% (ratioCI.944690..974583), but G128 is1.749% worse than stronger64K C64 (CI1.002730..1.032126). Grouping helps at matched calibration but does not beat the best current per-channel result; this does not test group128 at64K. All129result files/624artifacts verified,196packing/112forwardchecks pass,28hidden checkpoints finite, allF/C8/C64 development outputs exact. Runtime/other recipes unchanged, groupDSPspeed N/A, no promotion. Allnewprimary/reserve data exposed and excluded from future training/independent tests. Active none,next232. PC051 now activates separately registered AWQ-style input-channel equalization; prospective draft uses strongerC64 control.

Full docs/experiments/EXP-0231-RESULTS.md and PROFILE.md. Conditional AWQ draft docs/AWQ_CONDITIONAL_DRAFT.md; official main pin read-only resolved d6e797a42b9ef7778de8ee2352116e0f48a78d61 (not yet downloaded). Register before new source/model work.
