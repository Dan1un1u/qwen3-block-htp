# EXP0255 running handoff

Authority PC069. Source codex/exp-0255-residual-calibration at6c53dc2743681a2dea05c6e93d32e513befd8d9a, clean/pushed. Source implementation b8330a3; following commit adds reporting/audit only.

New256document development/final panels prepared and independently audited. Dataset freeze SHA256 08079800d96284f6e6f6b48f1aa76cfe2d01920b06ee7d1c449b925b1d80ae38. All128 calibration documents collected on actual R3 wide_nr64 A8 trajectory, first79 tokens. Fixed vector fit completed: R_MSE changes55residual sites, R_FAN55, S_MSE2. Parameters freeze SHA256 6b5e874e7746cc134e645289dd591f856335fb35a023c69f88f03510034708bc. Independent NumPy objective/FP16 U8 rounding checks184cases passed. No candidate PPL conclusion yet.

Long runner started numerical -> reproduction -> development -> final. Inspect live processes ps and logs before resuming; never duplicate or overwrite. Current root tool session19428 is observational, process authoritative. Main script scripts/residual_exp0255.py phases reproduction/development/final resumes per-score files and completed weight proofs. Parameters/data/calibration are frozen; do not rerun prepare/freeze/calibrate/numerical over existing evidence.

Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0255. Logs numerical.log,reproduction.log,development.log,final.log. Four historical controls must reproduce sealed EXP0253. Eight new development arms fixed; selection.json automatically freezes minimum worst-mode deltaNLL among B/R_MSE/R_FAN before final. Final F/C64/B/selected/S_MSE two modes. All original gates unchanged. No hardware work this phase; pending old denseR3 whole-layer correctness unchanged.

After runner passes, run scripts/report_exp0255.py (independent audit plus paired bootstrap report). Then inspect conclusions, add NEXT_DIRECTION/ARTIFACT_PROVENANCE, seal all evidence, close memory index/status/CONTEXT, commitpush source/memory, bootstrap. Do not claim completion before this closure. No promotion.

## Development complete; final running

Four historical controls exact; eight development scores complete. R_MSE float28.481816->29.575748 but integer31.371519->29.523674; R_FAN float27.924172 but integer32.227583. Fixed worst-mode rule selects B (original parameters), selection SHA256 b225f7ca5c4d7ac4a568efd943cb266fd8745177f3d687716043ecb7c07fb531. Do not alter selection or add unselected residual candidates to current final. Final fixed six unique arms F,C64,B_float/int,S_MSE_float/int started with successful preflight. Need complete final, audit/report/seal/memory closure. Residual integer MSE improvement is development-only, not independently confirmed; discuss future integer-primary independent validation after closure. Postfit calibration-only analysis: R_MSE/R_FAN scales+inverses identical all56sites;43zeros differ;10074/117440512 decoded values differ, all outside common quantizer interval. Evidence calibration_zero_point_attribution.json; no additional fitting/scoring.
