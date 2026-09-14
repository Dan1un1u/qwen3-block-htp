# L32-0021 completed optimization / chain3 numerical stop
No active experiment. Source closure `4ff8ec9ea24a9511b602cb1d9f8c364da76508b4`; single-layer profile/native
`5adc46b774a23df9138f1070560fb68a28c165a1`. Three-layer source d77364464d9c73910b3d074b0da6ac4308fbe1ba.
Read docs/LLAMA32_ROTATION_PIPELINE_FOLLOWUP.md and docs/experiments/L32-0021.md.
Retain candidate3: dense H16 operand orientation produces contiguous SP2
consumer tiles; two-worker finish/caller next-layout overlap,8-row LUT pipeline,
paired token-load reuse, vector H16 constants. FP16/SP2/FP32 math unchanged.
Final code consumes VTCM directly; intermediate vector array stack staging found
in superseded candidates was removed.20 affected R4/both runs retained but not
physical/performance acceptance. Final assembly has scalar stack slots only.
All12 isolated layer0/7/15 OFF/R3/R4/both numeric and physical checks pass;
exact0020 output hashes preserved. Fixed10 cyclic five-arm warmed comparison:
OFF1959.78/1464.19us, R31899.49/1400.94, R42159.80/1507.40,
both2090.70/1497.78, sealed0020 both2159.67/1469.60 (prefill/decode).
Combined vsOFF +6.68%/+2.29%, CI upper8.69%/5.67% PASS.
Paired vs oldboth prefill -3.19% (CIratio.950082-.987010); decode inconclusive.
R4-alone prefill+10.21%,CIupper12.29% FAIL. No gate weakening/repeated formal run.
Extended combination to3 layers with fresh original-BF16 Down folds for1/2,
same frozen8train-window method, existing verified layer0 fold. OFFchain3 exact.
Combinedchain3 independent min-rowcos .791505/.911935 FAIL despite finite
execution,8MiB, correctcache structure/prefix. Cache value mismatches retained.
Independent software with only actual audited layer0 substituted already
reproduces large deviation (layer2mincos .803272/.908457). This supports rounding
amplification; it is not a complete conditional native oracle or proof that every
remaining difference is explained. Do not mislabel this as introduced speed
optimization error: isolated outputs match old code, and software reproduces much
of the problem without cross-layer hardware scheduling.
Next discuss locating sensitivity at next-layer Norm/A8,QKV/attention/FFN boundaries.
Keep optimized pipeline fixed. No full16/frontend/E2E/PPL; no default/quality promotion.
Ledger `/mnt/d/llm_exp/results/llama32-htp/l32-0021/evidence-ledger-checkpoint-a01.json`
SHA256 `fd22398af6ddf91ce6fcf79a70601364b1f1bc59e173a9f49bec948a8ea2b1b9`;822files,17model manifests,111CLI,1242boundaries.
Prior0020 evidence reverified unchanged. Qwen and other Llama branch frozen.
