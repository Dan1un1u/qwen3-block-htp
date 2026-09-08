# EXP-0252 normalization and dense R3 joint ablation

NR64 (fixed64-bin Q30 reciprocal plus one integer Newton step) approaches exact integer normalization, passes independent hardware arithmetic and no-R3 speed gates. Dense R3 whole-layer gate remains failed even with exact normalization. No promotion, slice or full-model device execution.

Source 71f8a62449226127e6befb776abc5f572f1c4a88; runtime dd96c4aa6caed2c425237f1672d7dbec72ddcda0. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0252. Frozen original per-output-channel W4 and EXP0246 calibration/prefix/attention scales unchanged. C64 names the retained W4A16 quantization reference; this is not group64 deployment.

## Independent software PPL

Fresh128documents/2048targets,32 per en/zh wiki/news cell; all8 arms frozen before scoring. Existing6 development controls reproduce EXP0250 per-token scores. PPL is conditional software arithmetic, not DSP PPL.

|Configuration|PPL|Ratio vs F16|
|---|---:|---:|
|F|32.210139|1.0000|
|C64|32.948761|1.0229|
|OFF_wide_sole|61.643355|1.9138|
|OFF_wide_exact|53.344543|1.6561|
|OFF_wide_nr64|53.747922|1.6687|
|R3_wide_sole|43.539341|1.3517|
|R3_wide_exact|42.741873|1.3270|
|R3_wide_nr64|41.725860|1.2954|
OFF NR64/SOLE PPL ratio 0.871918,95% CI [0.8178711262660174, 0.9322152204144928]. Improvement supported on this panel.
R3 NR64/SOLE PPL ratio 0.958348,95% CI [0.9054445473728077, 1.0135484848728336]. Interval includes1; PPL improvement is not statistically established on this panel.
R3 NR64 remains29.54% above matched F16. All A8 arms fail unchanged quality point gates. NR64 versus exact PPL differences have intervals including1; a lower point estimate is not evidence that reciprocal approximation improves mathematical accuracy.

Paired document NLL/PPL ratios,95%intervals and language/domain cells: SOFTWARE_REPORT.md. Candidate fixed before any final scoring; no final-data selection. Existing overall5%/eachcell10% quality gates remain unchanged.

## Same-input hardware error amplification

|Normalizer|Probability max LSB|AV max LSB|O max LSB|Residual max LSB|Final max LSB|Final cosine|
|---|---:|---:|---:|---:|---:|---:|
|control|64|35|8|8|11|0.99272220|
|exact|42|20|6|6|7|0.99432103|
|nr64|42|20|6|6|7|0.99432129|

These are actual HMX dense R3 versus independently verified Float64-dense reference, on identical layer input and each declared normalizer. Dense matrix1FP16ULP+minnormal and Q/K1code gates pass, but whole2LSB/cosine0.999 does not. Exact normalization also fails; further reciprocal refinement alone cannot remove this captured residual divergence.

Each of12 arm x9step raw-QK/probability/AV boundary sets independently matches NumPy int64 exactly. Same actual Q/K under SOLE/exact/NR64 checked; O/residual/final are actual DSP captures. NR64 vs exact prefill probability difference<=1LSB, final up to3LSB; that is an approximation diagnostic, not its implementation oracle. HVX NR64 vs independent NR64 arithmetic and untimed scalar complete-layer reference are exact. Repeated NR64 output and independently packed K cache/verified V cache pass. Old EXP0251 controls reproduce.

## Hardware speed

Five short and ten rotated paired formal rounds, repeat1/repeat10,2970timed layerRPCs. Exact division and numerically ineligible R3 arms are audit-only. Primary repeat10 median paired-round mean Host wall, paired ratios/10000bootstrap seed252. No outlier or warmup deletion; repeat1 remains separately reported.

|Scope|SOLE us|NR64 us|Paired change|Ratio95% CI|
|---|---:|---:|---:|---|
|repeat1_prefill|1689.088|1784.193|+5.67%|[1.0087536476886059, 1.1457275028482432]|
|repeat1_decode|1026.361|982.301|+0.63%|[0.9199372887066766, 1.023728823744185]|
|repeat10_prefill|1548.815|1532.357|-1.40%|[0.9636816535913649, 1.0232566887468437]|
|repeat10_decode|954.346|960.539|+0.69%|[0.9929232788322736, 1.0210908932539067]|

Primary repeat10 intervals are below1.1; speed gate passes, no extra pairs required. Both include1, so no demonstrated speedup/penalty. Repeat1 prefill uncertainty is wider and does not replace the declared repeat10 primary.8MiB requested/acquired,peak6,682,752bytes,zero timed intermediate DDR/spill/audit,oneRPC and7nativeW4projections; every additive ledger closes exactly. Constant reciprocal table256bytes, per-row LUT construction and Newton arithmetic included.

## Limits and continuation

Current runtime ABI116, layer0 M64+eightM1, capacity72. No full-model device PPL/text/E2E or baseline promotion. Conditional further integration is not entered while R3 numerical failure and model-quality acceptance remain unresolved. No costly mode5 refinement, butterfly, altered weights/activation scales or hidden scalar fallback in timing.

All execution attempts succeeded. PPL model processes loaded before later hardware/reporting-only commits; core PPL source files are byte-identical since a6a8326. Per-score heads retained and verified. Exact probability division is a reference, not an assumed efficient device candidate. Source/evidence checks in independent_integrity_checks.json, provenance in ARTIFACT_PROVENANCE.json.
