# EXP-0323: dense fixed R3 with independently recalibrated Q/K

Host-only Qwen3-1.7B W4A8 Down16. Eight WikiText-2 training windows x2048 (16384 calibration tokens); frozen test subset8192 targets, 64 warmup and 2048 maximum window, identical to EXP0321/0322. Static per-layer/per-tensor Q/K A8 scales, symmetric z128. Dense normalized H128 after Q/K norm+RoPE, before A8. W4, Down16, V/AV, all other grids, prefix V, and output head unchanged. Prefix K transformed consistently from original decoded seed.

| Configuration | Integer core PPL | FP32 attention core PPL |
|---|---:|---:|
| Current frozen scales, no R3 |25.724948760|22.673814718|
| Recalibrated Q/K, no R3 |37.922363883|22.861365860|
| Recalibrated Q/K, dense R3 |21.780754540|20.161081667|

Matched BF16 teacher17.874597661; historical real device28.467047659 is a separate implementation tier, not a paired measurement of R3.

## Findings
R3 integer simulation improves PPL15.3322% versus current frozen integer control; all5 windows improve. Paired float-core OFF/R3 recalibration improves PPL11.8116%, all5 windows. This isolates evidence that rotated Q/K grids are useful independently of the integer core. R3 integer still21.8531% above matched BF16 teacher and above R3 FP core. No quality acceptance or device deployment claim.

The OFF/recal integer arm regresses to37.9224, whereas its FP core remains22.8614. Consequently the large difference37.9224 to21.7808 cannot be advertised as pure R3 improvement against the existing baseline. Q/K range selection uses tensor MSE and original choose_carrier multiplier/shift approximation; these objectives do not optimize downstream score rounding/clipping/softmax. Integer score coupling is a plausible explanation requiring targeted attribution, not established root cause. Recalibrated symmetric grids also differ from inherited asymmetric grids.

Calibrated Q and K sampled NMSE improve in28/28 layers; median OFF/R3 reduction5.34035x Q and16.11017x K. These are calibration statistics, not test quality metrics.

## Correctness/provenance
Unchanged integer and FP-core controls reproduce EXP0321/0322 per-token NLL exactly; all6 arms match all8192 target IDs, positions, windows and contexts. All quantities finite. Rotation orthogonality maxerror3.42285e-8 and prequant QK dot relative RMS2.32895e-7; independent Sylvester sign construction exact. New integer settings pass68 independent reference cases including long KV. Original W/package/prefix/rope/token hashes checked. Frozen calibration parameters unchanged through evaluation. No32-token shared span between selected train windows and full test corpus.

A preliminary strided calibration sampler was rejected because it aliases head coordinates. Its outputs are preserved under diagnostic-strided-sampling-not-evaluated and were never PPL-evaluated. Reported results use8192 seeded random coordinates per window/Q-or-K on a shared baseline trajectory; both OFF/R3 use same coordinates. Parameters frozen before PPL; no evaluation-based range tuning.

## Per-window PPL
| arm |w0|w1|w2|w3|w4|
|---|---:|---:|---:|---:|---:|
|control|15.890281893|27.545564934|32.665862415|27.916568379|52.785407969|
|attention_core_fp32|13.982357715|24.216803038|28.375687395|25.627211769|39.252154221|
|off_fp|14.093939635|24.610704695|28.789494355|25.256867742|42.418195694|
|r3_fp|11.998721890|21.489270724|26.376855665|23.032005247|30.470450638|
|off_int|26.176979346|37.767758292|47.743049856|40.988073138|63.597162962|
|r3_int|13.147816552|23.405184039|27.361646839|24.786790321|39.083166310|

## Next direction
Verify this fixed R3/QK-grid candidate on actual HMX stepwise (component, continuous blocks, then same full-model PPL). Keep prior candidate and static parameters frozen. Separately attribute score-carrier rounding/clipping on identical captured Q/K if pursuing further quality improvements. Do not learn rotations or alter W4/Down16 based on this result. Hardware rounding can change the outcome; no runtime/speed assertion yet.

Source unchanged486c3cc5244d6d047988cf00f521485a09f6e968; no hardware or baseline promotion. See README for reproduction and summary.json for exact results.
