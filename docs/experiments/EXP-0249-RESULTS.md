# EXP-0249 integer attention attribution

conditional full-model software PPL with independently validated integer attention, not DSP PPL

F16 PPL 29.722077; C64 W4A16 PPL 31.288773.

| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |
|---|---:|---:|---:|
| legacy | 42.152770 | 35.216624 | 0.8355 [0.7973, 0.8748] |
| carrier | 46.875280 | 38.290143 | 0.8169 [0.7767, 0.8581] |
| score | 7705.742457 | 6517.019677 | 0.8457 [0.7657, 0.9338] |
| exponent | 7461.603531 | 6136.562387 | 0.8224 [0.7374, 0.9172] |
| sole | 11039.825162 | 9232.957784 | 0.8363 [0.7448, 0.9355] |

## Conditional increments (positive means worse)

| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |
|---|---:|---:|
| carrier | +0.106190 [+0.064703, +0.147970] | +0.083674 [+0.045128, +0.121984] |
| score | +5.102231 [+4.952995, +5.253790] | +5.136980 [+4.984090, +5.295349] |
| exponent | -0.032195 [-0.134937, +0.072408] | -0.060152 [-0.147903, +0.028324] |
| sole | +0.391739 [+0.274261, +0.508967] | +0.408515 [+0.306240, +0.517066] |

## Cells

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 29.610551 | 22.749867 | 25.713699 | 45.053309 |
| C64 | 29.296772 | 24.725480 | 27.182712 | 48.673977 |
| OFF_legacy | 41.039643 | 31.569951 | 36.994355 | 65.870584 |
| OFF_carrier | 46.183503 | 35.340996 | 41.054248 | 72.052944 |
| OFF_score | 9818.350881 | 5729.381209 | 9538.642318 | 6570.904081 |
| OFF_exponent | 8330.031892 | 5921.098326 | 9467.369415 | 6638.201276 |
| OFF_sole | 14157.187634 | 8086.325976 | 14596.032934 | 8889.671540 |
| R3_legacy | 32.151300 | 29.419871 | 29.232624 | 55.626764 |
| R3_carrier | 36.582943 | 30.117397 | 33.181838 | 58.796499 |
| R3_score | 7073.670097 | 5524.531575 | 7648.353275 | 6035.148048 |
| R3_exponent | 6919.130955 | 4552.324276 | 7416.953772 | 6070.022517 |
| R3_sole | 11228.492524 | 7972.461681 | 10577.175954 | 7675.024591 |

## Evidence and limits

All four legacy development controls reproduce EXP0246 per-token NLL and top1 exactly. Independent int64/scalar and GPU reference agree, including actual EXP0042/EXP0248 prefill QK/probability/AV captures, random causal prefill/decode, all 56 layer configurations and 384 SOLE/exact division boundary checks.
Fresh 256 documents, 64 per language/domain cell, 4096 targets. Documents, text hashes and 32-token windows excluded against historical/calibration data. Frozen configurations and independent source reconstruction precede scoring. Repeat, causal, CE, immutable weights and prefix checks pass for every mode.
Carrier includes signed K clipping, V recentering, integer AV conversion, FP32-exact integer dot products and the existing probability normalization/scale discrepancy. Later score/exponent/SOLE increments change only the named factor relative to the preceding path. They are conditional effects, not independent additive component errors.
R3 vs OFF includes previously frozen Q/K recalibration. Linear, norm and dense R3 operations retain the software arithmetic and EOS uses offline raw-prefix KV. Thus full-device accuracy and speed remain unmeasured. No final-based tuning, no new quantization range, no promotion.
The failed initial parameter-freeze assertion is retained; it incorrectly assumed every layer probability scale was 1/255. Twenty-six of 28 frozen layers have slightly clipped probability ranges. Original dataset protocol and amended configuration protocol remain separately hashed.

E2E token/s: N/A (no device/full-model timing).

## Interpretation and next direction

The large full-model conditional loss first appears when native QK score conversion is enabled while standard Softmax is retained. This isolates the score conversion stage as a critical failure, not a requirement that rotation must damage Softmax. This experiment does not separately identify the PPL contribution of clipping versus coarse rounding within that conversion.
F3 maps absolute score codes to approximately [-11.09035,11.00371] natural-logit units. Intermediate rounding before multiplier makes the effective grid coarser than the nominal ln(2)/8 step. Subtracting the maximum after saturation cannot restore differences between scores already merged at the endpoint.
The retained EXP0248 device layer0 replay has real reconstructed logits from -11.1194 to 33.1454; 13.29% of valid entries saturate and 36.43% of query rows have a maximum above the fixed upper window. These are device-capture diagnostics, not new device PPL.
Paired Q/K orthogonal R3 preserves ideal QK logits. It can improve Q/K component quantization, but cannot in general make a too-narrow absolute score window sufficient. The OFF/R3 comparison here also includes the previously frozen Q/K recalibration.
Exponents can partly compensate existing errors, so a lower conditional PPL at that stage is not evidence that its standalone approximation error vanished. SOLE and integer exponent precision must be reconsidered after repairing score range. The first carrier increment bundles signed K clipping, V recentering, probability scale semantics and AV conversion; it does not locate those losses individually.

### Paired same-input score tracing on exposed development inputs

| Variant | Layers with >50% upper-clipped row maxima | Largest mean row probability L1 |
|---|---|---|
| OFF_carrier | [6, 11, 16, 18, 19, 20, 23, 24, 25] | L11: 0.8937 |
| R3_carrier | [6, 11, 13, 16, 19, 20, 23, 24] | L11: 0.8827 |

The trace replays the first four exposed development documents on each carrier trajectory. Capturing the trace preserves all per-token NLL and top1 outputs exactly. It does not select or tune a final arm. Per-layer raw scores and masks are retained in trace/.

Proposed next work: separate QK window saturation from conversion granularity in a newly frozen software diagnostic. First test computing row maxima on raw HMX U8 scores and multiplying the differences in at least16-bit arithmetic before exponent coding, avoiding the second U8 saturation. Since abs(raw delta)*multiplier <=255*18=4590, the score-difference arithmetic fits16 bits; this is an algorithmic bound, not a demonstrated kernel speed result. Retain wider first-stage scores only if the separate raw-conversion ablation proves necessary. Establish numerical PPL recovery first, then check a realizable hardware mapping and the unchanged single-layer >10% Host-wall stop. Keep C64 W4 and other recipes frozen. After attention is stable, revisit exponent/reciprocal precision and whether plain dense HMX R3 can avoid the expensive EXP0248 refinement. These are proposed follow-ups, not implemented or promoted in EXP0249.

Bootstrap intervals are pointwise paired document intervals, stratified by the four language/domain cells; they are not adjusted for multiple comparisons.

Figures: integer_attention_diagnosis.png and .pdf. Additional local arithmetic evidence: local_softmax_precision.json and hardware_score_range.json. The historical Softmax-only comparison uses valid entries only (rather than including padded zeros) and is not a paired R3-vs-OFF experiment.
